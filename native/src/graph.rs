//! Reverse-mode autodiff graph.
//!
//! Nodes are appended in execution order, but backward does not rely on that: it collects the
//! sub-graph reachable from the root and runs Kahn's algorithm, so a future scheduler may append
//! nodes from several queues without silently corrupting the gradient order.

use crate::error::{NativeError, Result};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    MatMul,
    Add,
    Sub,
    Mul,
    ReLU,
    Sigmoid,
    Tanh,
    Mse,
    Softmax,
    CrossEntropy,
    Gelu,
    Silu,
    LayerNorm,
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    Dropout,
    Embedding,
    BatchedMatMul,
    Permute,
    Reshape,
    GroupNorm,
}

impl Op {
    pub fn from_code(code: u8) -> Result<Self> {
        match code {
            1 => Ok(Self::MatMul),
            2 => Ok(Self::Add),
            3 => Ok(Self::ReLU),
            4 => Ok(Self::Sub),
            5 => Ok(Self::Mul),
            6 => Ok(Self::Sigmoid),
            7 => Ok(Self::Tanh),
            8 => Ok(Self::Mse),
            9 => Ok(Self::Softmax),
            10 => Ok(Self::CrossEntropy),
            11 => Ok(Self::Gelu),
            12 => Ok(Self::Silu),
            13 => Ok(Self::LayerNorm),
            14 => Ok(Self::BatchNorm2d),
            15 => Ok(Self::Conv2d),
            16 => Ok(Self::MaxPool2d),
            17 => Ok(Self::Dropout),
            18 => Ok(Self::Embedding),
            19 => Ok(Self::BatchedMatMul),
            20 => Ok(Self::Permute),
            21 => Ok(Self::Reshape),
            22 => Ok(Self::GroupNorm),
            other => Err(NativeError::unsupported(format!(
                "unknown op code {other}; expected 1..=22"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::MatMul => "MatMul",
            Self::Add => "Add",
            Self::Sub => "Sub",
            Self::Mul => "Mul",
            Self::ReLU => "ReLU",
            Self::Sigmoid => "Sigmoid",
            Self::Tanh => "Tanh",
            Self::Mse => "Mse",
            Self::Softmax => "Softmax",
            Self::CrossEntropy => "CrossEntropy",
            Self::Gelu => "Gelu",
            Self::Silu => "Silu",
            Self::LayerNorm => "LayerNorm",
            Self::BatchNorm2d => "BatchNorm2d",
            Self::Conv2d => "Conv2d",
            Self::MaxPool2d => "MaxPool2d",
            Self::Dropout => "Dropout",
            Self::Embedding => "Embedding",
            Self::BatchedMatMul => "BatchedMatMul",
            Self::Permute => "Permute",
            Self::Reshape => "Reshape",
            Self::GroupNorm => "GroupNorm",
        }
    }

    pub fn arity(self) -> usize {
        match self {
            Self::ReLU
            | Self::Sigmoid
            | Self::Tanh
            | Self::Gelu
            | Self::Silu
            | Self::Softmax
            | Self::MaxPool2d
            | Self::Dropout
            | Self::Permute
            | Self::Reshape => 1,
            Self::MatMul
            | Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Mse
            | Self::CrossEntropy
            | Self::Conv2d
            | Self::Embedding
            | Self::BatchedMatMul => 2,
            Self::LayerNorm | Self::GroupNorm => 3,
            Self::BatchNorm2d => 5,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Attrs {
    None,
    MatMul {
        m: usize,
        k: usize,
        n: usize,
    },
    Broadcast {
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
    },
    Softmax {
        rows: usize,
        dim: usize,
    },
    CrossEntropy {
        rows: usize,
        classes: usize,
    },
    LayerNorm {
        rows: usize,
        dim: usize,
        eps: f32,
    },
    BatchNorm {
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        eps: f32,
        #[allow(dead_code)]
        momentum: f32,
        training: bool,
    },
    Conv2d {
        n: usize,
        cin: usize,
        h: usize,
        w: usize,
        cout: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dw: usize,
        groups: usize,
        oh: usize,
        ow: usize,
    },
    GroupNorm {
        n: usize,
        c: usize,
        spatial: usize,
        groups: usize,
        eps: f32,
    },
    MaxPool2d {
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        oh: usize,
        ow: usize,
    },
    Dropout {
        #[allow(dead_code)]
        length: usize,
        probability: f32,
        training: bool,
        seed: u64,
    },
    Embedding {
        rows: usize,
        dim: usize,
    },
    BatchedMatMul {
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    },
    Permute {
        in_shape: Vec<usize>,
        perm: Vec<usize>,
    },
    Reshape {
        in_shape: Vec<usize>,
        #[allow(dead_code)]
        out_shape: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<usize>,
    pub output: usize,
    pub attrs: Attrs,
}

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
    produced: HashMap<usize, usize>,
    consumed: HashSet<usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    /// True when the address participates in the recorded graph as an input or an output.
    /// Releasing such a tensor invalidates the graph, so `Context::release_tensor` clears it.
    pub fn uses(&self, address: usize) -> bool {
        self.produced.contains_key(&address) || self.consumed.contains(&address)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.produced.clear();
        self.consumed.clear();
    }

    /// Validates a node before any output bytes are written. This is deliberately separate from
    /// `push` so a rejected op cannot leave a half-updated output tensor behind.
    pub fn check_push(&self, op: Op, inputs: &[usize], output: usize) -> Result<()> {
        if inputs.len() != op.arity() {
            return Err(NativeError::graph(format!(
                "{} expects {} inputs, received {}",
                op.name(),
                op.arity(),
                inputs.len()
            )));
        }
        if inputs.iter().any(|&input| input == output) {
            return Err(NativeError::graph(format!(
                "{} cannot write into one of its inputs (in-place ops are not differentiable)",
                op.name()
            )));
        }
        if self.produced.contains_key(&output) {
            return Err(NativeError::graph(format!(
                "tensor 0x{output:x} is already produced by the current graph; call clearGraph \
                 between optimisation steps"
            )));
        }
        if self.consumed.contains(&output) {
            return Err(NativeError::graph(format!(
                "tensor 0x{output:x} was already consumed as an input; overwriting it would \
                 break the recorded graph"
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn push(&mut self, node: Node) -> Result<()> {
        self.check_push(node.op, &node.inputs, node.output)?;
        self.push_unchecked(node);
        Ok(())
    }

    /// Caller must have run `check_push` for the same node.
    pub fn push_unchecked(&mut self, node: Node) {
        let index = self.nodes.len();
        self.produced.insert(node.output, index);
        for input in &node.inputs {
            self.consumed.insert(*input);
        }
        self.nodes.push(node);
    }

    /// Kahn topological sort of the sub-graph that actually contributes to `root`.
    /// The returned order has producers before consumers.
    pub fn topological_order(&self, root: usize) -> Result<Vec<usize>> {
        let Some(&root_index) = self.produced.get(&root) else {
            return Err(NativeError::graph(format!(
                "backward root 0x{root:x} was not produced by the current graph"
            )));
        };

        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = vec![root_index];
        while let Some(index) = stack.pop() {
            if reachable[index] {
                continue;
            }
            reachable[index] = true;
            for input in &self.nodes[index].inputs {
                if let Some(&producer) = self.produced.get(input) {
                    if !reachable[producer] {
                        stack.push(producer);
                    }
                }
            }
        }

        let mut indegree = vec![0usize; self.nodes.len()];
        let mut consumers: HashMap<usize, Vec<usize>> = HashMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if !reachable[index] {
                continue;
            }
            for input in &node.inputs {
                if let Some(&producer) = self.produced.get(input) {
                    if reachable[producer] {
                        indegree[index] += 1;
                        consumers.entry(producer).or_default().push(index);
                    }
                }
            }
        }

        let mut queue: VecDeque<usize> = (0..self.nodes.len())
            .filter(|&index| reachable[index] && indegree[index] == 0)
            .collect();
        let mut order = Vec::with_capacity(reachable.iter().filter(|&&value| value).count());
        while let Some(index) = queue.pop_front() {
            order.push(index);
            if let Some(next) = consumers.get(&index) {
                for &consumer in next {
                    indegree[consumer] -= 1;
                    if indegree[consumer] == 0 {
                        queue.push_back(consumer);
                    }
                }
            }
        }

        let expected = reachable.iter().filter(|&&value| value).count();
        if order.len() != expected {
            return Err(NativeError::graph(
                "cycle detected while topologically sorting the autodiff graph",
            ));
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: Op, inputs: Vec<usize>, output: usize) -> Node {
        Node {
            op,
            inputs,
            output,
            attrs: Attrs::None,
        }
    }

    #[test]
    fn topological_order_puts_inputs_first() {
        let mut graph = Graph::new();
        graph.push(node(Op::ReLU, vec![10], 11)).unwrap();
        graph.push(node(Op::Add, vec![11, 12], 13)).unwrap();
        graph.push(node(Op::Mul, vec![13, 10], 14)).unwrap();
        let order = graph.topological_order(14).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn ignores_nodes_outside_the_root_subgraph() {
        let mut graph = Graph::new();
        graph.push(node(Op::ReLU, vec![1], 2)).unwrap();
        graph.push(node(Op::ReLU, vec![3], 4)).unwrap();
        let order = graph.topological_order(4).unwrap();
        assert_eq!(order, vec![1]);
    }

    #[test]
    fn rejects_overwrites_and_in_place_ops() {
        let mut graph = Graph::new();
        graph.push(node(Op::ReLU, vec![1], 2)).unwrap();
        assert!(graph.push(node(Op::ReLU, vec![2], 3)).is_ok());
        assert!(graph.push(node(Op::ReLU, vec![3], 2)).is_err());
        assert!(graph.push(node(Op::Add, vec![4, 4], 4)).is_err());
    }

    #[test]
    fn rejects_root_without_producer() {
        let graph = Graph::new();
        assert!(graph.topological_order(99).is_err());
    }
}
