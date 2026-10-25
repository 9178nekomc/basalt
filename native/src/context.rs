//! Per-context ownership of tensors, the autodiff graph, rayon pools and optimizer state.
//!
//! A `Context` is created by Java and returned as an `Arc` raw pointer. Nothing in this module is
//! process-global: two contexts can run concurrently with independent graphs and thread pools.

use crate::error::{NativeError, Result};
use crate::graph::Graph;
use crate::ops;
use crate::tensor::{NativeTensor, TensorStore};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

#[derive(Default)]
struct AdamState {
    first: Vec<f32>,
    second: Vec<f32>,
}

pub struct Context {
    pub tensors: RwLock<TensorStore>,
    pub graph: Mutex<Graph>,
    pools: Mutex<HashMap<usize, Arc<ThreadPool>>>,
    optimizers: Mutex<HashMap<usize, AdamState>>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            tensors: RwLock::new(TensorStore::new()),
            graph: Mutex::new(Graph::new()),
            pools: Mutex::new(HashMap::new()),
            optimizers: Mutex::new(HashMap::new()),
        }
    }

    pub fn create_tensor(&self, shape: Vec<usize>, requires_grad: bool) -> Result<usize> {
        let tensor = NativeTensor::new(shape, requires_grad)?;
        self.tensors
            .write()
            .map_err(|_| NativeError::invalid("tensor store lock poisoned"))?
            .insert(tensor)
    }

    pub fn create_tensor_from_data(
        &self,
        shape: Vec<usize>,
        data: Vec<f32>,
        requires_grad: bool,
    ) -> Result<usize> {
        let tensor = NativeTensor::from_data(shape, data, requires_grad)?;
        self.tensors
            .write()
            .map_err(|_| NativeError::invalid("tensor store lock poisoned"))?
            .insert(tensor)
    }

    pub fn tensor(&self, address: usize) -> Result<Arc<RwLock<NativeTensor>>> {
        self.tensors
            .read()
            .map_err(|_| NativeError::invalid("tensor store lock poisoned"))?
            .get(address)
            .ok_or_else(|| NativeError::unknown_tensor(address))
    }

    pub fn release_tensor(&self, address: usize) -> Result<bool> {
        // Lock order is graph -> tensor store, matching queue execution and backward. Releasing a
        // tensor that appears in the graph invalidates the whole graph; clearing it here keeps a
        // later allocation from reusing the address under a stale node.
        {
            let mut graph = self
                .graph
                .lock()
                .map_err(|_| NativeError::invalid("graph lock poisoned"))?;
            if graph.uses(address) {
                graph.clear();
            }
        }
        let removed = self
            .tensors
            .write()
            .map_err(|_| NativeError::invalid("tensor store lock poisoned"))?
            .remove(address);
        if removed {
            if let Ok(mut optimizers) = self.optimizers.lock() {
                optimizers.remove(&address);
            }
        }
        Ok(removed)
    }

    pub fn clear_graph(&self) -> Result<()> {
        self.graph
            .lock()
            .map_err(|_| NativeError::invalid("graph lock poisoned"))?
            .clear();
        Ok(())
    }

    pub fn graph_len(&self) -> Result<usize> {
        Ok(self
            .graph
            .lock()
            .map_err(|_| NativeError::invalid("graph lock poisoned"))?
            .len())
    }

    pub fn pool_for(&self, threads: usize) -> Result<Arc<ThreadPool>> {
        let requested = if threads == 0 {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        } else {
            threads
        }
        .clamp(1, 256);
        let mut pools = self
            .pools
            .lock()
            .map_err(|_| NativeError::invalid("thread-pool registry lock poisoned"))?;
        if let Some(pool) = pools.get(&requested) {
            return Ok(Arc::clone(pool));
        }
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(requested)
                .thread_name(move |index| format!("basalt-tensor-{requested}-{index}"))
                .build()
                .map_err(|error| {
                    NativeError::invalid(format!("cannot create native thread pool: {error}"))
                })?,
        );
        pools.insert(requested, Arc::clone(&pool));
        Ok(pool)
    }

    pub fn apply_adam(
        &self,
        address: usize,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        step: u64,
    ) -> Result<()> {
        let tensor = self.tensor(address)?;
        let mut guard = tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        guard.ensure_grad();
        let length = guard.data.len();
        let mut optimizers = self
            .optimizers
            .lock()
            .map_err(|_| NativeError::invalid("optimizer state lock poisoned"))?;
        let state = optimizers.entry(address).or_insert_with(|| AdamState {
            first: vec![0.0; length],
            second: vec![0.0; length],
        });
        if state.first.len() != length {
            state.first = vec![0.0; length];
            state.second = vec![0.0; length];
        }
        let NativeTensor { data, grad, .. } = &mut *guard;
        ops::adam_step(
            data,
            grad,
            &mut state.first,
            &mut state.second,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
        );
        Ok(())
    }

    pub fn apply_sgd(
        &self,
        address: usize,
        learning_rate: f32,
        weight_decay: f32,
    ) -> Result<()> {
        let tensor = self.tensor(address)?;
        let mut guard = tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        guard.ensure_grad();
        let NativeTensor { data, grad, .. } = &mut *guard;
        ops::sgd_step(data, grad, learning_rate, weight_decay);
        Ok(())
    }

    pub fn fill(&self, address: usize, value: f32) -> Result<()> {
        if !value.is_finite() {
            return Err(NativeError::invalid("fill value must be finite"));
        }
        let tensor = self.tensor(address)?;
        let mut guard = tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        guard.data.fill(value);
        Ok(())
    }

    pub fn randomize(&self, address: usize, seed: u64, scale: f32) -> Result<()> {
        if !scale.is_finite() || scale < 0.0 {
            return Err(NativeError::invalid("random scale must be finite and non-negative"));
        }
        let tensor = self.tensor(address)?;
        let mut guard = tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        let mut state = seed | 1;
        for value in &mut guard.data {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
            *value = (unit * 2.0 - 1.0) * scale;
        }
        Ok(())
    }

    pub fn zero_grad(&self, address: usize) -> Result<()> {
        let tensor = self.tensor(address)?;
        let mut guard = tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        guard.zero_grad();
        Ok(())
    }

    /// Global L2 gradient clipping across a parameter set. Returns the pre-clip norm.
    pub fn clip_grad_norm(&self, addresses: &[usize], max_norm: f32) -> Result<f32> {
        if !max_norm.is_finite() || max_norm < 0.0 {
            return Err(NativeError::invalid(
                "gradient clipping norm must be finite and non-negative",
            ));
        }
        let mut tensors = Vec::with_capacity(addresses.len());
        let mut total = 0.0f64;
        for &address in addresses {
            let tensor = self.tensor(address)?;
            {
                let guard = tensor
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                for value in &guard.grad {
                    total += (*value as f64) * (*value as f64);
                }
            }
            tensors.push(tensor);
        }
        let norm = total.sqrt() as f32;
        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / (norm + 1.0e-6);
            for tensor in tensors {
                let mut guard = tensor
                    .write()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                for value in &mut guard.grad {
                    *value *= scale;
                }
            }
        }
        Ok(norm)
    }

    pub fn stats(&self) -> Result<(usize, usize, usize)> {
        let tensors = self
            .tensors
            .read()
            .map_err(|_| NativeError::invalid("tensor store lock poisoned"))?;
        let graph = self
            .graph
            .lock()
            .map_err(|_| NativeError::invalid("graph lock poisoned"))?;
        Ok((tensors.len(), tensors.live_bytes(), graph.len()))
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Ok(mut graph) = self.graph.lock() {
            graph.clear();
        }
        if let Ok(mut optimizers) = self.optimizers.lock() {
            optimizers.clear();
        }
        // TensorStore::drop releases the raw Arc reference for every remaining slot.
    }
}
