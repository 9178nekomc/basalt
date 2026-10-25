//! Command execution and reverse-mode autodiff for the full operator set.
//!
//! The wire command is 128 bytes so convolution, pooling, normalization and batchnorm can carry
//! their shape parameters inline without a second allocation or a pointer chase.

use crate::context::Context;
use crate::error::{NativeError, Result};
use crate::graph::{Attrs, Graph, Node, Op};
use crate::nn_ops;
use crate::ops;
use crate::tensor::NativeTensor;
use jni::sys::{jint, jlong};
use rayon::ThreadPool;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OpCommand {
    pub op_code: u8,
    pub flags: u8,
    pub _padding: [u8; 6],
    pub input_a: usize,
    pub input_b: usize,
    pub input_c: usize,
    pub input_d: usize,
    pub input_e: usize,
    pub output: usize,
    pub attrs: [i32; 16],
    pub _reserved: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<OpCommand>() == 128);

fn attr_f32(command: &OpCommand, index: usize) -> f32 {
    f32::from_bits(command.attrs[index] as u32)
}

fn attr_u64(command: &OpCommand, low: usize, high: usize) -> u64 {
    (command.attrs[low] as u32 as u64) | ((command.attrs[high] as u32 as u64) << 32)
}

fn input_addresses(command: &OpCommand, arity: usize) -> Result<Vec<usize>> {
    let candidates = [
        command.input_a,
        command.input_b,
        command.input_c,
        command.input_d,
        command.input_e,
    ];
    let mut inputs = Vec::with_capacity(arity);
    for (index, address) in candidates.iter().take(arity).enumerate() {
        if *address == 0 {
            return Err(NativeError::invalid(format!(
                "input {} of the command is null",
                index
            )));
        }
        inputs.push(*address);
    }
    Ok(inputs)
}

fn tensor_shape_of(tensor: &Arc<RwLock<NativeTensor>>) -> Result<Vec<usize>> {
    Ok(tensor
        .read()
        .map_err(|_| NativeError::invalid("tensor lock poisoned"))?
        .shape
        .clone())
}

fn tensor_len_of(tensor: &Arc<RwLock<NativeTensor>>) -> Result<usize> {
    Ok(tensor
        .read()
        .map_err(|_| NativeError::invalid("tensor lock poisoned"))?
        .data
        .len())
}

fn accumulate(map: &mut HashMap<usize, Vec<f32>>, address: usize, values: Vec<f32>) {
    match map.get_mut(&address) {
        Some(existing) => {
            for (slot, value) in existing.iter_mut().zip(values) {
                *slot += value;
            }
        }
        None => {
            map.insert(address, values);
        }
    }
}

pub fn execute_queue(
    context: &Context,
    graph: &mut Graph,
    address: *const u8,
    count: usize,
    pool: &ThreadPool,
) -> Result<()> {
    for index in 0..count {
        // SAFETY: the caller checked `count * 128 <= direct-buffer capacity`; read_unaligned
        // accepts any buffer alignment.
        let command = unsafe {
            (address.add(index * std::mem::size_of::<OpCommand>()) as *const OpCommand)
                .read_unaligned()
        };
        execute_command(context, graph, command, pool)?;
    }
    Ok(())
}

fn execute_command(
    context: &Context,
    graph: &mut Graph,
    command: OpCommand,
    pool: &ThreadPool,
) -> Result<()> {
    let op = Op::from_code(command.op_code)?;
    if command.output == 0 {
        return Err(NativeError::invalid("op output handle must not be null"));
    }
    let inputs = input_addresses(&command, op.arity())?;
    graph.check_push(op, &inputs, command.output)?;

    let output = context.tensor(command.output)?;
    let output_shape = tensor_shape_of(&output)?;
    let output_len = tensor_len_of(&output)?;

    match op {
        Op::MatMul => {
            let left = context.tensor(command.input_a)?;
            let right = context.tensor(command.input_b)?;
            let left_shape = tensor_shape_of(&left)?;
            let right_shape = tensor_shape_of(&right)?;
            if left_shape.len() != 2 || right_shape.len() != 2 {
                return Err(NativeError::shape("MatMul expects rank-2 operands"));
            }
            let (m, k) = (left_shape[0], left_shape[1]);
            let (k2, n) = (right_shape[0], right_shape[1]);
            if k != k2 || output_shape != vec![m, n] || output_len != m * n {
                return Err(NativeError::shape(format!(
                    "MatMul shape mismatch: {:?} x {:?} -> {:?}",
                    left_shape, right_shape, output_shape
                )));
            }
            let left_guard = left
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let right_guard = right
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            ops::gemm_f32(
                &left_guard.data,
                &right_guard.data,
                &mut output_guard.data,
                m,
                k,
                n,
                Some(pool),
            );
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::MatMul { m, k, n },
            });
        }
        Op::Add | Op::Sub | Op::Mul => {
            let left = context.tensor(command.input_a)?;
            let right = context.tensor(command.input_b)?;
            let left_shape = tensor_shape_of(&left)?;
            let right_shape = tensor_shape_of(&right)?;
            let broadcast = ops::broadcast_shape(&left_shape, &right_shape)?;
            if output_shape != broadcast || output_len != broadcast.iter().product::<usize>() {
                return Err(NativeError::shape(format!(
                    "{} output shape must be {:?}",
                    op.name(),
                    broadcast
                )));
            }
            let left_guard = left
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let right_guard = right
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            ops::binary_forward(
                op,
                &left_guard.data,
                &left_shape,
                &right_guard.data,
                &right_shape,
                &mut output_guard.data,
                &broadcast,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Broadcast {
                    a_shape: left_shape,
                    b_shape: right_shape,
                },
            });
        }
        Op::ReLU | Op::Sigmoid | Op::Tanh => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            if output_shape != input_shape || output_len != input_shape.iter().product::<usize>() {
                return Err(NativeError::shape(format!(
                    "{} output shape must be {:?}",
                    op.name(),
                    input_shape
                )));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            ops::unary_forward(op, &input_guard.data, &mut output_guard.data)?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::None,
            });
        }
        Op::Gelu | Op::Silu => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            if output_shape != input_shape || output_len != input_shape.iter().product::<usize>() {
                return Err(NativeError::shape("activation output shape mismatch"));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            if op == Op::Gelu {
                nn_ops::gelu_forward(&input_guard.data, &mut output_guard.data);
            } else {
                nn_ops::silu_forward(&input_guard.data, &mut output_guard.data);
            }
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::None,
            });
        }
        Op::Mse => {
            let prediction = context.tensor(command.input_a)?;
            let target = context.tensor(command.input_b)?;
            if tensor_shape_of(&prediction)? != tensor_shape_of(&target)? {
                return Err(NativeError::shape("MSE expects equal shapes"));
            }
            if output_shape != vec![1] || output_len != 1 {
                return Err(NativeError::shape("MSE output must be scalar"));
            }
            let prediction_guard = prediction
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let target_guard = target
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            ops::mse_forward(
                &prediction_guard.data,
                &target_guard.data,
                &mut output_guard.data,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::None,
            });
        }
        Op::Softmax => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            let dim = command.attrs[0].max(0) as usize;
            if dim == 0 || input_shape.last().copied() != Some(dim) || output_shape != input_shape {
                return Err(NativeError::shape("softmax expects the last dimension to match"));
            }
            let rows = output_len / dim;
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::softmax_forward(&input_guard.data, &mut output_guard.data, rows, dim)?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Softmax { rows, dim },
            });
        }
        Op::CrossEntropy => {
            let logits = context.tensor(command.input_a)?;
            let targets = context.tensor(command.input_b)?;
            let logits_shape = tensor_shape_of(&logits)?;
            if logits_shape.len() != 2 {
                return Err(NativeError::shape("cross-entropy expects rank-2 logits"));
            }
            let (rows, classes) = (logits_shape[0], logits_shape[1]);
            let target_len = tensor_len_of(&targets)?;
            if target_len != rows || output_shape != vec![1] || output_len != 1 {
                return Err(NativeError::shape("cross-entropy target/output shape mismatch"));
            }
            let logits_guard = logits
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let target_guard = targets
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::cross_entropy_forward(
                &logits_guard.data,
                &target_guard.data,
                &mut output_guard.data,
                rows,
                classes,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::CrossEntropy { rows, classes },
            });
        }
        Op::LayerNorm => {
            let input = context.tensor(command.input_a)?;
            let gamma = context.tensor(command.input_b)?;
            let beta = context.tensor(command.input_c)?;
            let input_shape = tensor_shape_of(&input)?;
            let dim = input_shape.last().copied().unwrap_or(0);
            if dim == 0
                || output_shape != input_shape
                || tensor_shape_of(&gamma)? != vec![dim]
                || tensor_shape_of(&beta)? != vec![dim]
            {
                return Err(NativeError::shape("layer-norm shape mismatch"));
            }
            let rows = output_len / dim;
            let eps = attr_f32(&command, 0);
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let gamma_guard = gamma
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let beta_guard = beta
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::layer_norm_forward(
                &input_guard.data,
                &gamma_guard.data,
                &beta_guard.data,
                &mut output_guard.data,
                rows,
                dim,
                eps,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::LayerNorm { rows, dim, eps },
            });
        }
        Op::BatchNorm2d => {
            let input = context.tensor(command.input_a)?;
            let gamma = context.tensor(command.input_b)?;
            let beta = context.tensor(command.input_c)?;
            let running_mean = context.tensor(command.input_d)?;
            let running_var = context.tensor(command.input_e)?;
            let input_shape = tensor_shape_of(&input)?;
            if input_shape.len() != 4 {
                return Err(NativeError::shape("batch-norm expects NCHW input"));
            }
            let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
            if output_shape != input_shape
                || tensor_shape_of(&gamma)? != vec![c]
                || tensor_shape_of(&beta)? != vec![c]
                || tensor_shape_of(&running_mean)? != vec![c]
                || tensor_shape_of(&running_var)? != vec![c]
            {
                return Err(NativeError::shape("batch-norm parameter shape mismatch"));
            }
            let eps = attr_f32(&command, 0);
            let momentum = attr_f32(&command, 1);
            let training = command.attrs[2] != 0;
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let gamma_guard = gamma
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let beta_guard = beta
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut mean_guard = running_mean
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut var_guard = running_var
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::batch_norm_forward(
                &input_guard.data,
                &gamma_guard.data,
                &beta_guard.data,
                &mut mean_guard.data,
                &mut var_guard.data,
                &mut output_guard.data,
                n,
                c,
                h,
                w,
                eps,
                momentum,
                training,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::BatchNorm {
                    n,
                    c,
                    h,
                    w,
                    eps,
                    momentum,
                    training,
                },
            });
        }
        Op::Conv2d => {
            let input = context.tensor(command.input_a)?;
            let weight = context.tensor(command.input_b)?;
            let input_shape = tensor_shape_of(&input)?;
            let weight_shape = tensor_shape_of(&weight)?;
            if input_shape.len() != 4 || weight_shape.len() != 4 {
                return Err(NativeError::shape("conv2d expects NCHW input and OIHW weight"));
            }
            let (n, cin, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
            let (cout, weight_cin, kh, kw) = (
                weight_shape[0],
                weight_shape[1],
                weight_shape[2],
                weight_shape[3],
            );
            let sh = command.attrs[2].max(1) as usize;
            let sw = command.attrs[3].max(1) as usize;
            let ph = command.attrs[4].max(0) as usize;
            let pw = command.attrs[5].max(0) as usize;
            let dh = command.attrs[6].max(1) as usize;
            let dw = command.attrs[7].max(1) as usize;
            let groups = command.attrs[8].max(1) as usize;
            if cin % groups != 0 || cout % groups != 0 || cin / groups != weight_cin || kh == 0 || kw == 0 {
                return Err(NativeError::shape("conv2d channel/group/kernel mismatch"));
            }
            let eff_kh = dh * (kh - 1) + 1;
            let eff_kw = dw * (kw - 1) + 1;
            let oh = (h + 2 * ph).checked_sub(eff_kh).ok_or_else(|| {
                NativeError::shape("conv2d dilated kernel is larger than the padded input")
            })? / sh
                + 1;
            let ow = (w + 2 * pw).checked_sub(eff_kw).ok_or_else(|| {
                NativeError::shape("conv2d dilated kernel is larger than the padded input")
            })? / sw
                + 1;
            if output_shape != vec![n, cout, oh, ow] || output_len != n * cout * oh * ow {
                return Err(NativeError::shape(format!(
                    "conv2d output must be {:?}",
                    vec![n, cout, oh, ow]
                )));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let weight_guard = weight
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::conv2d_forward(
                &input_guard.data,
                &weight_guard.data,
                &mut output_guard.data,
                n,
                cin,
                h,
                w,
                cout,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
                groups,
                oh,
                ow,
                Some(pool),
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Conv2d {
                    n,
                    cin,
                    h,
                    w,
                    cout,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    oh,
                    ow,
                },
            });
        }
        Op::GroupNorm => {
            let input = context.tensor(command.input_a)?;
            let gamma = context.tensor(command.input_b)?;
            let beta = context.tensor(command.input_c)?;
            let input_shape = tensor_shape_of(&input)?;
            if input_shape.len() < 2 {
                return Err(NativeError::shape("group-norm expects rank >= 2 input [N, C, ...]"));
            }
            let n = input_shape[0];
            let c = input_shape[1];
            let spatial: usize = input_shape[2..].iter().product::<usize>().max(1);
            let groups = command.attrs[0].max(1) as usize;
            let eps = attr_f32(&command, 1);
            if c % groups != 0
                || output_shape != input_shape
                || tensor_shape_of(&gamma)? != vec![c]
                || tensor_shape_of(&beta)? != vec![c]
            {
                return Err(NativeError::shape("group-norm parameter/channel shape mismatch"));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let gamma_guard = gamma
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let beta_guard = beta
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::group_norm_forward(
                &input_guard.data,
                &gamma_guard.data,
                &beta_guard.data,
                &mut output_guard.data,
                n,
                c,
                spatial,
                groups,
                eps,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::GroupNorm {
                    n,
                    c,
                    spatial,
                    groups,
                    eps,
                },
            });
        }
        Op::MaxPool2d => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            if input_shape.len() != 4 {
                return Err(NativeError::shape("max-pool expects NCHW input"));
            }
            let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
            let kh = command.attrs[0].max(1) as usize;
            let kw = command.attrs[1].max(1) as usize;
            let sh = command.attrs[2].max(1) as usize;
            let sw = command.attrs[3].max(1) as usize;
            if kh > h || kw > w {
                return Err(NativeError::shape("max-pool kernel is larger than the input"));
            }
            let oh = (h - kh) / sh + 1;
            let ow = (w - kw) / sw + 1;
            if output_shape != vec![n, c, oh, ow] || output_len != n * c * oh * ow {
                return Err(NativeError::shape(format!(
                    "max-pool output must be {:?}",
                    vec![n, c, oh, ow]
                )));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::max_pool_forward(
                &input_guard.data,
                &mut output_guard.data,
                n,
                c,
                h,
                w,
                kh,
                kw,
                sh,
                sw,
                oh,
                ow,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::MaxPool2d {
                    n,
                    c,
                    h,
                    w,
                    kh,
                    kw,
                    sh,
                    sw,
                    oh,
                    ow,
                },
            });
        }
        Op::Dropout => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            if output_shape != input_shape {
                return Err(NativeError::shape("dropout output shape mismatch"));
            }
            let probability = attr_f32(&command, 0);
            let training = command.attrs[1] != 0;
            let seed = attr_u64(&command, 2, 3);
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::dropout_forward(
                &input_guard.data,
                &mut output_guard.data,
                probability,
                training,
                seed,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Dropout {
                    length: output_len,
                    probability,
                    training,
                    seed,
                },
            });
        }
        Op::Embedding => {
            let indices = context.tensor(command.input_a)?;
            let weight = context.tensor(command.input_b)?;
            let indices_shape = tensor_shape_of(&indices)?;
            let weight_shape = tensor_shape_of(&weight)?;
            if weight_shape.len() != 2 {
                return Err(NativeError::shape("embedding weight must be rank 2"));
            }
            let rows = tensor_len_of(&indices)?;
            let dim = weight_shape[1];
            let mut expected = indices_shape.clone();
            expected.push(dim);
            if output_shape != expected || output_len != rows * dim {
                return Err(NativeError::shape("embedding output shape mismatch"));
            }
            let indices_guard = indices
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let weight_guard = weight
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::embedding_forward(
                &indices_guard.data,
                &weight_guard.data,
                &mut output_guard.data,
                rows,
                dim,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Embedding { rows, dim },
            });
        }
        Op::BatchedMatMul => {
            let left = context.tensor(command.input_a)?;
            let right = context.tensor(command.input_b)?;
            let left_shape = tensor_shape_of(&left)?;
            let right_shape = tensor_shape_of(&right)?;
            if left_shape.len() != 3 || right_shape.len() != 3 {
                return Err(NativeError::shape("batched-matmul expects rank-3 operands"));
            }
            let (batch, m, k) = (left_shape[0], left_shape[1], left_shape[2]);
            let (batch2, k2, n) = (right_shape[0], right_shape[1], right_shape[2]);
            if batch != batch2 || k != k2 || output_shape != vec![batch, m, n] {
                return Err(NativeError::shape("batched-matmul shape mismatch"));
            }
            let left_guard = left
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let right_guard = right
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::batched_matmul_forward(
                &left_guard.data,
                &right_guard.data,
                &mut output_guard.data,
                batch,
                m,
                k,
                n,
                Some(pool),
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::BatchedMatMul { batch, m, k, n },
            });
        }
        Op::Permute => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            let rank = command.attrs[0].max(0) as usize;
            if rank == 0 || rank != input_shape.len() {
                return Err(NativeError::shape("permute rank mismatch"));
            }
            let mut perm = Vec::with_capacity(rank);
            for index in 0..rank {
                perm.push(command.attrs[1 + index].max(0) as usize);
            }
            let mut expected = vec![0usize; rank];
            for index in 0..rank {
                if perm[index] >= rank {
                    return Err(NativeError::shape("permute index out of range"));
                }
                expected[index] = input_shape[perm[index]];
            }
            if output_shape != expected {
                return Err(NativeError::shape("permute output shape mismatch"));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::permute_forward(
                &input_guard.data,
                &mut output_guard.data,
                &input_shape,
                &perm,
            )?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Permute { in_shape: input_shape, perm },
            });
        }
        Op::Reshape => {
            let input = context.tensor(command.input_a)?;
            let input_shape = tensor_shape_of(&input)?;
            let rank = command.attrs[0].max(0) as usize;
            if rank == 0 || rank > 4 {
                return Err(NativeError::shape("reshape rank must be in [1, 4]"));
            }
            let mut out_shape = Vec::with_capacity(rank);
            for index in 0..rank {
                let dim = command.attrs[1 + index];
                if dim <= 0 {
                    return Err(NativeError::shape("reshape dimensions must be positive"));
                }
                out_shape.push(dim as usize);
            }
            if output_shape != out_shape || output_len != tensor_len_of(&input)? {
                return Err(NativeError::shape("reshape output shape mismatch"));
            }
            let input_guard = input
                .read()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            let mut output_guard = output
                .write()
                .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
            nn_ops::reshape_forward(&input_guard.data, &mut output_guard.data)?;
            graph.push_unchecked(Node {
                op,
                inputs,
                output: command.output,
                attrs: Attrs::Reshape {
                    in_shape: input_shape,
                    out_shape,
                },
            });
        }
    }
    Ok(())
}

pub fn backward(context: &Context, root: jlong, threads: jint) -> Result<()> {
    let root_address = root as usize;
    let graph = context
        .graph
        .lock()
        .map_err(|_| NativeError::invalid("graph lock poisoned"))?;
    let order = graph.topological_order(root_address)?;
    let root_tensor = context.tensor(root_address)?;
    let root_len = tensor_len_of(&root_tensor)?;
    let pool = context.pool_for(threads.max(1) as usize)?;

    let mut gradients: HashMap<usize, Vec<f32>> = HashMap::new();
    gradients.insert(root_address, vec![1.0; root_len]);

    for &index in order.iter().rev() {
        let node = graph.node(index);
        let Some(grad_output) = gradients.get(&node.output).cloned() else {
            continue;
        };
        match node.op {
            Op::MatMul => {
                let Attrs::MatMul { m, k, n } = node.attrs else {
                    return Err(NativeError::graph("MatMul node is missing metadata"));
                };
                let left = context.tensor(node.inputs[0])?;
                let right = context.tensor(node.inputs[1])?;
                let left_guard = left
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let right_guard = right
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_left = vec![0.0f32; m * k];
                let mut grad_right = vec![0.0f32; k * n];
                ops::gemm_nt(
                    &grad_output,
                    &right_guard.data,
                    &mut grad_left,
                    m,
                    k,
                    n,
                    Some(&pool),
                );
                ops::gemm_tn(
                    &left_guard.data,
                    &grad_output,
                    &mut grad_right,
                    m,
                    k,
                    n,
                    Some(&pool),
                );
                drop(left_guard);
                drop(right_guard);
                accumulate(&mut gradients, node.inputs[0], grad_left);
                accumulate(&mut gradients, node.inputs[1], grad_right);
            }
            Op::Add | Op::Sub | Op::Mul => {
                let Attrs::Broadcast { a_shape, b_shape } = &node.attrs else {
                    return Err(NativeError::graph("binary node is missing metadata"));
                };
                let left = context.tensor(node.inputs[0])?;
                let right = context.tensor(node.inputs[1])?;
                let left_guard = left
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let right_guard = right
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_left = vec![0.0f32; left_guard.data.len()];
                let mut grad_right = vec![0.0f32; right_guard.data.len()];
                let output_shape = ops::broadcast_shape(a_shape, b_shape)?;
                ops::binary_backward(
                    node.op,
                    &grad_output,
                    &output_shape,
                    &left_guard.data,
                    a_shape,
                    &right_guard.data,
                    b_shape,
                    &mut grad_left,
                    &mut grad_right,
                )?;
                drop(left_guard);
                drop(right_guard);
                accumulate(&mut gradients, node.inputs[0], grad_left);
                accumulate(&mut gradients, node.inputs[1], grad_right);
            }
            Op::ReLU | Op::Sigmoid | Op::Tanh => {
                let input = context.tensor(node.inputs[0])?;
                let output = context.tensor(node.output)?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let output_guard = output
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; input_guard.data.len()];
                ops::unary_backward(
                    node.op,
                    &input_guard.data,
                    &output_guard.data,
                    &grad_output,
                    &mut grad_input,
                )?;
                drop(input_guard);
                drop(output_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
            Op::Gelu | Op::Silu => {
                let input = context.tensor(node.inputs[0])?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; input_guard.data.len()];
                if node.op == Op::Gelu {
                    nn_ops::gelu_backward(&input_guard.data, &grad_output, &mut grad_input);
                } else {
                    nn_ops::silu_backward(&input_guard.data, &grad_output, &mut grad_input);
                }
                drop(input_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
            Op::Mse => {
                let prediction = context.tensor(node.inputs[0])?;
                let target = context.tensor(node.inputs[1])?;
                let prediction_guard = prediction
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let target_guard = target
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_prediction = vec![0.0f32; prediction_guard.data.len()];
                let mut grad_target = vec![0.0f32; target_guard.data.len()];
                ops::mse_backward(
                    &prediction_guard.data,
                    &target_guard.data,
                    &grad_output,
                    &mut grad_prediction,
                    &mut grad_target,
                )?;
                drop(prediction_guard);
                drop(target_guard);
                accumulate(&mut gradients, node.inputs[0], grad_prediction);
                accumulate(&mut gradients, node.inputs[1], grad_target);
            }
            Op::Softmax => {
                let Attrs::Softmax { rows, dim } = node.attrs else {
                    return Err(NativeError::graph("softmax node is missing metadata"));
                };
                let output = context.tensor(node.output)?;
                let output_guard = output
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; output_guard.data.len()];
                nn_ops::softmax_backward(
                    &output_guard.data,
                    &grad_output,
                    &mut grad_input,
                    rows,
                    dim,
                );
                drop(output_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
            Op::CrossEntropy => {
                let Attrs::CrossEntropy { rows, classes } = node.attrs else {
                    return Err(NativeError::graph("cross-entropy node is missing metadata"));
                };
                let logits = context.tensor(node.inputs[0])?;
                let targets = context.tensor(node.inputs[1])?;
                let logits_guard = logits
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let target_guard = targets
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_logits = vec![0.0f32; rows * classes];
                nn_ops::cross_entropy_backward(
                    &logits_guard.data,
                    &target_guard.data,
                    &grad_output,
                    &mut grad_logits,
                    rows,
                    classes,
                );
                drop(logits_guard);
                drop(target_guard);
                accumulate(&mut gradients, node.inputs[0], grad_logits);
            }
            Op::LayerNorm => {
                let Attrs::LayerNorm { rows, dim, eps } = node.attrs else {
                    return Err(NativeError::graph("layer-norm node is missing metadata"));
                };
                let input = context.tensor(node.inputs[0])?;
                let gamma = context.tensor(node.inputs[1])?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let gamma_guard = gamma
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; rows * dim];
                let mut grad_gamma = vec![0.0f32; dim];
                let mut grad_beta = vec![0.0f32; dim];
                nn_ops::layer_norm_backward(
                    &input_guard.data,
                    &gamma_guard.data,
                    &grad_output,
                    &mut grad_input,
                    &mut grad_gamma,
                    &mut grad_beta,
                    rows,
                    dim,
                    eps,
                );
                drop(input_guard);
                drop(gamma_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
                accumulate(&mut gradients, node.inputs[1], grad_gamma);
                accumulate(&mut gradients, node.inputs[2], grad_beta);
            }
            Op::BatchNorm2d => {
                let Attrs::BatchNorm {
                    n,
                    c,
                    h,
                    w,
                    eps,
                    training,
                    ..
                } = node.attrs
                else {
                    return Err(NativeError::graph("batch-norm node is missing metadata"));
                };
                let input = context.tensor(node.inputs[0])?;
                let gamma = context.tensor(node.inputs[1])?;
                let running_mean = context.tensor(node.inputs[3])?;
                let running_var = context.tensor(node.inputs[4])?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let gamma_guard = gamma
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mean_guard = running_mean
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let var_guard = running_var
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; n * c * h * w];
                let mut grad_gamma = vec![0.0f32; c];
                let mut grad_beta = vec![0.0f32; c];
                nn_ops::batch_norm_backward(
                    &input_guard.data,
                    &gamma_guard.data,
                    &grad_output,
                    &mut grad_input,
                    &mut grad_gamma,
                    &mut grad_beta,
                    &mean_guard.data,
                    &var_guard.data,
                    n,
                    c,
                    h,
                    w,
                    eps,
                    training,
                );
                drop(input_guard);
                drop(gamma_guard);
                drop(mean_guard);
                drop(var_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
                accumulate(&mut gradients, node.inputs[1], grad_gamma);
                accumulate(&mut gradients, node.inputs[2], grad_beta);
            }
            Op::Conv2d => {
                let Attrs::Conv2d {
                    n,
                    cin,
                    h,
                    w,
                    cout,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    oh,
                    ow,
                } = node.attrs
                else {
                    return Err(NativeError::graph("conv2d node is missing metadata"));
                };
                let input = context.tensor(node.inputs[0])?;
                let weight = context.tensor(node.inputs[1])?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let weight_guard = weight
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; n * cin * h * w];
                let mut grad_weight = vec![0.0f32; cout * (cin / groups.max(1)) * kh * kw];
                nn_ops::conv2d_backward_input(
                    &input_guard.data,
                    &weight_guard.data,
                    &grad_output,
                    &mut grad_input,
                    n,
                    cin,
                    h,
                    w,
                    cout,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    oh,
                    ow,
                    Some(&pool),
                )?;
                nn_ops::conv2d_backward_weight(
                    &input_guard.data,
                    &grad_output,
                    &mut grad_weight,
                    n,
                    cin,
                    h,
                    w,
                    cout,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    oh,
                    ow,
                    Some(&pool),
                )?;
                drop(input_guard);
                drop(weight_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
                accumulate(&mut gradients, node.inputs[1], grad_weight);
            }
            Op::GroupNorm => {
                let Attrs::GroupNorm {
                    n,
                    c,
                    spatial,
                    groups,
                    eps,
                } = node.attrs
                else {
                    return Err(NativeError::graph("group-norm node is missing metadata"));
                };
                let input = context.tensor(node.inputs[0])?;
                let gamma = context.tensor(node.inputs[1])?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let gamma_guard = gamma
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; n * c * spatial];
                let mut grad_gamma = vec![0.0f32; c];
                let mut grad_beta = vec![0.0f32; c];
                nn_ops::group_norm_backward(
                    &input_guard.data,
                    &gamma_guard.data,
                    &grad_output,
                    &mut grad_input,
                    &mut grad_gamma,
                    &mut grad_beta,
                    n,
                    c,
                    spatial,
                    groups,
                    eps,
                );
                drop(input_guard);
                drop(gamma_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
                accumulate(&mut gradients, node.inputs[1], grad_gamma);
                accumulate(&mut gradients, node.inputs[2], grad_beta);
            }
            Op::MaxPool2d => {
                let Attrs::MaxPool2d {
                    n,
                    c,
                    h,
                    w,
                    kh,
                    kw,
                    sh,
                    sw,
                    oh,
                    ow,
                } = node.attrs
                else {
                    return Err(NativeError::graph("max-pool node is missing metadata"));
                };
                let input = context.tensor(node.inputs[0])?;
                let input_guard = input
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_input = vec![0.0f32; n * c * h * w];
                nn_ops::max_pool_backward(
                    &input_guard.data,
                    &grad_output,
                    &mut grad_input,
                    n,
                    c,
                    h,
                    w,
                    kh,
                    kw,
                    sh,
                    sw,
                    oh,
                    ow,
                );
                drop(input_guard);
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
            Op::Dropout => {
                let Attrs::Dropout {
                    probability,
                    training,
                    seed,
                    ..
                } = node.attrs
                else {
                    return Err(NativeError::graph("dropout node is missing metadata"));
                };
                let mut grad_input = vec![0.0f32; grad_output.len()];
                nn_ops::dropout_backward(
                    &grad_output,
                    &mut grad_input,
                    probability,
                    training,
                    seed,
                );
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
            Op::Embedding => {
                let Attrs::Embedding { rows, dim } = node.attrs else {
                    return Err(NativeError::graph("embedding node is missing metadata"));
                };
                let indices = context.tensor(node.inputs[0])?;
                let weight = context.tensor(node.inputs[1])?;
                let indices_guard = indices
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let weight_guard = weight
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_weight = vec![0.0f32; weight_guard.data.len()];
                nn_ops::embedding_backward(
                    &indices_guard.data,
                    &grad_output,
                    &mut grad_weight,
                    rows,
                    dim,
                );
                drop(indices_guard);
                drop(weight_guard);
                accumulate(&mut gradients, node.inputs[1], grad_weight);
            }
            Op::BatchedMatMul => {
                let Attrs::BatchedMatMul { batch, m, k, n } = node.attrs else {
                    return Err(NativeError::graph("batched-matmul node is missing metadata"));
                };
                let left = context.tensor(node.inputs[0])?;
                let right = context.tensor(node.inputs[1])?;
                let left_guard = left
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let right_guard = right
                    .read()
                    .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
                let mut grad_left = vec![0.0f32; batch * m * k];
                let mut grad_right = vec![0.0f32; batch * k * n];
                nn_ops::batched_matmul_backward_a(
                    &right_guard.data,
                    &grad_output,
                    &mut grad_left,
                    batch,
                    m,
                    k,
                    n,
                    Some(&pool),
                );
                nn_ops::batched_matmul_backward_b(
                    &left_guard.data,
                    &grad_output,
                    &mut grad_right,
                    batch,
                    m,
                    k,
                    n,
                    Some(&pool),
                );
                drop(left_guard);
                drop(right_guard);
                accumulate(&mut gradients, node.inputs[0], grad_left);
                accumulate(&mut gradients, node.inputs[1], grad_right);
            }
            Op::Permute => {
                let Attrs::Permute { in_shape, perm } = node.attrs.clone() else {
                    return Err(NativeError::graph("permute node is missing metadata"));
                };
                let mut grad_input = vec![0.0f32; in_shape.iter().product::<usize>()];
                nn_ops::permute_backward(&grad_output, &mut grad_input, &in_shape, &perm)?;
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
            Op::Reshape => {
                let Attrs::Reshape { in_shape, .. } = node.attrs.clone() else {
                    return Err(NativeError::graph("reshape node is missing metadata"));
                };
                let mut grad_input = vec![0.0f32; in_shape.iter().product::<usize>()];
                nn_ops::reshape_backward(&grad_output, &mut grad_input)?;
                accumulate(&mut gradients, node.inputs[0], grad_input);
            }
        }
    }

    for (address, gradient) in gradients {
        let Ok(tensor) = context.tensor(address) else {
            continue;
        };
        let mut guard = tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        if guard.requires_grad {
            guard.accumulate_grad(&gradient)?;
        }
    }
    Ok(())
}
