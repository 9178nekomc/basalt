//! Forward and backward kernels for the eight supported ops.

use crate::error::{NativeError, Result};
use crate::simd::{axpy, dot};
use rayon::prelude::*;
use rayon::ThreadPool;

const GEMM_ROW_BLOCK: usize = 6;
const PARALLEL_GEMM_WORK: u64 = 262_144;

pub fn gemm_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    c.fill(0.0);
    let work = (m as u64) * (n as u64) * (k as u64);
    match pool {
        Some(pool) if m >= GEMM_ROW_BLOCK && work >= PARALLEL_GEMM_WORK => {
            let chunk_rows = GEMM_ROW_BLOCK;
            pool.install(|| {
                c.par_chunks_mut(chunk_rows * n)
                    .enumerate()
                    .for_each(|(block, chunk)| {
                        let start = block * chunk_rows;
                        gemm_rows(a, b, chunk, start, chunk.len() / n, k, n);
                    });
            });
        }
        _ => gemm_rows(a, b, c, 0, m, k, n),
    }
}

/// The required `i -> k -> j` loop, with `j` innermost and vectorised by `axpy`.
fn gemm_rows(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    row_start: usize,
    rows: usize,
    k: usize,
    n: usize,
) {
    for local_row in 0..rows {
        let a_row = &a[(row_start + local_row) * k..(row_start + local_row + 1) * k];
        let c_row = &mut c[local_row * n..(local_row + 1) * n];
        for inner in 0..k {
            let alpha = a_row[inner];
            if alpha != 0.0 {
                axpy(c_row, &b[inner * n..(inner + 1) * n], alpha);
            }
        }
    }
}

/// `c[m,n] = a[m,k] * b[n,k]^T` (forward, overwrites `c`). Used by convolution im2col GEMM.
pub fn gemm_forward_nt(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    c.fill(0.0);
    let work = (m as u64) * (n as u64) * (k as u64);
    if let Some(pool) = pool {
        if work >= PARALLEL_GEMM_WORK && m >= 2 {
            pool.install(|| {
                c.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(row, output)| {
                        let left = &a[row * k..(row + 1) * k];
                        for column in 0..n {
                            output[column] = dot(left, &b[column * k..(column + 1) * k]);
                        }
                    });
            });
            return;
        }
    }
    for row in 0..m {
        let left = &a[row * k..(row + 1) * k];
        for column in 0..n {
            c[row * n + column] = dot(left, &b[column * k..(column + 1) * k]);
        }
    }
}

/// `c[m,n] = a[m,k] * b[k,n]` (forward, overwrites `c`). Used by convolution backward-input.
pub fn gemm_forward_nn(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    c.fill(0.0);
    let work = (m as u64) * (n as u64) * (k as u64);
    let update = |row: usize, output: &mut [f32]| {
        for inner in 0..k {
            let alpha = a[row * k + inner];
            if alpha != 0.0 {
                axpy(output, &b[inner * n..(inner + 1) * n], alpha);
            }
        }
    };
    if let Some(pool) = pool {
        if work >= PARALLEL_GEMM_WORK && m >= 2 {
            pool.install(|| {
                c.par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(row, output)| update(row, output));
            });
            return;
        }
    }
    for row in 0..m {
        update(row, &mut c[row * n..(row + 1) * n]);
    }
}

/// `grad_a[m,k] += grad_c[m,n] * b[k,n]^T`.
pub fn gemm_nt(
    grad_c: &[f32],
    b: &[f32],
    grad_a: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    pool: Option<&ThreadPool>,
) {
    let work = (m as u64) * (k as u64) * (n as u64);
    match pool {
        Some(pool) if work >= PARALLEL_GEMM_WORK && m >= 2 => {
            pool.install(|| {
                grad_a
                    .par_chunks_mut(k)
                    .enumerate()
                    .for_each(|(row, output)| {
                        let left = &grad_c[row * n..(row + 1) * n];
                        for column in 0..k {
                            let right = &b[column * n..(column + 1) * n];
                            output[column] += dot(left, right);
                        }
                    });
            });
        }
        _ => {
            for row in 0..m {
                let left = &grad_c[row * n..(row + 1) * n];
                for column in 0..k {
                    let right = &b[column * n..(column + 1) * n];
                    grad_a[row * k + column] += dot(left, right);
                }
            }
        }
    }
}

/// `grad_b[k,n] += a[m,k]^T * grad_c[m,n]`.
pub fn gemm_tn(
    a: &[f32],
    grad_c: &[f32],
    grad_b: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    pool: Option<&ThreadPool>,
) {
    let work = (m as u64) * (k as u64) * (n as u64);
    match pool {
        Some(pool) if work >= PARALLEL_GEMM_WORK && k >= 2 => {
            pool.install(|| {
                grad_b
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(column, output)| {
                        for row in 0..m {
                            let alpha = a[row * k + column];
                            if alpha != 0.0 {
                                axpy(output, &grad_c[row * n..(row + 1) * n], alpha);
                            }
                        }
                    });
            });
        }
        _ => {
            for column in 0..k {
                let output = &mut grad_b[column * n..(column + 1) * n];
                for row in 0..m {
                    let alpha = a[row * k + column];
                    if alpha != 0.0 {
                        axpy(output, &grad_c[row * n..(row + 1) * n], alpha);
                    }
                }
            }
        }
    }
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut shape = vec![1usize; rank];
    for index in 0..rank {
        let a_dim = if index < rank - a.len() {
            1
        } else {
            a[index - (rank - a.len())]
        };
        let b_dim = if index < rank - b.len() {
            1
        } else {
            b[index - (rank - b.len())]
        };
        shape[index] = if a_dim == b_dim {
            a_dim
        } else if a_dim == 1 {
            b_dim
        } else if b_dim == 1 {
            a_dim
        } else {
            return Err(NativeError::shape(format!(
                "cannot broadcast shapes {:?} and {:?}",
                a, b
            )));
        };
    }
    Ok(shape)
}

fn aligned_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let offset = out_shape.len() - shape.len();
    let mut strides = vec![0usize; out_shape.len()];
    let mut stride = 1usize;
    for dimension in (0..shape.len()).rev() {
        let broadcast = shape[dimension] == 1 && out_shape[offset + dimension] != 1;
        strides[offset + dimension] = if broadcast { 0 } else { stride };
        stride *= shape[dimension];
    }
    strides
}

fn is_same_shape(a: &[usize], b: &[usize]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(left, right)| left == right)
}

fn is_scalar(shape: &[usize]) -> bool {
    shape.len() == 1 && shape[0] == 1
}

pub fn binary_forward(
    op: crate::graph::Op,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out: &mut [f32],
    out_shape: &[usize],
) -> Result<()> {
    use crate::graph::Op;
    let expected = out_shape.iter().product::<usize>();
    if out.len() != expected {
        return Err(NativeError::shape("binary output length mismatch"));
    }
    if is_same_shape(a_shape, b_shape) {
        match op {
            Op::Add => {
                for index in 0..out.len() {
                    out[index] = a[index] + b[index];
                }
            }
            Op::Sub => {
                for index in 0..out.len() {
                    out[index] = a[index] - b[index];
                }
            }
            Op::Mul => {
                for index in 0..out.len() {
                    out[index] = a[index] * b[index];
                }
            }
            Op::Div => {
                for index in 0..out.len() {
                    out[index] = a[index] / b[index];
                }
            }
            Op::Pow => {
                for index in 0..out.len() {
                    out[index] = a[index].powf(b[index]);
                }
            }
            _ => return Err(NativeError::invalid("binary_forward received a non-binary op")),
        }
        return Ok(());
    }
    if is_scalar(b_shape) {
        let scalar = b[0];
        match op {
            Op::Add => out.iter_mut().zip(a).for_each(|(o, x)| *o = *x + scalar),
            Op::Sub => out.iter_mut().zip(a).for_each(|(o, x)| *o = *x - scalar),
            Op::Mul => out.iter_mut().zip(a).for_each(|(o, x)| *o = *x * scalar),
            Op::Div => out.iter_mut().zip(a).for_each(|(o, x)| *o = *x / scalar),
            Op::Pow => out.iter_mut().zip(a).for_each(|(o, x)| *o = x.powf(scalar)),
            _ => return Err(NativeError::invalid("binary_forward received a non-binary op")),
        }
        return Ok(());
    }
    if is_scalar(a_shape) {
        let scalar = a[0];
        match op {
            Op::Add => out.iter_mut().zip(b).for_each(|(o, x)| *o = scalar + *x),
            Op::Sub => out.iter_mut().zip(b).for_each(|(o, x)| *o = scalar - *x),
            Op::Mul => out.iter_mut().zip(b).for_each(|(o, x)| *o = scalar * *x),
            Op::Div => out.iter_mut().zip(b).for_each(|(o, x)| *o = scalar / *x),
            Op::Pow => out.iter_mut().zip(b).for_each(|(o, x)| *o = scalar.powf(*x)),
            _ => return Err(NativeError::invalid("binary_forward received a non-binary op")),
        }
        return Ok(());
    }

    let rank = out_shape.len();
    let a_strides = aligned_strides(a_shape, out_shape);
    let b_strides = aligned_strides(b_shape, out_shape);
    let mut coordinates = vec![0usize; rank];
    for output_index in 0..out.len() {
        let mut a_index = 0usize;
        let mut b_index = 0usize;
        for dimension in 0..rank {
            a_index += coordinates[dimension] * a_strides[dimension];
            b_index += coordinates[dimension] * b_strides[dimension];
        }
        out[output_index] = match op {
            Op::Add => a[a_index] + b[b_index],
            Op::Sub => a[a_index] - b[b_index],
            Op::Mul => a[a_index] * b[b_index],
            Op::Div => a[a_index] / b[b_index],
            Op::Pow => a[a_index].powf(b[b_index]),
            _ => return Err(NativeError::invalid("binary_forward received a non-binary op")),
        };
        for dimension in (0..rank).rev() {
            coordinates[dimension] += 1;
            if coordinates[dimension] < out_shape[dimension] {
                break;
            }
            coordinates[dimension] = 0;
        }
    }
    Ok(())
}

/// Accumulates input gradients for Add/Sub/Mul. `grad_a` and `grad_b` are already zeroed.
#[allow(clippy::too_many_arguments)]
pub fn binary_backward(
    op: crate::graph::Op,
    grad_out: &[f32],
    out_shape: &[usize],
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    grad_a: &mut [f32],
    grad_b: &mut [f32],
) -> Result<()> {
    use crate::graph::Op;
    if is_same_shape(a_shape, b_shape) {
        match op {
            Op::Add => {
                for index in 0..grad_out.len() {
                    grad_a[index] += grad_out[index];
                    grad_b[index] += grad_out[index];
                }
            }
            Op::Sub => {
                for index in 0..grad_out.len() {
                    grad_a[index] += grad_out[index];
                    grad_b[index] -= grad_out[index];
                }
            }
            Op::Mul => {
                for index in 0..grad_out.len() {
                    grad_a[index] += grad_out[index] * b[index];
                    grad_b[index] += grad_out[index] * a[index];
                }
            }
            Op::Div => {
                for index in 0..grad_out.len() {
                    let denominator = b[index];
                    grad_a[index] += grad_out[index] / denominator;
                    grad_b[index] -= grad_out[index] * a[index] / (denominator * denominator);
                }
            }
            Op::Pow => {
                for index in 0..grad_out.len() {
                    let base = a[index];
                    let exponent = b[index];
                    grad_a[index] += grad_out[index] * exponent * base.powf(exponent - 1.0);
                    grad_b[index] += grad_out[index] * base.powf(exponent) * base.ln();
                }
            }
            _ => return Err(NativeError::invalid("binary_backward received a non-binary op")),
        }
        return Ok(());
    }

    let rank = out_shape.len();
    let a_strides = aligned_strides(a_shape, out_shape);
    let b_strides = aligned_strides(b_shape, out_shape);
    let mut coordinates = vec![0usize; rank];
    for output_index in 0..grad_out.len() {
        let mut a_index = 0usize;
        let mut b_index = 0usize;
        for dimension in 0..rank {
            a_index += coordinates[dimension] * a_strides[dimension];
            b_index += coordinates[dimension] * b_strides[dimension];
        }
        let gradient = grad_out[output_index];
        match op {
            Op::Add => {
                grad_a[a_index] += gradient;
                grad_b[b_index] += gradient;
            }
            Op::Sub => {
                grad_a[a_index] += gradient;
                grad_b[b_index] -= gradient;
            }
            Op::Mul => {
                grad_a[a_index] += gradient * b[b_index];
                grad_b[b_index] += gradient * a[a_index];
            }
            Op::Div => {
                grad_a[a_index] += gradient / b[b_index];
                grad_b[b_index] -= gradient * a[a_index] / (b[b_index] * b[b_index]);
            }
            Op::Pow => {
                let base = a[a_index];
                let exponent = b[b_index];
                grad_a[a_index] += gradient * exponent * base.powf(exponent - 1.0);
                grad_b[b_index] += gradient * base.powf(exponent) * base.ln();
            }
            _ => return Err(NativeError::invalid("binary_backward received a non-binary op")),
        }
        for dimension in (0..rank).rev() {
            coordinates[dimension] += 1;
            if coordinates[dimension] < out_shape[dimension] {
                break;
            }
            coordinates[dimension] = 0;
        }
    }
    Ok(())
}

/// 沿单一维度归约。输出长度是 outer * inner（keepdim 只改形状，不改元素数）。
///
/// 布局按行主序：[outer, axis, inner]。每个 (outer, inner) 组合是一组要折叠的数据。
pub fn reduce_dim_forward(
    op: crate::graph::Op,
    input: &[f32],
    shape: &[usize],
    dim: usize,
    out: &mut [f32],
) -> Result<()> {
    use crate::graph::Op;
    if dim >= shape.len() {
        return Err(NativeError::shape(format!(
            "reduction dim {dim} out of range for rank {}",
            shape.len()
        )));
    }
    let axis = shape[dim];
    if axis == 0 {
        return Err(NativeError::shape("cannot reduce an empty axis"));
    }
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    if input.len() != outer * axis * inner {
        return Err(NativeError::shape("reduce input length mismatch"));
    }
    if out.len() != outer * inner {
        return Err(NativeError::shape("reduce output length mismatch"));
    }

    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis * inner + i;
            let mut acc = if op == Op::MaxDim {
                f32::NEG_INFINITY
            } else {
                0.0f32
            };
            for k in 0..axis {
                let value = input[base + k * inner];
                match op {
                    Op::SumDim | Op::MeanDim => acc += value,
                    Op::MaxDim => {
                        if value > acc {
                            acc = value;
                        }
                    }
                    _ => return Err(NativeError::invalid("reduce_dim_forward: wrong op")),
                }
            }
            out[o * inner + i] = if op == Op::MeanDim {
                acc / axis as f32
            } else {
                acc
            };
        }
    }
    Ok(())
}

/// 按轴归约的反向。`grad_in` 已经清零，这里累加。
///
/// Max 的反向只把梯度给到该组第一个最大值的位置（并列时取最先出现的）。
pub fn reduce_dim_backward(
    op: crate::graph::Op,
    input: &[f32],
    shape: &[usize],
    dim: usize,
    grad_out: &[f32],
    grad_in: &mut [f32],
) -> Result<()> {
    use crate::graph::Op;
    if dim >= shape.len() {
        return Err(NativeError::shape("reduce_dim_backward: dim out of range"));
    }
    let axis = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    if grad_out.len() != outer * inner || grad_in.len() != outer * axis * inner {
        return Err(NativeError::shape("reduce_dim_backward length mismatch"));
    }

    for o in 0..outer {
        for i in 0..inner {
            let base = o * axis * inner + i;
            let gradient = grad_out[o * inner + i];
            match op {
                Op::SumDim => {
                    for k in 0..axis {
                        grad_in[base + k * inner] += gradient;
                    }
                }
                Op::MeanDim => {
                    let share = gradient / axis as f32;
                    for k in 0..axis {
                        grad_in[base + k * inner] += share;
                    }
                }
                Op::MaxDim => {
                    let mut best = 0usize;
                    let mut best_value = input[base];
                    for k in 1..axis {
                        let value = input[base + k * inner];
                        if value > best_value {
                            best_value = value;
                            best = k;
                        }
                    }
                    grad_in[base + best * inner] += gradient;
                }
                _ => return Err(NativeError::invalid("reduce_dim_backward: wrong op")),
            }
        }
    }
    Ok(())
}

/// 全量归约：把任意形状压成一个标量。
///
/// 没有归约就没法手写 softmax 的分母、也没法把任意损失归一化——
/// 这是「自己写模型」最常缺的一块。
pub fn reduce_forward(op: crate::graph::Op, input: &[f32], out: &mut [f32]) -> Result<()> {
    use crate::graph::Op;
    if out.len() != 1 {
        return Err(NativeError::shape("reduce output must be a single element"));
    }
    let mut sum = 0.0f64;
    for value in input {
        sum += *value as f64;
    }
    out[0] = match op {
        Op::SumAll => sum as f32,
        Op::MeanAll => {
            if input.is_empty() {
                return Err(NativeError::shape("mean of an empty tensor"));
            }
            (sum / input.len() as f64) as f32
        }
        _ => return Err(NativeError::invalid("reduce_forward received a non-reduce op")),
    };
    Ok(())
}

/// 归约的反向：把标量梯度摊回输入形状。
pub fn reduce_backward(
    op: crate::graph::Op,
    input_len: usize,
    grad_out: &[f32],
    grad_in: &mut [f32],
) -> Result<()> {
    use crate::graph::Op;
    if grad_out.len() != 1 || grad_in.len() != input_len {
        return Err(NativeError::shape("reduce backward length mismatch"));
    }
    let scale = match op {
        Op::SumAll => grad_out[0],
        Op::MeanAll => {
            if input_len == 0 {
                return Err(NativeError::shape("mean backward of an empty tensor"));
            }
            grad_out[0] / input_len as f32
        }
        _ => return Err(NativeError::invalid("reduce_backward received a non-reduce op")),
    };
    for slot in grad_in.iter_mut() {
        *slot += scale;
    }
    Ok(())
}

pub fn unary_forward(op: crate::graph::Op, input: &[f32], out: &mut [f32]) -> Result<()> {
    use crate::graph::Op;
    if input.len() != out.len() {
        return Err(NativeError::shape("unary input/output length mismatch"));
    }
    match op {
        Op::ReLU => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = value.max(0.0);
            }
        }
        Op::Sigmoid => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = stable_sigmoid(*value);
            }
        }
        Op::Tanh => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = value.tanh();
            }
        }
        Op::Exp => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = value.exp();
            }
        }
        Op::Log => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = value.ln();
            }
        }
        Op::Sqrt => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = value.sqrt();
            }
        }
        Op::Abs => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = value.abs();
            }
        }
        Op::Neg => {
            for (output, value) in out.iter_mut().zip(input) {
                *output = -value;
            }
        }
        _ => return Err(NativeError::invalid("unary_forward received a non-unary op")),
    }
    Ok(())
}

pub fn unary_backward(
    op: crate::graph::Op,
    input: &[f32],
    output: &[f32],
    grad_out: &[f32],
    grad_in: &mut [f32],
) -> Result<()> {
    use crate::graph::Op;
    if input.len() != output.len() || input.len() != grad_out.len() || input.len() != grad_in.len()
    {
        return Err(NativeError::shape("unary backward length mismatch"));
    }
    match op {
        Op::ReLU => {
            for index in 0..input.len() {
                if input[index] > 0.0 {
                    grad_in[index] += grad_out[index];
                }
            }
        }
        Op::Sigmoid => {
            for index in 0..input.len() {
                let value = output[index];
                grad_in[index] += grad_out[index] * value * (1.0 - value);
            }
        }
        Op::Exp => {
            // d/dx e^x = e^x，直接用前向输出
            for index in 0..input.len() {
                grad_in[index] += grad_out[index] * output[index];
            }
        }
        Op::Log => {
            // d/dx ln(x) = 1/x；x <= 0 时结果为 inf/NaN，由调用方决定是否加 eps
            for index in 0..input.len() {
                grad_in[index] += grad_out[index] / input[index];
            }
        }
        Op::Sqrt => {
            // d/dx sqrt(x) = 1 / (2 sqrt(x))，用前向输出避免重算
            for index in 0..input.len() {
                grad_in[index] += grad_out[index] / (2.0 * output[index]);
            }
        }
        Op::Abs => {
            // 在 0 处取次梯度 0
            for index in 0..input.len() {
                let sign = if input[index] > 0.0 {
                    1.0
                } else if input[index] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                grad_in[index] += grad_out[index] * sign;
            }
        }
        Op::Neg => {
            for index in 0..input.len() {
                grad_in[index] -= grad_out[index];
            }
        }
        Op::Tanh => {
            for index in 0..input.len() {
                let value = output[index];
                grad_in[index] += grad_out[index] * (1.0 - value * value);
            }
        }
        _ => return Err(NativeError::invalid("unary_backward received a non-unary op")),
    }
    Ok(())
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

pub fn mse_forward(prediction: &[f32], target: &[f32], out: &mut [f32]) -> Result<f32> {
    if prediction.len() != target.len() || out.len() != 1 || prediction.is_empty() {
        return Err(NativeError::shape(
            "MSE expects equal-length prediction/target and a scalar output",
        ));
    }
    let mut sum = 0.0f64;
    for (predicted, expected) in prediction.iter().zip(target) {
        let delta = (*predicted - *expected) as f64;
        sum += delta * delta;
    }
    let loss = sum / prediction.len() as f64;
    out[0] = loss as f32;
    Ok(loss as f32)
}

pub fn mse_backward(
    prediction: &[f32],
    target: &[f32],
    grad_out: &[f32],
    grad_prediction: &mut [f32],
    grad_target: &mut [f32],
) -> Result<()> {
    if prediction.len() != target.len()
        || grad_prediction.len() != prediction.len()
        || grad_target.len() != prediction.len()
        || grad_out.len() != 1
    {
        return Err(NativeError::shape("MSE backward length mismatch"));
    }
    let scale = 2.0 * grad_out[0] / prediction.len() as f32;
    for index in 0..prediction.len() {
        let gradient = (prediction[index] - target[index]) * scale;
        grad_prediction[index] += gradient;
        grad_target[index] -= gradient;
    }
    Ok(())
}

/// In-place SGD with decoupled weight decay: `p -= lr * (g + wd * p)`.
pub fn sgd_step(data: &mut [f32], grad: &[f32], learning_rate: f32, weight_decay: f32) {
    if grad.len() != data.len() {
        return;
    }
    for index in 0..data.len() {
        data[index] -= learning_rate * (grad[index] + weight_decay * data[index]);
    }
}

/// In-place Adam with bias correction and decoupled weight decay.
#[allow(clippy::too_many_arguments)]
pub fn adam_step(
    data: &mut [f32],
    grad: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u64,
) {
    if grad.len() != data.len()
        || first_moment.len() != data.len()
        || second_moment.len() != data.len()
        || step == 0
    {
        return;
    }
    let bias1 = 1.0 - beta1.powi(step as i32);
    let bias2 = 1.0 - beta2.powi(step as i32);
    for index in 0..data.len() {
        let gradient = grad[index];
        first_moment[index] = beta1 * first_moment[index] + (1.0 - beta1) * gradient;
        second_moment[index] = beta2 * second_moment[index] + (1.0 - beta2) * gradient * gradient;
        let corrected_first = first_moment[index] / bias1;
        let corrected_second = second_moment[index] / bias2;
        data[index] -= learning_rate
            * (corrected_first / (corrected_second.sqrt() + epsilon) + weight_decay * data[index]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Op;

    fn reference_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..k {
                    sum = a[row * k + inner].mul_add(b[inner * n + column], sum);
                }
                c[row * n + column] = sum;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_reference_including_tails() {
        for &(m, k, n) in &[(1, 1, 1), (2, 3, 5), (7, 9, 11), (13, 4, 17), (31, 33, 9)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.17).cos()).collect();
            let mut c = vec![0.0f32; m * n];
            gemm_f32(&a, &b, &mut c, m, k, n, None);
            let expected = reference_gemm(&a, &b, m, k, n);
            for (left, right) in c.iter().zip(&expected) {
                assert!((left - right).abs() <= 1e-4 * (1.0 + right.abs()));
            }
        }
    }

    #[test]
    fn broadcast_add_and_backward() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![10.0, 20.0, 30.0];
        let mut out = vec![0.0; 6];
        binary_forward(Op::Add, &a, &[2, 3], &b, &[3], &mut out, &[2, 3]).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        let mut grad_a = vec![0.0; 6];
        let mut grad_b = vec![0.0; 3];
        binary_backward(
            Op::Add,
            &[1.0; 6],
            &[2, 3],
            &a,
            &[2, 3],
            &b,
            &[3],
            &mut grad_a,
            &mut grad_b,
        )
        .unwrap();
        assert_eq!(grad_a, vec![1.0; 6]);
        assert_eq!(grad_b, vec![2.0; 3]);
    }

    #[test]
    fn mse_gradient_is_two_over_n() {
        let prediction = vec![1.0, 2.0, 3.0, 4.0];
        let target = vec![0.0, 0.0, 0.0, 0.0];
        let mut loss = [0.0f32];
        mse_forward(&prediction, &target, &mut loss).unwrap();
        assert!((loss[0] - 7.5).abs() < 1e-6);
        let mut grad_prediction = vec![0.0; 4];
        let mut grad_target = vec![0.0; 4];
        mse_backward(
            &prediction,
            &target,
            &[1.0],
            &mut grad_prediction,
            &mut grad_target,
        )
        .unwrap();
        assert!((grad_prediction[0] - 0.5).abs() < 1e-6);
        assert!((grad_prediction[3] - 2.0).abs() < 1e-6);
        assert!((grad_target[3] + 2.0).abs() < 1e-6);
    }
    fn unary_value(op: crate::graph::Op, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; input.len()];
        unary_forward(op, input, &mut out).expect("unary forward");
        out
    }

    fn unary_grad(op: crate::graph::Op, input: &[f32]) -> Vec<f32> {
        let out = unary_value(op, input);
        let grad_out = vec![1.0f32; input.len()];
        let mut grad_in = vec![0.0f32; input.len()];
        unary_backward(op, input, &out, &grad_out, &mut grad_in).expect("unary backward");
        grad_in
    }

    #[test]
    fn exp_log_sqrt_roundtrip() {
        use crate::graph::Op;
        let input = [0.0f32, 1.0, 2.0, -1.0];
        let exp = unary_value(Op::Exp, &input);
        assert!((exp[1] - std::f32::consts::E).abs() < 1e-6);
        assert!((exp[0] - 1.0).abs() < 1e-6);

        let log = unary_value(Op::Log, &exp);
        for index in 0..input.len() {
            assert!((log[index] - input[index]).abs() < 1e-5, "exp/log roundtrip at {index}");
        }

        let sqrt = unary_value(Op::Sqrt, &[4.0, 9.0, 0.25]);
        assert!((sqrt[0] - 2.0).abs() < 1e-6);
        assert!((sqrt[1] - 3.0).abs() < 1e-6);
        assert!((sqrt[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn abs_and_neg_backward() {
        use crate::graph::Op;
        let input = [-2.0f32, 0.0, 3.0];
        assert_eq!(unary_value(Op::Abs, &input), vec![2.0, 0.0, 3.0]);
        // 0 处取次梯度 0
        assert_eq!(unary_grad(Op::Abs, &input), vec![-1.0, 0.0, 1.0]);
        assert_eq!(unary_value(Op::Neg, &input), vec![2.0, 0.0, -3.0]);
        assert_eq!(unary_grad(Op::Neg, &input), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn exp_log_sqrt_gradients() {
        use crate::graph::Op;
        let input = [0.5f32, 1.5];
        let exp_grad = unary_grad(Op::Exp, &input);
        for index in 0..input.len() {
            assert!((exp_grad[index] - input[index].exp()).abs() < 1e-5);
        }
        let log_grad = unary_grad(Op::Log, &input);
        for index in 0..input.len() {
            assert!((log_grad[index] - 1.0 / input[index]).abs() < 1e-5);
        }
        let sqrt_grad = unary_grad(Op::Sqrt, &input);
        for index in 0..input.len() {
            assert!((sqrt_grad[index] - 0.5 / input[index].sqrt()).abs() < 1e-5);
        }
    }

    #[test]
    fn div_and_pow_forward_backward() {
        use crate::graph::Op;
        let a = [4.0f32, 9.0];
        let b = [2.0f32, 3.0];
        let mut out = vec![0.0f32; 2];
        binary_forward(Op::Div, &a, &[2], &b, &[2], &mut out, &[2]).expect("div forward");
        assert_eq!(out, vec![2.0, 3.0]);

        let mut grad_a = vec![0.0f32; 2];
        let mut grad_b = vec![0.0f32; 2];
        binary_backward(Op::Div, &[1.0, 1.0], &[2], &a, &[2], &b, &[2], &mut grad_a, &mut grad_b)
            .expect("div backward");
        assert!((grad_a[0] - 0.5).abs() < 1e-6);
        assert!((grad_b[0] + 1.0).abs() < 1e-6); // -a/b^2 = -4/4 = -1

        binary_forward(Op::Pow, &[2.0, 3.0], &[2], &[3.0, 2.0], &[2], &mut out, &[2])
            .expect("pow forward");
        assert_eq!(out, vec![8.0, 9.0]);

        grad_a = vec![0.0f32; 2];
        grad_b = vec![0.0f32; 2];
        binary_backward(Op::Pow, &[1.0, 1.0], &[2], &[2.0, 3.0], &[2], &[3.0, 2.0], &[2],
                        &mut grad_a, &mut grad_b).expect("pow backward");
        // d/da a^b = b a^(b-1) -> 3*4=12, 2*3=6
        assert!((grad_a[0] - 12.0).abs() < 1e-4);
        assert!((grad_a[1] - 6.0).abs() < 1e-4);
        // d/db a^b = a^b ln a
        assert!((grad_b[0] - 8.0 * 2.0f32.ln()).abs() < 1e-4);
        assert!((grad_b[1] - 9.0 * 3.0f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn div_broadcasts_like_add() {
        use crate::graph::Op;
        // [2,2] / [2] -> 每行除以同一向量
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, 2.0];
        let mut out = vec![0.0f32; 4];
        binary_forward(Op::Div, &a, &[2, 2], &b, &[2], &mut out, &[2, 2]).expect("broadcast div");
        assert_eq!(out, vec![1.0, 1.0, 3.0, 2.0]);
    }
    #[test]
    fn reduce_sum_and_mean() {
        use crate::graph::Op;
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 1];
        reduce_forward(Op::SumAll, &input, &mut out).expect("sum");
        assert!((out[0] - 10.0).abs() < 1e-6);
        reduce_forward(Op::MeanAll, &input, &mut out).expect("mean");
        assert!((out[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn reduce_backward_broadcasts_scalar() {
        use crate::graph::Op;
        let mut grad_in = vec![0.0f32; 4];
        reduce_backward(Op::SumAll, 4, &[2.0], &mut grad_in).expect("sum backward");
        assert_eq!(grad_in, vec![2.0, 2.0, 2.0, 2.0]);

        let mut mean_grad = vec![0.0f32; 4];
        reduce_backward(Op::MeanAll, 4, &[2.0], &mut mean_grad).expect("mean backward");
        assert_eq!(mean_grad, vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn reduce_rejects_non_scalar_output() {
        use crate::graph::Op;
        let mut out = vec![0.0f32; 2];
        assert!(reduce_forward(Op::SumAll, &[1.0, 2.0], &mut out).is_err());
    }
    #[test]
    fn reduce_dim_sum_mean_max_forward() {
        use crate::graph::Op;
        // [2,3] 按行主序：[[1,2,3],[4,5,6]]
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0f32; 2];

        reduce_dim_forward(Op::SumDim, &input, &[2, 3], 1, &mut out).expect("sum dim1");
        assert_eq!(out, vec![6.0, 15.0]);

        reduce_dim_forward(Op::MeanDim, &input, &[2, 3], 1, &mut out).expect("mean dim1");
        assert_eq!(out, vec![2.0, 5.0]);

        reduce_dim_forward(Op::MaxDim, &input, &[2, 3], 1, &mut out).expect("max dim1");
        assert_eq!(out, vec![3.0, 6.0]);

        // 按列归约，输出长度是 inner = 3
        let mut col = vec![0.0f32; 3];
        reduce_dim_forward(Op::SumDim, &input, &[2, 3], 0, &mut col).expect("sum dim0");
        assert_eq!(col, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_dim_backward_routes_gradients() {
        use crate::graph::Op;
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut grad_in = vec![0.0f32; 6];
        reduce_dim_backward(Op::SumDim, &input, &[2, 3], 1, &[10.0, 20.0], &mut grad_in)
            .expect("sum backward");
        assert_eq!(grad_in, vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);

        let mut mean_grad = vec![0.0f32; 6];
        reduce_dim_backward(Op::MeanDim, &input, &[2, 3], 1, &[3.0, 6.0], &mut mean_grad)
            .expect("mean backward");
        assert_eq!(mean_grad, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // max 只给到该行最大值位置：第 0 行是下标 2，第 1 行是下标 2
        let mut max_grad = vec![0.0f32; 6];
        reduce_dim_backward(Op::MaxDim, &input, &[2, 3], 1, &[1.0, 1.0], &mut max_grad)
            .expect("max backward");
        assert_eq!(max_grad, vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn reduce_dim_rejects_bad_shapes() {
        use crate::graph::Op;
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 2];
        // dim 越界
        assert!(reduce_dim_forward(Op::SumDim, &input, &[2, 2], 5, &mut out).is_err());
        // 输出长度不对
        let mut wrong = vec![0.0f32; 3];
        assert!(reduce_dim_forward(Op::SumDim, &input, &[2, 2], 1, &mut wrong).is_err());
    }
}
