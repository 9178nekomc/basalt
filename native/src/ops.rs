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
}
