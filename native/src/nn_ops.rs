//! Neural-network operators: convolution, pooling, normalization, softmax/CE, embedding,
//! dropout, batched matmul, permute and reshape. Every forward has a matching backward.

use crate::error::{NativeError, Result};
use crate::ops::{gemm_f32, gemm_forward_nn, gemm_forward_nt, gemm_nt, gemm_tn};
use rayon::prelude::*;
use rayon::ThreadPool;

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn softmax_forward(input: &[f32], out: &mut [f32], rows: usize, dim: usize) -> Result<()> {
    if dim == 0 || rows == 0 || input.len() != rows * dim || out.len() != input.len() {
        return Err(NativeError::shape("softmax shape mismatch"));
    }
    for row in 0..rows {
        let base = row * dim;
        let mut maximum = f32::NEG_INFINITY;
        for index in 0..dim {
            maximum = maximum.max(input[base + index]);
        }
        let mut sum = 0.0f32;
        for index in 0..dim {
            let value = (input[base + index] - maximum).exp();
            out[base + index] = value;
            sum += value;
        }
        if !(sum > 0.0) || !sum.is_finite() {
            return Err(NativeError::invalid("softmax denominator is not finite"));
        }
        let inverse = 1.0 / sum;
        for index in 0..dim {
            out[base + index] *= inverse;
        }
    }
    Ok(())
}

pub fn softmax_backward(
    output: &[f32],
    grad_out: &[f32],
    grad_in: &mut [f32],
    rows: usize,
    dim: usize,
) {
    for row in 0..rows {
        let base = row * dim;
        let mut dot_product = 0.0f32;
        for index in 0..dim {
            dot_product += grad_out[base + index] * output[base + index];
        }
        for index in 0..dim {
            let value = output[base + index];
            grad_in[base + index] += value * (grad_out[base + index] - dot_product);
        }
    }
}

pub fn cross_entropy_forward(
    logits: &[f32],
    targets: &[f32],
    out: &mut [f32],
    rows: usize,
    classes: usize,
) -> Result<f32> {
    if rows == 0 || classes == 0 || logits.len() != rows * classes || targets.len() != rows {
        return Err(NativeError::shape("cross-entropy shape mismatch"));
    }
    if out.len() != 1 {
        return Err(NativeError::shape("cross-entropy output must be scalar"));
    }
    let mut total = 0.0f64;
    for row in 0..rows {
        let target = targets[row];
        if !target.is_finite() || target < 0.0 || target >= classes as f32 {
            return Err(NativeError::invalid(format!(
                "cross-entropy target {target} is outside [0, {classes})"
            )));
        }
        let base = row * classes;
        let mut maximum = f32::NEG_INFINITY;
        for index in 0..classes {
            maximum = maximum.max(logits[base + index]);
        }
        let mut sum = 0.0f64;
        for index in 0..classes {
            sum += ((logits[base + index] - maximum) as f64).exp();
        }
        let log_sum_exp = maximum as f64 + sum.ln();
        total += log_sum_exp - logits[base + target as usize] as f64;
    }
    let loss = total / rows as f64;
    out[0] = loss as f32;
    Ok(loss as f32)
}

pub fn cross_entropy_backward(
    logits: &[f32],
    targets: &[f32],
    grad_out: &[f32],
    grad_logits: &mut [f32],
    rows: usize,
    classes: usize,
) {
    let scale = grad_out[0] / rows as f32;
    for row in 0..rows {
        let base = row * classes;
        let mut maximum = f32::NEG_INFINITY;
        for index in 0..classes {
            maximum = maximum.max(logits[base + index]);
        }
        let mut sum = 0.0f32;
        for index in 0..classes {
            let value = (logits[base + index] - maximum).exp();
            grad_logits[base + index] = value;
            sum += value;
        }
        let inverse = 1.0 / sum;
        for index in 0..classes {
            grad_logits[base + index] *= inverse;
        }
        grad_logits[base + targets[row] as usize] -= 1.0;
        for index in 0..classes {
            grad_logits[base + index] *= scale;
        }
    }
}

pub fn gelu_forward(input: &[f32], out: &mut [f32]) {
    let constant = (2.0f32 / std::f32::consts::PI).sqrt();
    for index in 0..input.len() {
        let value = input[index];
        let inner = constant * (value + 0.044_715 * value * value * value);
        out[index] = 0.5 * value * (1.0 + inner.tanh());
    }
}

pub fn gelu_backward(input: &[f32], grad_out: &[f32], grad_in: &mut [f32]) {
    let constant = (2.0f32 / std::f32::consts::PI).sqrt();
    for index in 0..input.len() {
        let value = input[index];
        let inner = constant * (value + 0.044_715 * value * value * value);
        let tanh = inner.tanh();
        let first = 0.5 * (1.0 + tanh);
        let derivative = 0.5 * (1.0 - tanh * tanh) * constant
            * (1.0 + 3.0 * 0.044_715 * value * value);
        grad_in[index] += grad_out[index] * (first + value * derivative);
    }
}

pub fn silu_forward(input: &[f32], out: &mut [f32]) {
    for index in 0..input.len() {
        let value = input[index];
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        out[index] = value * sigmoid;
    }
}

pub fn silu_backward(input: &[f32], grad_out: &[f32], grad_in: &mut [f32]) {
    for index in 0..input.len() {
        let value = input[index];
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        grad_in[index] += grad_out[index] * sigmoid * (1.0 + value * (1.0 - sigmoid));
    }
}

pub fn layer_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<()> {
    if dim == 0 || input.len() != rows * dim || gamma.len() != dim || beta.len() != dim {
        return Err(NativeError::shape("layer-norm shape mismatch"));
    }
    for row in 0..rows {
        let base = row * dim;
        let mut mean = 0.0f64;
        for index in 0..dim {
            mean += input[base + index] as f64;
        }
        mean /= dim as f64;
        let mut variance = 0.0f64;
        for index in 0..dim {
            let delta = input[base + index] as f64 - mean;
            variance += delta * delta;
        }
        variance /= dim as f64;
        let inverse = 1.0 / (variance + eps as f64).sqrt();
        for index in 0..dim {
            let normalized = ((input[base + index] as f64 - mean) * inverse) as f32;
            out[base + index] = normalized * gamma[index] + beta[index];
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(
    input: &[f32],
    gamma: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
    grad_gamma: &mut [f32],
    grad_beta: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) {
    for row in 0..rows {
        let base = row * dim;
        let mut mean = 0.0f64;
        for index in 0..dim {
            mean += input[base + index] as f64;
        }
        mean /= dim as f64;
        let mut variance = 0.0f64;
        for index in 0..dim {
            let delta = input[base + index] as f64 - mean;
            variance += delta * delta;
        }
        variance /= dim as f64;
        let inverse = 1.0 / (variance + eps as f64).sqrt();
        let mut sum_dy = 0.0f32;
        let mut sum_dy_yhat = 0.0f32;
        for index in 0..dim {
            let normalized = ((input[base + index] as f64 - mean) * inverse) as f32;
            let dy = grad_out[base + index];
            sum_dy += dy;
            sum_dy_yhat += dy * normalized;
            grad_gamma[index] += dy * normalized;
            grad_beta[index] += dy;
        }
        for index in 0..dim {
            let normalized = ((input[base + index] as f64 - mean) * inverse) as f32;
            let dy = grad_out[base + index];
            let dx = inverse as f32 / dim as f32
                * (dim as f32 * dy * gamma[index] - sum_dy * gamma[index]
                    - normalized * sum_dy_yhat * gamma[index]);
            grad_input[base + index] += dx;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    running_mean: &mut [f32],
    running_var: &mut [f32],
    out: &mut [f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
    momentum: f32,
    training: bool,
) -> Result<()> {
    let spatial = n * h * w;
    if spatial == 0 || input.len() != n * c * h * w || gamma.len() != c || beta.len() != c {
        return Err(NativeError::shape("batch-norm shape mismatch"));
    }
    for channel in 0..c {
        let mut mean = 0.0f64;
        if training {
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        mean += input[((batch * c + channel) * h + row) * w + column] as f64;
                    }
                }
            }
            mean /= spatial as f64;
            let mut variance = 0.0f64;
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let delta = input[((batch * c + channel) * h + row) * w + column] as f64
                            - mean;
                        variance += delta * delta;
                    }
                }
            }
            variance /= spatial as f64;
            let momentum64 = momentum as f64;
            running_mean[channel] = ((1.0 - momentum64) * running_mean[channel] as f64
                + momentum64 * mean) as f32;
            let unbiased = if spatial > 1 {
                variance * spatial as f64 / (spatial - 1) as f64
            } else {
                variance
            };
            running_var[channel] = ((1.0 - momentum64) * running_var[channel] as f64
                + momentum64 * unbiased) as f32;
            let inverse = 1.0 / (variance + eps as f64).sqrt();
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let index = ((batch * c + channel) * h + row) * w + column;
                        let normalized = ((input[index] as f64 - mean) * inverse) as f32;
                        out[index] = normalized * gamma[channel] + beta[channel];
                    }
                }
            }
        } else {
            let inverse = 1.0 / (running_var[channel] as f64 + eps as f64).sqrt();
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let index = ((batch * c + channel) * h + row) * w + column;
                        let normalized = ((input[index] as f64 - running_mean[channel] as f64)
                            * inverse) as f32;
                        out[index] = normalized * gamma[channel] + beta[channel];
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn batch_norm_backward(
    input: &[f32],
    gamma: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
    grad_gamma: &mut [f32],
    grad_beta: &mut [f32],
    running_mean: &[f32],
    running_var: &[f32],
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    eps: f32,
    training: bool,
) {
    let spatial = n * h * w;
    for channel in 0..c {
        if training {
            let mut mean = 0.0f64;
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        mean += input[((batch * c + channel) * h + row) * w + column] as f64;
                    }
                }
            }
            mean /= spatial as f64;
            let mut variance = 0.0f64;
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let delta = input[((batch * c + channel) * h + row) * w + column] as f64
                            - mean;
                        variance += delta * delta;
                    }
                }
            }
            variance /= spatial as f64;
            let inverse = 1.0 / (variance + eps as f64).sqrt();
            let mut sum_dy = 0.0f32;
            let mut sum_dy_yhat = 0.0f32;
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let index = ((batch * c + channel) * h + row) * w + column;
                        let normalized = ((input[index] as f64 - mean) * inverse) as f32;
                        let dy = grad_out[index];
                        sum_dy += dy;
                        sum_dy_yhat += dy * normalized;
                        grad_gamma[channel] += dy * normalized;
                        grad_beta[channel] += dy;
                    }
                }
            }
            let scale = inverse as f32 / spatial as f32;
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let index = ((batch * c + channel) * h + row) * w + column;
                        let normalized = ((input[index] as f64 - mean) * inverse) as f32;
                        grad_input[index] += scale
                            * (spatial as f32 * grad_out[index] * gamma[channel]
                                - sum_dy * gamma[channel]
                                - normalized * sum_dy_yhat * gamma[channel]);
                    }
                }
            }
        } else {
            let inverse = (1.0 / (running_var[channel] as f64 + eps as f64).sqrt()) as f32;
            for batch in 0..n {
                for row in 0..h {
                    for column in 0..w {
                        let index = ((batch * c + channel) * h + row) * w + column;
                        grad_input[index] += grad_out[index] * gamma[channel] * inverse;
                        grad_gamma[channel] += grad_out[index]
                            * ((input[index] - running_mean[channel]) * inverse);
                        grad_beta[channel] += grad_out[index];
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_forward(
    input: &[f32],
    weight: &[f32],
    out: &mut [f32],
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
    pool: Option<&ThreadPool>,
) -> Result<()> {
    let groups = groups.max(1);
    if cin % groups != 0 || cout % groups != 0 {
        return Err(NativeError::shape("conv2d channels must be divisible by groups"));
    }
    let cin_per_group = cin / groups;
    let cout_per_group = cout / groups;
    let kernel = cin_per_group * kh * kw;
    if input.len() != n * cin * h * w
        || weight.len() != cout * kernel
        || out.len() != n * cout * oh * ow
    {
        return Err(NativeError::shape("conv2d shape mismatch"));
    }
    if groups == 1 && dh == 1 && dw == 1 {
        let rows = n * oh * ow;
        let mut column = vec![0.0f32; rows * kernel];
        im2col(
            input, &mut column, n, cin, h, w, kh, kw, sh, sw, ph, pw, oh, ow,
        );
        gemm_forward_nt(&column, weight, out, rows, cout, kernel, pool);
        return Ok(());
    }
    for batch in 0..n {
        for g in 0..groups {
            for oc in 0..cout_per_group {
                let cout_idx = g * cout_per_group + oc;
                let w_base = cout_idx * kernel;
                for out_r in 0..oh {
                    for out_c in 0..ow {
                        let mut sum = 0.0f32;
                        for ic in 0..cin_per_group {
                            let cin_idx = g * cin_per_group + ic;
                            let in_base = ((batch * cin + cin_idx) * h) * w;
                            let w_ic_base = w_base + ic * kh * kw;
                            for kr in 0..kh {
                                let in_r = out_r as isize * sh as isize
                                    + kr as isize * dh as isize
                                    - ph as isize;
                                if in_r < 0 || in_r >= h as isize {
                                    continue;
                                }
                                for kc in 0..kw {
                                    let in_c = out_c as isize * sw as isize
                                        + kc as isize * dw as isize
                                        - pw as isize;
                                    if in_c < 0 || in_c >= w as isize {
                                        continue;
                                    }
                                    sum += input[in_base + in_r as usize * w + in_c as usize]
                                        * weight[w_ic_base + kr * kw + kc];
                                }
                            }
                        }
                        out[((batch * cout + cout_idx) * oh + out_r) * ow + out_c] = sum;
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_input(
    input: &[f32],
    weight: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
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
    pool: Option<&ThreadPool>,
) -> Result<()> {
    let groups = groups.max(1);
    let cin_per_group = cin / groups;
    let cout_per_group = cout / groups;
    let kernel = cin_per_group * kh * kw;
    if groups == 1 && dh == 1 && dw == 1 {
        let rows = n * oh * ow;
        let mut grad_column = vec![0.0f32; rows * kernel];
        gemm_forward_nn(grad_out, weight, &mut grad_column, rows, kernel, cout, pool);
        col2im(
            &grad_column, grad_input, n, cin, h, w, kh, kw, sh, sw, ph, pw, oh, ow,
        );
        let _ = input;
        return Ok(());
    }
    for batch in 0..n {
        for g in 0..groups {
            for oc in 0..cout_per_group {
                let cout_idx = g * cout_per_group + oc;
                let w_base = cout_idx * kernel;
                for out_r in 0..oh {
                    for out_c in 0..ow {
                        let go = grad_out[((batch * cout + cout_idx) * oh + out_r) * ow + out_c];
                        for ic in 0..cin_per_group {
                            let cin_idx = g * cin_per_group + ic;
                            let in_base = ((batch * cin + cin_idx) * h) * w;
                            let w_ic_base = w_base + ic * kh * kw;
                            for kr in 0..kh {
                                let in_r = out_r as isize * sh as isize
                                    + kr as isize * dh as isize
                                    - ph as isize;
                                if in_r < 0 || in_r >= h as isize {
                                    continue;
                                }
                                for kc in 0..kw {
                                    let in_c = out_c as isize * sw as isize
                                        + kc as isize * dw as isize
                                        - pw as isize;
                                    if in_c < 0 || in_c >= w as isize {
                                        continue;
                                    }
                                    grad_input[in_base + in_r as usize * w + in_c as usize] +=
                                        go * weight[w_ic_base + kr * kw + kc];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = input;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_backward_weight(
    input: &[f32],
    grad_out: &[f32],
    grad_weight: &mut [f32],
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
    pool: Option<&ThreadPool>,
) -> Result<()> {
    let groups = groups.max(1);
    let cin_per_group = cin / groups;
    let cout_per_group = cout / groups;
    let kernel = cin_per_group * kh * kw;
    if groups == 1 && dh == 1 && dw == 1 {
        let rows = n * oh * ow;
        let mut column = vec![0.0f32; rows * kernel];
        im2col(
            input, &mut column, n, cin, h, w, kh, kw, sh, sw, ph, pw, oh, ow,
        );
        let update = |channel: usize, target: &mut [f32]| {
            for inner in 0..kernel {
                let mut sum = 0.0f32;
                for row in 0..rows {
                    sum += grad_out[row * cout + channel] * column[row * kernel + inner];
                }
                target[inner] += sum;
            }
        };
        if let Some(pool) = pool {
            if rows * kernel * cout > 262_144 && cout >= 2 {
                pool.install(|| {
                    grad_weight
                        .par_chunks_mut(kernel)
                        .enumerate()
                        .for_each(|(channel, target)| update(channel, target));
                });
                return Ok(());
            }
        }
        for channel in 0..cout {
            update(channel, &mut grad_weight[channel * kernel..(channel + 1) * kernel]);
        }
        return Ok(());
    }
    for batch in 0..n {
        for g in 0..groups {
            for oc in 0..cout_per_group {
                let cout_idx = g * cout_per_group + oc;
                let w_base = cout_idx * kernel;
                for out_r in 0..oh {
                    for out_c in 0..ow {
                        let go = grad_out[((batch * cout + cout_idx) * oh + out_r) * ow + out_c];
                        for ic in 0..cin_per_group {
                            let cin_idx = g * cin_per_group + ic;
                            let in_base = ((batch * cin + cin_idx) * h) * w;
                            let w_ic_base = w_base + ic * kh * kw;
                            for kr in 0..kh {
                                let in_r = out_r as isize * sh as isize
                                    + kr as isize * dh as isize
                                    - ph as isize;
                                if in_r < 0 || in_r >= h as isize {
                                    continue;
                                }
                                for kc in 0..kw {
                                    let in_c = out_c as isize * sw as isize
                                        + kc as isize * dw as isize
                                        - pw as isize;
                                    if in_c < 0 || in_c >= w as isize {
                                        continue;
                                    }
                                    grad_weight[w_ic_base + kr * kw + kc] +=
                                        go * input[in_base + in_r as usize * w + in_c as usize];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_forward(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    n: usize,
    c: usize,
    spatial: usize,
    groups: usize,
    eps: f32,
) -> Result<()> {
    if groups == 0 || c % groups != 0 || spatial == 0 || input.len() != n * c * spatial
        || gamma.len() != c || beta.len() != c
    {
        return Err(NativeError::shape("group-norm shape mismatch"));
    }
    let channels_per_group = c / groups;
    let group_elems = channels_per_group * spatial;
    for batch in 0..n {
        for g in 0..groups {
            let c_start = g * channels_per_group;
            let mut mean = 0.0f64;
            for ch in 0..channels_per_group {
                let base = ((batch * c) + c_start + ch) * spatial;
                for s in 0..spatial {
                    mean += input[base + s] as f64;
                }
            }
            mean /= group_elems as f64;
            let mut var = 0.0f64;
            for ch in 0..channels_per_group {
                let base = ((batch * c) + c_start + ch) * spatial;
                for s in 0..spatial {
                    let d = input[base + s] as f64 - mean;
                    var += d * d;
                }
            }
            var /= group_elems as f64;
            let inv_std = 1.0 / (var + eps as f64).sqrt();
            for ch in 0..channels_per_group {
                let channel = c_start + ch;
                let base = ((batch * c) + channel) * spatial;
                let gm = gamma[channel];
                let bt = beta[channel];
                for s in 0..spatial {
                    let x_hat = ((input[base + s] as f64 - mean) * inv_std) as f32;
                    out[base + s] = x_hat * gm + bt;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_backward(
    input: &[f32],
    gamma: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
    grad_gamma: &mut [f32],
    grad_beta: &mut [f32],
    n: usize,
    c: usize,
    spatial: usize,
    groups: usize,
    eps: f32,
) {
    let channels_per_group = c / groups;
    let group_elems = channels_per_group * spatial;
    for batch in 0..n {
        for g in 0..groups {
            let c_start = g * channels_per_group;
            let mut mean = 0.0f64;
            for ch in 0..channels_per_group {
                let base = ((batch * c) + c_start + ch) * spatial;
                for s in 0..spatial {
                    mean += input[base + s] as f64;
                }
            }
            mean /= group_elems as f64;
            let mut var = 0.0f64;
            for ch in 0..channels_per_group {
                let base = ((batch * c) + c_start + ch) * spatial;
                for s in 0..spatial {
                    let d = input[base + s] as f64 - mean;
                    var += d * d;
                }
            }
            var /= group_elems as f64;
            let inv_std = (1.0 / (var + eps as f64).sqrt()) as f32;
            let mut sum_dy_gamma = 0.0f32;
            let mut sum_dy_gamma_xhat = 0.0f32;
            for ch in 0..channels_per_group {
                let channel = c_start + ch;
                let base = ((batch * c) + channel) * spatial;
                let gm = gamma[channel];
                for s in 0..spatial {
                    let x_hat = ((input[base + s] as f64 - mean) as f32) * inv_std;
                    let dy = grad_out[base + s];
                    grad_gamma[channel] += dy * x_hat;
                    grad_beta[channel] += dy;
                    sum_dy_gamma += dy * gm;
                    sum_dy_gamma_xhat += dy * gm * x_hat;
                }
            }
            let scale = inv_std / (group_elems as f32);
            for ch in 0..channels_per_group {
                let channel = c_start + ch;
                let base = ((batch * c) + channel) * spatial;
                let gm = gamma[channel];
                for s in 0..spatial {
                    let x_hat = ((input[base + s] as f64 - mean) as f32) * inv_std;
                    let dy = grad_out[base + s];
                    grad_input[base + s] += scale
                        * ((group_elems as f32) * dy * gm - sum_dy_gamma - x_hat * sum_dy_gamma_xhat);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn im2col(
    input: &[f32],
    column: &mut [f32],
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) {
    let kernel = cin * kh * kw;
    for batch in 0..n {
        for out_row in 0..oh {
            for out_column in 0..ow {
                let row = (batch * oh + out_row) * ow + out_column;
                for channel in 0..cin {
                    for kernel_row in 0..kh {
                        let input_row = out_row as isize * sh as isize + kernel_row as isize
                            - ph as isize;
                        for kernel_column in 0..kw {
                            let input_column = out_column as isize * sw as isize
                                + kernel_column as isize
                                - pw as isize;
                            let target = row * kernel
                                + (channel * kh + kernel_row) * kw
                                + kernel_column;
                            column[target] = if input_row >= 0
                                && input_row < h as isize
                                && input_column >= 0
                                && input_column < w as isize
                            {
                                input[((batch * cin + channel) * h + input_row as usize) * w
                                    + input_column as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn col2im(
    column: &[f32],
    input: &mut [f32],
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    oh: usize,
    ow: usize,
) {
    let kernel = cin * kh * kw;
    for batch in 0..n {
        for out_row in 0..oh {
            for out_column in 0..ow {
                let row = (batch * oh + out_row) * ow + out_column;
                for channel in 0..cin {
                    for kernel_row in 0..kh {
                        let input_row = out_row as isize * sh as isize + kernel_row as isize
                            - ph as isize;
                        if input_row < 0 || input_row >= h as isize {
                            continue;
                        }
                        for kernel_column in 0..kw {
                            let input_column = out_column as isize * sw as isize
                                + kernel_column as isize
                                - pw as isize;
                            if input_column < 0 || input_column >= w as isize {
                                continue;
                            }
                            let source = row * kernel
                                + (channel * kh + kernel_row) * kw
                                + kernel_column;
                            input[((batch * cin + channel) * h + input_row as usize) * w
                                + input_column as usize] += column[source];
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_forward(
    input: &[f32],
    out: &mut [f32],
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
) -> Result<()> {
    if input.len() != n * c * h * w || out.len() != n * c * oh * ow {
        return Err(NativeError::shape("max-pool shape mismatch"));
    }
    for batch in 0..n {
        for channel in 0..c {
            for out_row in 0..oh {
                for out_column in 0..ow {
                    let mut maximum = f32::NEG_INFINITY;
                    for kernel_row in 0..kh {
                        let input_row = out_row * sh + kernel_row;
                        if input_row >= h {
                            continue;
                        }
                        for kernel_column in 0..kw {
                            let input_column = out_column * sw + kernel_column;
                            if input_column >= w {
                                continue;
                            }
                            maximum = maximum.max(
                                input[((batch * c + channel) * h + input_row) * w + input_column],
                            );
                        }
                    }
                    out[((batch * c + channel) * oh + out_row) * ow + out_column] = maximum;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool_backward(
    input: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
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
) {
    for batch in 0..n {
        for channel in 0..c {
            for out_row in 0..oh {
                for out_column in 0..ow {
                    let mut maximum = f32::NEG_INFINITY;
                    let mut maximum_row = 0usize;
                    let mut maximum_column = 0usize;
                    for kernel_row in 0..kh {
                        let input_row = out_row * sh + kernel_row;
                        if input_row >= h {
                            continue;
                        }
                        for kernel_column in 0..kw {
                            let input_column = out_column * sw + kernel_column;
                            if input_column >= w {
                                continue;
                            }
                            let value = input
                                [((batch * c + channel) * h + input_row) * w + input_column];
                            if value > maximum {
                                maximum = value;
                                maximum_row = input_row;
                                maximum_column = input_column;
                            }
                        }
                    }
                    grad_input[((batch * c + channel) * h + maximum_row) * w + maximum_column] +=
                        grad_out[((batch * c + channel) * oh + out_row) * ow + out_column];
                }
            }
        }
    }
}

fn dropout_mask(length: usize, probability: f32, training: bool, seed: u64) -> Vec<f32> {
    if !training || probability <= 0.0 {
        return vec![1.0; length];
    }
    let mut state = seed | 1;
    let scale = 1.0 / (1.0 - probability);
    let mut mask = vec![0.0f32; length];
    for value in &mut mask {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        *value = if unit >= probability { scale } else { 0.0 };
    }
    mask
}

pub fn dropout_forward(
    input: &[f32],
    out: &mut [f32],
    probability: f32,
    training: bool,
    seed: u64,
) -> Result<()> {
    if !(0.0..1.0).contains(&probability) {
        return Err(NativeError::invalid("dropout probability must be in [0, 1)"));
    }
    let mask = dropout_mask(input.len(), probability, training, seed);
    for index in 0..input.len() {
        out[index] = input[index] * mask[index];
    }
    Ok(())
}

pub fn dropout_backward(
    grad_out: &[f32],
    grad_in: &mut [f32],
    probability: f32,
    training: bool,
    seed: u64,
) {
    let mask = dropout_mask(grad_out.len(), probability, training, seed);
    for index in 0..grad_out.len() {
        grad_in[index] += grad_out[index] * mask[index];
    }
}

pub fn embedding_forward(
    indices: &[f32],
    weight: &[f32],
    out: &mut [f32],
    rows: usize,
    dim: usize,
) -> Result<()> {
    let vocab = weight.len() / dim;
    if weight.len() % dim != 0 || out.len() != rows * dim || indices.len() != rows {
        return Err(NativeError::shape("embedding shape mismatch"));
    }
    for row in 0..rows {
        let index = indices[row];
        if !index.is_finite() || index < 0.0 || index >= vocab as f32 {
            return Err(NativeError::invalid(format!(
                "embedding index {index} is outside [0, {vocab})"
            )));
        }
        let index = index as usize;
        out[row * dim..(row + 1) * dim]
            .copy_from_slice(&weight[index * dim..(index + 1) * dim]);
    }
    Ok(())
}

pub fn embedding_backward(
    indices: &[f32],
    grad_out: &[f32],
    grad_weight: &mut [f32],
    rows: usize,
    dim: usize,
) {
    for row in 0..rows {
        let index = indices[row] as usize;
        for column in 0..dim {
            grad_weight[index * dim + column] += grad_out[row * dim + column];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batched_matmul_forward(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    pool: Option<&ThreadPool>,
) -> Result<()> {
    if a.len() != batch * m * k || b.len() != batch * k * n || out.len() != batch * m * n {
        return Err(NativeError::shape("batched-matmul shape mismatch"));
    }
    for index in 0..batch {
        gemm_f32(
            &a[index * m * k..(index + 1) * m * k],
            &b[index * k * n..(index + 1) * k * n],
            &mut out[index * m * n..(index + 1) * m * n],
            m,
            k,
            n,
            pool,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn batched_matmul_backward_a(
    b: &[f32],
    grad_out: &[f32],
    grad_a: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    pool: Option<&ThreadPool>,
) {
    for index in 0..batch {
        gemm_nt(
            &grad_out[index * m * n..(index + 1) * m * n],
            &b[index * k * n..(index + 1) * k * n],
            &mut grad_a[index * m * k..(index + 1) * m * k],
            m,
            k,
            n,
            pool,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn batched_matmul_backward_b(
    a: &[f32],
    grad_out: &[f32],
    grad_b: &mut [f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    pool: Option<&ThreadPool>,
) {
    for index in 0..batch {
        gemm_tn(
            &a[index * m * k..(index + 1) * m * k],
            &grad_out[index * m * n..(index + 1) * m * n],
            &mut grad_b[index * k * n..(index + 1) * k * n],
            m,
            k,
            n,
            pool,
        );
    }
}

pub fn permute_forward(
    input: &[f32],
    out: &mut [f32],
    in_shape: &[usize],
    perm: &[usize],
) -> Result<()> {
    let rank = in_shape.len();
    if rank == 0 || perm.len() != rank || input.len() != numel(in_shape) || out.len() != input.len()
    {
        return Err(NativeError::shape("permute shape mismatch"));
    }
    let mut seen = vec![false; rank];
    for &dimension in perm {
        if dimension >= rank || seen[dimension] {
            return Err(NativeError::shape("permute is not a valid permutation"));
        }
        seen[dimension] = true;
    }
    let mut out_shape = vec![0usize; rank];
    for index in 0..rank {
        out_shape[index] = in_shape[perm[index]];
    }
    let in_strides = strides(in_shape);
    let out_strides = strides(&out_shape);
    let mut coordinates = vec![0usize; rank];
    for output_index in 0..out.len() {
        let mut input_index = 0usize;
        for dimension in 0..rank {
            input_index += coordinates[dimension] * in_strides[perm[dimension]];
        }
        out[output_index] = input[input_index];
        let _ = out_strides;
        increment(&mut coordinates, &out_shape);
    }
    Ok(())
}

pub fn permute_backward(
    grad_out: &[f32],
    grad_in: &mut [f32],
    in_shape: &[usize],
    perm: &[usize],
) -> Result<()> {
    let rank = in_shape.len();
    let mut out_shape = vec![0usize; rank];
    for index in 0..rank {
        out_shape[index] = in_shape[perm[index]];
    }
    let in_strides = strides(in_shape);
    let mut coordinates = vec![0usize; rank];
    for output_index in 0..grad_out.len() {
        let mut input_index = 0usize;
        for dimension in 0..rank {
            input_index += coordinates[dimension] * in_strides[perm[dimension]];
        }
        grad_in[input_index] += grad_out[output_index];
        increment(&mut coordinates, &out_shape);
    }
    Ok(())
}

pub fn reshape_forward(input: &[f32], out: &mut [f32]) -> Result<()> {
    if input.len() != out.len() {
        return Err(NativeError::shape("reshape requires equal element counts"));
    }
    out.copy_from_slice(input);
    Ok(())
}

pub fn reshape_backward(grad_out: &[f32], grad_in: &mut [f32]) -> Result<()> {
    if grad_out.len() != grad_in.len() {
        return Err(NativeError::shape("reshape backward requires equal element counts"));
    }
    for index in 0..grad_in.len() {
        grad_in[index] += grad_out[index];
    }
    Ok(())
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    strides
}

fn increment(coordinates: &mut [usize], shape: &[usize]) {
    for dimension in (0..shape.len()).rev() {
        coordinates[dimension] += 1;
        if coordinates[dimension] < shape[dimension] {
            return;
        }
        coordinates[dimension] = 0;
    }
}
