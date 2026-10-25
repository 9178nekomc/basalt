//! Hand-written x86-64 kernels used by the tensor engine.
//!
//! The GEMM loop is intentionally `i -> k -> j` with `j` innermost, as required by the cache
//! model: `b[k, :]` is streamed once per row block and `c[i, :]` stays resident in L1. Rows are
//! processed six at a time so each `b` cache line feeds six FMAs instead of one. AVX2/FMA is
//! selected at runtime; every kernel has a scalar fallback for non-x86 hosts and for CPUs without
//! FMA.

use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
pub fn has_avx2_fma() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(not(target_arch = "x86_64"))]
pub fn has_avx2_fma() -> bool {
    false
}

/// `y += alpha * x`, with AVX2/FMA when available.
#[inline]
pub fn axpy(y: &mut [f32], x: &[f32], alpha: f32) {
    let length = y.len().min(x.len());
    if length == 0 || alpha == 0.0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: the function checks the CPU feature bits before entering the intrinsic
            // block, and all loads/stores are bounded by `length`.
            unsafe {
                axpy_avx2(&mut y[..length], &x[..length], alpha);
            }
            return;
        }
    }
    for index in 0..length {
        y[index] += alpha * x[index];
    }
}

/// Dot product with AVX2/FMA when available.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let length = a.len().min(b.len());
    if length == 0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // SAFETY: feature detection precedes the intrinsic block; bounds are clamped.
            return unsafe { dot_avx2(&a[..length], &b[..length]) };
        }
    }
    let mut sum = 0.0f32;
    for index in 0..length {
        sum = a[index].mul_add(b[index], sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(y: &mut [f32], x: &[f32], alpha: f32) {
    use std::arch::x86_64::*;
    let length = y.len();
    let mut index = 0usize;
    let broadcast = _mm256_set1_ps(alpha);
    while index + 32 <= length {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(index));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(index + 8));
        let x2 = _mm256_loadu_ps(x.as_ptr().add(index + 16));
        let x3 = _mm256_loadu_ps(x.as_ptr().add(index + 24));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(index));
        let y1 = _mm256_loadu_ps(y.as_ptr().add(index + 8));
        let y2 = _mm256_loadu_ps(y.as_ptr().add(index + 16));
        let y3 = _mm256_loadu_ps(y.as_ptr().add(index + 24));
        _mm256_storeu_ps(y.as_mut_ptr().add(index), _mm256_fmadd_ps(broadcast, x0, y0));
        _mm256_storeu_ps(
            y.as_mut_ptr().add(index + 8),
            _mm256_fmadd_ps(broadcast, x1, y1),
        );
        _mm256_storeu_ps(
            y.as_mut_ptr().add(index + 16),
            _mm256_fmadd_ps(broadcast, x2, y2),
        );
        _mm256_storeu_ps(
            y.as_mut_ptr().add(index + 24),
            _mm256_fmadd_ps(broadcast, x3, y3),
        );
        index += 32;
    }
    while index + 8 <= length {
        let xv = _mm256_loadu_ps(x.as_ptr().add(index));
        let yv = _mm256_loadu_ps(y.as_ptr().add(index));
        _mm256_storeu_ps(
            y.as_mut_ptr().add(index),
            _mm256_fmadd_ps(broadcast, xv, yv),
        );
        index += 8;
    }
    while index < length {
        *y.get_unchecked_mut(index) += alpha * *x.get_unchecked(index);
        index += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let length = a.len();
    let mut index = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    while index + 32 <= length {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(index)),
            _mm256_loadu_ps(b.as_ptr().add(index)),
            acc0,
        );
        acc1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(index + 8)),
            _mm256_loadu_ps(b.as_ptr().add(index + 8)),
            acc1,
        );
        acc2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(index + 16)),
            _mm256_loadu_ps(b.as_ptr().add(index + 16)),
            acc2,
        );
        acc3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(index + 24)),
            _mm256_loadu_ps(b.as_ptr().add(index + 24)),
            acc3,
        );
        index += 32;
    }
    while index + 8 <= length {
        acc0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(index)),
            _mm256_loadu_ps(b.as_ptr().add(index)),
            acc0,
        );
        index += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc0);
    let mut sum = 0.0f32;
    for lane in lanes {
        sum += lane;
    }
    while index < length {
        sum = a.get_unchecked(index).mul_add(*b.get_unchecked(index), sum);
        index += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_axpy(y: &mut [f32], x: &[f32], alpha: f32) {
        for index in 0..y.len() {
            y[index] += alpha * x[index];
        }
    }

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for index in 0..a.len() {
            sum = a[index].mul_add(b[index], sum);
        }
        sum
    }

    #[test]
    fn axpy_matches_reference_for_all_tail_lengths() {
        for length in 0..=97 {
            let x: Vec<f32> = (0..length).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut expected = vec![0.25f32; length];
            let mut actual = expected.clone();
            reference_axpy(&mut expected, &x, -1.75);
            axpy(&mut actual, &x, -1.75);
            for (left, right) in expected.iter().zip(&actual) {
                assert!((left - right).abs() <= 1e-6, "length={length}");
            }
        }
    }

    #[test]
    fn dot_matches_reference() {
        for length in 0..=131 {
            let a: Vec<f32> = (0..length).map(|i| (i as f32 * 0.11).cos()).collect();
            let b: Vec<f32> = (0..length).map(|i| (i as f32 * 0.07).sin()).collect();
            let expected = reference_dot(&a, &b);
            let actual = dot(&a, &b);
            let tolerance = 1e-4 * (1.0 + expected.abs());
            assert!((expected - actual).abs() <= tolerance, "length={length}");
        }
    }
}
