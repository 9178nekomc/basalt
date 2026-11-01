//! Blocked GEMM with an AVX2/FMA register-tiled micro-kernel.
//!
//! The previous implementation walked `i -> k -> j` and issued one rank-1 update per
//! inner step: for every multiply-add it touched an `n`-wide row of B, so the loop ran
//! at memory bandwidth rather than at the CPU's FLOP capability (measured ~18 GFLOPS on
//! a 4-core Zen 3, roughly 8% of peak).
//!
//! This module keeps an 8x8 tile of the result in registers, so each pair of loaded
//! vectors feeds 8 FMAs instead of 1. B is packed into a contiguous panel per (kc, nc)
//! block so the micro-kernel reads sequential memory, and the packed panel is reused
//! across every row block.
//!
//! Everything falls back to a portable scalar kernel when AVX2/FMA is absent or the
//! matrix is too small to tile.

use rayon::prelude::*;
use rayon::ThreadPool;

/// Rows of C held in registers by the micro-kernel.
pub const MR: usize = 6;
/// Columns of C held in registers by the micro-kernel.
pub const NR: usize = 16;
/// Depth of the packed panel; keeps the working set inside L2.
const KC: usize = 512;
/// Rows of C per parallel task.
const MC: usize = 30;   // 必须是 MR 的整数倍
/// Columns of C per packed panel.
const NC: usize = 256;  // 必须是 NR 的整数倍

/// Which operand layout B uses.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// B is `[k, n]` row-major.
    Nn,
    /// B is `[n, k]` row-major (the transpose of the logical operand).
    Nt,
}

/// C = A * B, with A `[m, k]` row-major and C `[m, n]` row-major.
///
/// `c` is overwritten, not accumulated into.
pub fn matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    layout: Layout,
    pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    c.fill(0.0);
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    // 开关：用于在分块逻辑与微内核之间做二分定位，也可在没有 AVX2 的机器上强制标量
    let force_scalar = std::env::var("BASALT_GEMM_SCALAR").is_ok();

    #[cfg(target_arch = "x86_64")]
    {
        if !force_scalar && avx2_available() {
            // SAFETY: guarded by the runtime feature check above.
            unsafe { blocked(a, b, c, m, k, n, layout, pool) };
            return;
        }
    }
    let _ = force_scalar;
    scalar(a, b, c, m, k, n, layout, pool);
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Pack `B[pc..pc+kc, jc..jc+nc]` into a `kc * nc` row-major panel.
fn pack_b(b: &[f32], k: usize, n: usize, pc: usize, jc: usize, kc: usize, nc: usize,
          layout: Layout, panel: &mut [f32]) {
    match layout {
        Layout::Nn => {
            for p in 0..kc {
                let src = (pc + p) * n + jc;
                panel[p * nc..p * nc + nc].copy_from_slice(&b[src..src + nc]);
            }
        }
        Layout::Nt => {
            // B is [n, k]: gather a column of the logical matrix into a panel row.
            for p in 0..kc {
                let row = &mut panel[p * nc..p * nc + nc];
                for j in 0..nc {
                    row[j] = b[(jc + j) * k + pc + p];
                }
            }
        }
    }
}

/// 6x16 register tile: C[0..6, 0..16] += A[0..6, 0..k] * panel[0..k, 0..16].
///
/// 每步 k 只加载 6 个 A 标量（广播）与 2 个 B 向量，却发出 12 次 FMA，
/// 因此每个 FMA 只摊到 0.67 次加载——比 8x8 的 1.1 次低得多，这是吞吐的关键。
///
/// # Safety
/// Caller must guarantee AVX2 + FMA, that `a` has `6 * lda` readable floats,
/// `panel` has `k * nc` readable floats, and `c` has `6 * ldc` writable floats.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn kernel_6x16(
    k: usize,
    a: *const f32,
    lda: usize,
    panel: *const f32,
    nc: usize,
    c: *mut f32,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    let mut c00 = _mm256_loadu_ps(c);
    let mut c01 = _mm256_loadu_ps(c.add(8));
    let mut c10 = _mm256_loadu_ps(c.add(ldc));
    let mut c11 = _mm256_loadu_ps(c.add(ldc + 8));
    let mut c20 = _mm256_loadu_ps(c.add(2 * ldc));
    let mut c21 = _mm256_loadu_ps(c.add(2 * ldc + 8));
    let mut c30 = _mm256_loadu_ps(c.add(3 * ldc));
    let mut c31 = _mm256_loadu_ps(c.add(3 * ldc + 8));
    let mut c40 = _mm256_loadu_ps(c.add(4 * ldc));
    let mut c41 = _mm256_loadu_ps(c.add(4 * ldc + 8));
    let mut c50 = _mm256_loadu_ps(c.add(5 * ldc));
    let mut c51 = _mm256_loadu_ps(c.add(5 * ldc + 8));

    for p in 0..k {
        let b0 = _mm256_loadu_ps(panel.add(p * nc));
        let b1 = _mm256_loadu_ps(panel.add(p * nc + 8));
        let a0 = _mm256_broadcast_ss(&*a.add(p));
        let a1 = _mm256_broadcast_ss(&*a.add(lda + p));
        let a2 = _mm256_broadcast_ss(&*a.add(2 * lda + p));
        let a3 = _mm256_broadcast_ss(&*a.add(3 * lda + p));
        let a4 = _mm256_broadcast_ss(&*a.add(4 * lda + p));
        let a5 = _mm256_broadcast_ss(&*a.add(5 * lda + p));
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    _mm256_storeu_ps(c, c00);
    _mm256_storeu_ps(c.add(8), c01);
    _mm256_storeu_ps(c.add(ldc), c10);
    _mm256_storeu_ps(c.add(ldc + 8), c11);
    _mm256_storeu_ps(c.add(2 * ldc), c20);
    _mm256_storeu_ps(c.add(2 * ldc + 8), c21);
    _mm256_storeu_ps(c.add(3 * ldc), c30);
    _mm256_storeu_ps(c.add(3 * ldc + 8), c31);
    _mm256_storeu_ps(c.add(4 * ldc), c40);
    _mm256_storeu_ps(c.add(4 * ldc + 8), c41);
    _mm256_storeu_ps(c.add(5 * ldc), c50);
    _mm256_storeu_ps(c.add(5 * ldc + 8), c51);
}

/// # Safety
/// Caller must guarantee AVX2 + FMA and the same bounds as [`kernel_6x16`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    layout: Layout,
    pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(MC % MR, 0, "MC 必须是 MR 的整数倍，否则尾块越界");
    debug_assert_eq!(NC % NR, 0, "NC 必须是 NR 的整数倍，否则尾块越界");
    let m_main = m - m % MR;
    let n_main = n - n % NR;
    let mut panel = vec![0.0f32; KC * NC];

    let mut jc = 0;
    while jc < n_main {
        let nc = NC.min(n_main - jc);
        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            pack_b(b, k, n, pc, jc, kc, nc, layout, &mut panel[..kc * nc]);

            let panel_ref = &panel[..kc * nc];
            let rows = m_main / MC + usize::from(m_main % MC != 0);

            let run = |block: usize, chunk: &mut [f32]| {
                let ic = block * MC;
                let mc = MC.min(m_main - ic);
                let mut ir = 0;
                while ir < mc {
                    let mut jr = 0;
                    while jr < nc {
                        // SAFETY: indices are inside the main (tiled) region and the
                        // panel covers exactly kc x nc floats.
                        unsafe {
                            kernel_6x16(
                                kc,
                                a.as_ptr().add((ic + ir) * k + pc),
                                k,
                                panel_ref.as_ptr().add(jr),
                                nc,
                                chunk.as_mut_ptr().add(ir * n + jc + jr),
                                n,
                            );
                        }
                        jr += NR;
                    }
                    ir += MR;
                }
            };

            match pool {
                Some(pool) if rows > 1 => pool.install(|| {
                    c.par_chunks_mut(MC * n)
                        .take(rows)
                        .enumerate()
                        .for_each(|(block, chunk)| run(block, chunk));
                }),
                _ => {
                    for block in 0..rows {
                        let start = block * MC * n;
                        let end = (start + MC * n).min(c.len());
                        run(block, &mut c[start..end]);
                    }
                }
            }

            pc += kc;
        }
        jc += nc;
    }

    // Remainder columns: rows already tiled, but columns beyond n_main.
    if n_main < n {
        for i in 0..m_main {
            for j in n_main..n {
                c[i * n + j] = dot_row(a, b, i, j, k, n, layout);
            }
        }
    }
    // Remainder rows: everything past m_main.
    for i in m_main..m {
        for j in 0..n {
            c[i * n + j] = dot_row(a, b, i, j, k, n, layout);
        }
    }
}

fn dot_row(a: &[f32], b: &[f32], i: usize, j: usize, k: usize, n: usize, layout: Layout) -> f32 {
    let row = &a[i * k..(i + 1) * k];
    let mut sum = 0.0f32;
    match layout {
        Layout::Nn => {
            for p in 0..k {
                sum += row[p] * b[p * n + j];
            }
        }
        Layout::Nt => {
            let column = &b[j * k..(j + 1) * k];
            for p in 0..k {
                sum += row[p] * column[p];
            }
        }
    }
    sum
}

/// Portable path: same blocking structure, scalar micro-kernel.
fn scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    layout: Layout,
    pool: Option<&ThreadPool>,
) {
    let row_block = MC;
    let work = (m as u64) * (n as u64) * (k as u64);
    let run = |chunk: &mut [f32], start: usize| {
        let rows = chunk.len() / n;
        for local in 0..rows {
            let i = start + local;
            for j in 0..n {
                chunk[local * n + j] = dot_row(a, b, i, j, k, n, layout);
            }
        }
    };

    match pool {
        Some(pool) if work >= 262_144 && m > row_block => pool.install(|| {
            c.par_chunks_mut(row_block * n)
                .enumerate()
                .for_each(|(block, chunk)| run(chunk, block * row_block));
        }),
        _ => {
            let mut start = 0;
            while start < m {
                let end = (start + row_block).min(m);
                run(&mut c[start * n..end * n], start);
                start = end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, layout: Layout) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let bv = match layout {
                        Layout::Nn => b[p * n + j],
                        Layout::Nt => b[j * k + p],
                    };
                    sum += a[i * k + p] * bv;
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    fn run_case(m: usize, k: usize, n: usize, layout: Layout) {
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
        let b: Vec<f32> = match layout {
            Layout::Nn => (0..k * n).map(|i| ((i * 5 % 11) as f32) - 5.0).collect(),
            Layout::Nt => (0..n * k).map(|i| ((i * 5 % 11) as f32) - 5.0).collect(),
        };
        let mut out = vec![0.0f32; m * n];
        matmul(&a, &b, &mut out, m, k, n, layout, None);
        let expected = reference(&a, &b, m, k, n, layout);
        for index in 0..m * n {
            assert!(
                (out[index] - expected[index]).abs() < 1e-3,
                "mismatch at {index}: {} vs {} (m={m} k={k} n={n})",
                out[index],
                expected[index]
            );
        }
    }

    #[test]
    fn matches_reference_for_aligned_sizes() {
        run_case(64, 64, 64, Layout::Nn);
        run_case(64, 64, 64, Layout::Nt);
        run_case(128, 32, 96, Layout::Nn);
    }

    #[test]
    fn matches_reference_with_remainders() {
        // 非 8 的倍数：覆盖行尾、列尾与 k 尾
        run_case(13, 7, 11, Layout::Nn);
        run_case(13, 7, 11, Layout::Nt);
        run_case(9, 33, 17, Layout::Nn);
        run_case(1, 1, 1, Layout::Nn);
        run_case(3, 5, 2, Layout::Nt);
    }

    #[test]
    fn matches_reference_with_thread_pool() {
        // 线程池路径与单线程路径必须给出相同结果——段错误就是在这条路径上暴露的
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for (m, k, n) in [(512usize, 512usize, 512usize), (128, 256, 128), (100, 200, 100)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) - 8.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) - 6.0).collect();
            let mut out = vec![0.0f32; m * n];
            matmul(&a, &b, &mut out, m, k, n, Layout::Nn, Some(&pool));
            let expected = reference(&a, &b, m, k, n, Layout::Nn);
            for index in 0..m * n {
                assert!((out[index] - expected[index]).abs() < 1e-2,
                        "pooled mismatch at {index} for {m}x{k}x{n}");
            }
        }
    }

    #[test]
    fn handles_large_k_panel_splits() {
        // k 超过 KC，强制走多面板
        run_case(16, 700, 16, Layout::Nn);
        run_case(16, 700, 16, Layout::Nt);
    }
}
