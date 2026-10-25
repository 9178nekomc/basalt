//! JSL native kernels: AVX2/FMA GEMM and K-Means assignment behind a minimal JNI surface.
//!
//! Design notes:
//! * Arrays are pinned with `GetDoubleArrayElements`; inputs use `NoCopyBack`, outputs use
//!   `Commit`. The JVM may hand back a copy, but HotSpot returns a direct pointer for primitive
//!   arrays that are not in the middle of a GC relocation.
//! * Every entry point catches panics and returns a negative status instead of unwinding across
//!   the FFI boundary. Java turns a non-zero status into an exception.
//! * GEMM transposes B once into an `n x k` panel, then evaluates contiguous AVX2 dot products.
//!   Rows are split across scoped OS threads; the caller decides the parallelism.

use jni::objects::{JByteArray, JClass, JDoubleArray, JIntArray, ReleaseMode};
use jni::sys::{jint, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

const STATUS_OK: jint = 0;
const STATUS_PANIC: jint = -2;
const STATUS_BAD_ARGS: jint = -1;

#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_available(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    JNI_TRUE as jint
}

#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_version(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    1
}

/// `int gemm(double[] a, double[] b, double[] c, int m, int k, int n, int threads)`
#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_gemm(
    mut env: JNIEnv,
    _class: JClass,
    a_array: JDoubleArray,
    b_array: JDoubleArray,
    c_array: JDoubleArray,
    m: jint,
    k: jint,
    n: jint,
    threads: jint,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (m, k, n) = (m as usize, k as usize, n as usize);
        if m == 0 || k == 0 || n == 0 {
            return STATUS_BAD_ARGS;
        }
        let a = match unsafe { env.get_array_elements(&a_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let b = match unsafe { env.get_array_elements(&b_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let mut c = match unsafe { env.get_array_elements(&c_array, ReleaseMode::CopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return STATUS_BAD_ARGS;
        }
        gemm(&a, &b, &mut c, m, k, n, threads.max(1) as usize);
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

/// `int kmeansAssign(double[] data, double[] centers, int rows, int clusters, int features,
///                   int threads, int[] assignments, double[] distances)`
#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_kmeansAssign(
    mut env: JNIEnv,
    _class: JClass,
    data_array: JDoubleArray,
    centers_array: JDoubleArray,
    rows: jint,
    clusters: jint,
    features: jint,
    threads: jint,
    assignments_array: JIntArray,
    distances_array: JDoubleArray,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (rows, clusters, features) = (rows as usize, clusters as usize, features as usize);
        if rows == 0 || clusters == 0 || features == 0 {
            return STATUS_BAD_ARGS;
        }
        let data = match unsafe { env.get_array_elements(&data_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let centers = match unsafe { env.get_array_elements(&centers_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let mut assignments =
            match unsafe { env.get_array_elements(&assignments_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let mut distances =
            match unsafe { env.get_array_elements(&distances_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        if data.len() != rows * features
            || centers.len() != clusters * features
            || assignments.len() != rows
            || distances.len() != rows
        {
            return STATUS_BAD_ARGS;
        }
        kmeans_assign(
            &data,
            &centers,
            &mut assignments,
            &mut distances,
            rows,
            clusters,
            features,
            threads.max(1) as usize,
        );
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

/// `int forestPredict(double[] data, int rows, int columns, int[] feature, double[] threshold,
///                    int[] left, int[] right, double[] value, int[] treeOffsets,
///                    double[] output, double scale)`
///
/// Tree arrays are concatenated; `treeOffsets[t]` is the first node of tree `t`, with
/// `treeOffsets[trees]` the total node count.
#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_forestPredict(
    mut env: JNIEnv,
    _class: JClass,
    data_array: JDoubleArray,
    rows: jint,
    columns: jint,
    feature_array: JIntArray,
    threshold_array: JDoubleArray,
    left_array: JIntArray,
    right_array: JIntArray,
    value_array: JDoubleArray,
    offsets_array: JIntArray,
    output_array: JDoubleArray,
    scale: f64,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (rows, columns) = (rows as usize, columns as usize);
        if rows == 0 || columns == 0 {
            return STATUS_BAD_ARGS;
        }
        let data = match unsafe { env.get_array_elements(&data_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let feature = match unsafe { env.get_array_elements(&feature_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let threshold =
            match unsafe { env.get_array_elements(&threshold_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let left = match unsafe { env.get_array_elements(&left_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let right = match unsafe { env.get_array_elements(&right_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let value = match unsafe { env.get_array_elements(&value_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let offsets =
            match unsafe { env.get_array_elements(&offsets_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let mut output =
            match unsafe { env.get_array_elements(&output_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        if data.len() != rows * columns
            || output.len() != rows
            || feature.len() != threshold.len()
            || feature.len() != left.len()
            || feature.len() != right.len()
            || feature.len() != value.len()
            || offsets.len() < 2
        {
            return STATUS_BAD_ARGS;
        }
        forest_predict(
            &data, rows, columns, &feature, &threshold, &left, &right, &value, &offsets,
            &mut output, scale,
        );
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

fn gemm(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize, threads: usize) {
    let workers = threads.max(1).min(m.max(1)).min((m / 128).max(1));
    if workers <= 1 || m < 32 {
        unsafe {
            matrixmultiply::dgemm(
                m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1,
                0.0, c.as_mut_ptr(), n as isize, 1,
            );
        }
        return;
    }
    let chunk = (m + workers - 1) / workers;
    let pool = pool_for(workers);
    pool.install(|| {
        c.par_chunks_mut(chunk * n).enumerate().for_each(|(index, c_chunk)| {
            let start = index * chunk;
            let rows = c_chunk.len() / n;
            unsafe {
                matrixmultiply::dgemm(
                    rows, k, n, 1.0, a.as_ptr().add(start * k), k as isize, 1,
                    b.as_ptr(), n as isize, 1, 0.0, c_chunk.as_mut_ptr(), n as isize, 1,
                );
            }
        });
    });
}

pub(crate) fn pool_for(threads: usize) -> Arc<ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let pools = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = pools.lock().expect("native thread-pool registry poisoned");
    guard
        .entry(threads)
        .or_insert_with(|| {
            Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(move |index| format!("jsl-native-{threads}-{index}"))
                    .build()
                    .expect("cannot create native thread pool"),
            )
        })
        .clone()
}

#[target_feature(enable = "avx2,fma")]
unsafe fn squared_distance_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let length = a.len();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut index = 0usize;
    while index + 8 <= length {
        let d0 = _mm256_sub_pd(
            _mm256_loadu_pd(a_ptr.add(index)),
            _mm256_loadu_pd(b_ptr.add(index)),
        );
        let d1 = _mm256_sub_pd(
            _mm256_loadu_pd(a_ptr.add(index + 4)),
            _mm256_loadu_pd(b_ptr.add(index + 4)),
        );
        acc0 = _mm256_fmadd_pd(d0, d0, acc0);
        acc1 = _mm256_fmadd_pd(d1, d1, acc1);
        index += 8;
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    let mut lanes = [0.0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc0);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    while index < length {
        let delta = *a_ptr.add(index) - *b_ptr.add(index);
        sum = delta.mul_add(delta, sum);
        index += 1;
    }
    sum
}

fn kmeans_assign(
    data: &[f64],
    centers: &[f64],
    assignments: &mut [i32],
    distances: &mut [f64],
    rows: usize,
    clusters: usize,
    features: usize,
    threads: usize,
) {
    let avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    let workers = threads.max(1).min(rows.max(1));
    let chunk = (rows + workers - 1) / workers;
    let assign_chunk = |assignment_chunk: &mut [i32], distance_chunk: &mut [f64], start: usize| {
        for local in 0..assignment_chunk.len() {
            let row = start + local;
            let offset = row * features;
            let point = &data[offset..offset + features];
            let mut best = 0usize;
            let mut best_distance = f64::INFINITY;
            for cluster in 0..clusters {
                let center = &centers[cluster * features..(cluster + 1) * features];
                let distance = if avx {
                    unsafe { squared_distance_avx2(point, center) }
                } else {
                    let mut sum = 0.0;
                    for feature in 0..features {
                        let delta = point[feature] - center[feature];
                        sum = delta.mul_add(delta, sum);
                    }
                    sum
                };
                if distance < best_distance {
                    best_distance = distance;
                    best = cluster;
                }
            }
            assignment_chunk[local] = best as i32;
            distance_chunk[local] = best_distance;
        }
    };
    if workers <= 1 {
        assign_chunk(assignments, distances, 0);
        return;
    }
    let pool = pool_for(workers);
    pool.install(|| {
        assignments
            .par_chunks_mut(chunk)
            .zip(distances.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(index, (assignment_chunk, distance_chunk))| {
                assign_chunk(assignment_chunk, distance_chunk, index * chunk);
            });
    });
}

/// `int forestVote(double[] data, int rows, int columns, int[] feature, double[] threshold,
///                  int[] left, int[] right, double[] value, int[] offsets, double[] labels,
///                  double[] votes, double scale)`
#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_forestVote(
    mut env: JNIEnv,
    _class: JClass,
    data_array: JDoubleArray,
    rows: jint,
    columns: jint,
    feature_array: JIntArray,
    threshold_array: JDoubleArray,
    left_array: JIntArray,
    right_array: JIntArray,
    value_array: JDoubleArray,
    offsets_array: JIntArray,
    labels_array: JDoubleArray,
    votes_array: JDoubleArray,
    scale: f64,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (rows, columns) = (rows as usize, columns as usize);
        if rows == 0 || columns == 0 {
            return STATUS_BAD_ARGS;
        }
        let data = match unsafe { env.get_array_elements(&data_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let feature = match unsafe { env.get_array_elements(&feature_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let threshold =
            match unsafe { env.get_array_elements(&threshold_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let left = match unsafe { env.get_array_elements(&left_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let right = match unsafe { env.get_array_elements(&right_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let value = match unsafe { env.get_array_elements(&value_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let offsets =
            match unsafe { env.get_array_elements(&offsets_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let labels =
            match unsafe { env.get_array_elements(&labels_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let mut votes =
            match unsafe { env.get_array_elements(&votes_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let classes = labels.len();
        if classes == 0
            || data.len() != rows * columns
            || votes.len() != rows * classes
            || feature.len() != threshold.len()
            || feature.len() != left.len()
            || feature.len() != right.len()
            || feature.len() != value.len()
            || offsets.len() < 2
        {
            return STATUS_BAD_ARGS;
        }
        forest_vote(
            &data, rows, columns, &feature, &threshold, &left, &right, &value, &offsets,
            &labels, &mut votes, scale,
        );
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[allow(clippy::too_many_arguments)]
fn forest_predict(
    data: &[f64],
    rows: usize,
    columns: usize,
    feature: &[i32],
    threshold: &[f64],
    left: &[i32],
    right: &[i32],
    value: &[f64],
    offsets: &[i32],
    output: &mut [f64],
    scale: f64,
) {
    let tree_count = offsets.len() - 1;
    for row in 0..rows {
        let offset = row * columns;
        let mut sum = 0.0;
        for tree in 0..tree_count {
            let mut node = offsets[tree] as usize;
            loop {
                let split_feature = feature[node];
                if split_feature < 0 {
                    sum += scale * value[node];
                    break;
                }
                node = if data[offset + split_feature as usize] <= threshold[node] {
                    left[node] as usize
                } else {
                    right[node] as usize
                };
            }
        }
        output[row] += sum;
    }
}

#[allow(clippy::too_many_arguments)]
fn forest_vote(
    data: &[f64],
    rows: usize,
    columns: usize,
    feature: &[i32],
    threshold: &[f64],
    left: &[i32],
    right: &[i32],
    value: &[f64],
    offsets: &[i32],
    labels: &[f64],
    votes: &mut [f64],
    scale: f64,
) {
    let tree_count = offsets.len() - 1;
    let classes = labels.len();
    for tree in 0..tree_count {
        let base = offsets[tree] as usize;
        for row in 0..rows {
            let offset = row * columns;
            let mut node = base;
            loop {
                let split_feature = feature[node];
                if split_feature < 0 {
                    let label = value[node];
                    let mut class = 0usize;
                    while class < classes && labels[class] != label {
                        class += 1;
                    }
                    if class < classes {
                        votes[row * classes + class] += scale;
                    }
                    break;
                }
                node = if data[offset + split_feature as usize] <= threshold[node] {
                    left[node] as usize
                } else {
                    right[node] as usize
                };
            }
        }
    }
}

/// `int histogramAccumulate(double[] gradients, double[] hessians, byte[] bins, int[] samples,
///                          int[] activeFeatures, int featureCount, int maxBins, int threads,
///                          double[] outGradient, double[] outHessian, int[] outCount)`
///
/// Output slots are indexed by `feature * maxBins + bin` and must be pre-cleared by the caller
/// for the active feature ranges. Sample rows index into the full `bins` matrix using
/// `featureCount` as the row stride.
#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_histogramAccumulate(
    mut env: JNIEnv,
    _class: JClass,
    gradients_array: JDoubleArray,
    hessians_array: JDoubleArray,
    bins_array: JByteArray,
    samples_array: JIntArray,
    active_features_array: JIntArray,
    feature_count: jint,
    max_bins: jint,
    threads: jint,
    out_gradient_array: JDoubleArray,
    out_hessian_array: JDoubleArray,
    out_count_array: JIntArray,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (feature_count, max_bins) = (feature_count as usize, max_bins as usize);
        if feature_count == 0 || max_bins == 0 {
            return STATUS_BAD_ARGS;
        }
        let gradients =
            match unsafe { env.get_array_elements(&gradients_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let hessians =
            match unsafe { env.get_array_elements(&hessians_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let bins = match unsafe { env.get_array_elements(&bins_array, ReleaseMode::NoCopyBack) } {
            Ok(array) => array,
            Err(_) => return STATUS_BAD_ARGS,
        };
        let samples =
            match unsafe { env.get_array_elements(&samples_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let active_features =
            match unsafe { env.get_array_elements(&active_features_array, ReleaseMode::NoCopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let mut out_gradient =
            match unsafe { env.get_array_elements(&out_gradient_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let mut out_hessian =
            match unsafe { env.get_array_elements(&out_hessian_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let mut out_count =
            match unsafe { env.get_array_elements(&out_count_array, ReleaseMode::CopyBack) } {
                Ok(array) => array,
                Err(_) => return STATUS_BAD_ARGS,
            };
        let total = feature_count * max_bins;
        if active_features.is_empty()
            || out_gradient.len() != total
            || out_hessian.len() != total
            || out_count.len() != total
            || bins.len() % feature_count != 0
        {
            return STATUS_BAD_ARGS;
        }
        for feature in active_features.iter() {
            if *feature < 0 || (*feature as usize) >= feature_count {
                return STATUS_BAD_ARGS;
            }
        }
        histogram_accumulate(
            &gradients,
            &hessians,
            &bins,
            &samples,
            &active_features,
            feature_count,
            max_bins,
            threads.max(1) as usize,
            &mut out_gradient,
            &mut out_hessian,
            &mut out_count,
        );
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn histogram_accumulate(
    gradients: &[f64],
    hessians: &[f64],
    bins: &[i8],
    samples: &[i32],
    active_features: &[i32],
    feature_count: usize,
    max_bins: usize,
    threads: usize,
    out_gradient: &mut [f64],
    out_hessian: &mut [f64],
    out_count: &mut [i32],
) {
    if samples.is_empty() {
        return;
    }
    let workers = threads.min(samples.len()).max(1);
    let total = feature_count * max_bins;
    if workers <= 1 || samples.len() < 1_024 {
        accumulate_histogram_range(
            gradients, hessians, bins, samples, active_features, feature_count, max_bins,
            0, samples.len(), out_gradient, out_hessian, out_count,
        );
        return;
    }
    let chunk = (samples.len() + workers - 1) / workers;
    let pool = pool_for(workers);
    pool.install(|| {
        let partials: Vec<(Vec<f64>, Vec<f64>, Vec<i32>)> = (0..workers)
            .into_par_iter()
            .map(|worker| {
                let from = worker * chunk;
                let to = ((worker + 1) * chunk).min(samples.len());
                let mut gradient = vec![0.0f64; total];
                let mut hessian = vec![0.0f64; total];
                let mut count = vec![0i32; total];
                accumulate_histogram_range(
                    gradients, hessians, bins, samples, active_features, feature_count,
                    max_bins, from, to, &mut gradient, &mut hessian, &mut count,
                );
                (gradient, hessian, count)
            })
            .collect();
        for (gradient, hessian, count) in partials {
            for index in 0..total {
                out_gradient[index] += gradient[index];
                out_hessian[index] += hessian[index];
                out_count[index] += count[index];
            }
        }
    });
}

#[derive(Clone, Copy, Default)]
struct HistogramBucket {
    gradient: f32,
    hessian: f32,
    count: f32,
}

/// Interleaved variant used by the persistent session. Keeping gradient, hessian and count in one
/// 24-byte bucket means a feature update touches one cache line instead of three separate arrays.
#[allow(clippy::too_many_arguments)]
pub(crate) fn histogram_accumulate_interleaved(
    gradients: &[f64],
    hessians: &[f64],
    bins: &[i8],
    samples: &[i32],
    active_features: &[i32],
    feature_count: usize,
    max_bins: usize,
    threads: usize,
    out_gradient: &mut [f64],
    out_hessian: &mut [f64],
    out_count: &mut [i32],
) {
    if samples.is_empty() {
        return;
    }
    let total = feature_count * max_bins;
    let workers = threads.min(samples.len() / 2_048).max(1);
    let mut buckets = if workers <= 1 {
        let mut local = vec![HistogramBucket::default(); total];
        accumulate_bucket_range(
            gradients, hessians, bins, samples, active_features, feature_count, max_bins,
            0, samples.len(), &mut local,
        );
        local
    } else {
        let chunk = (samples.len() + workers - 1) / workers;
        let partials: Vec<Vec<HistogramBucket>> = pool_for(workers).install(|| {
            (0..workers)
                .into_par_iter()
                .map(|worker| {
                    let from = worker * chunk;
                    let to = ((worker + 1) * chunk).min(samples.len());
                    let mut local = vec![HistogramBucket::default(); total];
                    accumulate_bucket_range(
                        gradients, hessians, bins, samples, active_features, feature_count,
                        max_bins, from, to, &mut local,
                    );
                    local
                })
                .collect()
        });
        let mut merged = vec![HistogramBucket::default(); total];
        for partial in partials {
            for index in 0..total {
                merged[index].gradient += partial[index].gradient;
                merged[index].hessian += partial[index].hessian;
                merged[index].count += partial[index].count;
            }
        }
        merged
    };
    for index in 0..total {
        let bucket = &mut buckets[index];
        out_gradient[index] += bucket.gradient as f64;
        out_hessian[index] += bucket.hessian as f64;
        out_count[index] += bucket.count as i32;
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_bucket_range(
    gradients: &[f64],
    hessians: &[f64],
    bins: &[i8],
    samples: &[i32],
    active_features: &[i32],
    feature_count: usize,
    max_bins: usize,
    from: usize,
    to: usize,
    buckets: &mut [HistogramBucket],
) {
    for index in from..to {
        let sample = samples[index] as usize;
        let row_base = sample * feature_count;
        let gradient = gradients[sample] as f32;
        let hessian = hessians[sample] as f32;
        let mut position = 0usize;
        while position + 1 < active_features.len() {
            let left_feature = active_features[position] as usize;
            let right_feature = active_features[position + 1] as usize;
            let left_bucket =
                left_feature * max_bins + (bins[row_base + left_feature] as u8 as usize);
            let right_bucket =
                right_feature * max_bins + (bins[row_base + right_feature] as u8 as usize);
            {
                let target = &mut buckets[left_bucket];
                target.gradient += gradient;
                target.hessian += hessian;
                target.count += 1.0;
            }
            {
                let target = &mut buckets[right_bucket];
                target.gradient += gradient;
                target.hessian += hessian;
                target.count += 1.0;
            }
            position += 2;
        }
        if position < active_features.len() {
            let feature = active_features[position] as usize;
            let bucket = feature * max_bins + (bins[row_base + feature] as u8 as usize);
            let target = &mut buckets[bucket];
            target.gradient += gradient;
            target.hessian += hessian;
            target.count += 1.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn accumulate_histogram_range(
    gradients: &[f64],
    hessians: &[f64],
    bins: &[i8],
    samples: &[i32],
    active_features: &[i32],
    feature_count: usize,
    max_bins: usize,
    from: usize,
    to: usize,
    out_gradient: &mut [f64],
    out_hessian: &mut [f64],
    out_count: &mut [i32],
) {
    for index in from..to {
        let sample = samples[index] as usize;
        let row_base = sample * feature_count;
        let gradient = gradients[sample];
        let hessian = hessians[sample];
        for feature in active_features {
            let feature = *feature as usize;
            let bucket = feature * max_bins + (bins[row_base + feature] as u8 as usize);
            out_gradient[bucket] += gradient;
            out_hessian[bucket] += hessian;
            out_count[bucket] += 1;
        }
    }
}

#[allow(dead_code)]
fn _keep_jni_false() -> jint {
    JNI_FALSE as jint
}
