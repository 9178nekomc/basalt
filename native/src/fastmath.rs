//! Classical-model fast paths that benefit from native memory layout but do not need autodiff.
//!
//! These kernels keep the existing Java implementations as the reference and fallback. Each one
//! is a pure function over contiguous arrays: no global state, no tensor handles, no hidden
//! allocation retained after the call.

use crate::classic::pool_for;
use crate::error::{STATUS_INVALID, STATUS_OK, STATUS_PANIC};
use crate::simd::has_avx2_fma;
use jni::objects::{JClass, JDoubleArray, JIntArray, ReleaseMode};
use jni::sys::{jboolean, jint, jlong, JNI_FALSE};
use jni::JNIEnv;
use rayon::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn cholesky_solve(mut matrix: Vec<f64>, rhs: Vec<f64>, size: usize) -> Vec<f64> {
    for row in 0..size {
        for column in 0..=row {
            let mut sum = matrix[row * size + column];
            for inner in 0..column {
                sum -= matrix[row * size + inner] * matrix[column * size + inner];
            }
            if row == column {
                matrix[row * size + column] = sum.max(1e-12).sqrt();
            } else {
                matrix[row * size + column] = sum / matrix[column * size + column];
            }
        }
    }
    let mut intermediate = vec![0.0f64; size];
    for row in 0..size {
        let mut sum = rhs[row];
        for inner in 0..row {
            sum -= matrix[row * size + inner] * intermediate[inner];
        }
        intermediate[row] = sum / matrix[row * size + row];
    }
    let mut solution = vec![0.0f64; size];
    for row in (0..size).rev() {
        let mut sum = intermediate[row];
        for inner in (row + 1)..size {
            sum -= matrix[inner * size + row] * solution[inner];
        }
        solution[row] = sum / matrix[row * size + row];
    }
    solution
}

#[allow(clippy::too_many_arguments)]
fn ridge_fit_impl(
    data: &[f64],
    rows: usize,
    columns: usize,
    targets: &[f64],
    alpha: f64,
    fit_intercept: bool,
    threads: usize,
    weights: &mut [f64],
    intercept: &mut [f64],
) -> Result<(), String> {
    if rows == 0 || columns == 0 {
        return Err(invalid("ridge requires at least one row and one column"));
    }
    if data.len() != rows * columns || targets.len() != rows {
        return Err(invalid("ridge array length mismatch"));
    }
    if weights.len() != columns || intercept.len() != 1 {
        return Err(invalid("ridge output length mismatch"));
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(invalid("ridge alpha must be finite and non-negative"));
    }
    let workers = threads.max(1).min(rows).min(64);
    let chunk = (rows + workers - 1) / workers;
    let pool = pool_for(workers);

    let (column_sums, target_sum) = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let from = worker * chunk;
                let to = ((worker + 1) * chunk).min(rows);
                let mut sums = vec![0.0f64; columns];
                let mut target = 0.0f64;
                for row in from..to {
                    let base = row * columns;
                    for column in 0..columns {
                        sums[column] += data[base + column];
                    }
                    target += targets[row];
                }
                (sums, target)
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .fold(
        (vec![0.0f64; columns], 0.0f64),
        |(mut left, left_target), (right, right_target)| {
            for column in 0..columns {
                left[column] += right[column];
            }
            (left, left_target + right_target)
        },
    );

    let means: Vec<f64> = if fit_intercept {
        column_sums.iter().map(|sum| sum / rows as f64).collect()
    } else {
        vec![0.0; columns]
    };
    let target_mean = if fit_intercept {
        target_sum / rows as f64
    } else {
        0.0
    };

    let partials: Vec<(Vec<f64>, Vec<f64>)> = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let from = worker * chunk;
                let to = ((worker + 1) * chunk).min(rows);
                let mut normal = vec![0.0f64; columns * columns];
                let mut rhs = vec![0.0f64; columns];
                let mut centered = vec![0.0f64; columns];
                for row in from..to {
                    let base = row * columns;
                    let target = targets[row] - target_mean;
                    for column in 0..columns {
                        centered[column] = data[base + column] - means[column];
                    }
                    for left in 0..columns {
                        let value = centered[left];
                        rhs[left] += value * target;
                        for right in left..columns {
                            normal[left * columns + right] += value * centered[right];
                        }
                    }
                }
                (normal, rhs)
            })
            .collect::<Vec<_>>()
    });

    let mut normal = vec![0.0f64; columns * columns];
    let mut rhs = vec![0.0f64; columns];
    for (partial_normal, partial_rhs) in partials {
        for index in 0..normal.len() {
            normal[index] += partial_normal[index];
        }
        for index in 0..rhs.len() {
            rhs[index] += partial_rhs[index];
        }
    }
    for left in 0..columns {
        for right in 0..left {
            normal[left * columns + right] = normal[right * columns + left];
        }
        normal[left * columns + left] += alpha;
    }
    let solution = cholesky_solve(normal, rhs, columns);
    weights.copy_from_slice(&solution);
    intercept[0] = target_mean
        - means
            .iter()
            .zip(&solution)
            .map(|(mean, weight)| mean * weight)
            .sum::<f64>();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gaussian_nb_fit_impl(
    data: &[f64],
    rows: usize,
    columns: usize,
    class_indices: &[i32],
    class_count: usize,
    means: &mut [f64],
    variances: &mut [f64],
    log_priors: &mut [f64],
    counts: &mut [i32],
    variance_smoothing: f64,
) -> Result<(), String> {
    if rows == 0 || columns == 0 || class_count == 0 {
        return Err(invalid("GaussianNB requires rows, columns and classes"));
    }
    if data.len() != rows * columns || class_indices.len() != rows {
        return Err(invalid("GaussianNB array length mismatch"));
    }
    if means.len() != class_count * columns
        || variances.len() != means.len()
        || log_priors.len() != class_count
        || counts.len() != class_count
    {
        return Err(invalid("GaussianNB output length mismatch"));
    }
    counts.fill(0);
    means.fill(0.0);
    for row in 0..rows {
        let class = class_indices[row];
        if class < 0 || class as usize >= class_count {
            return Err(invalid("GaussianNB class index out of range"));
        }
        let class = class as usize;
        counts[class] += 1;
        let base = row * columns;
        let target = class * columns;
        for column in 0..columns {
            means[target + column] += data[base + column];
        }
    }
    for class in 0..class_count {
        if counts[class] == 0 {
            return Err(invalid("GaussianNB received an empty class"));
        }
        let inverse = 1.0 / counts[class] as f64;
        let base = class * columns;
        for column in 0..columns {
            means[base + column] *= inverse;
        }
        log_priors[class] = (counts[class] as f64 / rows as f64).ln();
    }

    variances.fill(0.0);
    for row in 0..rows {
        let class = class_indices[row] as usize;
        let base = row * columns;
        let target = class * columns;
        for column in 0..columns {
            let delta = data[base + column] - means[target + column];
            variances[target + column] += delta * delta;
        }
    }
    let mut maximum = 0.0f64;
    for class in 0..class_count {
        let inverse = 1.0 / counts[class] as f64;
        let base = class * columns;
        for column in 0..columns {
            variances[base + column] *= inverse;
            maximum = maximum.max(variances[base + column]);
        }
    }
    let mut epsilon = variance_smoothing * maximum;
    if !(epsilon > 0.0) {
        epsilon = variance_smoothing;
    }
    for value in variances.iter_mut() {
        *value += epsilon;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn assign_chunk(
    data: &[f64],
    centers: &[f64],
    assignments: &mut [i32],
    distances: &mut [f64],
    start: usize,
    end: usize,
    clusters: usize,
    features: usize,
) -> (Vec<f64>, Vec<i32>) {
    let mut sums = vec![0.0f64; clusters * features];
    let mut counts = vec![0i32; clusters];
    for (local, row) in (start..end).enumerate() {
        let base = row * features;
        let point = &data[base..base + features];
        let mut best = 0usize;
        let mut best_distance = f64::INFINITY;
        let mut cluster = 0usize;
        // Two centers per pass reuse the point loads and double the independent FMA chains.
        while cluster + 1 < clusters {
            let left = &centers[cluster * features..(cluster + 1) * features];
            let right = &centers[(cluster + 1) * features..(cluster + 2) * features];
            let (left_distance, right_distance) = squared_distance_pair(point, left, right);
            if left_distance < best_distance {
                best_distance = left_distance;
                best = cluster;
            }
            if right_distance < best_distance {
                best_distance = right_distance;
                best = cluster + 1;
            }
            cluster += 2;
        }
        if cluster < clusters {
            let center = &centers[cluster * features..(cluster + 1) * features];
            let distance = squared_distance(point, center);
            if distance < best_distance {
                best_distance = distance;
                best = cluster;
            }
        }
        assignments[local] = best as i32;
        distances[local] = best_distance;
        let target = best * features;
        for feature in 0..features {
            sums[target + feature] += point[feature];
        }
        counts[best] += 1;
    }
    (sums, counts)
}

#[allow(clippy::too_many_arguments)]
fn kmeans_fit_impl(
    data: &[f64],
    rows: usize,
    clusters: usize,
    features: usize,
    restarts: usize,
    max_iterations: usize,
    tolerance: f64,
    seed: u64,
    threads: usize,
    centers_out: &mut [f64],
    inertia_out: &mut [f64],
    iterations_out: &mut [i32],
) -> Result<(), String> {
    if rows == 0 || clusters == 0 || features == 0 {
        return Err(invalid("KMeans requires rows, clusters and features"));
    }
    if clusters > rows {
        return Err(invalid("cluster count cannot exceed sample count"));
    }
    if restarts == 0 || max_iterations == 0 {
        return Err(invalid("KMeans restarts and iterations must be positive"));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid("KMeans tolerance must be finite and non-negative"));
    }
    if data.len() != rows * features
        || centers_out.len() != clusters * features
        || inertia_out.len() != 1
        || iterations_out.len() != 1
    {
        return Err(invalid("KMeans array length mismatch"));
    }
    let workers = threads.max(1).min(rows).min(64);
    let chunk = (rows + workers - 1) / workers;
    let pool = pool_for(workers);
    let mut best_centers = vec![0.0f64; clusters * features];
    let mut best_inertia = f64::INFINITY;
    let mut best_iterations = 0usize;

    for restart in 0..restarts {
        let restart_seed = seed ^ (restart as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut centers = init_plus_plus(data, rows, clusters, features, restart_seed);
        let mut assignments = vec![0i32; rows];
        let mut distances = vec![0.0f64; rows];
        let mut counts = vec![0i32; clusters];
        let mut completed = 0usize;

        for iteration in 1..=max_iterations {
            completed = iteration;
            let previous = centers.clone();
            let partials: Vec<(Vec<f64>, Vec<i32>)> = if workers == 1 {
                vec![assign_chunk(
                    data, &centers, &mut assignments, &mut distances, 0, rows, clusters, features,
                )]
            } else {
                pool.install(|| {
                    assignments
                        .par_chunks_mut(chunk)
                        .zip(distances.par_chunks_mut(chunk))
                        .enumerate()
                        .map(|(index, (assignment_chunk, distance_chunk))| {
                            let start = index * chunk;
                            let end = start + assignment_chunk.len();
                            assign_chunk(
                                data, &centers, assignment_chunk, distance_chunk, start, end,
                                clusters, features,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            };
            let mut sums = vec![0.0f64; clusters * features];
            counts.fill(0);
            for (partial_sums, partial_counts) in partials {
                for index in 0..sums.len() {
                    sums[index] += partial_sums[index];
                }
                for cluster in 0..clusters {
                    counts[cluster] += partial_counts[cluster];
                }
            }
            for cluster in 0..clusters {
                let count = counts[cluster];
                if count > 0 {
                    let inverse = 1.0 / count as f64;
                    let base = cluster * features;
                    for feature in 0..features {
                        centers[base + feature] = sums[base + feature] * inverse;
                    }
                }
            }
            repair_empty(
                data, &mut centers, &mut assignments, &mut distances, &mut counts, rows, clusters,
                features,
            );
            let mut shift = 0.0f64;
            for index in 0..centers.len() {
                let delta = centers[index] - previous[index];
                shift += delta * delta;
            }
            if shift <= tolerance * tolerance {
                break;
            }
        }

        let inertia = compute_inertia(
            data, &centers, rows, clusters, features, workers, chunk, &pool,
        );
        if inertia < best_inertia {
            best_inertia = inertia;
            best_centers.copy_from_slice(&centers);
            best_iterations = completed;
        }
    }

    centers_out.copy_from_slice(&best_centers);
    inertia_out[0] = best_inertia;
    iterations_out[0] = best_iterations as i32;
    Ok(())
}

fn init_plus_plus(
    data: &[f64],
    rows: usize,
    clusters: usize,
    features: usize,
    seed: u64,
) -> Vec<f64> {
    let mut state = seed | 1;
    let mut centers = vec![0.0f64; clusters * features];
    let first = (next_u64(&mut state) % rows as u64) as usize;
    centers[..features].copy_from_slice(&data[first * features..(first + 1) * features]);
    let mut distances = vec![f64::INFINITY; rows];
    for cluster in 1..clusters {
        let previous = (cluster - 1) * features;
        let mut sum = 0.0f64;
        for row in 0..rows {
            let distance = squared_distance(
                &data[row * features..(row + 1) * features],
                &centers[previous..previous + features],
            );
            distances[row] = distances[row].min(distance);
            sum += distances[row];
        }
        let selected = if sum <= 0.0 {
            cluster % rows
        } else {
            let threshold = next_unit(&mut state) * sum;
            let mut cumulative = 0.0f64;
            let mut chosen = rows - 1;
            for row in 0..rows {
                cumulative += distances[row];
                if cumulative >= threshold {
                    chosen = row;
                    break;
                }
            }
            chosen
        };
        let target = cluster * features;
        centers[target..target + features]
            .copy_from_slice(&data[selected * features..(selected + 1) * features]);
    }
    centers
}

fn next_u64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

fn next_unit(state: &mut u64) -> f64 {
    (next_u64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

#[allow(clippy::too_many_arguments)]
fn repair_empty(
    data: &[f64],
    centers: &mut [f64],
    assignments: &mut [i32],
    distances: &mut [f64],
    counts: &mut [i32],
    rows: usize,
    clusters: usize,
    features: usize,
) {
    let mut reserved = vec![false; rows];
    for cluster in 0..clusters {
        if counts[cluster] != 0 {
            continue;
        }
        let mut farthest = usize::MAX;
        let mut farthest_distance = -1.0f64;
        for row in 0..rows {
            if !reserved[row] && distances[row] > farthest_distance {
                farthest_distance = distances[row];
                farthest = row;
            }
        }
        if farthest == usize::MAX {
            break;
        }
        reserved[farthest] = true;
        let donor = assignments[farthest] as usize;
        if counts[donor] <= 1 {
            continue;
        }
        let old_count = counts[donor] as f64;
        counts[donor] -= 1;
        let inverse = 1.0 / (old_count - 1.0);
        for feature in 0..features {
            let index = donor * features + feature;
            centers[index] = (centers[index] * old_count - data[farthest * features + feature])
                * inverse;
        }
        assignments[farthest] = cluster as i32;
        for feature in 0..features {
            centers[cluster * features + feature] = data[farthest * features + feature];
        }
        counts[cluster] = 1;
        distances[farthest] = 0.0;
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_inertia(
    data: &[f64],
    centers: &[f64],
    rows: usize,
    clusters: usize,
    features: usize,
    workers: usize,
    chunk: usize,
    pool: &rayon::ThreadPool,
) -> f64 {
    let partials: Vec<f64> = pool.install(|| {
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let from = worker * chunk;
                let to = ((worker + 1) * chunk).min(rows);
                let mut total = 0.0f64;
                for row in from..to {
                    let point = &data[row * features..(row + 1) * features];
                    let mut best = f64::INFINITY;
                    for cluster in 0..clusters {
                        let center = &centers[cluster * features..(cluster + 1) * features];
                        best = best.min(squared_distance(point, center));
                    }
                    total += best;
                }
                total
            })
            .collect::<Vec<_>>()
    });
    partials.iter().sum()
}

fn squared_distance_pair(a: &[f64], b: &[f64], c: &[f64]) -> (f64, f64) {
    if has_avx2_fma() {
        // SAFETY: `has_avx2_fma` checks the CPU features before the intrinsic path.
        return unsafe { squared_distance_pair_avx2(a, b, c) };
    }
    let mut left = 0.0f64;
    let mut right = 0.0f64;
    for index in 0..a.len() {
        let delta_left = a[index] - b[index];
        let delta_right = a[index] - c[index];
        left = delta_left.mul_add(delta_left, left);
        right = delta_right.mul_add(delta_right, right);
    }
    (left, right)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn squared_distance_pair_avx2(a: &[f64], b: &[f64], c: &[f64]) -> (f64, f64) {
    use std::arch::x86_64::*;
    let length = a.len();
    let mut index = 0usize;
    let mut acc_left = _mm256_setzero_pd();
    let mut acc_right = _mm256_setzero_pd();
    while index + 4 <= length {
        let point = _mm256_loadu_pd(a.as_ptr().add(index));
        let delta_left = _mm256_sub_pd(point, _mm256_loadu_pd(b.as_ptr().add(index)));
        let delta_right = _mm256_sub_pd(point, _mm256_loadu_pd(c.as_ptr().add(index)));
        acc_left = _mm256_fmadd_pd(delta_left, delta_left, acc_left);
        acc_right = _mm256_fmadd_pd(delta_right, delta_right, acc_right);
        index += 4;
    }
    let mut left_lanes = [0.0f64; 4];
    let mut right_lanes = [0.0f64; 4];
    _mm256_storeu_pd(left_lanes.as_mut_ptr(), acc_left);
    _mm256_storeu_pd(right_lanes.as_mut_ptr(), acc_right);
    let mut left = left_lanes[0] + left_lanes[1] + left_lanes[2] + left_lanes[3];
    let mut right = right_lanes[0] + right_lanes[1] + right_lanes[2] + right_lanes[3];
    while index < length {
        let point = *a.get_unchecked(index);
        let delta_left = point - *b.get_unchecked(index);
        let delta_right = point - *c.get_unchecked(index);
        left = delta_left.mul_add(delta_left, left);
        right = delta_right.mul_add(delta_right, right);
        index += 1;
    }
    (left, right)
}

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    if has_avx2_fma() {
        // SAFETY: `has_avx2_fma` checks the CPU features before the intrinsic path.
        return unsafe { squared_distance_avx2(a, b) };
    }
    let mut sum = 0.0f64;
    for index in 0..a.len() {
        let delta = a[index] - b[index];
        sum = delta.mul_add(delta, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn squared_distance_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let length = a.len();
    let mut index = 0usize;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    while index + 8 <= length {
        let d0 = _mm256_sub_pd(
            _mm256_loadu_pd(a.as_ptr().add(index)),
            _mm256_loadu_pd(b.as_ptr().add(index)),
        );
        let d1 = _mm256_sub_pd(
            _mm256_loadu_pd(a.as_ptr().add(index + 4)),
            _mm256_loadu_pd(b.as_ptr().add(index + 4)),
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
        let delta = *a.get_unchecked(index) - *b.get_unchecked(index);
        sum = delta.mul_add(delta, sum);
        index += 1;
    }
    sum
}

#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_ridgeFit(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    data_array: JDoubleArray<'_>,
    rows: jint,
    columns: jint,
    target_array: JDoubleArray<'_>,
    alpha: f64,
    fit_intercept: jboolean,
    threads: jint,
    weights_array: JDoubleArray<'_>,
    intercept_array: JDoubleArray<'_>,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let rows = rows as usize;
        let columns = columns as usize;
        // Pin + memcpy once, then release the critical section before rayon starts. Holding a
        // critical section across worker threads is forbidden by the JNI specification.
        let data = {
            let guard = match unsafe {
                env.get_array_elements_critical(&data_array, ReleaseMode::NoCopyBack)
            } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
            guard.to_vec()
        };
        let targets = {
            let guard = match unsafe {
                env.get_array_elements_critical(&target_array, ReleaseMode::NoCopyBack)
            } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
            guard.to_vec()
        };
        let mut weights =
            match unsafe { env.get_array_elements(&weights_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut intercept =
            match unsafe { env.get_array_elements(&intercept_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        match ridge_fit_impl(
            &data,
            rows,
            columns,
            &targets,
            alpha,
            fit_intercept != JNI_FALSE,
            threads.max(1) as usize,
            &mut weights,
            &mut intercept,
        ) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID,
        }
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_gaussianNbFit(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    data_array: JDoubleArray<'_>,
    rows: jint,
    columns: jint,
    class_array: JIntArray<'_>,
    class_count: jint,
    means_array: JDoubleArray<'_>,
    variances_array: JDoubleArray<'_>,
    priors_array: JDoubleArray<'_>,
    counts_array: JIntArray<'_>,
    variance_smoothing: f64,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let rows = rows as usize;
        let columns = columns as usize;
        let class_count = class_count as usize;
        let data = match unsafe { env.get_array_elements(&data_array, ReleaseMode::NoCopyBack) } {
            Ok(value) => value,
            Err(_) => return STATUS_INVALID,
        };
        let classes = match unsafe { env.get_array_elements(&class_array, ReleaseMode::NoCopyBack) } {
            Ok(value) => value,
            Err(_) => return STATUS_INVALID,
        };
        let mut means =
            match unsafe { env.get_array_elements(&means_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut variances =
            match unsafe { env.get_array_elements(&variances_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut priors =
            match unsafe { env.get_array_elements(&priors_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut counts =
            match unsafe { env.get_array_elements(&counts_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        match gaussian_nb_fit_impl(
            &data,
            rows,
            columns,
            &classes,
            class_count,
            &mut means,
            &mut variances,
            &mut priors,
            &mut counts,
            variance_smoothing,
        ) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID,
        }
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_kmeansFit(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    data_array: JDoubleArray<'_>,
    rows: jint,
    clusters: jint,
    features: jint,
    restarts: jint,
    max_iterations: jint,
    tolerance: f64,
    seed: jlong,
    threads: jint,
    centers_array: JDoubleArray<'_>,
    inertia_array: JDoubleArray<'_>,
    iterations_array: JIntArray<'_>,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let data = match unsafe { env.get_array_elements(&data_array, ReleaseMode::NoCopyBack) } {
            Ok(value) => value,
            Err(_) => return STATUS_INVALID,
        };
        let mut centers =
            match unsafe { env.get_array_elements(&centers_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut inertia =
            match unsafe { env.get_array_elements(&inertia_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut iterations =
            match unsafe { env.get_array_elements(&iterations_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        match kmeans_fit_impl(
            &data,
            rows.max(0) as usize,
            clusters.max(0) as usize,
            features.max(0) as usize,
            restarts.max(0) as usize,
            max_iterations.max(0) as usize,
            tolerance,
            seed as u64,
            threads.max(1) as usize,
            &mut centers,
            &mut inertia,
            &mut iterations,
        ) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID,
        }
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_recovers_linear_model() {
        let rows = 200usize;
        let columns = 3usize;
        let mut data = vec![0.0f64; rows * columns];
        let mut targets = vec![0.0f64; rows];
        for row in 0..rows {
            let x = row as f64 / rows as f64;
            data[row * columns] = x;
            data[row * columns + 1] = x * x;
            data[row * columns + 2] = 1.0 - x;
            targets[row] = 1.5 + 2.0 * x - 0.5 * x * x + 0.25 * (1.0 - x);
        }
        let mut weights = vec![0.0; columns];
        let mut intercept = [0.0];
        ridge_fit_impl(
            &data, rows, columns, &targets, 1e-8, true, 4, &mut weights, &mut intercept,
        )
        .unwrap();
        // The three columns are linearly dependent (x + (1-x) = 1), so the coefficient vector is
        // not unique; predictions are the invariant that matters.
        for row in 0..rows {
            let predicted = intercept[0]
                + (0..columns)
                    .map(|column| weights[column] * data[row * columns + column])
                    .sum::<f64>();
            assert!((predicted - targets[row]).abs() < 1e-6);
        }
    }

    #[test]
    fn kmeans_fit_finds_two_clusters() {
        let data = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            10.0, 10.0, 11.0, 10.0, 10.0, 11.0, 11.0, 11.0,
        ];
        let mut centers = vec![0.0f64; 4];
        let mut inertia = [0.0f64];
        let mut iterations = [0i32];
        kmeans_fit_impl(
            &data, 8, 2, 2, 2, 50, 1e-10, 7, 2, &mut centers, &mut inertia, &mut iterations,
        )
        .unwrap();
        let mut pairs = vec![(centers[0], centers[1]), (centers[2], centers[3])];
        pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
        assert!((pairs[0].0 - 0.5).abs() < 1e-9 && (pairs[0].1 - 0.5).abs() < 1e-9);
        assert!((pairs[1].0 - 10.5).abs() < 1e-9 && (pairs[1].1 - 10.5).abs() < 1e-9);
        assert!(inertia[0] <= 4.0 + 1e-9);
        assert!(iterations[0] >= 1);
    }

    #[test]
    fn gaussian_nb_recovers_moments() {
        let rows = 100usize;
        let columns = 2usize;
        let mut data = vec![0.0f64; rows * columns];
        let mut classes = vec![0i32; rows];
        for row in 0..rows {
            let class = if row < 50 { 0 } else { 1 };
            classes[row] = class;
            let offset = if class == 0 { 0.0 } else { 4.0 };
            data[row * columns] = offset + (row % 5) as f64;
            data[row * columns + 1] = offset - (row % 3) as f64;
        }
        let mut means = vec![0.0; 2 * columns];
        let mut variances = vec![0.0; 2 * columns];
        let mut priors = vec![0.0; 2];
        let mut counts = vec![0i32; 2];
        gaussian_nb_fit_impl(
            &data, rows, columns, &classes, 2, &mut means, &mut variances, &mut priors,
            &mut counts, 1e-9,
        )
        .unwrap();
        assert_eq!(counts, vec![50, 50]);
        assert!((priors[0] - 0.5f64.ln()).abs() < 1e-12);
        assert!((means[0] - 2.0).abs() < 1e-9);
        assert!((means[columns] - 6.0).abs() < 1e-9);
    }

}
