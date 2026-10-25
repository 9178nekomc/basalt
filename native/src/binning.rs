//! Quantile binning in native code.
//!
//! The Java reference sorts each feature column and binary-searches every value. Doing the same
//! work in Rust removes `Matrix.get` bounds checks and keeps the temporary column contiguous. The
//! threshold construction is deliberately bit-for-bit identical to the Java path:
//! `position = rows * bin / maxBins - 1`, midpoint `left/2 + right/2`, fallback `left`, and
//! consecutive duplicate thresholds are dropped.

use crate::error::{STATUS_INVALID, STATUS_OK, STATUS_PANIC};
use jni::objects::{JByteArray, JClass, JDoubleArray, JIntArray, ReleaseMode};
use jni::sys::jint;
use jni::JNIEnv;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[allow(clippy::too_many_arguments)]
fn bin_features_impl(
    data: &[f64],
    rows: usize,
    columns: usize,
    max_bins: usize,
    binned: &mut [i8],
    thresholds: &mut [f64],
    counts: &mut [i32],
) -> Result<(), String> {
    if rows < 2 || columns == 0 {
        return Err("binning requires at least two rows and one column".to_string());
    }
    if max_bins < 2 || max_bins > 255 {
        return Err("maxBins must be in [2, 255]".to_string());
    }
    if data.len() != rows * columns
        || binned.len() != rows * columns
        || thresholds.len() != columns * (max_bins - 1)
        || counts.len() != columns
    {
        return Err("binning array length mismatch".to_string());
    }
    let mut column = vec![0.0f64; rows];
    for feature in 0..columns {
        for row in 0..rows {
            column[row] = data[row * columns + feature];
        }
        column.sort_unstable_by(f64::total_cmp);
        let base = feature * (max_bins - 1);
        let mut count = 0usize;
        for bin in 1..max_bins {
            if count >= max_bins - 1 {
                break;
            }
            let mut position = (rows as u128 * bin as u128 / max_bins as u128) as usize;
            position = position.saturating_sub(1).min(rows - 2);
            let left = column[position];
            let right = column[position + 1];
            if left == right {
                continue;
            }
            let midpoint = left / 2.0 + right / 2.0;
            let threshold = if midpoint >= left && midpoint < right {
                midpoint
            } else {
                left
            };
            if count == 0 || threshold != thresholds[base + count - 1] {
                thresholds[base + count] = threshold;
                count += 1;
            }
        }
        counts[feature] = count as i32;
        let feature_thresholds = &thresholds[base..base + count];
        for row in 0..rows {
            let value = data[row * columns + feature];
            let insertion = feature_thresholds.partition_point(|&threshold| threshold < value);
            binned[row * columns + feature] = insertion as i8;
        }
    }
    Ok(())
}

#[no_mangle]
pub extern "system" fn Java_jsl_core_NativeKernels_binFeatures(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    data_array: JDoubleArray<'_>,
    rows: jint,
    columns: jint,
    max_bins: jint,
    binned_array: JByteArray<'_>,
    threshold_array: JDoubleArray<'_>,
    count_array: JIntArray<'_>,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let rows = rows.max(0) as usize;
        let columns = columns.max(0) as usize;
        let max_bins = max_bins.max(0) as usize;
        let data = match unsafe { env.get_array_elements(&data_array, ReleaseMode::NoCopyBack) } {
            Ok(value) => value,
            Err(_) => return STATUS_INVALID,
        };
        let mut binned =
            match unsafe { env.get_array_elements(&binned_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut thresholds =
            match unsafe { env.get_array_elements(&threshold_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut counts =
            match unsafe { env.get_array_elements(&count_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let expected_thresholds = columns.saturating_mul(max_bins.saturating_sub(1));
        if binned.len() != rows.saturating_mul(columns)
            || thresholds.len() != expected_thresholds
            || counts.len() != columns
        {
            return STATUS_INVALID;
        }
        match bin_features_impl(
            &data,
            rows,
            columns,
            max_bins,
            &mut binned,
            &mut thresholds,
            &mut counts,
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
    fn bins_match_expected_thresholds() {
        let rows = 8usize;
        let columns = 2usize;
        let data = vec![
            0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0, 5.0, 15.0, 6.0, 16.0, 7.0,
            17.0,
        ];
        let max_bins = 4usize;
        let mut binned = vec![0i8; rows * columns];
        let mut thresholds = vec![0.0f64; columns * (max_bins - 1)];
        let mut counts = vec![0i32; columns];
        bin_features_impl(
            &data,
            rows,
            columns,
            max_bins,
            &mut binned,
            &mut thresholds,
            &mut counts,
        )
        .unwrap();
        assert_eq!(counts, vec![3, 3]);
        for row in 0..rows {
            assert_eq!(binned[row * columns], row as i8 / 2);
            assert_eq!(binned[row * columns + 1], row as i8 / 2);
        }
    }
}
