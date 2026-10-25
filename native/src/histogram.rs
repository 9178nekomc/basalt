//! Persistent native histogram session used by `HistogramGradientBoosting`.
//!
//! The one-shot `histogramAccumulate` entry point copies the binned matrix (640 KiB for the
//! reference dataset) on every node. A tree with 31 leaves performs roughly 30 accumulations, and
//! 100 trees therefore copy hundreds of megabytes. The session keeps the binned matrix, gradients
//! and hessians in native memory for the whole fit; each node only transfers its sample indices,
//! active feature list and the small histogram outputs.

use crate::classic::histogram_accumulate_interleaved;
use crate::error::{STATUS_INVALID, STATUS_OK, STATUS_PANIC};
use jni::objects::{JByteArray, JClass, JDoubleArray, JIntArray, ReleaseMode};
use jni::sys::{jint, jlong};
use jni::JNIEnv;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

pub struct HistogramSession {
    bins: Vec<i8>,
    gradients: Vec<f64>,
    hessians: Vec<f64>,
    feature_count: usize,
    max_bins: usize,
}

fn registry() -> &'static Mutex<HashMap<usize, Arc<Mutex<HistogramSession>>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<Mutex<HistogramSession>>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_registry() -> MutexGuard<'static, HashMap<usize, Arc<Mutex<HistogramSession>>>> {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn session_from_ptr(pointer: jlong) -> Option<Arc<Mutex<HistogramSession>>> {
    if pointer == 0 {
        return None;
    }
    lock_registry().get(&(pointer as usize)).cloned()
}

#[no_mangle]
pub extern "system" fn Java_basalt_core_NativeKernels_histogramSessionCreate(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    bins_array: JByteArray<'_>,
    gradients_array: JDoubleArray<'_>,
    hessians_array: JDoubleArray<'_>,
    feature_count: jint,
    max_bins: jint,
) -> jlong {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let feature_count = feature_count.max(0) as usize;
        let max_bins = max_bins.max(0) as usize;
        if feature_count == 0 || max_bins == 0 {
            return 0;
        }
        let bins = match unsafe { env.get_array_elements(&bins_array, ReleaseMode::NoCopyBack) } {
            Ok(value) => value.to_vec(),
            Err(_) => return 0,
        };
        let gradients =
            match unsafe { env.get_array_elements(&gradients_array, ReleaseMode::NoCopyBack) } {
                Ok(value) => value.to_vec(),
                Err(_) => return 0,
            };
        let hessians =
            match unsafe { env.get_array_elements(&hessians_array, ReleaseMode::NoCopyBack) } {
                Ok(value) => value.to_vec(),
                Err(_) => return 0,
            };
        if bins.len() % feature_count != 0 || gradients.len() != hessians.len() {
            return 0;
        }
        let session = Arc::new(Mutex::new(HistogramSession {
            bins,
            gradients,
            hessians,
            feature_count,
            max_bins,
        }));
        let address = Arc::as_ptr(&session) as usize;
        lock_registry().insert(address, Arc::clone(&session));
        let raw = Arc::into_raw(session);
        raw as jlong
    }));
    result.unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_basalt_core_NativeKernels_histogramSessionUpdate(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    session: jlong,
    gradients_array: JDoubleArray<'_>,
    hessians_array: JDoubleArray<'_>,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(session) = session_from_ptr(session) else {
            return STATUS_INVALID;
        };
        let gradients =
            match unsafe { env.get_array_elements(&gradients_array, ReleaseMode::NoCopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let hessians =
            match unsafe { env.get_array_elements(&hessians_array, ReleaseMode::NoCopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut guard = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if gradients.len() != guard.gradients.len() || hessians.len() != guard.hessians.len() {
            return STATUS_INVALID;
        }
        guard.gradients.copy_from_slice(&gradients);
        guard.hessians.copy_from_slice(&hessians);
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub extern "system" fn Java_basalt_core_NativeKernels_histogramSessionAccumulate(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    session: jlong,
    samples_array: JIntArray<'_>,
    active_features_array: JIntArray<'_>,
    threads: jint,
    out_gradient_array: JDoubleArray<'_>,
    out_hessian_array: JDoubleArray<'_>,
    out_count_array: JIntArray<'_>,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let Some(session) = session_from_ptr(session) else {
            return STATUS_INVALID;
        };
        let samples =
            match unsafe { env.get_array_elements(&samples_array, ReleaseMode::NoCopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let active =
            match unsafe { env.get_array_elements(&active_features_array, ReleaseMode::NoCopyBack) }
            {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut out_gradient =
            match unsafe { env.get_array_elements(&out_gradient_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut out_hessian =
            match unsafe { env.get_array_elements(&out_hessian_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let mut out_count =
            match unsafe { env.get_array_elements(&out_count_array, ReleaseMode::CopyBack) } {
                Ok(value) => value,
                Err(_) => return STATUS_INVALID,
            };
        let guard = session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let total = guard.feature_count * guard.max_bins;
        if active.is_empty()
            || out_gradient.len() != total
            || out_hessian.len() != total
            || out_count.len() != total
        {
            return STATUS_INVALID;
        }
        if active
            .iter()
            .any(|feature| *feature < 0 || *feature as usize >= guard.feature_count)
        {
            return STATUS_INVALID;
        }
        histogram_accumulate_interleaved(
            &guard.gradients,
            &guard.hessians,
            &guard.bins,
            &samples,
            &active,
            guard.feature_count,
            guard.max_bins,
            threads.max(1) as usize,
            &mut out_gradient,
            &mut out_hessian,
            &mut out_count,
        );
        STATUS_OK
    }));
    result.unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub extern "system" fn Java_basalt_core_NativeKernels_histogramSessionDestroy(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    session: jlong,
) -> jint {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if session == 0 {
            return 0;
        }
        let address = session as usize;
        let removed = lock_registry().remove(&address);
        if removed.is_none() {
            return 1;
        }
        // SAFETY: registry entry + raw pointer are the only two strong references.
        unsafe {
            drop(Arc::from_raw(address as *const Mutex<HistogramSession>));
        }
        0
    }));
    result.unwrap_or(STATUS_PANIC)
}
