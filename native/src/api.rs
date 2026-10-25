//! JNI surface for the tensor engine.
//!
//! Java never sees a Rust reference: it holds an opaque `long` address. Every entry point first
//! resolves that address through the context registry and the context's slot map, so a stale or
//! double-released handle is rejected before any dereference.

use crate::context::Context;
use crate::error::{NativeError, Result, STATUS_CONTEXT, STATUS_INVALID};
use crate::engine;
use crate::tensor::NativeTensor;
use jni::objects::{JByteBuffer, JClass, JFloatArray, JIntArray, JLongArray};
use jni::sys::{jboolean, jdouble, jfloat, jint, jintArray, jlong, jsize};
use jni::JNIEnv;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

const VERSION: jint = 2;

fn registry() -> &'static Mutex<HashMap<usize, Arc<Context>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<Context>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_registry() -> MutexGuard<'static, HashMap<usize, Arc<Context>>> {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn context_from_ptr(pointer: jlong) -> Result<Arc<Context>> {
    if pointer == 0 {
        return Err(NativeError::new(STATUS_CONTEXT, "null context handle"));
    }
    let address = pointer as usize;
    lock_registry().get(&address).cloned().ok_or_else(|| {
        NativeError::new(
            STATUS_CONTEXT,
            format!("context 0x{address:x} is not live (already closed?)"),
        )
    })
}

fn jni_error(error: jni::errors::Error) -> NativeError {
    NativeError::new(STATUS_INVALID, format!("JNI call failed: {error}"))
}

fn throw_error(env: &mut JNIEnv<'_>, error: &NativeError) {
    let _ = env.throw_new(error.exception_class(), error.to_string());
}

fn run_jni<'local, T>(
    env: &mut JNIEnv<'local>,
    default: T,
    action: impl FnOnce(&mut JNIEnv<'local>) -> Result<T>,
) -> T {
    match catch_unwind(AssertUnwindSafe(|| action(env))) {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            throw_error(env, &error);
            default
        }
        Err(_) => {
            let _ = env.throw_new(
                "basalt/BasaltRuntimeException",
                "native panic caught at the JNI boundary",
            );
            default
        }
    }
}

fn read_int_array(env: &JNIEnv<'_>, array: &JIntArray<'_>) -> Result<Vec<i32>> {
    let length = env
        .get_array_length(array)
        .map_err(jni_error)? as usize;
    let mut buffer = vec![0i32; length];
    env.get_int_array_region(array, 0, &mut buffer)
        .map_err(jni_error)?;
    Ok(buffer)
}

fn read_float_array(env: &JNIEnv<'_>, array: &JFloatArray<'_>) -> Result<Vec<f32>> {
    let length = env
        .get_array_length(array)
        .map_err(jni_error)? as usize;
    let mut buffer = vec![0.0f32; length];
    env.get_float_array_region(array, 0, &mut buffer)
        .map_err(jni_error)?;
    Ok(buffer)
}

fn read_shape(env: &JNIEnv<'_>, array: &JIntArray<'_>) -> Result<Vec<usize>> {
    let raw = read_int_array(env, array)?;
    if raw.is_empty() {
        return Err(NativeError::shape("tensor shape must not be empty"));
    }
    let mut shape = Vec::with_capacity(raw.len());
    for dimension in raw {
        if dimension <= 0 {
            return Err(NativeError::shape(format!(
                "tensor dimensions must be positive, received {dimension}"
            )));
        }
        shape.push(dimension as usize);
    }
    Ok(shape)
}

fn tensor_shape_of(tensor: &Arc<RwLock<NativeTensor>>) -> Result<Vec<usize>> {
    Ok(tensor
        .read()
        .map_err(|_| NativeError::invalid("tensor lock poisoned"))?
        .shape
        .clone())
}

#[no_mangle]
#[allow(non_snake_case)]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_available(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jint {
    run_jni(&mut env, 0, |_env| Ok(1))
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_version(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jint {
    run_jni(&mut env, 0, |_env| Ok(VERSION))
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_createContext(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
) -> jlong {
    run_jni(&mut env, 0, |_env| {
        let context = Arc::new(Context::new());
        let address = Arc::as_ptr(&context) as usize;
        lock_registry().insert(address, Arc::clone(&context));
        let raw = Arc::into_raw(context);
        Ok(raw as jlong)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_destroyContext(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        if context == 0 {
            return Ok(0);
        }
        let address = context as usize;
        let removed = lock_registry().remove(&address);
        if removed.is_none() {
            return Ok(1);
        }
        // SAFETY: the registry entry and the raw pointer are the only two strong references.
        // Removing the registry entry first guarantees this drop runs at most once.
        unsafe {
            drop(Arc::from_raw(address as *const Context));
        }
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_contextStats(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    output: JLongArray<'_>,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |env| {
        let context = context_from_ptr(context)?;
        let (tensors, bytes, graph) = context.stats()?;
        let length = env.get_array_length(&output).map_err(jni_error)? as usize;
        if length < 3 {
            return Err(NativeError::invalid(
                "contextStats expects a long[3] output array",
            ));
        }
        let values = [tensors as i64, bytes as i64, graph as i64];
        env.set_long_array_region(&output, 0, &values)
            .map_err(jni_error)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_clearGraph(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        context_from_ptr(context)?.clear_graph()?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_graphSize(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
) -> jint {
    run_jni(&mut env, 0, |_env| {
        Ok(context_from_ptr(context)?.graph_len()? as jint)
    })
}

// ------------------------------------------------------------------ tensor lifecycle

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_createTensor(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    shape: JIntArray<'_>,
    requires_grad: jboolean,
) -> jlong {
    run_jni(&mut env, 0, |env| {
        let context = context_from_ptr(context)?;
        let shape = read_shape(env, &shape)?;
        let address = context.create_tensor(shape, requires_grad != 0)?;
        Ok(address as jlong)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_createTensorFromData(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    data: JFloatArray<'_>,
    shape: JIntArray<'_>,
    requires_grad: jboolean,
) -> jlong {
    run_jni(&mut env, 0, |env| {
        let context = context_from_ptr(context)?;
        let shape = read_shape(env, &shape)?;
        let data = read_float_array(env, &data)?;
        let address = context.create_tensor_from_data(shape, data, requires_grad != 0)?;
        Ok(address as jlong)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_setTensorData(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    data: JFloatArray<'_>,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |env| {
        let context = context_from_ptr(context)?;
        let values = read_float_array(env, &data)?;
        let tensor = context.tensor(tensor as usize)?;
        tensor
            .write()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?
            .set_data(&values)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_tensorData(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    output: JFloatArray<'_>,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |env| {
        let context = context_from_ptr(context)?;
        let tensor = context.tensor(tensor as usize)?;
        let guard = tensor
            .read()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        let length = env.get_array_length(&output).map_err(jni_error)? as usize;
        if length != guard.data.len() {
            return Err(NativeError::shape(format!(
                "output array has {length} elements, tensor has {}",
                guard.data.len()
            )));
        }
        env.set_float_array_region(&output, 0, &guard.data)
            .map_err(jni_error)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_tensorGrad(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    output: JFloatArray<'_>,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |env| {
        let context = context_from_ptr(context)?;
        let tensor = context.tensor(tensor as usize)?;
        let guard = tensor
            .read()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        let length = env.get_array_length(&output).map_err(jni_error)? as usize;
        if length != guard.data.len() {
            return Err(NativeError::shape(format!(
                "output array has {length} elements, tensor has {}",
                guard.data.len()
            )));
        }
        if guard.grad.is_empty() {
            let zeros = vec![0.0f32; guard.data.len()];
            env.set_float_array_region(&output, 0, &zeros)
                .map_err(jni_error)?;
        } else {
            env.set_float_array_region(&output, 0, &guard.grad)
                .map_err(jni_error)?;
        }
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_tensorShape(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
) -> jintArray {
    run_jni(&mut env, std::ptr::null_mut(), |env| {
        let context = context_from_ptr(context)?;
        let tensor = context.tensor(tensor as usize)?;
        let shape = tensor_shape_of(&tensor)?;
        let output = env
            .new_int_array(shape.len() as jsize)
            .map_err(jni_error)?;
        let dimensions: Vec<i32> = shape.iter().map(|&value| value as i32).collect();
        env.set_int_array_region(&output, 0, &dimensions)
            .map_err(jni_error)?;
        Ok(output.into_raw())
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_fillTensor(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    value: jfloat,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        context_from_ptr(context)?.fill(tensor as usize, value)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_randomizeTensor(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    seed: jlong,
    scale: jfloat,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        context_from_ptr(context)?.randomize(tensor as usize, seed as u64, scale)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_zeroGrad(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        context_from_ptr(context)?.zero_grad(tensor as usize)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_releaseTensor(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        let context = context_from_ptr(context)?;
        if tensor == 0 {
            return Ok(1);
        }
        let removed = context.release_tensor(tensor as usize)?;
        Ok(if removed { 0 } else { 1 })
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_sgdStep(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    learning_rate: jfloat,
    weight_decay: jfloat,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(NativeError::invalid(
                "learning rate must be finite and non-negative",
            ));
        }
        context_from_ptr(context)?.apply_sgd(tensor as usize, learning_rate, weight_decay)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_adamStep(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
    learning_rate: jfloat,
    beta1: jfloat,
    beta2: jfloat,
    epsilon: jfloat,
    weight_decay: jfloat,
    step: jlong,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err(NativeError::invalid(
                "learning rate must be finite and non-negative",
            ));
        }
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(NativeError::invalid("Adam betas must be in [0, 1)"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(NativeError::invalid("Adam epsilon must be positive"));
        }
        if step <= 0 {
            return Err(NativeError::invalid("Adam step must be positive"));
        }
        context_from_ptr(context)?.apply_adam(
            tensor as usize,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step as u64,
        )?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_clipGradNorm(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensors: JLongArray<'_>,
    max_norm: jfloat,
) -> jdouble {
    run_jni(&mut env, 0.0, |env| {
        let context = context_from_ptr(context)?;
        let length = env.get_array_length(&tensors).map_err(jni_error)? as usize;
        let mut values = vec![0i64; length];
        env.get_long_array_region(&tensors, 0, &mut values)
            .map_err(jni_error)?;
        let addresses: Vec<usize> = values.into_iter().map(|value| value as usize).collect();
        Ok(context.clip_grad_norm(&addresses, max_norm)? as f64)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_tensorNorm(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
) -> jdouble {
    run_jni(&mut env, 0.0, |_env| {
        let context = context_from_ptr(context)?;
        let tensor = context.tensor(tensor as usize)?;
        let guard = tensor
            .read()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        let mut sum = 0.0f64;
        for value in &guard.data {
            sum += (*value as f64) * (*value as f64);
        }
        Ok(sum.sqrt())
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_tensorGradNorm(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
) -> jdouble {
    run_jni(&mut env, 0.0, |_env| {
        let context = context_from_ptr(context)?;
        let tensor = context.tensor(tensor as usize)?;
        let guard = tensor
            .read()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        let mut sum = 0.0f64;
        for value in &guard.grad {
            sum += (*value as f64) * (*value as f64);
        }
        Ok(sum.sqrt())
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_tensorIsFinite(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    tensor: jlong,
) -> jboolean {
    run_jni(&mut env, 0, |_env| {
        let context = context_from_ptr(context)?;
        let tensor = context.tensor(tensor as usize)?;
        let guard = tensor
            .read()
            .map_err(|_| NativeError::invalid("tensor lock poisoned"))?;
        Ok(if guard.data.iter().all(|value| value.is_finite()) {
            1
        } else {
            0
        })
    })
}

// ------------------------------------------------------------------ queue and backward

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_executeQueue(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    commands: JByteBuffer<'_>,
    count: jint,
    threads: jint,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |env| {
        let context = context_from_ptr(context)?;
        if count < 0 {
            return Err(NativeError::invalid("command count must be non-negative"));
        }
        let count = count as usize;
        let address = env.get_direct_buffer_address(&commands).map_err(jni_error)?;
        let capacity = env
            .get_direct_buffer_capacity(&commands)
            .map_err(jni_error)?;
        let required = count
            .checked_mul(std::mem::size_of::<engine::OpCommand>())
            .ok_or_else(|| NativeError::invalid("command buffer size overflows usize"))?;
        if required > capacity {
            return Err(NativeError::invalid(format!(
                "direct command buffer holds {capacity} bytes, {required} required"
            )));
        }
        if count > 0 && address.is_null() {
            return Err(NativeError::invalid(
                "direct command buffer address is null",
            ));
        }
        let pool = context.pool_for(threads.max(1) as usize)?;
        let mut graph = context
            .graph
            .lock()
            .map_err(|_| NativeError::invalid("graph lock poisoned"))?;
        engine::execute_queue(&context, &mut graph, address, count, &pool)?;
        Ok(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_basalt_nativebridge_NativeBindings_backward(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    context: jlong,
    root: jlong,
    threads: jint,
) -> jint {
    run_jni(&mut env, STATUS_INVALID, |_env| {
        let context = context_from_ptr(context)?;
        engine::backward(&context, root, threads)?;
        Ok(0)
    })
}
