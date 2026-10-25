//! Error type shared by the tensor engine and the JNI boundary.
//!
//! The numeric codes are part of the Java contract: every native entry point returns `0` on
//! success and a negative code on failure, and also throws the matching `JSLRuntimeException`
//! subclass with the same message so a Java caller cannot accidentally ignore a failed operation.

use std::fmt;

pub const STATUS_OK: i32 = 0;
pub const STATUS_INVALID: i32 = -1;
pub const STATUS_SHAPE: i32 = -2;
pub const STATUS_UNKNOWN_TENSOR: i32 = -3;
pub const STATUS_CONTEXT: i32 = -4;
pub const STATUS_GRAPH: i32 = -5;
pub const STATUS_PANIC: i32 = -6;
pub const STATUS_OOM: i32 = -7;
pub const STATUS_UNSUPPORTED: i32 = -8;

#[derive(Debug, Clone)]
pub struct NativeError {
    pub code: i32,
    pub message: String,
}

impl NativeError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(STATUS_INVALID, message)
    }

    pub fn shape(message: impl Into<String>) -> Self {
        Self::new(STATUS_SHAPE, message)
    }

    pub fn graph(message: impl Into<String>) -> Self {
        Self::new(STATUS_GRAPH, message)
    }

    pub fn oom(message: impl Into<String>) -> Self {
        Self::new(STATUS_OOM, message)
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(STATUS_UNSUPPORTED, message)
    }

    pub fn unknown_tensor(address: usize) -> Self {
        Self::new(
            STATUS_UNKNOWN_TENSOR,
            format!("tensor handle 0x{address:x} is not live in this context"),
        )
    }

    pub fn exception_class(&self) -> &'static str {
        match self.code {
            STATUS_INVALID => "jsl/JSLParameterException",
            STATUS_SHAPE => "jsl/JSLShapeException",
            STATUS_OOM => "jsl/JSLMemoryException",
            STATUS_UNSUPPORTED => "jsl/JSLUnsupportedOperationException",
            _ => "jsl/JSLRuntimeException",
        }
    }
}

impl fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} (code {})", self.message, self.code)
    }
}

impl std::error::Error for NativeError {}

pub type Result<T> = std::result::Result<T, NativeError>;
