//! Device tags for tensor storage.
//!
//! The runtime keeps every tensor on exactly one device and refuses operations that mix
//! devices. Only the CPU backend exists today; `Cuda` is reserved so that the transfer
//! path, the validation rules and the Java API are already exercised by tests, and a GPU
//! backend can plug in without changing call sites.
//!
//! Deliberately no CUDA code lives here: a backend that cannot be executed on the build
//! machine would be untested by definition.

use crate::error::{NativeError, Result};

/// Where a tensor's storage lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Host memory, computed by the Rust kernels.
    Cpu,
    /// Reserved for a CUDA backend; requesting it currently fails with a clear error.
    Cuda(u32),
}

impl Device {
    /// Wire tag used by the JNI layer.
    pub fn tag(self) -> i32 {
        match self {
            Device::Cpu => 0,
            // 1000 + index keeps the CPU tag stable and leaves room for more backends.
            Device::Cuda(index) => 1000 + index as i32,
        }
    }

    /// Inverse of [`Device::tag`].
    pub fn from_tag(tag: i32) -> Result<Self> {
        match tag {
            0 => Ok(Device::Cpu),
            1000..=1999 => Ok(Device::Cuda((tag - 1000) as u32)),
            other => Err(NativeError::invalid(format!("unknown device tag {other}"))),
        }
    }

    pub fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Human readable name, used in error messages.
    pub fn name(self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(index) => format!("cuda:{index}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_round_trip() {
        for device in [Device::Cpu, Device::Cuda(0), Device::Cuda(3)] {
            assert_eq!(Device::from_tag(device.tag()).unwrap(), device);
        }
    }

    #[test]
    fn unknown_tag_is_rejected() {
        assert!(Device::from_tag(-1).is_err());
        assert!(Device::from_tag(5000).is_err());
    }
}
