//! Tensor storage owned by a native `Context`.
//!
//! Every tensor is an `Arc<RwLock<NativeTensor>>`. Java receives the address of the `Arc`'s
//! allocation and never dereferences it; all native entry points resolve the address through the
//! context's slot map first. That indirection is what makes a stale or double-released handle a
//! recoverable error instead of undefined behaviour.
//!
//! There is intentionally no `Box::leak` and no process-wide tensor pool. The only process-wide
//! structure in this crate is the context registry in `api.rs`, which validates context pointers
//! at the FFI boundary.

use crate::error::{NativeError, Result};
use slotmap::{new_key_type, SlotMap};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

new_key_type! {
    /// Stable key inside a context's tensor store.
    pub struct TensorKey;
}

/// The public tensor payload. `grad` starts empty and is materialised on first use.
#[derive(Debug, Clone)]
pub struct NativeTensor {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
}

impl NativeTensor {
    pub fn new(shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let elements = checked_numel(&shape)?;
        Ok(Self {
            data: vec![0.0; elements],
            grad: Vec::new(),
            shape,
            requires_grad,
        })
    }

    pub fn from_data(shape: Vec<usize>, data: Vec<f32>, requires_grad: bool) -> Result<Self> {
        let elements = checked_numel(&shape)?;
        if data.len() != elements {
            return Err(NativeError::shape(format!(
                "data length {} does not match shape {:?} ({} elements)",
                data.len(),
                shape,
                elements
            )));
        }
        if let Some(index) = data.iter().position(|value| !value.is_finite()) {
            return Err(NativeError::invalid(format!(
                "tensor data contains a non-finite value at index {index}"
            )));
        }
        Ok(Self {
            data,
            grad: Vec::new(),
            shape,
            requires_grad,
        })
    }

    pub fn bytes(&self) -> usize {
        (self.data.capacity() + self.grad.capacity()) * std::mem::size_of::<f32>()
    }

    /// Lazily allocates the gradient buffer. Kept separate from `new` so inference-only tensors
    /// never pay for a second buffer.
    pub fn ensure_grad(&mut self) {
        if self.grad.len() != self.data.len() {
            self.grad = vec![0.0; self.data.len()];
        }
    }

    pub fn zero_grad(&mut self) {
        if !self.grad.is_empty() {
            self.grad.fill(0.0);
        }
    }

    pub fn accumulate_grad(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.data.len() {
            return Err(NativeError::shape(format!(
                "gradient length {} does not match tensor length {}",
                values.len(),
                self.data.len()
            )));
        }
        self.ensure_grad();
        for (slot, value) in self.grad.iter_mut().zip(values) {
            *slot += *value;
        }
        Ok(())
    }

    pub fn set_data(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.data.len() {
            return Err(NativeError::shape(format!(
                "data length {} does not match tensor length {}",
                values.len(),
                self.data.len()
            )));
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(NativeError::invalid(format!(
                "tensor data contains a non-finite value at index {index}"
            )));
        }
        self.data.copy_from_slice(values);
        Ok(())
    }
}

pub fn checked_numel(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(NativeError::shape("tensor shape must not be empty"));
    }
    let mut elements = 1usize;
    for &dimension in shape {
        if dimension == 0 {
            return Err(NativeError::shape("tensor dimensions must be positive"));
        }
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| NativeError::shape("tensor element count overflows usize"))?;
    }
    if elements > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(NativeError::shape(
            "tensor allocation exceeds the addressable byte range",
        ));
    }
    Ok(elements)
}

struct TensorSlot {
    tensor: Arc<RwLock<NativeTensor>>,
    address: usize,
}

/// Slot map + address index. The address index is the FFI-safe lookup; the slot map owns the
/// storage and makes bulk teardown deterministic.
#[derive(Default)]
pub struct TensorStore {
    slots: SlotMap<TensorKey, TensorSlot>,
    by_address: HashMap<usize, TensorKey>,
    live_bytes: usize,
}

impl TensorStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `tensor`, returns the address Java will hold. One strong reference is deliberately
    /// transferred to the raw pointer; the store keeps a second one for lookup and teardown.
    pub fn insert(&mut self, tensor: NativeTensor) -> Result<usize> {
        let bytes = tensor.bytes();
        let arc = Arc::new(RwLock::new(tensor));
        let address = Arc::as_ptr(&arc) as usize;
        let raw = Arc::into_raw(arc.clone());
        debug_assert_eq!(raw as usize, address);
        let key = self.slots.insert(TensorSlot {
            tensor: arc,
            address,
        });
        self.by_address.insert(address, key);
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        Ok(address)
    }

    pub fn get(&self, address: usize) -> Option<Arc<RwLock<NativeTensor>>> {
        let key = *self.by_address.get(&address)?;
        self.slots.get(key).map(|slot| Arc::clone(&slot.tensor))
    }

    /// Removes the slot and drops the raw-pointer strong reference. Returns `false` when the
    /// address is unknown, which is the normal idempotent path for a double `close()`.
    pub fn remove(&mut self, address: usize) -> bool {
        let Some(key) = self.by_address.remove(&address) else {
            return false;
        };
        let Some(slot) = self.slots.remove(key) else {
            return false;
        };
        let bytes = slot
            .tensor
            .read()
            .map(|tensor| tensor.bytes())
            .unwrap_or(0);
        // SAFETY: the address was produced by `Arc::into_raw` in `insert`, and `remove` runs at
        // most once per address because the slot-map key is removed from `by_address` first.
        unsafe {
            drop(Arc::from_raw(address as *const RwLock<NativeTensor>));
        }
        drop(slot);
        self.live_bytes = self.live_bytes.saturating_sub(bytes);
        true
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    /// Drops every live tensor. Called from `Context::drop`; the slot map is replaced so a
    /// partially failed teardown can never leave a stale address behind.
    pub fn clear(&mut self) {
        let slots = std::mem::take(&mut self.slots);
        for (_, slot) in slots {
            // SAFETY: one raw strong reference exists per slot, and `clear` consumes the store.
            unsafe {
                drop(Arc::from_raw(slot.address as *const RwLock<NativeTensor>));
            }
            drop(slot);
        }
        self.by_address.clear();
        self.live_bytes = 0;
    }
}

impl Drop for TensorStore {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_lookup_remove_is_single_shot() {
        let mut store = TensorStore::new();
        let tensor = NativeTensor::new(vec![2, 3], true).unwrap();
        let address = store.insert(tensor).unwrap();
        assert!(store.get(address).is_some());
        assert_eq!(store.len(), 1);
        assert!(store.remove(address));
        assert!(store.get(address).is_none());
        assert!(!store.remove(address));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn rejects_zero_and_overflowing_shapes() {
        assert!(checked_numel(&[2, 0]).is_err());
        assert!(checked_numel(&[]).is_err());
        assert!(checked_numel(&[usize::MAX, 2]).is_err());
    }

    #[test]
    fn gradient_is_lazy() {
        let mut tensor = NativeTensor::new(vec![4], true).unwrap();
        assert!(tensor.grad.is_empty());
        tensor.accumulate_grad(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.grad, vec![1.0, 2.0, 3.0, 4.0]);
        tensor.zero_grad();
        assert_eq!(tensor.grad, vec![0.0; 4]);
    }
}
