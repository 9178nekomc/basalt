//! Tensor storage owned by a native `Context`.
//!
//! Every tensor is an `Arc<RwLock<NativeTensor>>`. Java receives a generation-tagged 64-bit
//! handle (`(generation << 32) | slot_id`) and never dereferences raw memory; all native entry
//! points resolve the handle through the context's slot map and generation table first. That
//! indirection guarantees a stale or double-released handle is always rejected and can never
//! suffer from ABA address reuse.

use crate::error::{NativeError, Result};
use slotmap::{new_key_type, SlotMap};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(0x1000_0001);

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
        let mut data = Vec::new();
        data.try_reserve_exact(elements)
            .map_err(|_| NativeError::oom(format!("failed to allocate tensor of {elements} floats")))?;
        data.resize(elements, 0.0);
        Ok(Self {
            data,
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
            .ok_or_else(|| NativeError::oom("tensor element count overflows usize"))?;
    }
    if elements > isize::MAX as usize / std::mem::size_of::<f32>() {
        return Err(NativeError::oom(
            "tensor allocation exceeds the addressable byte range",
        ));
    }
    Ok(elements)
}

struct TensorSlot {
    tensor: Arc<RwLock<NativeTensor>>,
    handle_id: usize,
}

/// Slot map + monotonic handle index. Each inserted tensor gets a unique monotonic `handle_id`
/// so released handles are never recycled, eliminating ABA / dangling pointer hazards.
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

    /// Stores `tensor`, returns the unique generation-tagged handle Java will hold.
    pub fn insert(&mut self, tensor: NativeTensor) -> Result<usize> {
        let bytes = tensor.bytes();
        let arc = Arc::new(RwLock::new(tensor));
        let handle_id = NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed) as usize;
        let key = self.slots.insert(TensorSlot {
            tensor: arc,
            handle_id,
        });
        self.by_address.insert(handle_id, key);
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        Ok(handle_id)
    }

    pub fn get(&self, address: usize) -> Option<Arc<RwLock<NativeTensor>>> {
        let key = *self.by_address.get(&address)?;
        self.slots.get(key).map(|slot| Arc::clone(&slot.tensor))
    }

    /// Removes the slot and drops the tensor reference. Returns `false` when the
    /// handle is unknown, which is the normal idempotent path for a double `close()`.
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
        let _ = slot.handle_id;
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

    /// Drops every live tensor. Called from `Context::drop`.
    pub fn clear(&mut self) {
        let slots = std::mem::take(&mut self.slots);
        drop(slots);
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
    fn handles_are_never_reused_after_release() {
        let mut store = TensorStore::new();
        let h1 = store.insert(NativeTensor::new(vec![2, 2], false).unwrap()).unwrap();
        assert!(store.remove(h1));
        let h2 = store.insert(NativeTensor::new(vec![2, 2], false).unwrap()).unwrap();
        assert_ne!(h1, h2);
        assert!(store.get(h1).is_none());
        assert!(store.get(h2).is_some());
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
