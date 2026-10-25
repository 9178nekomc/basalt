//! Basalt native runtime.
//!
//! `classic` keeps the original array-based kernels (f64 GEMM, K-Means assignment, forest
//! prediction, histogram accumulation) used by the classical estimators. The tensor engine lives
//! in `tensor` + `graph` + `ops` + `context` and is exposed through `api`.

mod api;
mod binning;
mod classic;
mod context;
mod engine;
mod error;
mod fastmath;
mod graph;
mod histogram;
mod nn_ops;
mod ops;
mod simd;
mod tensor;
