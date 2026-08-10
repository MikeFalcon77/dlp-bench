//! Detector implementations.
//!
//! Each is a [`crate::detector::Detector`] and depends on nothing in
//! `eval/` or `perf/` — these are the implementations we would ship in an
//! endpoint agent, and the benchmark is a consumer of them.

pub mod deterministic;

/// Shared ONNX Runtime layer.
///
/// Lifted out of `gliner2` once a second ONNX detector existed. All `ort`
/// types in this crate live here, so every model runs on one runtime version
/// with one thread-pool and one optimisation configuration — without that, a
/// latency comparison between detectors means nothing.
#[cfg(feature = "onnx")]
pub mod onnx;

/// Word segmentation shared by the span detectors.
#[cfg(feature = "onnx")]
pub mod words;

#[cfg(feature = "onnx")]
pub mod gliner1;

#[cfg(feature = "onnx")]
pub mod gliner2;

/// Generic BIO token-classification detector.
#[cfg(feature = "onnx")]
pub mod token_class;
