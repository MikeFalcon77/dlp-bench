//! `dlp-bench` — a reproducible benchmark for local PII detection models
//! intended for endpoint DLP agents and AI gateways.
//!
//! The library half is deliberately usable outside the benchmark: the
//! [`detector::Detector`] implementations here are the ones we would ship in an
//! agent, and they depend on nothing in [`eval`] or [`perf`].
//!
//! # Layering
//!
//! ```text
//!   entity / span / labels     canonical taxonomy, byte-offset spans
//!         |
//!   detector::Detector         the only abstraction the evaluator sees
//!         |
//!   detectors::*               deterministic, gliner2, (hybrid)
//!         |
//!   dataset -> eval -> report  quality
//!   dataset -> perf  -> report latency / memory
//! ```

pub mod dataset;
pub mod detector;
pub mod detectors;
pub mod entity;
pub mod error;
pub mod eval;
pub mod labels;
pub mod perf;
pub mod report;
pub mod runmeta;
pub mod span;

pub use detector::{ArtifactInfo, Detector};
pub use entity::{Detection, DetectionSource, EntityType};
pub use error::{DetectorError, Error, Result};
