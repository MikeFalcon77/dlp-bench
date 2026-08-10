//! Crate-wide error type.
//!
//! Follows the quarry convention of a `thiserror` enum plus a `Result` alias
//! per crate, but keeps model/runtime failures structured rather than
//! stringly-typed — `quarry-models` flattens every `ort`/`fastembed` error with
//! `.to_string()`, which loses the information you most want when a session
//! fails to build.

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    // Both sources are boxed: inlining them makes `Error` 136 bytes, which
    // `clippy::result_large_err` rightly flags on every `Result<T>` in the
    // binary. The error path is cold; one allocation there costs nothing.
    #[error("dataset {path}: {source}")]
    Dataset {
        path: PathBuf,
        #[source]
        source: Box<crate::dataset::DatasetError>,
    },

    #[error("detector {detector}: {source}")]
    Detector {
        detector: String,
        #[source]
        source: Box<DetectorError>,
    },

    #[error("unknown detector {0:?}; available: {available}", available = .1.join(", "))]
    UnknownDetector(String, Vec<String>),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Config(String),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Error::Config(msg.into())
    }

    pub fn detector(detector: impl Into<String>, source: DetectorError) -> Self {
        Error::Detector {
            detector: detector.into(),
            source: Box::new(source),
        }
    }

    pub fn dataset(path: impl Into<PathBuf>, source: crate::dataset::DatasetError) -> Self {
        Error::Dataset {
            path: path.into(),
            source: Box::new(source),
        }
    }
}

/// Failures from a detector, at load time or during inference.
///
/// Kept separate from [`Error`] so detectors can be lifted into an endpoint
/// agent without dragging the benchmark CLI's error type along.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DetectorError {
    /// A required model artifact is missing. Carries the remedy, because the
    /// single most common failure is "you did not run fetch-models.sh".
    #[error("missing model artifact {path}\n  run: bash scripts/fetch-models.sh {model_id}")]
    MissingArtifact { path: PathBuf, model_id: String },

    /// An artifact is present but is not the pinned one. Refusing to run is
    /// deliberate: benchmarking an unknown artifact produces numbers nobody can
    /// reproduce.
    #[error("artifact {path} failed verification: {detail}")]
    ArtifactMismatch { path: PathBuf, detail: String },

    /// The ONNX graph does not have the input/output contract we compiled
    /// against. Fails loudly rather than producing plausible garbage.
    #[error("model graph contract violated: {0}")]
    GraphContract(String),

    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("inference: {0}")]
    Inference(String),

    /// A detector produced a span that does not validate against its own input.
    /// Always a bug in the detector.
    #[error("detector emitted invalid span at index {index}: {source}")]
    InvalidSpan {
        index: usize,
        #[source]
        source: crate::span::SpanError,
    },

    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{0}")]
    Config(String),
}

impl DetectorError {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        DetectorError::Io {
            path: path.into(),
            source,
        }
    }

    pub fn config(msg: impl Into<String>) -> Self {
        DetectorError::Config(msg.into())
    }

    pub fn inference(msg: impl Into<String>) -> Self {
        DetectorError::Inference(msg.into())
    }

    pub fn graph(msg: impl Into<String>) -> Self {
        DetectorError::GraphContract(msg.into())
    }
}
