//! Shared helpers for the model-dependent tests.
//!
//! These tests need a 446 MB artifact that is not in git. Rather than
//! `#[ignore]` them — which makes it too easy for the whole GLiNER2 suite to
//! rot unnoticed — they skip with a printed hint when the model is absent and
//! run for real when it is present.

use std::path::Path;

use tokenizers::Tokenizer;

pub const MODEL_DIR: &str = "models/gliner2-pii-onnx";
pub const MODEL_PATH: &str = "models/gliner2-pii-onnx/model.quantized.onnx";
pub const TOKENIZER_PATH: &str = "models/gliner2-pii-onnx/tokenizer.json";

/// Whether the pinned artifacts are on disk.
pub fn model_present() -> bool {
    Path::new(MODEL_PATH).is_file() && Path::new(TOKENIZER_PATH).is_file()
}

pub fn load_tokenizer() -> Tokenizer {
    Tokenizer::from_file(TOKENIZER_PATH).expect("tokenizer.json must load")
}
