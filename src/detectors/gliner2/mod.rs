//! Fastino GLiNER2 PII detector.
//!
//! Runs `fastino/gliner2-privacy-filter-PII-multi` locally on CPU via a fused
//! single-graph ONNX export (`yethdev/gliner2-pii-onnx`), with no Python at
//! runtime.
//!
//! # Provenance and its caveats
//!
//! The export is community-produced and unaffiliated with Fastino. Its card
//! claims span-set parity with the PyTorch engine at thresholds 0.3/0.5/0.7 and
//! a max probability drift of 0.0057, but it also reports a *higher* SPY F1
//! than the original (.535 vs .477), which a lossy weight-quantised export
//! should not achieve. We treat those numbers as unverified marketing and rely
//! on our own measurements. See the README's "known limitations".
//!
//! # Structure
//!
//! * [`graph`] — all `ort` types; shared with any future ONNX detector.
//! * [`words`] — text preparation and the reference word splitter.
//! * [`prompt`] — schema prefix and per-document encoder input.
//! * [`decode`] — span extraction and per-label suppression.

pub mod decode;
pub mod prompt;
pub mod words;

#[cfg(test)]
pub(crate) mod test_support;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokenizers::Tokenizer;

use self::prompt::{EncodedInput, PromptPrefix, WordIdCache};
use super::onnx::{ExecutionProviderChoice, GraphContract, GraphInput, OrtGraph, ThreadConfig};
use crate::detector::{ArtifactInfo, Detector, finish};
use crate::entity::{Detection, DetectionSource};
use crate::error::DetectorError;
use crate::labels::LabelMap;

/// Artifact file names inside the model directory.
pub const MODEL_FILE: &str = "model.quantized.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The pinned artifact, so a benchmark result names exactly what produced it.
pub const MODEL_REPO: &str = "yethdev/gliner2-pii-onnx";
pub const MODEL_REVISION: &str = "7d82c70220c2b0891d4bda612402602b63be1c61";
pub const MODEL_SHA256: &str = "489edc522a34b8c6a0630c4710998abf0227b11acf45600ba2bce95bcd943930";
pub const MODEL_SIZE_BYTES: u64 = 446_391_629;
pub const BASE_MODEL: &str = "fastino/gliner2-privacy-filter-PII-multi";

const CONTRACT: GraphContract = GraphContract {
    inputs: &[
        "input_ids",
        "attention_mask",
        "text_word_indices",
        "schema_special_indices",
    ],
    outputs: &["span_probs", "count_logits"],
};

/// The model's span-width axis. Asserted against the first real output.
const EXPECTED_MAX_WIDTH: usize = 8;
/// The count head's class count.
const EXPECTED_COUNT_CLASSES: usize = 20;

#[derive(Debug, Clone)]
pub struct Gliner2Config {
    pub model_dir: PathBuf,
    pub threshold: f32,
    pub threads: ThreadConfig,
    /// Execution provider. CPU by default; `CoreMl` routes what it can to the
    /// Apple GPU/Neural Engine and falls back to CPU for the rest.
    pub provider: ExecutionProviderChoice,
    /// Which label schema to condition on. Also the mapping to canonical types.
    pub schema: &'static LabelMap,
    /// Verify the model file's sha256 at load.
    ///
    /// On by default: benchmarking an unverified artifact produces numbers
    /// nobody can reproduce. Costs ~0.4 s for 446 MB.
    pub verify_checksum: bool,
}

impl Gliner2Config {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            threshold: 0.5,
            threads: ThreadConfig::default(),
            provider: ExecutionProviderChoice::default(),
            schema: &crate::labels::GLINER2_PII,
            verify_checksum: true,
        }
    }
}

pub struct Gliner2Detector {
    /// `Mutex` so the detector satisfies `Detector::detect(&self)` while
    /// `Session::run` needs `&mut`. For the throughput benchmark we build N
    /// independent detectors rather than sharing one, which measures real
    /// per-thread scaling and exposes the N x RSS cost honestly.
    ///
    /// `Arc` so a threshold sweep can produce a re-tuned view without reloading
    /// 446 MB of weights per sweep point — the threshold is a decode-time
    /// parameter, so the session and prefix are genuinely shared.
    inner: Arc<Mutex<Inner>>,
    prefix: Arc<PromptPrefix>,
    config: Gliner2Config,
    name: String,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

struct Inner {
    graph: OrtGraph,
    tokenizer: Tokenizer,
    cache: WordIdCache,
    enc: EncodedInput,
}

impl Gliner2Detector {
    /// Load the model. Expensive; call once.
    pub fn load(config: Gliner2Config) -> Result<Self, DetectorError> {
        let model_path = config.model_dir.join(MODEL_FILE);
        let tokenizer_path = config.model_dir.join(TOKENIZER_FILE);
        let model_id = config
            .model_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("gliner2-pii-onnx")
            .to_owned();

        for p in [&model_path, &tokenizer_path] {
            if !p.is_file() {
                return Err(DetectorError::MissingArtifact {
                    path: p.clone(),
                    model_id: model_id.clone(),
                });
            }
        }

        if config.verify_checksum {
            verify_sha256(&model_path, MODEL_SHA256)?;
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| DetectorError::Tokenizer(format!("{}: {e}", tokenizer_path.display())))?;

        // A tokenizer whose special-token ids differ is not this model's
        // tokenizer, and every index we compute would be silently wrong.
        assert_special_tokens(&tokenizer)?;

        let prefix = prompt::build_prefix(&tokenizer, &config.schema.labels())?;
        let graph = OrtGraph::load_with(&model_path, &CONTRACT, config.threads, config.provider)?;

        let name = if config.schema.model() == "gliner2-pii" {
            "gliner2-pii".to_owned()
        } else {
            format!("gliner2-pii[{}]", config.schema.model())
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                graph,
                tokenizer,
                cache: WordIdCache::new(50_000),
                enc: EncodedInput::default(),
            })),
            prefix: Arc::new(prefix),
            config,
            name,
            model_path,
            tokenizer_path,
        })
    }

    pub fn config(&self) -> &Gliner2Config {
        &self.config
    }

    /// Prefix length in tokens — the fixed per-call cost imposed by the schema.
    ///
    /// Surfaced because it is the dominant term in short-prompt latency and the
    /// perf report needs to show it rather than let the reader assume latency
    /// scales with document length alone.
    pub fn prefix_tokens(&self) -> usize {
        self.prefix.len()
    }

    /// ONNX graph I/O description, for `dlp-bench inspect`.
    pub fn describe_graph(&self) -> String {
        self.inner.lock().expect("gliner2 mutex").graph.describe()
    }
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), DetectorError> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).map_err(|e| DetectorError::io(path, e))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| DetectorError::io(path, e))?;
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if got != expected {
        return Err(DetectorError::ArtifactMismatch {
            path: path.to_path_buf(),
            detail: format!("sha256 {got} != pinned {expected}; re-run scripts/fetch-models.sh"),
        });
    }
    Ok(())
}

fn assert_special_tokens(tok: &Tokenizer) -> Result<(), DetectorError> {
    use prompt::special;
    for (name, want) in [
        (special::P, special::P_ID),
        (special::E, special::E_ID),
        (special::SEP_TEXT, special::SEP_TEXT_ID),
    ] {
        let ids = prompt::encode_word(tok, name)?;
        if ids != vec![i64::from(want)] {
            return Err(DetectorError::Tokenizer(format!(
                "special token {name} encodes to {ids:?}, expected [{want}]; \
                 this is not the tokenizer this detector was built against"
            )));
        }
    }
    Ok(())
}

impl Detector for Gliner2Detector {
    fn name(&self) -> &str {
        &self.name
    }

    fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
        let prepared = words::prepare(text);
        if prepared.words.is_empty() {
            return Ok(Vec::new());
        }

        let mut inner = self.inner.lock().map_err(|_| {
            DetectorError::inference("gliner2 session mutex poisoned by a previous panic")
        })?;
        let Inner {
            graph,
            tokenizer,
            cache,
            enc,
        } = &mut *inner;

        prompt::build_input(&self.prefix, &prepared.words, tokenizer, cache, enc)?;

        let outputs = graph.run_i64(
            &[
                GraphInput::batched("input_ids", &enc.input_ids),
                GraphInput::batched("attention_mask", &enc.attention_mask),
                GraphInput::flat("text_word_indices", &enc.text_word_indices),
                GraphInput::flat(
                    "schema_special_indices",
                    &self.prefix.schema_special_indices,
                ),
            ],
            CONTRACT.outputs,
        )?;
        let (span_probs, count_logits) = (&outputs[0], &outputs[1]);

        let dims = decode::SpanDims::from_tensor(span_probs).ok_or_else(|| {
            DetectorError::graph(format!(
                "span_probs must be rank 3, got shape {:?}",
                span_probs.shape
            ))
        })?;
        // Catching a transposed or resized axis here is the difference between
        // a clear error and hundreds of plausible-looking wrong spans.
        if dims.max_width != EXPECTED_MAX_WIDTH {
            return Err(DetectorError::graph(format!(
                "expected span width axis {EXPECTED_MAX_WIDTH}, got {} (shape {:?})",
                dims.max_width, span_probs.shape
            )));
        }
        if dims.num_labels != self.prefix.num_labels {
            return Err(DetectorError::graph(format!(
                "graph returned {} labels but {} were requested",
                dims.num_labels, self.prefix.num_labels
            )));
        }
        if count_logits.shape.last() != Some(&EXPECTED_COUNT_CLASSES) {
            return Err(DetectorError::graph(format!(
                "count_logits must end in {EXPECTED_COUNT_CLASSES} classes, got {:?}",
                count_logits.shape
            )));
        }

        let raw = decode::decode(
            span_probs,
            dims,
            count_logits,
            &prepared.words,
            &prepared.text,
            self.config.threshold,
        );

        // Offsets are into the prepared text, which may carry a synthetic
        // trailing '.'. Clip to the caller's text so a returned span is always
        // sliceable against what they passed in.
        let limit = prepared.original_len();
        let source = DetectionSource::model(self.name.clone());
        let mut out = Vec::with_capacity(raw.len());
        for s in raw {
            let Some(entity_type) = self.config.schema.map_index(s.label_idx) else {
                // Structurally impossible: label_idx comes from the schema we
                // built. Fail loudly rather than drop it silently.
                return Err(DetectorError::inference(format!(
                    "decoded label index {} is outside the {}-label schema",
                    s.label_idx,
                    self.config.schema.len()
                )));
            };
            let end = s.end.min(limit);
            if s.start >= end {
                continue; // the span was entirely the synthetic '.'
            }
            let Some((start, end)) = crate::span::trim_span(&prepared.text, s.start, end) else {
                continue;
            };
            out.push(Detection::new(
                entity_type,
                start,
                end,
                s.score,
                source.clone(),
            ));
        }

        finish(text, out)
    }

    fn threshold(&self) -> Option<f32> {
        Some(self.config.threshold)
    }

    fn with_threshold(&self, threshold: f32) -> Option<Box<dyn Detector>> {
        // Shares the loaded session and prefix. Reloading per sweep point would
        // add ~1 s x 18 points of pure overhead and measure nothing useful;
        // the threshold only affects decoding.
        let mut config = self.config.clone();
        config.threshold = threshold;
        Some(Box::new(Self {
            inner: Arc::clone(&self.inner),
            prefix: Arc::clone(&self.prefix),
            config,
            name: self.name.clone(),
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
        }))
    }

    fn artifact_info(&self) -> Vec<ArtifactInfo> {
        vec![
            ArtifactInfo {
                name: "onnx".into(),
                path: self.model_path.display().to_string(),
                repo: Some(MODEL_REPO.into()),
                revision: Some(MODEL_REVISION.into()),
                sha256: Some(MODEL_SHA256.into()),
                size_bytes: Some(MODEL_SIZE_BYTES),
            },
            ArtifactInfo {
                name: "tokenizer".into(),
                path: self.tokenizer_path.display().to_string(),
                repo: Some(MODEL_REPO.into()),
                revision: Some(MODEL_REVISION.into()),
                sha256: None,
                size_bytes: std::fs::metadata(&self.tokenizer_path)
                    .ok()
                    .map(|m| m.len()),
            },
        ]
    }

    fn model_size_bytes(&self) -> Option<u64> {
        Some(MODEL_SIZE_BYTES)
    }

    fn fixed_prompt_tokens(&self) -> Option<usize> {
        Some(self.prefix.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::{MODEL_DIR, model_present};

    fn detector(threshold: f32) -> Option<Gliner2Detector> {
        if !model_present() {
            eprintln!("skipping: model absent, run `bash scripts/fetch-models.sh`");
            return None;
        }
        let mut cfg = Gliner2Config::new(MODEL_DIR);
        cfg.threshold = threshold;
        cfg.threads = ThreadConfig::with_intra(4);
        cfg.verify_checksum = false; // covered by its own test; keeps this fast
        Some(Gliner2Detector::load(cfg).expect("model must load"))
    }

    #[test]
    fn loads_and_reports_its_artifacts() {
        let Some(d) = detector(0.5) else { return };
        assert_eq!(d.name(), "gliner2-pii");
        assert_eq!(d.threshold(), Some(0.5));
        assert_eq!(d.model_size_bytes(), Some(MODEL_SIZE_BYTES));

        let a = d.artifact_info();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].revision.as_deref(), Some(MODEL_REVISION));
        assert!(
            a[0].revision.as_deref() != Some("main"),
            "revision must be pinned"
        );
    }

    #[test]
    fn a_wrong_model_directory_names_the_fetch_script() {
        let msg = match Gliner2Detector::load(Gliner2Config::new("models/nope")) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a nonexistent model directory must not load"),
        };
        assert!(msg.contains("fetch-models.sh"), "unhelpful error: {msg}");
    }
}
