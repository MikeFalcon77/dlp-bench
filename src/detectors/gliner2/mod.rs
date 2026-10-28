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
#[cfg(feature = "candle")]
use super::candle::gliner2::Gliner2Candle;
use super::onnx::{GraphContract, GraphInput, OrtGraph, SessionOptions, TensorOut, ThreadConfig};
use crate::detector::{ArtifactInfo, Detector, finish};
use crate::entity::{Detection, DetectionSource};
use crate::error::DetectorError;
use crate::labels::LabelMap;
use crate::runtime::Backend;
use crate::stages::{Stage, StageTimer};

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
    /// How the ONNX Runtime session is built: execution provider, CoreML
    /// settings, dimension overrides, profiling. Defaults to the CPU baseline.
    pub session: SessionOptions,
    /// Pad every encoder input to this fixed size.
    ///
    /// Must agree with the `seq_len` / `num_words` dimension overrides in
    /// [`Self::session`]; `load` refuses a configuration where they differ,
    /// because a mismatch surfaces as a shape error on every single call.
    pub fixed_shape: Option<prompt::FixedShape>,
    /// Which label schema to condition on. Also the mapping to canonical types.
    pub schema: &'static LabelMap,
    /// Which library runs the graph. Candle rebuilds it from the same ONNX
    /// file; prompt construction and decoding are shared either way.
    pub backend: Backend,
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
            session: SessionOptions::default(),
            fixed_shape: None,
            schema: &crate::labels::GLINER2_PII,
            backend: Backend::Ort,
            verify_checksum: true,
        }
    }
}

pub struct Gliner2Detector {
    /// `Arc` so a threshold sweep can produce a re-tuned view without reloading
    /// 446 MB of weights per sweep point — the threshold is a decode-time
    /// parameter, so the engine and prefix are genuinely shared.
    engine: Arc<Engine>,
    /// Tokenizer and word cache, locked only while one document is encoded so
    /// that concurrent callers of a concurrency-capable engine do not queue
    /// behind each other's inference.
    prep: Arc<Mutex<Prep>>,
    prefix: Arc<PromptPrefix>,
    config: Gliner2Config,
    name: String,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

struct Prep {
    tokenizer: Tokenizer,
    cache: WordIdCache,
}

/// What turns an encoded prompt into `(span_probs, count_logits)`.
enum Engine {
    /// `Mutex` because `Session::run` needs `&mut`: `ort` refuses concurrent
    /// runs on one session, since parts of ONNX Runtime (some EP allocators,
    /// statistics) are not thread-safe. Concurrency means one session per
    /// worker, which is what the throughput and concurrency benchmarks build.
    Ort(Mutex<OrtGraph>),
    /// No lock: the model is immutable after load and its forward pass takes
    /// `&self`, so one loaded copy serves any number of threads.
    #[cfg(feature = "candle")]
    Candle(Box<Gliner2Candle>),
}

impl Engine {
    fn run(
        &self,
        enc: &EncodedInput,
        prefix: &PromptPrefix,
        timer: &mut StageTimer,
    ) -> Result<(TensorOut, TensorOut), DetectorError> {
        match self {
            Engine::Ort(graph) => {
                let mut graph = graph.lock().map_err(|_| {
                    DetectorError::inference("gliner2 session mutex poisoned by a previous panic")
                })?;
                let mut outputs = graph.run_i64_timed(
                    &[
                        GraphInput::batched("input_ids", &enc.input_ids),
                        GraphInput::batched("attention_mask", &enc.attention_mask),
                        GraphInput::flat("text_word_indices", &enc.text_word_indices),
                        GraphInput::flat("schema_special_indices", &prefix.schema_special_indices),
                    ],
                    CONTRACT.outputs,
                    timer,
                )?;
                let count = outputs.pop().expect("contract guarantees two outputs");
                let spans = outputs.pop().expect("contract guarantees two outputs");
                Ok((spans, count))
            }
            #[cfg(feature = "candle")]
            Engine::Candle(model) => {
                let t_in = timer.start();
                let mut t_run = None;
                let out = model.run(
                    &enc.input_ids,
                    &enc.text_word_indices,
                    &prefix.schema_special_indices,
                    &mut || {
                        timer.record(Stage::InputCopy, t_in);
                        t_run = timer.start();
                    },
                );
                // Readback is inside the run: Candle has no separate output
                // copy to attribute.
                timer.record(Stage::SessionRun, t_run);
                out
            }
        }
    }
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
        assert_fixed_shape_matches_overrides(&config, prefix.num_labels)?;
        let engine = match config.backend {
            Backend::Ort => Engine::Ort(Mutex::new(OrtGraph::load_with_options(
                &model_path,
                &CONTRACT,
                config.threads,
                &config.session,
            )?)),
            Backend::Candle(device) => load_candle(&config, &model_path, device)?,
        };

        let mut name = if config.schema.model() == "gliner2-pii" {
            "gliner2-pii".to_owned()
        } else {
            format!("gliner2-pii[{}]", config.schema.model())
        };
        if let Backend::Candle(device) = config.backend {
            name = format!("{name}-candle-{}", device.as_str());
        }

        Ok(Self {
            engine: Arc::new(engine),
            prep: Arc::new(Mutex::new(Prep {
                tokenizer,
                cache: WordIdCache::new(50_000),
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
        match &*self.engine {
            Engine::Ort(graph) => graph.lock().expect("gliner2 mutex").describe(),
            #[cfg(feature = "candle")]
            Engine::Candle(model) => model.describe(),
        }
    }

    /// Encoder input sizes for `text`, before any padding.
    ///
    /// Surfaced by `inspect` so the sweep script can pick bucket sizes from
    /// measured sub-token counts rather than from a guess at the ratio between
    /// whitespace words and SentencePiece pieces.
    pub fn encoded_size(&self, text: &str) -> Result<(usize, usize), DetectorError> {
        let prepared = words::prepare(text);
        let enc = self.encode(&prepared.words, None)?;
        Ok((enc.real_seq_len, enc.real_num_words))
    }

    /// Encoder input for `words`, holding the tokenizer lock only for this.
    ///
    /// A fresh buffer per call rather than a reused one: under concurrent
    /// callers a shared buffer would extend the lock over inference, and the
    /// allocation is a few hundred integers.
    fn encode(
        &self,
        words: &[words::Word],
        pad: Option<prompt::FixedShape>,
    ) -> Result<EncodedInput, DetectorError> {
        let mut prep = self.prep.lock().map_err(|_| {
            DetectorError::inference("gliner2 tokenizer mutex poisoned by a previous panic")
        })?;
        let Prep { tokenizer, cache } = &mut *prep;
        let mut enc = EncodedInput::default();
        prompt::build_input(&self.prefix, words, tokenizer, cache, pad, &mut enc)?;
        Ok(enc)
    }

    /// Raw `(span_probs, count_logits)` for `text`, optionally right-padded.
    ///
    /// Exists for the padding-equivalence gate in `tests/static_shapes.rs`,
    /// which has to compare tensors rather than decoded spans: decoding hides
    /// small numeric drift behind a threshold, and the count head is a hard
    /// on/off gate whose collapse would look like "no detections" rather than
    /// like a broken run.
    ///
    /// Deliberately not on the [`Detector`] trait — nothing on the benchmark
    /// path needs raw tensors, and putting it there would invite a second,
    /// divergent inference path.
    pub fn raw_outputs(
        &self,
        text: &str,
        pad: Option<prompt::FixedShape>,
    ) -> Result<(TensorOut, TensorOut), DetectorError> {
        let prepared = words::prepare(text);
        if prepared.words.is_empty() {
            return Err(DetectorError::inference("no words to encode"));
        }
        let enc = self.encode(&prepared.words, pad)?;
        self.engine
            .run(&enc, &self.prefix, &mut StageTimer::disabled())
    }

    /// Words `text` segments into, for sizing a `raw_outputs` comparison.
    pub fn word_count(&self, text: &str) -> usize {
        words::prepare(text).words.len()
    }
}

#[cfg(feature = "candle")]
fn load_candle(
    config: &Gliner2Config,
    model_path: &Path,
    device: crate::runtime::CandleDevice,
) -> Result<Engine, DetectorError> {
    // Padding is a static-shape experiment for ONNX Runtime's CoreML EP; the
    // Candle path runs unpadded and builds no attention mask.
    if config.fixed_shape.is_some() {
        return Err(DetectorError::config(
            "fixed-shape padding is an ONNX Runtime experiment; the candle backend runs unpadded"
                .to_owned(),
        ));
    }
    Ok(Engine::Candle(Box::new(Gliner2Candle::load(
        model_path,
        device,
        config.threads,
    )?)))
}

#[cfg(not(feature = "candle"))]
fn load_candle(
    _: &Gliner2Config,
    _: &Path,
    _: crate::runtime::CandleDevice,
) -> Result<Engine, DetectorError> {
    Err(DetectorError::config(
        "the candle backend is not compiled into this binary; rebuild with --features candle"
            .to_owned(),
    ))
}

/// Refuse a config whose padding and dimension overrides disagree.
///
/// These are two halves of one decision — the session is *told* the shape and
/// the encoder must *produce* it — and nothing downstream can recover if they
/// drift. Caught once at load rather than as an identical shape error on every
/// subsequent call.
fn assert_fixed_shape_matches_overrides(
    config: &Gliner2Config,
    num_labels: usize,
) -> Result<(), DetectorError> {
    let want = |name: &str| config.session.dimension_override(name);
    let mismatch = |name: &str, padded: usize, declared: Option<i64>| match declared {
        Some(d) if d as usize != padded => Err(DetectorError::config(format!(
            "--dim-override {name}={d} but inputs are padded to {padded}; \
             every call would fail on a shape mismatch"
        ))),
        _ => Ok(()),
    };

    match config.fixed_shape {
        Some(f) => {
            mismatch("seq_len", f.seq_len, want("seq_len"))?;
            mismatch("num_words", f.num_words, want("num_words"))?;
        }
        None => {
            for name in ["seq_len", "num_words"] {
                if let Some(d) = want(name) {
                    return Err(DetectorError::config(format!(
                        "--dim-override {name}={d} pins the graph but no padding was \
                         configured; inputs would stay variable-length and fail"
                    )));
                }
            }
        }
    }

    // `schema_special_indices` is already fixed-length for a given schema, so
    // an override is free — but only at the right value.
    if let Some(d) = want("num_labels_plus_1")
        && d as usize != num_labels + 1
    {
        return Err(DetectorError::config(format!(
            "--dim-override num_labels_plus_1={d} but the {num_labels}-label schema \
             produces {} indices",
            num_labels + 1
        )));
    }
    Ok(())
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
        self.detect_staged(text, &mut StageTimer::disabled())
    }

    fn supports_stage_timing(&self) -> bool {
        true
    }

    fn detect_staged(
        &self,
        text: &str,
        timer: &mut StageTimer,
    ) -> Result<Vec<Detection>, DetectorError> {
        let t_prepare = timer.start();
        let prepared = words::prepare(text);
        timer.record(Stage::Prepare, t_prepare);
        if prepared.words.is_empty() {
            return Ok(Vec::new());
        }

        // Lock waits land in Tokenize (the tokenizer lock) and SessionRun (the
        // ORT session lock). Both are uncontended in the single-caller latency
        // measurement; under concurrent callers the wait is part of what a
        // caller pays, so it belongs in the number.
        let t_tok = timer.start();
        let enc = self.encode(&prepared.words, self.config.fixed_shape)?;
        timer.record(Stage::Tokenize, t_tok);

        let (span_probs, count_logits) = self.engine.run(&enc, &self.prefix, timer)?;
        let (span_probs, count_logits) = (&span_probs, &count_logits);

        let t_decode = timer.start();
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

        let done = finish(text, out);
        timer.record(Stage::Decode, t_decode);
        done
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
            engine: Arc::clone(&self.engine),
            prep: Arc::clone(&self.prep),
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

    fn runtime_info(&self) -> Option<crate::runtime::RuntimeInfo> {
        match &*self.engine {
            Engine::Ort(graph) => Some(graph.lock().ok()?.runtime_info()),
            #[cfg(feature = "candle")]
            Engine::Candle(model) => Some(model.runtime_info()),
        }
    }

    fn fixed_prompt_tokens(&self) -> Option<usize> {
        Some(self.prefix.len())
    }

    fn concurrent_detect(&self) -> bool {
        match &*self.engine {
            Engine::Ort(_) => false,
            #[cfg(feature = "candle")]
            Engine::Candle(_) => true,
        }
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

    /// **The claim the Candle backend rests on**: the same file, rebuilt as a
    /// Candle network, reproduces the graph's own outputs.
    ///
    /// Raw tensors rather than decoded spans, for the reason `raw_outputs`
    /// gives: a threshold hides drift, and the count gate is all-or-nothing.
    /// Every `(word, width, label)` cell is compared — including the spans
    /// that run past the text, which the export rewrites to `(0, 0)` and so
    /// probe that rewrite too.
    #[cfg(feature = "candle")]
    #[test]
    fn candle_reproduces_the_onnx_graph_outputs() {
        let Some(ort) = detector(0.5) else { return };
        let mut cfg = Gliner2Config::new(MODEL_DIR);
        cfg.verify_checksum = false;
        cfg.backend = Backend::Candle(crate::runtime::CandleDevice::Cpu);
        let candle = Gliner2Detector::load(cfg).expect("candle must load the same file");
        assert_eq!(candle.name(), "gliner2-pii-candle-cpu");

        // The long document puts token distances past 128, where DeBERTa's
        // log-bucketed relative positions start, and past 512, where they clamp.
        let long = crate::perf::synthetic_document(300);
        for text in [
            "Contact John Smith at john@example.com or +1 415 555 0100.",
            "Свяжитесь с Михаил Ястребцов, IBAN DE89 3704 0044 0532 0130 00",
            "version 1.2.3 released, see issue #4411",
            "postgres://admin:s3cretpw99@prod-db.internal:5432/app",
            long.as_str(),
        ] {
            let (a_spans, a_count) = ort.raw_outputs(text, None).unwrap();
            let (b_spans, b_count) = candle.raw_outputs(text, None).unwrap();
            assert_eq!(a_spans.shape, b_spans.shape, "{text:?}");
            assert_eq!(a_count.shape, b_count.shape, "{text:?}");
            let max = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            };
            let dp = max(&a_spans.data, &b_spans.data);
            let dc = max(&a_count.data, &b_count.data);
            assert!(dp < 1e-3, "span_probs differ by {dp} on {text:?}");
            assert!(dc < 1e-2, "count_logits differ by {dc} on {text:?}");

            let strip = |d: Vec<Detection>| {
                d.into_iter()
                    .map(|x| (x.entity_type, x.start, x.end))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                strip(ort.detect(text).unwrap()),
                strip(candle.detect(text).unwrap()),
                "{text:?}"
            );
        }
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
