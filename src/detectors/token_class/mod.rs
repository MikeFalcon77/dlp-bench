//! Generic BIO token-classification PII detector.
//!
//! One forward pass over the text, one label per sub-token, spans assembled by
//! BIO decoding. No label prompt — which is the whole reason this detector
//! exists in the benchmark: [`super::gliner2`] pays 77 tokens of schema prefix
//! on every call, re-encoded by all 12 encoder layers, and the open question is
//! how much that zero-shot flexibility actually costs.
//!
//! The comparison is deliberately controlled: the shipped configuration uses
//! `Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2`, which has the **same
//! `mdeberta-v3-base` backbone** as the GLiNER2 export — same 12 layers, same
//! hidden size, same tokenizer family. The architecture of the head and the
//! absence of a prompt are the only substantive differences.
//!
//! # On the name
//!
//! The task asked for "AI4Privacy ModernBERT". No ModernBERT PII model has an
//! ONNX export (five candidates checked, all safetensors-only), and AI4Privacy
//! publish no first-party ModernBERT. This is the closest available artifact
//! that satisfies the actual requirement — an ai4privacy/OpenPII-trained token
//! classifier runnable from Rust — and it is named for what it is.

pub mod decode;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokenizers::Tokenizer;

use super::onnx::{ExecutionProviderChoice, GraphContract, GraphInput, OrtGraph, ThreadConfig};
use crate::detector::{ArtifactInfo, Detector, finish};
use crate::entity::{Detection, DetectionSource};
use crate::error::DetectorError;
use crate::labels::{LabelMap, UnmappedLabels};

pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const CONFIG_FILE: &str = "config.json";

/// One ONNX weight file at a given numeric precision.
#[derive(Debug, Clone, Copy)]
pub struct ModelVariant {
    pub precision: Precision,
    /// File name inside the model directory.
    pub file: &'static str,
    pub sha256: &'static str,
    pub size_bytes: u64,
    /// Set when the published artifact does not load, with the reason.
    ///
    /// Kept in the table rather than deleted: the variant exists upstream, and
    /// a reader comparing precisions needs to know why one is missing from the
    /// results. It also means `--variant fp16` produces an actionable message
    /// instead of a raw ONNX Runtime type error.
    pub broken: Option<&'static str>,
}

/// Numeric precision of a weight file.
///
/// A first-class axis rather than a filename detail: it is recorded in
/// [`ArtifactInfo`] and therefore in every JSON result, so a latency number can
/// never be attributed to the wrong precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp32,
    Fp16,
    Int8,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Fp16 => "fp16",
            Precision::Int8 => "int8",
        }
    }
}

/// Everything that differs between token-classification models.
///
/// Introduced once there were three of them. The decode path
/// ([`decode`], `read_id2label`, chunking, `softmax_argmax`) was already
/// model-agnostic, so this descriptor is the whole of the variation.
#[derive(Debug)]
pub struct TokenClassModel {
    pub id: &'static str,
    pub dir: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub variants: &'static [ModelVariant],
    /// Graph input names, in order. `load` compares these for exact equality.
    pub contract: GraphContract,
    /// BERT-family models take a third `token_type_ids` input; DeBERTa-v2 has
    /// `type_vocab_size: 0` and does not.
    pub needs_token_type_ids: bool,
    pub labels: &'static LabelMap,
    /// Notes surfaced by `inspect`, e.g. language coverage.
    pub note: &'static str,
}

impl TokenClassModel {
    pub fn variant(&self, precision: Precision) -> Option<&'static ModelVariant> {
        self.variants.iter().find(|v| v.precision == precision)
    }

    /// Variants that actually load.
    pub fn usable_variants(&self) -> impl Iterator<Item = &'static ModelVariant> {
        self.variants.iter().filter(|v| v.broken.is_none())
    }

    /// Default precision: the only one available if there is a single variant,
    /// otherwise fp32 so quality comparisons start from the reference weights.
    pub fn default_variant(&self) -> &'static ModelVariant {
        self.usable_variants()
            .find(|v| v.precision == Precision::Fp32)
            .or_else(|| self.usable_variants().next())
            .unwrap_or(&self.variants[0])
    }
}

const BIO_CONTRACT_2: GraphContract = GraphContract {
    inputs: &["input_ids", "attention_mask"],
    outputs: &["logits"],
};
const BIO_CONTRACT_3: GraphContract = GraphContract {
    inputs: &["input_ids", "attention_mask", "token_type_ids"],
    outputs: &["logits"],
};

/// ai4privacy/OpenPII on mdeberta-v3-base — the controlled comparison against
/// GLiNER2, which shares the same backbone.
pub static AI4PRIVACY_MDEBERTA: TokenClassModel = TokenClassModel {
    id: "ai4privacy-mdeberta",
    dir: "models/ai4privacy-mdeberta",
    repo: "Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2",
    revision: "316240516ad48a82ae1b13567670093901e41b5e",
    variants: &[ModelVariant {
        precision: Precision::Fp32,
        file: "model.onnx",
        sha256: "1c529747fec02680c565f8e840c0aed87ab2b67e8c6b993effdc4f2a2b57ed9a",
        size_bytes: 1_114_083_751,
        broken: None,
    }],
    contract: BIO_CONTRACT_2,
    needs_token_type_ids: false,
    labels: &crate::labels::AI4PRIVACY,
    note: "multilingual backbone, but the v2 training data does not carry it over",
};

/// MiniLM-L6: 6 layers x 384, the small end of off-the-shelf transformer NER.
pub static MINILM_L6_PII: TokenClassModel = TokenClassModel {
    id: "minilm-l6-pii",
    dir: "models/minilm-l6-pii",
    repo: "Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2",
    revision: "80a05dbac61de613ea01aee8542b465bb51c54b2",
    variants: &[ModelVariant {
        precision: Precision::Fp32,
        file: "model.onnx",
        sha256: "8341930b1c41fb05ffffc737c334cb78ff073c161a7806654a944d7355be893b",
        size_bytes: 90_542_099,
        broken: None,
    }],
    contract: BIO_CONTRACT_3,
    needs_token_type_ids: true,
    labels: &crate::labels::MINILM_PII,
    note: "English wordpiece (vocab 30522) — expect poor non-English recall",
};

/// DeBERTa-v3-xsmall, standing in for DataFog/pii-small-en (custom
/// CharCNN + CRF architecture, safetensors only, no export path here).
pub static DEBERTA_XSMALL_PII: TokenClassModel = TokenClassModel {
    id: "deberta-xsmall-pii",
    dir: "models/deberta-xsmall-pii",
    repo: "megaloxorafitis/pii-ner-deberta-v3-xsmall",
    revision: "54e8f7565f739dfa9febb58ddb89291342256995",
    variants: &[ModelVariant {
        // Published quantized-only; there is no fp32 weight file upstream.
        precision: Precision::Int8,
        file: "model.onnx",
        sha256: "b024fbfc5d5aeaefd3190a59b7a757a5d821103c75503b5da9c702109faacb2a",
        size_bytes: 87_334_561,
        broken: None,
    }],
    contract: BIO_CONTRACT_2,
    needs_token_type_ids: false,
    labels: &crate::labels::DEBERTA_XSMALL_PII,
    note: "no CRF (the DataFog model it substitutes for has one); only 9 types",
};

/// Max sub-tokens per forward pass.
///
/// The backbones use relative or absolute attention that is O(n^2) either way,
/// and the GLiNER2 measurements showed cost per word roughly doubling from 128
/// to 1024 words. Chunking keeps each pass in the cheap part of that curve.
const DEFAULT_CHUNK_TOKENS: usize = 256;
/// Sub-token overlap between chunks, so an entity on a boundary is seen whole
/// by at least one chunk.
const DEFAULT_CHUNK_OVERLAP: usize = 32;

#[derive(Debug, Clone)]
pub struct TokenClassConfig {
    pub model: &'static TokenClassModel,
    /// Overrides `model.dir` when set.
    pub model_dir: Option<PathBuf>,
    pub precision: Option<Precision>,
    pub threshold: f32,
    pub threads: ThreadConfig,
    pub provider: ExecutionProviderChoice,
    pub verify_checksum: bool,
    pub chunk_tokens: usize,
    pub chunk_overlap: usize,
    /// Merge adjacent same-type spans (`FIRSTNAME` + `LASTNAME` -> one
    /// `PERSON`). On by default; see [`decode::merge_adjacent`] for why.
    pub merge_adjacent: bool,
}

impl TokenClassConfig {
    pub fn new(model: &'static TokenClassModel) -> Self {
        Self {
            model,
            model_dir: None,
            precision: None,
            threshold: 0.5,
            threads: ThreadConfig::default(),
            provider: ExecutionProviderChoice::default(),
            verify_checksum: true,
            chunk_tokens: DEFAULT_CHUNK_TOKENS,
            chunk_overlap: DEFAULT_CHUNK_OVERLAP,
            merge_adjacent: true,
        }
    }

    pub fn dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(self.model.dir))
    }

    /// Resolve the requested precision, erroring if this model has no such
    /// variant rather than silently benchmarking a different weight file.
    pub fn resolve_variant(&self) -> Result<&'static ModelVariant, DetectorError> {
        match self.precision {
            None => Ok(self.model.default_variant()),
            Some(p) => self.model.variant(p).ok_or_else(|| {
                DetectorError::config(format!(
                    "{} has no {} variant; available: {}",
                    self.model.id,
                    p.as_str(),
                    self.model
                        .variants
                        .iter()
                        .map(|v| v.precision.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            }),
        }
    }

    pub fn labels(&self) -> &'static LabelMap {
        self.model.labels
    }
}

pub struct TokenClassDetector {
    inner: Arc<Mutex<Inner>>,
    /// `id -> BIO label`, read from the model's own `config.json`.
    id2label: Arc<Vec<String>>,
    config: TokenClassConfig,
    variant: &'static ModelVariant,
    name: String,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

struct Inner {
    graph: OrtGraph,
    tokenizer: Tokenizer,
    unmapped: UnmappedLabels,
}

impl TokenClassDetector {
    pub fn load(config: TokenClassConfig) -> Result<Self, DetectorError> {
        let variant = config.resolve_variant()?;
        let dir = config.dir();
        let model_path = dir.join(variant.file);
        let tokenizer_path = dir.join(TOKENIZER_FILE);
        let config_path = dir.join(CONFIG_FILE);
        let model_id = config.model.id.to_owned();

        for p in [&model_path, &tokenizer_path, &config_path] {
            if !p.is_file() {
                return Err(DetectorError::MissingArtifact {
                    path: p.clone(),
                    model_id: model_id.clone(),
                });
            }
        }

        if config.verify_checksum {
            verify_sha256(&model_path, variant.sha256)?;
        }

        // The label vocabulary comes from the model, never from a hard-coded
        // list: an id/label mismatch would silently relabel every prediction.
        // Note the ordering is not assumed — DeBERTa-xsmall puts `O` at index
        // 13, not 0, and gives five of its types a `B-` tag with no `I-`.
        let id2label = read_id2label(&config_path)?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| DetectorError::Tokenizer(format!("{}: {e}", tokenizer_path.display())))?;
        // We chunk ourselves and need every sub-token's offsets, so the
        // tokenizer must not truncate or pad behind our back.
        let _ = tokenizer.with_truncation(None);
        tokenizer.with_padding(None);

        let graph = OrtGraph::load_with(
            &model_path,
            &config.model.contract,
            config.threads,
            config.provider,
        )?;

        let name = if config.model.variants.len() > 1 {
            format!("{}-{}", config.model.id, variant.precision.as_str())
        } else {
            config.model.id.to_owned()
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                graph,
                tokenizer,
                unmapped: UnmappedLabels::default(),
            })),
            id2label: Arc::new(id2label),
            config,
            variant,
            name,
            model_path,
            tokenizer_path,
        })
    }

    pub fn config(&self) -> &TokenClassConfig {
        &self.config
    }

    pub fn variant(&self) -> &'static ModelVariant {
        self.variant
    }

    pub fn model(&self) -> &'static TokenClassModel {
        self.config.model
    }

    pub fn num_labels(&self) -> usize {
        self.id2label.len()
    }

    /// Model labels seen at runtime that the mapping table does not cover.
    pub fn unmapped_labels(&self) -> UnmappedLabels {
        self.inner
            .lock()
            .map(|i| i.unmapped.clone())
            .unwrap_or_default()
    }

    pub fn describe_graph(&self) -> String {
        self.inner
            .lock()
            .expect("token_class mutex")
            .graph
            .describe()
    }
}

fn read_id2label(path: &Path) -> Result<Vec<String>, DetectorError> {
    let text = std::fs::read_to_string(path).map_err(|e| DetectorError::io(path, e))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| DetectorError::config(format!("{}: {e}", path.display())))?;
    let map = json
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or_else(|| DetectorError::config(format!("{}: no id2label object", path.display())))?;

    let mut out = vec![String::new(); map.len()];
    for (k, v) in map {
        let idx: usize = k
            .parse()
            .map_err(|_| DetectorError::config(format!("id2label key {k:?} is not an index")))?;
        let label = v
            .as_str()
            .ok_or_else(|| DetectorError::config(format!("id2label[{k}] is not a string")))?;
        if idx >= out.len() {
            return Err(DetectorError::config(format!(
                "id2label index {idx} out of range for {} labels",
                out.len()
            )));
        }
        out[idx] = label.to_owned();
    }
    if out.iter().any(String::is_empty) {
        return Err(DetectorError::config("id2label has gaps".to_owned()));
    }
    Ok(out)
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

/// Softmax over one row, returning `(argmax index, probability)`.
///
/// Max-subtracted for numerical stability; the raw logits from a 111-class head
/// routinely exceed 15.
fn softmax_argmax(row: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    let max = row[best];
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    (best, if sum > 0.0 { 1.0 / sum } else { 0.0 })
}

impl Detector for TokenClassDetector {
    fn name(&self) -> &str {
        &self.name
    }

    fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut inner = self.inner.lock().map_err(|_| {
            DetectorError::inference("token_class session mutex poisoned by a previous panic")
        })?;
        let Inner {
            graph,
            tokenizer,
            unmapped,
        } = &mut *inner;

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| DetectorError::Tokenizer(format!("encoding failed: {e}")))?;

        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();
        let special = encoding.get_special_tokens_mask();
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let mut tagged: Vec<decode::TaggedSpan> = Vec::new();

        // Chunk over sub-tokens with overlap. Each chunk is a full forward pass
        // with its own [CLS]/[SEP] already present in the encoding, so we slice
        // the id stream directly and let the model see a shorter sequence.
        let stride = self
            .config
            .chunk_tokens
            .saturating_sub(self.config.chunk_overlap)
            .max(1);
        let mut start = 0usize;
        loop {
            let end = (start + self.config.chunk_tokens).min(ids.len());
            let chunk_ids: Vec<i64> = ids[start..end].iter().map(|&i| i64::from(i)).collect();
            let mask = vec![1i64; chunk_ids.len()];

            // BERT-family models take a third all-zeros segment input; the
            // DeBERTa-v2 ones have `type_vocab_size: 0` and reject it. The
            // graph contract is compared for exact equality at load, so a
            // mismatch here would already have failed loudly.
            let mut inputs = vec![
                GraphInput::batched("input_ids", &chunk_ids),
                GraphInput::batched("attention_mask", &mask),
            ];
            let type_ids;
            if self.config.model.needs_token_type_ids {
                type_ids = vec![0i64; chunk_ids.len()];
                inputs.push(GraphInput::batched("token_type_ids", &type_ids));
            }
            let outputs = graph.run_i64(&inputs, self.config.model.contract.outputs)?;
            let logits = &outputs[0];

            let n_labels = *logits.shape.last().ok_or_else(|| {
                DetectorError::graph("logits tensor has no dimensions".to_owned())
            })?;
            if n_labels != self.id2label.len() {
                return Err(DetectorError::graph(format!(
                    "model emits {n_labels} classes but config.json declares {}",
                    self.id2label.len()
                )));
            }
            if logits.data.len() != chunk_ids.len() * n_labels {
                return Err(DetectorError::graph(format!(
                    "logits shape {:?} does not match {} tokens x {n_labels} classes",
                    logits.shape,
                    chunk_ids.len()
                )));
            }

            let mut tags: Vec<decode::TokenTag<'_>> = Vec::with_capacity(chunk_ids.len());
            for (i, tok) in (start..end).enumerate() {
                // [CLS]/[SEP] carry offsets (0, 0) and no text; feeding them to
                // the decoder would open spans at byte 0.
                if special.get(tok).copied().unwrap_or(0) == 1 {
                    continue;
                }
                let (s, e) = offsets[tok];
                if s >= e {
                    continue;
                }
                let row = &logits.data[i * n_labels..(i + 1) * n_labels];
                let (best, prob) = softmax_argmax(row);
                let label = self.id2label[best].as_str();
                if let Some((_, base)) = decode::split_bio(label)
                    && self.config.model.labels.map(base).is_none()
                {
                    unmapped.record(base);
                }
                tags.push(decode::TokenTag {
                    label,
                    score: prob,
                    start: s,
                    end: e,
                });
            }

            let schema = self.config.model.labels;
            tagged.extend(decode::decode_bio(
                &tags,
                &|base| schema.map(base),
                self.config.threshold,
            ));

            if end >= ids.len() {
                break;
            }
            start += stride;
        }

        // Overlapping chunks legitimately re-detect the same entity; merging
        // same-type neighbours also collapses those duplicates.
        let mut spans = decode::trim_spans(tagged, text);
        if self.config.merge_adjacent {
            spans = decode::merge_adjacent(spans, text, 3);
        } else {
            spans = decode::merge_adjacent(spans, text, 0);
        }

        let source = DetectionSource::model(self.name.clone());
        let out = spans
            .into_iter()
            .map(|s| Detection::new(s.entity_type, s.start, s.end, s.score, source.clone()))
            .collect();
        finish(text, out)
    }

    fn threshold(&self) -> Option<f32> {
        Some(self.config.threshold)
    }

    fn with_threshold(&self, threshold: f32) -> Option<Box<dyn Detector>> {
        let mut config = self.config.clone();
        config.threshold = threshold;
        Some(Box::new(Self {
            inner: Arc::clone(&self.inner),
            id2label: Arc::clone(&self.id2label),
            config,
            variant: self.variant,
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
                repo: Some(self.config.model.repo.into()),
                revision: Some(self.config.model.revision.into()),
                sha256: Some(self.variant.sha256.into()),
                size_bytes: Some(self.variant.size_bytes),
            },
            ArtifactInfo {
                name: "tokenizer".into(),
                path: self.tokenizer_path.display().to_string(),
                repo: Some(self.config.model.repo.into()),
                revision: Some(self.config.model.revision.into()),
                sha256: None,
                size_bytes: std::fs::metadata(&self.tokenizer_path)
                    .ok()
                    .map(|m| m.len()),
            },
        ]
    }

    fn model_size_bytes(&self) -> Option<u64> {
        Some(self.variant.size_bytes)
    }

    /// None: this is the point of the comparison. A token classifier pays no
    /// per-call prompt overhead at all.
    fn fixed_prompt_tokens(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = "models/ai4privacy-mdeberta";

    fn present() -> bool {
        model_present(&AI4PRIVACY_MDEBERTA)
    }

    /// Every descriptor's default weight file, for the table-wide tests.
    fn model_present(m: &TokenClassModel) -> bool {
        Path::new(m.dir).join(m.default_variant().file).is_file()
    }

    const ALL_MODELS: &[&TokenClassModel] =
        &[&AI4PRIVACY_MDEBERTA, &MINILM_L6_PII, &DEBERTA_XSMALL_PII];

    #[test]
    fn softmax_is_stable_and_normalised() {
        let (i, p) = softmax_argmax(&[1.0, 5.0, 2.0]);
        assert_eq!(i, 1);
        assert!((0.0..=1.0).contains(&p));
        assert!((p - 0.936).abs() < 0.01, "p = {p}");

        // Large logits must not overflow to NaN.
        let (i, p) = softmax_argmax(&[100.0, 300.0, 50.0]);
        assert_eq!(i, 1);
        assert!(p.is_finite() && p > 0.99, "p = {p}");

        // A flat distribution over 111 classes.
        let (_, p) = softmax_argmax(&vec![0.0f32; 111]);
        assert!((p - 1.0 / 111.0).abs() < 1e-5, "p = {p}");
    }

    #[test]
    fn id2label_is_read_from_the_model_and_gaps_are_rejected() {
        if !present() {
            eprintln!("SKIP: run `bash scripts/fetch-models.sh ai4privacy-mdeberta`");
            return;
        }
        let labels = read_id2label(Path::new(DIR).join(CONFIG_FILE).as_path()).unwrap();
        assert_eq!(labels.len(), 111, "1 O + 2 x 56 types");
        assert_eq!(labels[0], "O");
        assert!(labels.iter().any(|l| l == "B-EMAIL"));
        assert!(labels.iter().any(|l| l == "I-EMAIL"));

        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("config.json");
        std::fs::write(&bad, r#"{"id2label": {"0": "O", "2": "B-X"}}"#).unwrap();
        assert!(
            read_id2label(&bad).is_err(),
            "a gap in id2label must be rejected"
        );
    }

    /// **The assumption the whole detector rests on.**
    ///
    /// The Rust `tokenizers` crate reports sub-token offsets as UTF-8 byte
    /// offsets into the original string; the Python binding converts them to
    /// char offsets. Porting code between the two silently corrupts every span
    /// on non-ASCII input, so this is verified against real multibyte text
    /// rather than taken on trust.
    #[test]
    fn tokenizer_offsets_are_utf8_byte_offsets_not_char_offsets() {
        if !present() {
            eprintln!("SKIP: model absent");
            return;
        }
        let tok = Tokenizer::from_file(Path::new(DIR).join(TOKENIZER_FILE)).unwrap();

        for text in [
            "Contact John Smith at john@example.com",
            "Свяжитесь с Михаил Ястребцов",
            "Jörg Müller wohnt in München",
            "José García vive en Málaga",
            "François Dupont habite à Paris",
            "王小明在台北工作",
            "emoji 👋 boundary test",
        ] {
            let enc = tok.encode(text, true).unwrap();
            let mut ranges = Vec::new();
            for (i, &(s, e)) in enc.get_offsets().iter().enumerate() {
                if enc.get_special_tokens_mask()[i] == 1 || s >= e {
                    continue;
                }
                assert!(e <= text.len(), "offset {e} past text len in {text:?}");
                assert!(
                    text.is_char_boundary(s) && text.is_char_boundary(e),
                    "offset ({s},{e}) is not a char boundary in {text:?} — \
                     these would be char offsets, not byte offsets"
                );
                ranges.push((s, e));
            }

            // Sub-token ranges may *overlap*: SentencePiece emits a bare `▁`
            // whose range is the first byte of the following token (for
            // "José": `▁` -> (0,1) and `José` -> (0,5)). So the invariant is
            // that their union covers every non-whitespace byte, not that
            // concatenating them reproduces the string.
            let covered = crate::span::merge_ranges(ranges);
            for (i, ch) in text.char_indices() {
                if ch.is_whitespace() {
                    continue;
                }
                assert!(
                    covered.iter().any(|&(s, e)| i >= s && i < e),
                    "byte {i} ({ch:?}) of {text:?} is not covered by any sub-token"
                );
            }
            // Nothing may reach past the end of the text.
            assert!(covered.last().is_none_or(|&(_, e)| e <= text.len()));
        }
    }

    #[test]
    fn missing_model_names_the_fetch_script() {
        let mut cfg = TokenClassConfig::new(&AI4PRIVACY_MDEBERTA);
        cfg.model_dir = Some(PathBuf::from("models/nope"));
        let msg = match TokenClassDetector::load(cfg) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("must not load"),
        };
        assert!(msg.contains("fetch-models.sh"), "{msg}");
    }

    /// Each descriptor must agree with the artifact actually on disk: the
    /// declared graph contract, the label count, and whether the model wants
    /// `token_type_ids`. A descriptor that drifts from its model produces
    /// confident nonsense, so this is checked for every entry in the table.
    #[test]
    fn every_descriptor_matches_its_real_artifact() {
        let mut checked = 0;
        for m in ALL_MODELS {
            if !model_present(m) {
                eprintln!("SKIP {}: run `bash scripts/fetch-models.sh {}`", m.id, m.id);
                continue;
            }
            let mut cfg = TokenClassConfig::new(m);
            cfg.verify_checksum = false;
            cfg.threads = ThreadConfig::single();
            let d = TokenClassDetector::load(cfg)
                .unwrap_or_else(|e| panic!("{} failed to load: {e}", m.id));

            // The label map must not claim labels the model cannot emit.
            let labels = read_id2label(Path::new(m.dir).join(CONFIG_FILE).as_path()).unwrap();
            assert_eq!(d.num_labels(), labels.len(), "{}", m.id);
            let bases: std::collections::BTreeSet<&str> = labels
                .iter()
                .filter_map(|l| decode::split_bio(l).map(|(_, b)| b))
                .collect();
            for mapped in m.labels.labels() {
                assert!(
                    bases.contains(mapped),
                    "{}: label map claims {mapped:?}, which the model cannot emit",
                    m.id
                );
            }

            // A round-trip through the graph proves the contract and the
            // token_type_ids flag are right — a wrong flag fails at load or at
            // the first run, never silently.
            d.detect("Contact John Smith at john@example.com")
                .unwrap_or_else(|e| panic!("{} inference failed: {e}", m.id));
            checked += 1;
        }
        if checked == 0 {
            eprintln!("no token-classification models present; nothing verified");
        }
    }

    #[test]
    fn requesting_an_unavailable_precision_is_an_error() {
        // DeBERTa-xsmall is published quantized-only; asking for fp32 must say
        // so rather than quietly load the int8 file.
        let mut cfg = TokenClassConfig::new(&DEBERTA_XSMALL_PII);
        cfg.precision = Some(Precision::Fp32);
        let msg = match cfg.resolve_variant() {
            Err(e) => e.to_string(),
            Ok(v) => panic!("unexpectedly resolved to {}", v.file),
        };
        assert!(msg.contains("no fp32 variant"), "{msg}");
        assert!(msg.contains("int8"), "must list what is available: {msg}");

        // ...and the default resolves to the only variant there is.
        let cfg = TokenClassConfig::new(&DEBERTA_XSMALL_PII);
        assert_eq!(cfg.resolve_variant().unwrap().precision, Precision::Int8);
    }

    #[test]
    fn descriptor_table_is_internally_consistent() {
        for m in ALL_MODELS {
            assert!(!m.variants.is_empty(), "{} has no variants", m.id);
            assert_eq!(
                m.contract.inputs.len(),
                if m.needs_token_type_ids { 3 } else { 2 },
                "{}: contract and needs_token_type_ids disagree",
                m.id
            );
            assert!(
                m.dir.ends_with(m.id),
                "{}: dir should end with the id",
                m.id
            );
            assert!(!m.labels.is_empty(), "{} has an empty label map", m.id);
            assert!(!m.note.is_empty(), "{} has no note", m.id);
        }
    }
}
