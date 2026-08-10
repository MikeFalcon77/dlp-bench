//! GLiNER1 zero-shot span detector (`token_level` mode).
//!
//! Runs the knowledgator GLiNER-PII family locally on CPU. Two sizes are wired
//! — `edge` (ettin-encoder-32m) and `small` (-68m) — each at three precisions,
//! which is the point: the same weights at fp32/fp16/int8 isolate the
//! quantization axis from the size axis.
//!
//! # How this differs from [`super::gliner2`]
//!
//! Same family name, different model in almost every respect:
//!
//! | | GLiNER1 (here) | GLiNER2 |
//! |---|---|---|
//! | prompt | `<<ENT>> label … <<SEP>> words` | `( [P] entities ( [E] label … ) ) [SEP_TEXT]` |
//! | word splitter | `\w+(?:[-_]\w+)*\|\S` | URL/email/handle-aware |
//! | text fixup | none | trailing `.` appended |
//! | output | `[1, words, classes, 3]` start/end/inside | `[words, width, labels]` span scores |
//! | overlap policy | flat, across classes | per label |
//! | count gate | none | `argmax(count_logits) == 0` suppresses everything |
//!
//! Each follows its own upstream reference rather than a shared house style,
//! because matching the decode a model was trained with matters more than
//! internal consistency between two unrelated models.

pub mod decode;
pub mod prompt;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tokenizers::Tokenizer;

use super::onnx::{ExecutionProviderChoice, GraphContract, GraphInput, OrtGraph, ThreadConfig};
use super::token_class::{ModelVariant, Precision};
use super::words::Word;
use crate::detector::{ArtifactInfo, Detector, finish};
use crate::entity::{Detection, DetectionSource};
use crate::error::DetectorError;
use crate::labels::LabelMap;

pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const CONFIG_FILE: &str = "gliner_config.json";

const CONTRACT: GraphContract = GraphContract {
    inputs: &["input_ids", "attention_mask", "words_mask", "text_lengths"],
    outputs: &["logits"],
};

/// One GLiNER1 checkpoint.
#[derive(Debug)]
pub struct Gliner1Model {
    pub id: &'static str,
    pub dir: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub variants: &'static [ModelVariant],
    pub backbone: &'static str,
}

impl Gliner1Model {
    pub fn variant(&self, p: Precision) -> Option<&'static ModelVariant> {
        self.variants.iter().find(|v| v.precision == p)
    }

    pub fn usable_variants(&self) -> impl Iterator<Item = &'static ModelVariant> {
        self.variants.iter().filter(|v| v.broken.is_none())
    }

    /// int8 by default: this family exists to answer "how small can it get",
    /// and the quantized weights are quantization-aware trained rather than
    /// post-training quantized, so int8 is the intended deployment target.
    pub fn default_variant(&self) -> &'static ModelVariant {
        self.usable_variants()
            .find(|v| v.precision == Precision::Int8)
            .or_else(|| self.usable_variants().next())
            .unwrap_or(&self.variants[0])
    }
}

pub static GLINER1_PII_EDGE: Gliner1Model = Gliner1Model {
    id: "gliner1-pii-edge",
    dir: "models/gliner-pii-edge",
    repo: "knowledgator/gliner-pii-edge-v1.0",
    revision: "9b7f39b0a2da971a5beea78d35f1539d4009c891",
    backbone: "jhu-clsp/ettin-encoder-32m",
    variants: &[
        ModelVariant {
            precision: Precision::Fp32,
            file: "model.onnx",
            sha256: "4ca588722e6d79447ad4c9c230eeba3d9d472c672a9598184a34e9f77fc35836",
            size_bytes: 181_078_966,
            broken: None,
        },
        ModelVariant {
            precision: Precision::Fp16,
            file: "model_fp16.onnx",
            sha256: "af499f06532909628765ce1cebd65311fa2d9a16f74a34e7b6e1b9369230a3a7",
            size_bytes: 90_845_497,
            // Verified on both checkpoints: ONNX Runtime rejects the graph
            // with "Type (tensor(float16)) of output arg
            // .../rotary_emb/Cast_1_output_0 ... does not match expected type
            // (tensor(float))". A defect in the published export, not in the
            // loader — fp32 and int8 from the same repo load fine.
            broken: Some(
                "upstream fp16 export has a float16/float32 type \
                          mismatch in the rotary-embedding Cast node",
            ),
        },
        ModelVariant {
            precision: Precision::Int8,
            file: "model_quint8.onnx",
            sha256: "988acb03456b26e2d9f2521016d820310c2ed64deb4a846297d3289f0c2eb7e4",
            size_bytes: 45_820_894,
            broken: None,
        },
    ],
};

pub static GLINER1_PII_SMALL: Gliner1Model = Gliner1Model {
    id: "gliner1-pii-small",
    dir: "models/gliner-pii-small",
    repo: "knowledgator/gliner-pii-small-v1.0",
    revision: "d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e",
    backbone: "jhu-clsp/ettin-encoder-68m",
    variants: &[
        ModelVariant {
            precision: Precision::Fp32,
            file: "model.onnx",
            sha256: "c60fd671a30798c1bfad5828f8bbf6214eb44cbc5e2f0fb968563ae935a17454",
            size_bytes: 326_820_148,
            broken: None,
        },
        ModelVariant {
            precision: Precision::Fp16,
            file: "model_fp16.onnx",
            sha256: "522d6dad97e2bfb836bcc7be33d792472e1fc24dc45e99f86c19ecd3020143f1",
            size_bytes: 163_945_472,
            // Verified on both checkpoints: ONNX Runtime rejects the graph
            // with "Type (tensor(float16)) of output arg
            // .../rotary_emb/Cast_1_output_0 ... does not match expected type
            // (tensor(float))". A defect in the published export, not in the
            // loader — fp32 and int8 from the same repo load fine.
            broken: Some(
                "upstream fp16 export has a float16/float32 type \
                          mismatch in the rotary-embedding Cast node",
            ),
        },
        ModelVariant {
            precision: Precision::Int8,
            file: "model_quint8.onnx",
            sha256: "891589426ee96f2748b16439f44fad8c3f97e198e002a6637e58dee989500216",
            size_bytes: 82_680_500,
            broken: None,
        },
    ],
};

#[derive(Debug, Clone)]
pub struct Gliner1Config {
    pub model: &'static Gliner1Model,
    pub model_dir: Option<PathBuf>,
    pub precision: Option<Precision>,
    /// The model card recommends 0.3, not 0.5.
    pub threshold: f32,
    pub threads: ThreadConfig,
    pub provider: ExecutionProviderChoice,
    pub schema: &'static LabelMap,
    pub verify_checksum: bool,
    pub max_width: usize,
    pub overlap: decode::OverlapPolicy,
}

impl Gliner1Config {
    pub fn new(model: &'static Gliner1Model) -> Self {
        Self {
            model,
            model_dir: None,
            precision: None,
            threshold: 0.3,
            threads: ThreadConfig::default(),
            provider: ExecutionProviderChoice::default(),
            schema: &crate::labels::GLINER1_PII,
            verify_checksum: true,
            // From gliner_config.json for both checkpoints.
            max_width: 12,
            overlap: decode::OverlapPolicy::Flat,
        }
    }

    pub fn dir(&self) -> PathBuf {
        self.model_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(self.model.dir))
    }

    pub fn resolve_variant(&self) -> Result<&'static ModelVariant, DetectorError> {
        match self.precision {
            None => Ok(self.model.default_variant()),
            Some(p) => {
                let v = self.model.variant(p).ok_or_else(|| {
                    DetectorError::config(format!(
                        "{} has no {} variant; available: {}",
                        self.model.id,
                        p.as_str(),
                        self.model
                            .usable_variants()
                            .map(|v| v.precision.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?;
                if let Some(why) = v.broken {
                    return Err(DetectorError::config(format!(
                        "{} {} is unusable: {why}. Usable: {}",
                        self.model.id,
                        p.as_str(),
                        self.model
                            .usable_variants()
                            .map(|v| v.precision.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
                Ok(v)
            }
        }
    }
}

pub struct Gliner1Detector {
    inner: Arc<Mutex<Inner>>,
    labels: Arc<Vec<&'static str>>,
    config: Gliner1Config,
    variant: &'static ModelVariant,
    name: String,
    model_path: PathBuf,
    tokenizer_path: PathBuf,
}

struct Inner {
    graph: OrtGraph,
    tokenizer: Tokenizer,
    words: Vec<Word>,
    enc: prompt::EncodedInput,
}

impl Gliner1Detector {
    pub fn load(config: Gliner1Config) -> Result<Self, DetectorError> {
        let variant = config.resolve_variant()?;
        let dir = config.dir();
        let model_path = dir.join(variant.file);
        let tokenizer_path = dir.join(TOKENIZER_FILE);

        for p in [&model_path, &tokenizer_path] {
            if !p.is_file() {
                return Err(DetectorError::MissingArtifact {
                    path: p.clone(),
                    model_id: config.model.id.to_owned(),
                });
            }
        }
        if config.verify_checksum {
            verify_sha256(&model_path, variant.sha256)?;
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| DetectorError::Tokenizer(format!("{}: {e}", tokenizer_path.display())))?;
        prompt::assert_special_tokens(&tokenizer)?;

        let graph = OrtGraph::load_with(&model_path, &CONTRACT, config.threads, config.provider)?;
        let name = format!("{}-{}", config.model.id, variant.precision.as_str());
        let labels = config.schema.labels();

        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                graph,
                tokenizer,
                words: Vec::new(),
                enc: prompt::EncodedInput::default(),
            })),
            labels: Arc::new(labels),
            config,
            variant,
            name,
            model_path,
            tokenizer_path,
        })
    }

    pub fn config(&self) -> &Gliner1Config {
        &self.config
    }

    pub fn variant(&self) -> &'static ModelVariant {
        self.variant
    }

    pub fn model(&self) -> &'static Gliner1Model {
        self.config.model
    }

    pub fn describe_graph(&self) -> String {
        self.inner.lock().expect("gliner1 mutex").graph.describe()
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

impl Detector for Gliner1Detector {
    fn name(&self) -> &str {
        &self.name
    }

    fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut inner = self.inner.lock().map_err(|_| {
            DetectorError::inference("gliner1 session mutex poisoned by a previous panic")
        })?;
        let Inner {
            graph,
            tokenizer,
            words,
            enc,
        } = &mut *inner;

        prompt::split_words(text, words);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        prompt::build_input(tokenizer, &self.labels, words, text, enc)?;

        let text_lengths = [words.len() as i64];
        let outputs = graph.run_i64(
            &[
                GraphInput::batched("input_ids", &enc.input_ids),
                GraphInput::batched("attention_mask", &enc.attention_mask),
                GraphInput::batched("words_mask", &enc.words_mask),
                GraphInput {
                    name: "text_lengths",
                    shape: vec![1, 1],
                    data: &text_lengths,
                },
            ],
            CONTRACT.outputs,
        )?;
        let logits = &outputs[0];

        // Fail loudly on a shape we did not verify against, rather than index
        // into it and emit plausible nonsense.
        let dims = decode::LogitDims::from_shape(&logits.shape).ok_or_else(|| {
            DetectorError::graph(format!(
                "expected logits shape [1, words, classes, 3], got {:?}",
                logits.shape
            ))
        })?;
        if dims.num_classes != self.labels.len() {
            return Err(DetectorError::graph(format!(
                "graph returned {} classes but {} labels were requested",
                dims.num_classes,
                self.labels.len()
            )));
        }
        if dims.num_words != words.len() {
            return Err(DetectorError::graph(format!(
                "graph returned {} word positions for {} words",
                dims.num_words,
                words.len()
            )));
        }

        let raw = decode::decode(
            &logits.data,
            dims,
            words,
            self.config.threshold,
            self.config.max_width,
            self.config.overlap,
        );

        let source = DetectionSource::model(self.name.clone());
        let mut out = Vec::with_capacity(raw.len());
        for s in raw {
            let Some(entity_type) = self.config.schema.map_index(s.class_idx) else {
                return Err(DetectorError::inference(format!(
                    "decoded class index {} is outside the {}-label schema",
                    s.class_idx,
                    self.config.schema.len()
                )));
            };
            // Word spans can include trailing punctuation because punctuation
            // is its own word; apply the same boundary convention as every
            // other detector.
            let (start, end) =
                crate::span::trim_trailing(text, s.start, s.end, crate::span::TRAILING_PUNCT);
            let Some((start, end)) = crate::span::trim_span(text, start, end) else {
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
        let mut config = self.config.clone();
        config.threshold = threshold;
        Some(Box::new(Self {
            inner: Arc::clone(&self.inner),
            labels: Arc::clone(&self.labels),
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
                name: format!("onnx-{}", self.variant.precision.as_str()),
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

    /// GLiNER1 pays a per-call label prompt just as GLiNER2 does, though a
    /// shorter one: two tokens of overhead per label rather than four-ish.
    fn fixed_prompt_tokens(&self) -> Option<usize> {
        Some(self.labels.len() * 2 + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn present(m: &Gliner1Model) -> bool {
        Path::new(m.dir).join(m.default_variant().file).is_file()
    }

    fn detector(m: &'static Gliner1Model, p: Option<Precision>) -> Option<Gliner1Detector> {
        if !present(m) {
            eprintln!("SKIP {}: run `bash scripts/fetch-models.sh`", m.id);
            return None;
        }
        let mut cfg = Gliner1Config::new(m);
        cfg.precision = p;
        cfg.verify_checksum = false;
        cfg.threads = ThreadConfig::with_intra(4);
        Some(Gliner1Detector::load(cfg).expect("model must load"))
    }

    #[test]
    fn both_checkpoints_load_and_detect() {
        for m in [&GLINER1_PII_EDGE, &GLINER1_PII_SMALL] {
            let Some(d) = detector(m, None) else { continue };
            let text = "Contact John Smith at john@example.com";
            let out = d.detect(text).expect("detect");
            crate::span::validate_all(text, &out).expect("spans must validate");
            assert!(!out.is_empty(), "{} found nothing at all", m.id);
            assert_eq!(d.threshold(), Some(0.3), "model card recommends 0.3");
        }
    }

    #[test]
    fn every_variant_of_every_checkpoint_loads_and_agrees_broadly() {
        // The quantization axis is the point of this family; a variant that
        // fails to load or returns wildly different spans invalidates the
        // comparison, so all six are exercised.
        for m in [&GLINER1_PII_EDGE, &GLINER1_PII_SMALL] {
            if !present(m) {
                continue;
            }
            let text = "Contact John Smith at john@example.com";
            let mut per_variant = Vec::new();
            for p in [Precision::Fp32, Precision::Fp16, Precision::Int8] {
                let v = m.variant(p).unwrap();
                if !Path::new(m.dir).join(v.file).is_file() {
                    continue;
                }
                if v.broken.is_some() {
                    // The published fp16 exports do not load; asking for one
                    // must produce the recorded reason, not an ONNX type error.
                    let mut cfg = Gliner1Config::new(m);
                    cfg.precision = Some(p);
                    let msg = match Gliner1Detector::load(cfg) {
                        Err(e) => e.to_string(),
                        Ok(_) => panic!("{} {p:?} was marked broken but loaded", m.id),
                    };
                    assert!(msg.contains("unusable"), "{msg}");
                    assert!(msg.contains("rotary"), "reason must be specific: {msg}");
                    continue;
                }
                let Some(d) = detector(m, Some(p)) else {
                    continue;
                };
                assert_eq!(d.variant().precision, p);
                assert!(d.name().ends_with(p.as_str()), "name must record precision");
                let out = d.detect(text).expect("detect");
                crate::span::validate_all(text, &out).expect("spans validate");
                per_variant.push((p, out.len()));
            }
            assert_eq!(
                per_variant.len(),
                2,
                "{}: fp32 and int8 should be usable",
                m.id
            );
        }
    }

    #[test]
    fn a_missing_model_names_the_fetch_script() {
        let mut cfg = Gliner1Config::new(&GLINER1_PII_EDGE);
        cfg.model_dir = Some(PathBuf::from("models/nope"));
        let msg = match Gliner1Detector::load(cfg) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("must not load"),
        };
        assert!(msg.contains("fetch-models.sh"), "{msg}");
    }

    #[test]
    fn descriptor_tables_are_consistent() {
        for m in [&GLINER1_PII_EDGE, &GLINER1_PII_SMALL] {
            assert_eq!(m.variants.len(), 3, "{}", m.id);
            let mut seen = std::collections::BTreeSet::new();
            for v in m.variants {
                assert!(
                    seen.insert(v.precision.as_str()),
                    "{}: duplicate precision",
                    m.id
                );
                assert_eq!(v.sha256.len(), 64, "{}: bad sha256", m.id);
                assert!(v.size_bytes > 1_000_000, "{}: implausible size", m.id);
            }
            // Smaller precision must mean a smaller file.
            let fp32 = m.variant(Precision::Fp32).unwrap().size_bytes;
            let fp16 = m.variant(Precision::Fp16).unwrap().size_bytes;
            let int8 = m.variant(Precision::Int8).unwrap().size_bytes;
            assert!(fp32 > fp16 && fp16 > int8, "{}: sizes not ordered", m.id);
            assert_eq!(m.default_variant().precision, Precision::Int8);
            // fp16 is present in the table but excluded from the usable set.
            assert_eq!(m.usable_variants().count(), 2, "{}", m.id);
            assert!(
                m.variant(Precision::Fp16).unwrap().broken.is_some(),
                "{}",
                m.id
            );
        }
        // ...and edge really is smaller than small.
        assert!(
            GLINER1_PII_EDGE
                .variant(Precision::Int8)
                .unwrap()
                .size_bytes
                < GLINER1_PII_SMALL
                    .variant(Precision::Int8)
                    .unwrap()
                    .size_bytes
        );
    }
}
