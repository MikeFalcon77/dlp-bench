//! Benchmark dataset representation, loading and validation.
//!
//! One internal JSONL format that every source — hand-authored AI-DLP fixtures,
//! PIIBench subsets, AI4Privacy/OpenPII exports — is converted into. Importers
//! are the only place that knows about a foreign schema or a foreign offset
//! convention.

pub mod import;
mod jsonl;
pub mod transform;

pub use jsonl::{DatasetError, load_jsonl, load_jsonl_str};

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::entity::EntityType;
use crate::span::TypedSpan;

/// What a dataset is for.
///
/// Required, not optional. Choosing a threshold on the same data used for the
/// final comparison is the easiest way to publish an inflated number, so the
/// role is part of the run metadata and the sweep command warns when it is
/// asked to calibrate on a `Test` set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetRole {
    /// Calibration / development. Thresholds may be chosen here.
    Dev,
    /// Held out for final comparison.
    Test,
}

/// Content shape of an example, used to slice metrics by input kind.
///
/// This is the endpoint-DLP-specific axis: a detector that is excellent on
/// prose and useless on YAML is useless for intercepting Cursor prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tag {
    Prose,
    Json,
    Yaml,
    SourceCode,
    Log,
    Sql,
    Config,
    AiPrompt,
    Adversarial,
    /// Contains no PII at all. Drives the false-positive measurement.
    Negative,
}

impl Tag {
    pub const ALL: &'static [Tag] = &[
        Tag::Prose,
        Tag::Json,
        Tag::Yaml,
        Tag::SourceCode,
        Tag::Log,
        Tag::Sql,
        Tag::Config,
        Tag::AiPrompt,
        Tag::Adversarial,
        Tag::Negative,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Tag::Prose => "prose",
            Tag::Json => "json",
            Tag::Yaml => "yaml",
            Tag::SourceCode => "source_code",
            Tag::Log => "log",
            Tag::Sql => "sql",
            Tag::Config => "config",
            Tag::AiPrompt => "ai_prompt",
            Tag::Adversarial => "adversarial",
            Tag::Negative => "negative",
        }
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A ground-truth entity annotation. Offsets are UTF-8 bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldEntity {
    #[serde(rename = "type")]
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
}

impl GoldEntity {
    pub fn span(&self) -> TypedSpan {
        TypedSpan::new(self.entity_type, self.start, self.end)
    }
}

/// One benchmark example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub text: String,
    #[serde(default)]
    pub entities: Vec<GoldEntity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<Tag>,
}

impl Example {
    pub fn has_tag(&self, tag: Tag) -> bool {
        self.tags.contains(&tag)
    }

    pub fn is_negative(&self) -> bool {
        self.entities.is_empty()
    }

    pub fn spans(&self) -> Vec<TypedSpan> {
        self.entities.iter().map(GoldEntity::span).collect()
    }
}

/// A loaded, validated dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub role: DatasetRole,
    pub examples: Vec<Example>,
    /// SHA-256 of the raw file bytes, recorded in run metadata so a result can
    /// be tied to exactly the data that produced it.
    pub sha256: String,
}

impl Dataset {
    /// Load from a path. Role is inferred from the filename (`*-dev.jsonl` ->
    /// `Dev`) unless overridden, because getting this wrong silently is exactly
    /// the failure mode we are guarding against.
    pub fn load(path: impl AsRef<Path>, role: Option<DatasetRole>) -> Result<Self, DatasetError> {
        let path = path.as_ref();
        let raw = std::fs::read(path).map_err(|e| DatasetError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let text = String::from_utf8(raw.clone()).map_err(|_| DatasetError::NotUtf8 {
            path: path.to_path_buf(),
        })?;

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dataset")
            .to_owned();
        let role = role.unwrap_or(if stem.contains("dev") || stem.contains("calib") {
            DatasetRole::Dev
        } else {
            DatasetRole::Test
        });

        let examples = load_jsonl_str(&text)?;
        Ok(Self {
            name: stem,
            role,
            examples,
            sha256: sha256_hex(&raw),
        })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Counts per canonical type across all gold annotations.
    pub fn entity_counts(&self) -> BTreeMap<EntityType, usize> {
        let mut m = BTreeMap::new();
        for ex in &self.examples {
            for e in &ex.entities {
                *m.entry(e.entity_type).or_insert(0) += 1;
            }
        }
        m
    }

    pub fn tag_counts(&self) -> BTreeMap<Tag, usize> {
        let mut m = BTreeMap::new();
        for ex in &self.examples {
            for &t in &ex.tags {
                *m.entry(t).or_insert(0) += 1;
            }
        }
        m
    }

    pub fn negative_count(&self) -> usize {
        self.examples.iter().filter(|e| e.is_negative()).count()
    }

    pub fn total_entities(&self) -> usize {
        self.examples.iter().map(|e| e.entities.len()).sum()
    }

    /// Deterministic subset by stride, for smoke runs. Seedless on purpose:
    /// a stride keeps the tag/type mix roughly intact without needing an RNG.
    pub fn take_stride(&self, limit: usize) -> Dataset {
        if limit == 0 || limit >= self.examples.len() {
            return self.clone();
        }
        let stride = self.examples.len().div_ceil(limit);
        let examples: Vec<_> = self
            .examples
            .iter()
            .step_by(stride)
            .take(limit)
            .cloned()
            .collect();
        Dataset {
            name: format!("{}#stride{}", self.name, limit),
            role: self.role,
            examples,
            sha256: self.sha256.clone(),
        }
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_is_inferred_from_filename_but_overridable() {
        let dir = tempfile::tempdir().unwrap();
        let dev = dir.path().join("dlp-dev.jsonl");
        let test = dir.path().join("dlp-v1.jsonl");
        let line = r#"{"id":"a","text":"hi","entities":[]}"#;
        std::fs::write(&dev, line).unwrap();
        std::fs::write(&test, line).unwrap();

        assert_eq!(Dataset::load(&dev, None).unwrap().role, DatasetRole::Dev);
        assert_eq!(Dataset::load(&test, None).unwrap().role, DatasetRole::Test);
        assert_eq!(
            Dataset::load(&dev, Some(DatasetRole::Test)).unwrap().role,
            DatasetRole::Test
        );
    }

    #[test]
    fn sha256_is_stable_and_content_dependent() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_ne!(sha256_hex(b"a"), sha256_hex(b"b"));
    }

    #[test]
    fn stride_subset_is_deterministic_and_bounded() {
        let examples: Vec<_> = (0..10)
            .map(|i| Example {
                id: format!("e{i}"),
                source: None,
                language: None,
                text: "x".into(),
                entities: vec![],
                tags: vec![],
            })
            .collect();
        let d = Dataset {
            name: "t".into(),
            role: DatasetRole::Test,
            examples,
            sha256: String::new(),
        };
        let a = d.take_stride(3);
        let b = d.take_stride(3);
        assert_eq!(a.examples.len(), 3);
        assert_eq!(
            a.examples.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            b.examples.iter().map(|e| e.id.as_str()).collect::<Vec<_>>()
        );
        assert_eq!(d.take_stride(0).len(), 10);
        assert_eq!(d.take_stride(99).len(), 10);
    }
}
