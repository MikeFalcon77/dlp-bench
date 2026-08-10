//! JSONL loading with strict validation.
//!
//! Every rejection carries the line number and the example id. A dataset that
//! loads but is subtly wrong (a span off by one byte, a duplicated id counted
//! twice) corrupts every number downstream, so the loader is deliberately
//! unforgiving and never repairs anything silently.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{Example, GoldEntity};
use crate::span::{self, SpanError};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DatasetError {
    #[error("io: {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not valid UTF-8")]
    NotUtf8 { path: PathBuf },

    #[error("line {line}: malformed JSON: {source}")]
    Json {
        line: usize,
        #[source]
        source: serde_json::Error,
    },

    #[error("line {line}: duplicate example id {id:?} (first seen on line {first_line})")]
    DuplicateId {
        line: usize,
        id: String,
        first_line: usize,
    },

    #[error("line {line}: example {id:?} has an empty id")]
    EmptyId { line: usize, id: String },

    #[error("line {line}: example {id:?} entity {index} ({entity_type}): {source}")]
    InvalidSpan {
        line: usize,
        id: String,
        index: usize,
        entity_type: String,
        #[source]
        source: SpanError,
    },

    #[error(
        "line {line}: example {id:?} has overlapping gold entities: \
         {a_type} [{a_start}..{a_end}) and {b_type} [{b_start}..{b_end})"
    )]
    OverlappingGold {
        line: usize,
        id: String,
        a_type: String,
        a_start: usize,
        a_end: usize,
        b_type: String,
        b_start: usize,
        b_end: usize,
    },

    #[error("dataset is empty")]
    Empty,
}

/// Load and validate examples from JSONL text.
///
/// Output is sorted by id so two runs over the same file iterate in the same
/// order regardless of how the file was assembled.
pub fn load_jsonl_str(text: &str) -> Result<Vec<Example>, DatasetError> {
    let mut examples = Vec::new();
    let mut seen: HashMap<String, usize> = HashMap::new();

    for (idx, raw) in text.lines().enumerate() {
        let line = idx + 1;
        let trimmed = raw.trim();
        // Blank lines and `#` comments are tolerated so fixtures can be
        // annotated; nothing else is.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let ex: Example =
            serde_json::from_str(trimmed).map_err(|source| DatasetError::Json { line, source })?;

        if ex.id.trim().is_empty() {
            return Err(DatasetError::EmptyId { line, id: ex.id });
        }
        if let Some(&first_line) = seen.get(&ex.id) {
            return Err(DatasetError::DuplicateId {
                line,
                id: ex.id,
                first_line,
            });
        }
        seen.insert(ex.id.clone(), line);

        validate_example(&ex, line)?;
        examples.push(ex);
    }

    if examples.is_empty() {
        return Err(DatasetError::Empty);
    }

    examples.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(examples)
}

pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Vec<Example>, DatasetError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    load_jsonl_str(&text)
}

fn validate_example(ex: &Example, line: usize) -> Result<(), DatasetError> {
    for (index, e) in ex.entities.iter().enumerate() {
        span::validate(&ex.text, e.start, e.end, false).map_err(|source| {
            DatasetError::InvalidSpan {
                line,
                id: ex.id.clone(),
                index,
                entity_type: e.entity_type.to_string(),
                source,
            }
        })?;
    }

    // Overlapping gold spans break the one-to-one matcher's assumptions and
    // make character-coverage double-count. Reject at load rather than produce
    // quietly wrong metrics.
    let mut sorted: Vec<&GoldEntity> = ex.entities.iter().collect();
    sorted.sort_by_key(|e| (e.start, e.end));
    for w in sorted.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b.start < a.end {
            return Err(DatasetError::OverlappingGold {
                line,
                id: ex.id.clone(),
                a_type: a.entity_type.to_string(),
                a_start: a.start,
                a_end: a.end,
                b_type: b.entity_type.to_string(),
                b_start: b.start,
                b_end: b.end,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Tag;
    use crate::entity::EntityType;

    const GOOD: &str = r#"
# a comment line is allowed
{"id":"example-001","source":"custom","language":"en","text":"Contact John Smith at john@example.com","entities":[{"type":"PERSON","start":8,"end":18},{"type":"EMAIL","start":22,"end":38}],"tags":["prose"]}

{"id":"example-002","text":"Fix issue #4411.","entities":[],"tags":["negative"]}
"#;

    #[test]
    fn loads_valid_jsonl_with_comments_and_blank_lines() {
        let v = load_jsonl_str(GOOD).unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].id, "example-001");
        assert_eq!(v[0].entities.len(), 2);
        assert_eq!(v[0].entities[0].entity_type, EntityType::Person);
        assert_eq!(v[0].tags, vec![Tag::Prose]);
        assert!(v[1].is_negative());

        // The spans in the fixture must actually slice to the right text —
        // this is the check that would have caught an off-by-one in the fixture.
        let ex = &v[0];
        assert_eq!(&ex.text[8..18], "John Smith");
        assert_eq!(&ex.text[22..38], "john@example.com");
    }

    #[test]
    fn output_is_sorted_by_id_regardless_of_file_order() {
        let s = concat!(
            r#"{"id":"z","text":"x","entities":[]}"#,
            "\n",
            r#"{"id":"a","text":"x","entities":[]}"#,
            "\n",
            r#"{"id":"m","text":"x","entities":[]}"#,
        );
        let v = load_jsonl_str(s).unwrap();
        assert_eq!(
            v.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "m", "z"]
        );
    }

    #[test]
    fn rejects_duplicate_ids_naming_both_lines() {
        let s = concat!(
            r#"{"id":"dup","text":"x","entities":[]}"#,
            "\n",
            r#"{"id":"dup","text":"y","entities":[]}"#,
        );
        match load_jsonl_str(s).unwrap_err() {
            DatasetError::DuplicateId {
                line,
                id,
                first_line,
            } => {
                assert_eq!((line, first_line), (2, 1));
                assert_eq!(id, "dup");
            }
            e => panic!("wrong error: {e}"),
        }
    }

    #[test]
    fn rejects_out_of_bounds_span() {
        let s = r#"{"id":"a","text":"short","entities":[{"type":"EMAIL","start":0,"end":99}]}"#;
        match load_jsonl_str(s).unwrap_err() {
            DatasetError::InvalidSpan {
                source,
                entity_type,
                ..
            } => {
                assert_eq!(entity_type, "EMAIL");
                assert!(matches!(source, SpanError::OutOfBounds { .. }));
            }
            e => panic!("wrong error: {e}"),
        }
    }

    #[test]
    fn rejects_span_splitting_a_multibyte_character() {
        // "Михаил" — every character is two bytes, so end=3 is interior.
        let s = r#"{"id":"a","text":"Михаил","entities":[{"type":"PERSON","start":0,"end":3}]}"#;
        match load_jsonl_str(s).unwrap_err() {
            DatasetError::InvalidSpan { source, .. } => {
                assert!(matches!(source, SpanError::EndNotCharBoundary { end: 3 }));
            }
            e => panic!("wrong error: {e}"),
        }
        // ...and accepts the correct byte span for the same text.
        let ok = r#"{"id":"a","text":"Михаил","entities":[{"type":"PERSON","start":0,"end":12}]}"#;
        let v = load_jsonl_str(ok).unwrap();
        assert_eq!(&v[0].text[0..12], "Михаил");
    }

    #[test]
    fn rejects_inverted_and_empty_spans() {
        let inv = r#"{"id":"a","text":"abcdef","entities":[{"type":"EMAIL","start":4,"end":2}]}"#;
        assert!(matches!(
            load_jsonl_str(inv).unwrap_err(),
            DatasetError::InvalidSpan {
                source: SpanError::Inverted { .. },
                ..
            }
        ));
        let empty = r#"{"id":"a","text":"abcdef","entities":[{"type":"EMAIL","start":2,"end":2}]}"#;
        assert!(matches!(
            load_jsonl_str(empty).unwrap_err(),
            DatasetError::InvalidSpan {
                source: SpanError::Empty { .. },
                ..
            }
        ));
    }

    #[test]
    fn rejects_overlapping_gold_entities() {
        let s = r#"{"id":"a","text":"john@example.com","entities":[{"type":"EMAIL","start":0,"end":16},{"type":"USERNAME","start":0,"end":4}]}"#;
        match load_jsonl_str(s).unwrap_err() {
            DatasetError::OverlappingGold { a_type, b_type, .. } => {
                assert_eq!((a_type.as_str(), b_type.as_str()), ("USERNAME", "EMAIL"));
            }
            e => panic!("wrong error: {e}"),
        }
        // Abutting spans are fine — they do not overlap.
        let ok = r#"{"id":"a","text":"abcdef","entities":[{"type":"EMAIL","start":0,"end":3},{"type":"URL","start":3,"end":6}]}"#;
        assert!(load_jsonl_str(ok).is_ok());
    }

    #[test]
    fn rejects_unknown_entity_label_without_coercing_it() {
        let s = r#"{"id":"a","text":"x","entities":[{"type":"SSN","start":0,"end":1}]}"#;
        let err = load_jsonl_str(s).unwrap_err();
        assert!(
            matches!(err, DatasetError::Json { line: 1, .. }),
            "got {err}"
        );
        assert!(err.to_string().contains("unknown variant"), "got {err}");
    }

    #[test]
    fn rejects_malformed_json_and_empty_id_and_empty_dataset() {
        assert!(matches!(
            load_jsonl_str("{not json}").unwrap_err(),
            DatasetError::Json { line: 1, .. }
        ));
        assert!(matches!(
            load_jsonl_str(r#"{"id":"  ","text":"x","entities":[]}"#).unwrap_err(),
            DatasetError::EmptyId { .. }
        ));
        assert!(matches!(
            load_jsonl_str("\n\n# only comments\n").unwrap_err(),
            DatasetError::Empty
        ));
    }

    #[test]
    fn round_trips_through_serde() {
        let v = load_jsonl_str(GOOD).unwrap();
        let line = serde_json::to_string(&v[0]).unwrap();
        let back: Example = serde_json::from_str(&line).unwrap();
        assert_eq!(back, v[0]);
    }
}
