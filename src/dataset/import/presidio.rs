//! Importer for `presidio-research`'s `synth_dataset_v2`.
//!
//! An independent, third-party, MIT-licensed corpus: 1 500 documents and 2 863
//! annotated spans, roughly thirty times our hand-written fixtures. It exists
//! here because several canonical types have a support of 1 in the local
//! fixtures, which makes their per-entity numbers meaningless.
//!
//! # The conversion that matters
//!
//! **Presidio offsets are Unicode character indices; ours are UTF-8 bytes.**
//! Verified against the corpus rather than assumed: interpreting them as bytes
//! mis-slices **581 of 2 863 spans**, because 320 spans sit after non-ASCII
//! text (the templates are English but the generated values are international
//! — `Krisztián Szöllösy`, `Koskikatu`). Every span is converted through
//! [`crate::span::char_to_byte_table`] and then *re-verified* against the
//! `entity_value` the corpus supplies, so a conversion error is a hard failure
//! rather than a quietly shifted annotation.
//!
//! # What this corpus is not
//!
//! Template-generated English prose. No developer secrets (no API key, token,
//! private key or connection string), no code, config, JSON, YAML or logs, and
//! no language metadata. It is a strong independent check on
//! `PERSON`/`ADDRESS`/`CREDIT_CARD`/`PHONE`/`EMAIL`/`IBAN`/`SSN`/`IP`, and it
//! answers nothing about the AI-DLP or multilingual questions. Report it
//! separately.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::dataset::{Example, GoldEntity, Tag};
use crate::labels::{PRESIDIO, PRESIDIO_UNMAPPED};

/// One record as the corpus stores it.
#[derive(Debug, Deserialize)]
struct PresidioRecord {
    full_text: String,
    #[serde(default)]
    spans: Vec<PresidioSpan>,
}

#[derive(Debug, Deserialize)]
struct PresidioSpan {
    entity_type: String,
    entity_value: String,
    /// **Character** index, not byte.
    start_position: usize,
    /// **Character** index, not byte.
    end_position: usize,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ImportError {
    #[error("reading {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("parsing presidio json: {0}")]
    Json(#[from] serde_json::Error),

    #[error(
        "record {index}: span {entity_type} [{start}..{end}) does not resolve to \
         {expected:?} (got {actual:?}) — the offset convention is not what this \
         importer assumes"
    )]
    OffsetMismatch {
        index: usize,
        entity_type: String,
        start: usize,
        end: usize,
        expected: String,
        actual: String,
    },

    #[error("input contained no usable records")]
    Empty,
}

/// What the conversion did, so the result is auditable rather than a black box.
#[derive(Debug, Default, Clone)]
pub struct ImportReport {
    pub records_read: usize,
    pub records_kept: usize,
    /// Dropped because every one of their entities was an unmapped type. They
    /// are *not* emitted as negatives — the text does contain PII, just of a
    /// kind we do not score, and counting a detection there as a false
    /// positive would be wrong.
    pub records_dropped_all_unmapped: usize,
    pub spans_read: usize,
    pub spans_kept: usize,
    /// Unmapped label -> occurrences.
    pub unmapped: BTreeMap<String, usize>,
    /// Dropped because they overlapped an already-accepted span. The corpus has
    /// none today; the guard exists because our evaluator's one-to-one matcher
    /// requires disjoint gold.
    pub spans_dropped_overlap: usize,
}

impl ImportReport {
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} of {} records kept ({} dropped: all entities unmapped)\n\
             {} of {} spans kept",
            self.records_kept,
            self.records_read,
            self.records_dropped_all_unmapped,
            self.spans_kept,
            self.spans_read
        );
        if self.spans_dropped_overlap > 0 {
            s.push_str(&format!(
                "\n{} spans dropped for overlapping an accepted span",
                self.spans_dropped_overlap
            ));
        }
        if !self.unmapped.is_empty() {
            s.push_str("\nunmapped types (not scored):");
            let reasons: BTreeMap<_, _> = PRESIDIO_UNMAPPED.iter().copied().collect();
            for (t, n) in &self.unmapped {
                s.push_str(&format!(
                    "\n  {t:<18} {n:>5}  {}",
                    reasons
                        .get(t.as_str())
                        .copied()
                        .unwrap_or("NO RECORDED REASON")
                ));
            }
        }
        s
    }
}

/// Convert the raw corpus into benchmark examples.
///
/// Ids are `presidio-NNNN` from the record's position in the source file, so
/// two imports of the same pinned file produce identical ids.
pub fn import(json: &str) -> Result<(Vec<Example>, ImportReport), ImportError> {
    import_with(json, true)
}

/// As [`import`], but with control over whether bare place names count.
///
/// `score_gpe` decides whether `GPE` — a standalone city or country, 411 of the
/// corpus's 2 863 spans — is scored as `ADDRESS`. It is **not** a cosmetic
/// switch, and measuring both settings is the honest thing to do: a model with
/// one broad `address` label (GLiNER2) and a model with granular
/// `city`/`state`/`country`/`postcode` labels (MiniLM) are affected very
/// differently by it, so the choice partly determines which of them "wins".
pub fn import_with(
    json: &str,
    score_gpe: bool,
) -> Result<(Vec<Example>, ImportReport), ImportError> {
    let records: Vec<PresidioRecord> = serde_json::from_str(json)?;
    let mut report = ImportReport {
        records_read: records.len(),
        ..Default::default()
    };
    let mut out = Vec::with_capacity(records.len());

    for (index, r) in records.iter().enumerate() {
        report.spans_read += r.spans.len();

        // One table per document; O(len) once instead of O(len) per span.
        let table = crate::span::char_to_byte_table(&r.full_text);

        let mut entities: Vec<GoldEntity> = Vec::with_capacity(r.spans.len());
        let mut had_any = false;

        for sp in &r.spans {
            let mapped = if sp.entity_type == "GPE" && !score_gpe {
                None
            } else {
                PRESIDIO.map(&sp.entity_type)
            };
            let Some(entity_type) = mapped else {
                *report.unmapped.entry(sp.entity_type.clone()).or_insert(0) += 1;
                had_any = true;
                continue;
            };

            let (Some(&start), Some(&end)) =
                (table.get(sp.start_position), table.get(sp.end_position))
            else {
                return Err(ImportError::OffsetMismatch {
                    index,
                    entity_type: sp.entity_type.clone(),
                    start: sp.start_position,
                    end: sp.end_position,
                    expected: sp.entity_value.clone(),
                    actual: "<out of range>".to_owned(),
                });
            };

            // Re-verify after conversion. The corpus ships the surface text,
            // so this is a free, total check that the char->byte mapping is
            // right for every single span rather than for the ones we happened
            // to eyeball.
            let actual = r.full_text.get(start..end).unwrap_or_default();
            if actual != sp.entity_value {
                return Err(ImportError::OffsetMismatch {
                    index,
                    entity_type: sp.entity_type.clone(),
                    start,
                    end,
                    expected: sp.entity_value.clone(),
                    actual: actual.to_owned(),
                });
            }
            if start >= end {
                continue;
            }
            entities.push(GoldEntity {
                entity_type,
                start,
                end,
            });
            had_any = true;
        }

        // The evaluator's one-to-one matcher assumes disjoint gold, and the
        // loader rejects overlaps outright. Keep the longer span of any pair.
        entities.sort_by_key(|e| (e.start, std::cmp::Reverse(e.end)));
        let mut disjoint: Vec<GoldEntity> = Vec::with_capacity(entities.len());
        for e in entities {
            match disjoint.last() {
                Some(prev) if e.start < prev.end => report.spans_dropped_overlap += 1,
                _ => disjoint.push(e),
            }
        }

        // A record whose entities were all unmapped is not a negative example:
        // it does contain PII, of a type we chose not to score. Emitting it as
        // a negative would punish any detector that notices.
        if disjoint.is_empty() && had_any {
            report.records_dropped_all_unmapped += 1;
            continue;
        }

        report.spans_kept += disjoint.len();
        report.records_kept += 1;
        out.push(Example {
            id: format!("presidio-{index:04}"),
            source: Some("presidio-research/synth_dataset_v2".to_owned()),
            // The corpus carries no language metadata. Templates are English;
            // only the generated values are international. Left unset rather
            // than asserted, so the per-language slice reports `unknown`
            // instead of a claim we cannot support.
            language: None,
            text: r.full_text.clone(),
            entities: disjoint,
            tags: vec![Tag::Prose],
        });
    }

    if out.is_empty() {
        return Err(ImportError::Empty);
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((out, report))
}

/// Take a deterministic, entity-stratified subset.
///
/// Stratifies by the rarest entity type each example contains, so a 200-example
/// subset still carries the `US_DRIVER_LICENSE` and `IP_ADDRESS` records rather
/// than 200 more `PERSON`s. Selection is a seeded shuffle within each stratum,
/// so the same `(limit, seed)` always yields the same subset.
pub fn subset(examples: &[Example], limit: usize, seed: u64) -> Vec<Example> {
    if limit == 0 || limit >= examples.len() {
        return examples.to_vec();
    }

    let mut freq: BTreeMap<crate::entity::EntityType, usize> = BTreeMap::new();
    for ex in examples {
        for e in &ex.entities {
            *freq.entry(e.entity_type).or_insert(0) += 1;
        }
    }

    // Bucket each example under the rarest type it contains.
    let mut strata: BTreeMap<Option<crate::entity::EntityType>, Vec<&Example>> = BTreeMap::new();
    for ex in examples {
        let key = ex
            .entities
            .iter()
            .map(|e| e.entity_type)
            .min_by_key(|t| freq.get(t).copied().unwrap_or(0));
        strata.entry(key).or_default().push(ex);
    }

    let mut rng = crate::dataset::transform::Rng::new(seed);
    let mut picked: Vec<Example> = Vec::with_capacity(limit);
    // Round-robin over strata so the rare ones are represented before the
    // budget is spent on the common ones.
    let mut queues: Vec<Vec<&Example>> = strata
        .into_values()
        .map(|mut v| {
            // Seeded Fisher-Yates.
            for i in (1..v.len()).rev() {
                v.swap(i, rng.below(i + 1));
            }
            v
        })
        .collect();
    queues.sort_by_key(Vec::len); // rarest strata first

    while picked.len() < limit && queues.iter().any(|q| !q.is_empty()) {
        for q in &mut queues {
            if picked.len() >= limit {
                break;
            }
            if let Some(ex) = q.pop() {
                picked.push(ex.clone());
            }
        }
    }

    picked.sort_by(|a, b| a.id.cmp(&b.id));
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityType;

    const RAW: &str = "datasets/raw/presidio-synth-v2.json";

    fn corpus() -> Option<String> {
        std::fs::read_to_string(RAW).ok()
    }

    #[test]
    fn char_offsets_are_converted_to_byte_offsets() {
        // The exact failure mode: an entity preceded by multibyte text. Under
        // byte indexing `start_position: 8` would land inside "Szöllösy".
        let json = r#"[{
            "full_text": "Krisztián Szöllösy wrote to a@b.co yesterday",
            "spans": [
                {"entity_type": "PERSON", "entity_value": "Krisztián Szöllösy",
                 "start_position": 0, "end_position": 18},
                {"entity_type": "EMAIL_ADDRESS", "entity_value": "a@b.co",
                 "start_position": 28, "end_position": 34}
            ]
        }]"#;
        let (ex, _) = import(json).unwrap();
        assert_eq!(ex.len(), 1);
        let t = &ex[0].text;
        let person = &ex[0].entities[0];
        let email = &ex[0].entities[1];
        assert_eq!(&t[person.start..person.end], "Krisztián Szöllösy");
        assert_eq!(&t[email.start..email.end], "a@b.co");

        // The name holds three 2-byte characters (á, ö, ö), so 18 characters
        // occupy 21 bytes. Every later offset is displaced by the same 3.
        assert_eq!("Krisztián Szöllösy".chars().count(), 18);
        assert_eq!("Krisztián Szöllösy".len(), 21);
        assert_eq!((person.start, person.end), (0, 21));
        assert_eq!((email.start, email.end), (28 + 3, 34 + 3));

        // The point of the test: byte offsets are NOT the char offsets.
        assert_ne!(person.end, 18, "char offsets leaked through unconverted");
    }

    #[test]
    fn a_wrong_offset_convention_is_a_hard_error_not_a_shifted_span() {
        // entity_value does not match what the offsets select.
        let json = r#"[{
            "full_text": "hello world",
            "spans": [{"entity_type": "PERSON", "entity_value": "world",
                       "start_position": 0, "end_position": 5}]
        }]"#;
        match import(json) {
            Err(ImportError::OffsetMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "world");
                assert_eq!(actual, "hello");
            }
            other => panic!("expected an offset mismatch, got {other:?}"),
        }
    }

    #[test]
    fn unmapped_types_are_counted_and_records_of_only_unmapped_types_are_dropped() {
        let json = r#"[
            {"full_text": "Acme Corp is a company",
             "spans": [{"entity_type": "ORGANIZATION", "entity_value": "Acme Corp",
                        "start_position": 0, "end_position": 9}]},
            {"full_text": "John Smith at Acme Corp",
             "spans": [{"entity_type": "PERSON", "entity_value": "John Smith",
                        "start_position": 0, "end_position": 10},
                       {"entity_type": "ORGANIZATION", "entity_value": "Acme Corp",
                        "start_position": 14, "end_position": 23}]}
        ]"#;
        let (ex, rep) = import(json).unwrap();
        // The organisation-only record must NOT become a negative example.
        assert_eq!(ex.len(), 1);
        assert_eq!(rep.records_dropped_all_unmapped, 1);
        assert_eq!(rep.unmapped.get("ORGANIZATION"), Some(&2));
        assert_eq!(ex[0].entities.len(), 1);
        assert_eq!(ex[0].entities[0].entity_type, EntityType::Person);
    }

    #[test]
    fn overlapping_gold_is_reduced_to_a_disjoint_set() {
        // Our evaluator's matcher requires disjoint gold; the loader rejects
        // overlaps. The longer span wins.
        let json = r#"[{
            "full_text": "12 Main Street Springfield",
            "spans": [
                {"entity_type": "STREET_ADDRESS", "entity_value": "12 Main Street Springfield",
                 "start_position": 0, "end_position": 26},
                {"entity_type": "GPE", "entity_value": "Springfield",
                 "start_position": 15, "end_position": 26}
            ]
        }]"#;
        let (ex, rep) = import(json).unwrap();
        assert_eq!(ex[0].entities.len(), 1);
        assert_eq!(rep.spans_dropped_overlap, 1);
        assert_eq!(
            &ex[0].text[ex[0].entities[0].start..ex[0].entities[0].end],
            "12 Main Street Springfield"
        );
    }

    #[test]
    fn import_is_deterministic() {
        let json = r#"[{"full_text": "John Smith", "spans":
            [{"entity_type":"PERSON","entity_value":"John Smith",
              "start_position":0,"end_position":10}]}]"#;
        let a = import(json).unwrap().0;
        let b = import(json).unwrap().0;
        assert_eq!(a, b);
        assert_eq!(a[0].id, "presidio-0000");
        assert_eq!(
            a[0].source.as_deref(),
            Some("presidio-research/synth_dataset_v2")
        );
        assert_eq!(
            a[0].language, None,
            "the corpus carries no language metadata"
        );
    }

    // --- the real corpus ---------------------------------------------------

    #[test]
    fn the_real_corpus_imports_with_every_span_verified() {
        let Some(json) = corpus() else {
            eprintln!("SKIP: run `bash scripts/fetch-datasets.sh`");
            return;
        };
        let (examples, rep) = import(&json).expect("import must succeed");
        eprintln!("{}", rep.summary());

        assert_eq!(rep.records_read, 1500);
        assert_eq!(rep.spans_read, 2863);
        assert!(rep.records_kept > 1200, "kept only {}", rep.records_kept);
        // Every span that survived was re-verified against `entity_value`
        // inside `import`, so reaching here at all is the assertion.
        assert!(rep.spans_kept > 1900, "kept only {} spans", rep.spans_kept);

        // Every unmapped type must have a recorded reason.
        let reasons: BTreeMap<_, _> = PRESIDIO_UNMAPPED.iter().copied().collect();
        for t in rep.unmapped.keys() {
            assert!(
                reasons.contains_key(t.as_str()),
                "{t} unmapped with no reason"
            );
        }

        // And the output must satisfy the loader's own rules.
        for ex in &examples {
            for e in &ex.entities {
                crate::span::validate(&ex.text, e.start, e.end, false)
                    .unwrap_or_else(|err| panic!("{}: {err}", ex.id));
            }
            let mut sorted = ex.entities.clone();
            sorted.sort_by_key(|e| (e.start, e.end));
            for w in sorted.windows(2) {
                assert!(
                    w[1].start >= w[0].end,
                    "{}: overlapping gold survived",
                    ex.id
                );
            }
        }
    }

    #[test]
    fn the_real_corpus_gives_meaningful_support_to_rare_types() {
        let Some(json) = corpus() else { return };
        let (examples, _) = import(&json).unwrap();
        let mut counts: BTreeMap<EntityType, usize> = BTreeMap::new();
        for ex in &examples {
            for e in &ex.entities {
                *counts.entry(e.entity_type).or_insert(0) += 1;
            }
        }
        eprintln!("{counts:?}");
        // The whole reason for importing it: the local fixtures have support 1
        // for several of these.
        assert!(counts[&EntityType::Person] > 500);
        assert!(counts[&EntityType::Address] > 500);
        assert!(counts[&EntityType::CreditCard] > 100);
        assert!(counts[&EntityType::Phone] > 50);
    }

    #[test]
    fn subsetting_is_deterministic_bounded_and_keeps_rare_types() {
        let Some(json) = corpus() else { return };
        let (examples, _) = import(&json).unwrap();

        let a = subset(&examples, 200, 42);
        let b = subset(&examples, 200, 42);
        assert_eq!(a.len(), 200);
        assert_eq!(
            a.iter().map(|e| &e.id).collect::<Vec<_>>(),
            b.iter().map(|e| &e.id).collect::<Vec<_>>(),
            "same (limit, seed) must give the same subset"
        );
        assert_ne!(
            a.iter().map(|e| &e.id).collect::<Vec<_>>(),
            subset(&examples, 200, 7)
                .iter()
                .map(|e| &e.id)
                .collect::<Vec<_>>(),
            "the seed must matter"
        );

        // Stratification: the rarest types survive a 10x reduction.
        let types: std::collections::BTreeSet<_> = a
            .iter()
            .flat_map(|e| e.entities.iter().map(|x| x.entity_type))
            .collect();
        let all: std::collections::BTreeSet<_> = examples
            .iter()
            .flat_map(|e| e.entities.iter().map(|x| x.entity_type))
            .collect();
        assert_eq!(
            types, all,
            "a stratified subset must retain every entity type"
        );

        assert_eq!(subset(&examples, 0, 42).len(), examples.len());
        assert_eq!(subset(&examples, 99_999, 42).len(), examples.len());
    }
}
