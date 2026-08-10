//! Quality evaluation.
//!
//! Depends only on [`crate::detector::Detector`] and [`crate::dataset`]. There
//! is deliberately no model-specific code here — if this module ever needs to
//! know which detector it is running, the abstraction has failed.

pub mod errors;
pub mod matching;
pub mod metrics;
pub mod sweep;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::dataset::{Dataset, Example, Tag};
use crate::detector::Detector;
use crate::entity::{Detection, EntityType};
use crate::error::DetectorError;
use crate::span::TypedSpan;

pub use errors::{ErrorCollector, ErrorKind, ErrorReport};
pub use matching::{Alignment, MatchMode, align};
pub use metrics::{Averages, CharCoverage, Counter, CoverageCounter, Prf, QualityMetrics};

/// The high-risk entity group whose recall is reported separately.
#[derive(Debug, Clone)]
pub struct CriticalSet(BTreeSet<EntityType>);

impl Default for CriticalSet {
    fn default() -> Self {
        CriticalSet(EntityType::DEFAULT_CRITICAL.iter().copied().collect())
    }
}

impl CriticalSet {
    pub fn new(types: impl IntoIterator<Item = EntityType>) -> Self {
        CriticalSet(types.into_iter().collect())
    }

    pub fn types(&self) -> Vec<EntityType> {
        self.0.iter().copied().collect()
    }

    fn inner(&self) -> &BTreeSet<EntityType> {
        &self.0
    }
}

/// Evaluation knobs.
#[derive(Debug, Clone, Default)]
pub struct EvalConfig {
    pub critical: CriticalSet,
    /// Cap on retained error samples. 0 disables sampling but still counts.
    pub error_limit: usize,
    /// Restrict scoring to types the detector's schema can actually produce.
    ///
    /// Off by default. When on, gold entities outside the schema are excluded
    /// rather than counted as false negatives — useful for asking "how good is
    /// this model at what we asked it for", misleading for "is this model
    /// enough for our policy". The report always records which was used.
    pub restrict_to_supported: Option<BTreeSet<EntityType>>,
}

/// Per-slice metrics, so a detector that is fine on prose and useless on YAML
/// is visible rather than averaged away.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SliceMetrics {
    pub documents: u64,
    pub exact_micro: Prf,
    pub overlap_micro: Prf,
    pub sensitive_char_recall: f64,
    pub false_positives: u64,
}

/// Complete quality result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evaluation {
    pub detector: String,
    pub dataset: String,
    pub dataset_role: crate::dataset::DatasetRole,
    pub dataset_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    pub quality: QualityMetrics,
    /// Metrics per content tag (`json`, `source_code`, `log`, ...).
    pub by_tag: BTreeMap<String, SliceMetrics>,
    /// Metrics per language.
    ///
    /// A DLP agent that only works in English is not deployable in most
    /// organisations, and an aggregate score hides that completely: a corpus
    /// that is 85% English scores well even when a detector is useless on the
    /// rest. Documents with no `language` field are grouped under `unknown`.
    pub by_language: BTreeMap<String, SliceMetrics>,
    /// Types excluded from scoring, when `restrict_to_supported` was used.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_types: Vec<EntityType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errors: Option<ErrorReport>,
    /// Wall time spent inside `Detector::detect`, excluding dataset parsing and
    /// metric computation. Indicative only — use `perf` for real latency.
    pub detect_wall_ms: f64,
}

/// Run a detector over a dataset and compute every quality metric.
pub fn evaluate(
    detector: &dyn Detector,
    dataset: &Dataset,
    cfg: &EvalConfig,
) -> Result<Evaluation, DetectorError> {
    let mut exact = Counter::default();
    let mut overlap = Counter::default();
    let mut coverage = CoverageCounter::default();
    let mut collector = ErrorCollector::new(cfg.error_limit);

    let mut by_tag: BTreeMap<Tag, TagAccumulator> = BTreeMap::new();
    let mut by_lang: BTreeMap<String, TagAccumulator> = BTreeMap::new();
    let mut gold_total = 0u64;
    let mut pred_total = 0u64;
    let mut negative_docs = 0u64;
    let mut clean_negative_docs = 0u64;
    let mut detect_ns = 0u128;

    for ex in &dataset.examples {
        let t0 = std::time::Instant::now();
        let predictions = detector.detect(&ex.text)?;
        detect_ns += t0.elapsed().as_nanos();

        let (gold, pred) = filter_to_scope(ex, &predictions, cfg);

        gold_total += gold.len() as u64;
        pred_total += pred.len() as u64;
        if ex.is_negative() {
            negative_docs += 1;
            if pred.is_empty() {
                clean_negative_docs += 1;
            }
        }

        let gold_spans: Vec<TypedSpan> = gold.iter().map(|g| g.0).collect();
        let pred_spans: Vec<TypedSpan> = pred.iter().map(|p| p.0).collect();

        let ex_align = align(&gold_spans, &pred_spans, MatchMode::Exact);
        let ov_align = align(&gold_spans, &pred_spans, MatchMode::Overlap);

        apply(&mut exact, &gold_spans, &pred_spans, &ex_align);
        apply(&mut overlap, &gold_spans, &pred_spans, &ov_align);

        coverage.add_document(
            &gold_spans
                .iter()
                .map(|s| (s.start, s.end))
                .collect::<Vec<_>>(),
            &pred_spans
                .iter()
                .map(|s| (s.start, s.end))
                .collect::<Vec<_>>(),
        );

        // Errors come from the overlap alignment: an exact-boundary miss that
        // still overlaps is a boundary problem, not a detection failure, and
        // listing it as a false negative would drown the real misses.
        if collector.is_enabled() {
            record_errors(&mut collector, ex, &gold_spans, &pred, &ov_align);
        }

        {
            // Language slice. Grouped under `unknown` when the dataset does
            // not say, so a corpus without language metadata degrades to one
            // bucket rather than silently reporting nothing.
            let lang = ex.language.clone().unwrap_or_else(|| "unknown".to_owned());
            by_lang
                .entry(lang)
                .or_default()
                .add(&ex_align, &ov_align, &gold_spans, &pred_spans);
        }

        let tags: &[Tag] = if ex.tags.is_empty() {
            &[Tag::Prose]
        } else {
            &ex.tags
        };
        for &tag in tags {
            by_tag
                .entry(tag)
                .or_default()
                .add(&ex_align, &ov_align, &gold_spans, &pred_spans);
        }
    }

    let documents = dataset.len() as u64;
    let fp_per_1k = if documents == 0 {
        0.0
    } else {
        overlap.total_fp() as f64 * 1000.0 / documents as f64
    };

    let quality = QualityMetrics {
        exact: Averages::from_counter(&exact),
        overlap: Averages::from_counter(&overlap),
        critical_recall: overlap.recall_over(cfg.critical.inner()),
        critical_types: cfg.critical.types(),
        character_coverage: coverage.finish(),
        false_positives_per_1k_docs: fp_per_1k,
        clean_negative_docs,
        negative_docs,
        documents,
        gold_entities: gold_total,
        predicted_entities: pred_total,
    };

    let excluded_types = match &cfg.restrict_to_supported {
        Some(supported) => {
            let mut v: Vec<_> = dataset
                .entity_counts()
                .keys()
                .copied()
                .filter(|t| !supported.contains(t))
                .collect();
            v.sort_unstable();
            v
        }
        None => Vec::new(),
    };

    Ok(Evaluation {
        detector: detector.name().to_owned(),
        dataset: dataset.name.clone(),
        dataset_role: dataset.role,
        dataset_sha256: dataset.sha256.clone(),
        threshold: detector.threshold(),
        quality,
        by_tag: by_tag
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.finish()))
            .collect(),
        by_language: by_lang.into_iter().map(|(k, v)| (k, v.finish())).collect(),
        excluded_types,
        errors: collector.is_enabled().then(|| collector.finish()),
        detect_wall_ms: detect_ns as f64 / 1e6,
    })
}

/// Reduce gold and predictions to the types in scope.
type Scoped<'a> = (
    Vec<(TypedSpan, EntityType)>,
    Vec<(TypedSpan, &'a Detection)>,
);

fn filter_to_scope<'a>(ex: &Example, predictions: &'a [Detection], cfg: &EvalConfig) -> Scoped<'a> {
    let keep = |t: EntityType| {
        cfg.restrict_to_supported
            .as_ref()
            .is_none_or(|s| s.contains(&t))
    };

    let gold = ex
        .entities
        .iter()
        .filter(|e| keep(e.entity_type))
        .map(|e| (e.span(), e.entity_type))
        .collect();

    // Predictions of an out-of-scope type are dropped too — otherwise
    // restricting the scope would turn every such prediction into a false
    // positive, which is not what "restrict to what we asked for" means.
    let pred = predictions
        .iter()
        .filter(|d| keep(d.entity_type))
        .map(|d| (TypedSpan::new(d.entity_type, d.start, d.end), d))
        .collect();

    (gold, pred)
}

fn apply(c: &mut Counter, gold: &[TypedSpan], pred: &[TypedSpan], a: &Alignment) {
    for &(gi, _) in &a.matched {
        c.add_tp(gold[gi].entity_type);
    }
    for &gi in &a.unmatched_gold {
        c.add_fn(gold[gi].entity_type);
    }
    for &pi in &a.unmatched_pred {
        c.add_fp(pred[pi].entity_type);
    }
}

fn record_errors(
    collector: &mut ErrorCollector,
    ex: &Example,
    gold: &[TypedSpan],
    pred: &[(TypedSpan, &Detection)],
    a: &Alignment,
) {
    let tags: Vec<String> = ex.tags.iter().map(ToString::to_string).collect();

    for &gi in &a.unmatched_gold {
        let g = gold[gi];
        // Anything the detector *did* say in this region: turns "missed it"
        // into "found it, called it a PERSON".
        let competing: Vec<_> = pred
            .iter()
            .filter(|(s, _)| s.overlap_len(g) > 0)
            .map(|(s, d)| errors::Competing {
                entity_type: s.entity_type,
                start: s.start,
                end: s.end,
                score: d.score,
            })
            .collect();
        collector.record(
            ErrorKind::FalseNegative,
            &ex.id,
            ex.source.as_deref(),
            &tags,
            &ex.text,
            g.entity_type,
            g.start,
            g.end,
            None,
            competing,
        );
    }

    for &pi in &a.unmatched_pred {
        let (s, d) = pred[pi];
        let competing: Vec<_> = gold
            .iter()
            .filter(|g| g.overlap_len(s) > 0)
            .map(|g| errors::Competing {
                entity_type: g.entity_type,
                start: g.start,
                end: g.end,
                score: 1.0,
            })
            .collect();
        collector.record(
            ErrorKind::FalsePositive,
            &ex.id,
            ex.source.as_deref(),
            &tags,
            &ex.text,
            s.entity_type,
            s.start,
            s.end,
            Some(d.score),
            competing,
        );
    }
}

#[derive(Debug, Default)]
struct TagAccumulator {
    documents: u64,
    exact_tp: u64,
    exact_fp: u64,
    exact_fn: u64,
    ov_tp: u64,
    ov_fp: u64,
    ov_fn: u64,
    coverage: CoverageCounter,
}

impl TagAccumulator {
    /// Fold one document into the slice.
    ///
    /// Shared by the tag and language slices so the two cannot drift apart —
    /// they answer different questions from identical arithmetic.
    fn add(
        &mut self,
        exact: &Alignment,
        overlap: &Alignment,
        gold: &[TypedSpan],
        pred: &[TypedSpan],
    ) {
        self.documents += 1;
        self.exact_tp += exact.tp() as u64;
        self.exact_fp += exact.fp() as u64;
        self.exact_fn += exact.fn_() as u64;
        self.ov_tp += overlap.tp() as u64;
        self.ov_fp += overlap.fp() as u64;
        self.ov_fn += overlap.fn_() as u64;
        self.coverage.add_document(
            &gold.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            &pred.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
        );
    }

    fn finish(self) -> SliceMetrics {
        SliceMetrics {
            documents: self.documents,
            exact_micro: Prf::from_counts(self.exact_tp, self.exact_fp, self.exact_fn),
            overlap_micro: Prf::from_counts(self.ov_tp, self.ov_fp, self.ov_fn),
            sensitive_char_recall: self.coverage.finish().sensitive_char_recall,
            false_positives: self.ov_fp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetRole, GoldEntity};
    use crate::entity::{DetectionSource, EntityType::*};

    /// `(entity type, start, end, score)` — one scripted detection.
    type ScriptedSpan = (EntityType, usize, usize, f32);
    /// `(document text, detections to return for it)`.
    type Script = (&'static str, Vec<ScriptedSpan>);

    /// A detector driven by a fixed script, so the evaluator can be tested
    /// exhaustively without any model.
    struct Scripted {
        by_text: BTreeMap<String, Vec<Detection>>,
    }

    impl Scripted {
        fn new(pairs: Vec<Script>) -> Self {
            Self {
                by_text: pairs
                    .into_iter()
                    .map(|(t, ds)| {
                        (
                            t.to_owned(),
                            ds.into_iter()
                                .map(|(ty, s, e, sc)| {
                                    Detection::new(ty, s, e, sc, DetectionSource::model("scripted"))
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            }
        }
    }

    impl Detector for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
            crate::detector::finish(text, self.by_text.get(text).cloned().unwrap_or_default())
        }
    }

    fn ds(examples: Vec<Example>) -> Dataset {
        Dataset {
            name: "t".into(),
            role: DatasetRole::Test,
            examples,
            sha256: "0".into(),
        }
    }

    fn ex(id: &str, text: &str, ents: Vec<(EntityType, usize, usize)>, tags: Vec<Tag>) -> Example {
        Example {
            id: id.into(),
            source: None,
            language: None,
            text: text.into(),
            entities: ents
                .into_iter()
                .map(|(entity_type, start, end)| GoldEntity {
                    entity_type,
                    start,
                    end,
                })
                .collect(),
            tags,
        }
    }

    #[test]
    fn perfect_detector_scores_one_everywhere() {
        let text = "mail john@example.com";
        let d = Scripted::new(vec![(text, vec![(Email, 5, 21, 0.9)])]);
        let data = ds(vec![ex("a", text, vec![(Email, 5, 21)], vec![Tag::Prose])]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();

        assert_eq!(r.quality.exact.micro.f1, 1.0);
        assert_eq!(r.quality.overlap.micro.f1, 1.0);
        assert_eq!(r.quality.character_coverage.sensitive_char_recall, 1.0);
        assert_eq!(r.quality.false_positives_per_1k_docs, 0.0);
        assert_eq!(
            (r.quality.gold_entities, r.quality.predicted_entities),
            (1, 1)
        );
    }

    #[test]
    fn boundary_miss_is_an_exact_failure_but_partial_protection() {
        // The metric that would be invisible if only F1 were reported.
        let text = "john.smith@example.com";
        let d = Scripted::new(vec![(text, vec![(Email, 5, 22, 0.8)])]);
        let data = ds(vec![ex("a", text, vec![(Email, 0, 22)], vec![])]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();

        assert_eq!(r.quality.exact.micro.f1, 0.0, "boundaries differ");
        assert_eq!(r.quality.overlap.micro.f1, 1.0, "entity was noticed");
        let cov = r.quality.character_coverage;
        assert!((cov.sensitive_char_recall - 17.0 / 22.0).abs() < 1e-12);
        assert!(cov.exposed_fraction > 0.0, "5 bytes left exposed");
    }

    #[test]
    fn negative_documents_drive_the_false_positive_rate() {
        let clean = "Fix issue #4411.";
        let noisy = "Version 1.23.4567 shipped";
        let d = Scripted::new(vec![
            (clean, vec![]),
            (noisy, vec![(Phone, 8, 17, 0.4)]), // a version string read as a phone
        ]);
        let data = ds(vec![
            ex("a", clean, vec![], vec![Tag::Negative]),
            ex("b", noisy, vec![], vec![Tag::Negative]),
        ]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();

        assert_eq!(r.quality.negative_docs, 2);
        assert_eq!(r.quality.clean_negative_docs, 1);
        assert_eq!(r.quality.false_positives_per_1k_docs, 500.0);
        assert_eq!(r.quality.overlap.micro.precision, 0.0);
        assert_eq!(r.quality.overlap.micro.recall, 1.0, "nothing to miss");
        // Over-masking is visible even though there is no gold span.
        assert_eq!(r.quality.character_coverage.over_masked_bytes, 9);
    }

    #[test]
    fn critical_recall_is_reported_separately_from_the_easy_types() {
        let t1 = "mail a@b.co";
        let t2 = "passport X1234567";
        let d = Scripted::new(vec![
            (t1, vec![(Email, 5, 11, 0.9)]),
            (t2, vec![]), // misses the passport
        ]);
        let data = ds(vec![
            ex("a", t1, vec![(Email, 5, 11)], vec![]),
            ex("b", t2, vec![(Passport, 9, 17)], vec![]),
        ]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();

        assert_eq!(r.quality.overlap.micro.recall, 0.5);
        assert_eq!(
            r.quality.critical_recall.recall, 0.0,
            "the passport was missed"
        );
        assert!(r.quality.critical_types.contains(&Passport));
    }

    #[test]
    fn per_tag_slices_expose_format_specific_failure() {
        let prose = "contact a@b.co";
        let yaml = "email: a@b.co";
        let d = Scripted::new(vec![
            (prose, vec![(Email, 8, 14, 0.9)]),
            (yaml, vec![]), // fails on YAML
        ]);
        let data = ds(vec![
            ex("a", prose, vec![(Email, 8, 14)], vec![Tag::Prose]),
            ex("b", yaml, vec![(Email, 7, 13)], vec![Tag::Yaml]),
        ]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();

        assert_eq!(r.by_tag["prose"].overlap_micro.recall, 1.0);
        assert_eq!(r.by_tag["yaml"].overlap_micro.recall, 0.0);
        assert_eq!(r.by_tag["yaml"].documents, 1);
    }

    /// A detector that works in English and nowhere else must be visible as
    /// such — this is the slice that catches an English-only tokenizer.
    #[test]
    fn per_language_slices_expose_an_english_only_detector() {
        let en = "contact a@b.co";
        let ru = "напишите на a@b.co";
        let d = Scripted::new(vec![
            (en, vec![(Email, 8, 14, 0.9)]),
            (ru, vec![]), // finds nothing outside English
        ]);
        let mut ex_en = ex("a", en, vec![(Email, 8, 14)], vec![]);
        ex_en.language = Some("en".into());
        let mut ex_ru = ex("b", ru, vec![(Email, 22, 28)], vec![]);
        ex_ru.language = Some("ru".into());
        let data = ds(vec![ex_en, ex_ru]);

        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();
        assert_eq!(r.by_language["en"].overlap_micro.recall, 1.0);
        assert_eq!(r.by_language["ru"].overlap_micro.recall, 0.0);
        assert_eq!(r.by_language["ru"].documents, 1);
        // ...while the aggregate looks merely mediocre.
        assert_eq!(r.quality.overlap.micro.recall, 0.5);
    }

    #[test]
    fn documents_without_a_language_are_grouped_under_unknown() {
        let t = "a@b.co";
        let d = Scripted::new(vec![(t, vec![(Email, 0, 6, 0.9)])]);
        let data = ds(vec![ex("a", t, vec![(Email, 0, 6)], vec![])]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();
        assert_eq!(r.by_language["unknown"].documents, 1);
    }

    #[test]
    fn error_samples_capture_both_directions_with_context() {
        let text = "call John Smith at 555";
        let d = Scripted::new(vec![(text, vec![(Phone, 19, 22, 0.3)])]);
        let data = ds(vec![ex("a", text, vec![(Person, 5, 15)], vec![])]);
        let cfg = EvalConfig {
            error_limit: 10,
            ..Default::default()
        };
        let r = evaluate(&d, &data, &cfg).unwrap();

        let e = r.errors.unwrap();
        assert_eq!(e.false_negatives_total, 1);
        assert_eq!(e.false_positives_total, 1);
        assert_eq!(e.false_negatives[0].text, "John Smith");
        assert!(e.false_negatives[0].context.contains("«John Smith»"));
        assert_eq!(e.false_positives[0].score, Some(0.3));
    }

    #[test]
    fn restricting_scope_excludes_gold_and_predictions_alike() {
        let text = "passport X1234567 mail a@b.co";
        let d = Scripted::new(vec![(
            text,
            vec![(Email, 23, 29, 0.9), (Passport, 9, 17, 0.8)],
        )]);
        let data = ds(vec![ex(
            "a",
            text,
            vec![(Passport, 9, 17), (Email, 23, 29)],
            vec![],
        )]);

        let unrestricted = evaluate(&d, &data, &EvalConfig::default()).unwrap();
        assert_eq!(unrestricted.quality.gold_entities, 2);

        let cfg = EvalConfig {
            restrict_to_supported: Some([Email].into_iter().collect()),
            ..Default::default()
        };
        let r = evaluate(&d, &data, &cfg).unwrap();
        assert_eq!(r.quality.gold_entities, 1, "passport gold excluded");
        assert_eq!(
            r.quality.predicted_entities, 1,
            "passport prediction excluded too"
        );
        assert_eq!(r.quality.overlap.micro.f1, 1.0);
        assert_eq!(
            r.excluded_types,
            vec![Passport],
            "exclusion is recorded, not hidden"
        );
    }

    #[test]
    fn empty_predictions_and_all_negative_dataset() {
        let text = "127.0.0.1:8080";
        let d = Scripted::new(vec![(text, vec![])]);
        let data = ds(vec![ex("a", text, vec![], vec![Tag::Negative])]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();

        assert_eq!(r.quality.exact.micro.f1, 1.0, "correct silence is perfect");
        assert_eq!(r.quality.false_positives_per_1k_docs, 0.0);
        assert_eq!(r.quality.character_coverage.sensitive_char_recall, 1.0);
        assert_eq!(r.quality.clean_negative_docs, 1);
    }

    #[test]
    fn unicode_documents_score_correctly() {
        let text = "Свяжитесь с Михаил Ястребцов";
        let start = text.find("Михаил").unwrap();
        let d = Scripted::new(vec![(text, vec![(Person, start, text.len(), 0.9)])]);
        let data = ds(vec![ex(
            "a",
            text,
            vec![(Person, start, text.len())],
            vec![],
        )]);
        let r = evaluate(&d, &data, &EvalConfig::default()).unwrap();
        assert_eq!(r.quality.exact.micro.f1, 1.0);
        assert_eq!(
            r.quality.character_coverage.sensitive_bytes,
            (text.len() - start) as u64
        );
    }

    #[test]
    fn evaluation_is_deterministic() {
        let text = "a@b.co and c@d.co";
        let d = Scripted::new(vec![(
            text,
            vec![
                (Email, 0, 6, 0.9),
                (Email, 11, 17, 0.8),
                (Person, 0, 6, 0.2),
            ],
        )]);
        let data = ds(vec![ex(
            "a",
            text,
            vec![(Email, 0, 6), (Email, 11, 17)],
            vec![],
        )]);
        let first =
            serde_json::to_string(&evaluate(&d, &data, &EvalConfig::default()).unwrap()).unwrap();
        for _ in 0..8 {
            let again =
                serde_json::to_string(&evaluate(&d, &data, &EvalConfig::default()).unwrap())
                    .unwrap();
            // detect_wall_ms varies; compare everything else.
            assert_eq!(strip_timing(&first), strip_timing(&again));
        }
    }

    fn strip_timing(s: &str) -> String {
        let mut v: serde_json::Value = serde_json::from_str(s).unwrap();
        v.as_object_mut().unwrap().remove("detect_wall_ms");
        v.to_string()
    }
}
