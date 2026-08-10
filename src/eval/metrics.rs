//! Quality metrics.
//!
//! Several metrics rather than one number, because they answer different
//! questions and a DLP decision needs all of them:
//!
//! * **exact** entity F1 — can the detector name the entity *and* its exact
//!   boundaries? Relevant for structured redaction and for tokenisation.
//! * **overlap** entity F1 — did it notice the entity at all?
//! * **sensitive-character recall** — what fraction of sensitive *bytes* would
//!   actually be masked? A prediction that catches `smith@example.com` out of
//!   `john.smith@example.com` is an exact failure and a 77% redaction success,
//!   and both facts matter.
//! * **critical recall** — recall restricted to the high-risk group, so a
//!   catastrophic `PASSPORT` miss cannot hide behind an easy `EMAIL`.
//! * **macro vs micro** — micro is dominated by frequent classes; macro tells
//!   you whether the detector is uniformly competent.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::entity::EntityType;

/// Precision/recall/F1 with the counts they were computed from.
///
/// Counts are kept because a precision of 1.00 from a single prediction and one
/// from ten thousand are not the same claim.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Prf {
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl Prf {
    pub fn from_counts(tp: u64, fp: u64, fn_: u64) -> Self {
        // Convention: a detector that predicts nothing on a document with no
        // gold entities is perfect, not undefined. Reporting 0.0 there would
        // punish correct silence on negative examples.
        let precision = ratio(tp, tp + fp);
        let recall = ratio(tp, tp + fn_);
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            tp,
            fp,
            fn_,
            precision,
            recall,
            f1,
        }
    }

    pub fn support(&self) -> u64 {
        self.tp + self.fn_
    }
}

fn ratio(num: u64, den: u64) -> f64 {
    if den == 0 {
        1.0
    } else {
        num as f64 / den as f64
    }
}

/// Running TP/FP/FN accumulator, overall and per entity type.
#[derive(Debug, Clone, Default)]
pub struct Counter {
    tp: u64,
    fp: u64,
    fn_: u64,
    per_type: BTreeMap<EntityType, (u64, u64, u64)>,
}

impl Counter {
    pub fn add_tp(&mut self, t: EntityType) {
        self.tp += 1;
        self.per_type.entry(t).or_default().0 += 1;
    }

    pub fn add_fp(&mut self, t: EntityType) {
        self.fp += 1;
        self.per_type.entry(t).or_default().1 += 1;
    }

    pub fn add_fn(&mut self, t: EntityType) {
        self.fn_ += 1;
        self.per_type.entry(t).or_default().2 += 1;
    }

    /// Micro-averaged: every entity instance counts once, so frequent types
    /// dominate. This is the "what fraction of all PII did we catch" number.
    pub fn micro(&self) -> Prf {
        Prf::from_counts(self.tp, self.fp, self.fn_)
    }

    /// Macro-averaged: every type counts once, regardless of frequency.
    ///
    /// Only types with non-zero support (i.e. that actually appear in the gold
    /// data) are averaged; including a type the dataset never exercises would
    /// drag the mean toward an arbitrary constant.
    pub fn macro_(&self) -> Prf {
        let per = self.per_entity();
        let scored: Vec<&Prf> = per.values().filter(|p| p.support() > 0).collect();
        if scored.is_empty() {
            return Prf::from_counts(0, 0, 0);
        }
        let n = scored.len() as f64;
        let precision = scored.iter().map(|p| p.precision).sum::<f64>() / n;
        let recall = scored.iter().map(|p| p.recall).sum::<f64>() / n;
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Prf {
            tp: self.tp,
            fp: self.fp,
            fn_: self.fn_,
            precision,
            recall,
            f1,
        }
    }

    pub fn per_entity(&self) -> BTreeMap<EntityType, Prf> {
        self.per_type
            .iter()
            .map(|(&t, &(tp, fp, fn_))| (t, Prf::from_counts(tp, fp, fn_)))
            .collect()
    }

    /// Recall over a subset of types, aggregated as one pooled number.
    ///
    /// Pooled rather than macro-averaged so a type with two instances does not
    /// carry the same weight as one with two hundred; for the critical group we
    /// care about total leaked instances.
    pub fn recall_over(&self, types: &BTreeSet<EntityType>) -> Prf {
        let (mut tp, mut fp, mut fn_) = (0, 0, 0);
        for (t, &(a, b, c)) in &self.per_type {
            if types.contains(t) {
                tp += a;
                fp += b;
                fn_ += c;
            }
        }
        Prf::from_counts(tp, fp, fn_)
    }

    pub fn total_fp(&self) -> u64 {
        self.fp
    }
}

/// Byte-level coverage of gold sensitive regions.
///
/// This is the redaction question, and it is deliberately **type-agnostic**: a
/// prediction that covers a credit card but calls it a `BANK_ACCOUNT` still
/// gets the card masked. Reporting it alongside typed F1 keeps the two concerns
/// from being confused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CharCoverage {
    /// Total bytes inside gold spans.
    pub sensitive_bytes: u64,
    /// Gold bytes covered by at least one prediction of any type.
    pub covered_bytes: u64,
    /// Predicted bytes that fall outside every gold span. The redaction cost of
    /// over-masking: how much benign text a redactor would destroy.
    pub over_masked_bytes: u64,
    /// `covered / sensitive`. 1.0 when there is nothing sensitive.
    pub sensitive_char_recall: f64,
    /// `1 - sensitive_char_recall`, i.e. the fraction left exposed.
    pub exposed_fraction: f64,
}

impl CharCoverage {
    fn finish(mut self) -> Self {
        self.sensitive_char_recall = ratio(self.covered_bytes, self.sensitive_bytes);
        self.exposed_fraction = 1.0 - self.sensitive_char_recall;
        self
    }
}

/// Accumulates byte coverage across documents.
#[derive(Debug, Clone, Default)]
pub struct CoverageCounter {
    sensitive: u64,
    covered: u64,
    over: u64,
}

impl CoverageCounter {
    /// Add one document. Ranges are byte ranges into that document's text.
    pub fn add_document(&mut self, gold: &[(usize, usize)], pred: &[(usize, usize)]) {
        let gold_m = crate::span::merge_ranges(gold.to_vec());
        let pred_m = crate::span::merge_ranges(pred.to_vec());

        let sensitive: usize = gold_m.iter().map(|(s, e)| e - s).sum();
        let predicted: usize = pred_m.iter().map(|(s, e)| e - s).sum();

        let mut covered = 0usize;
        for &(gs, ge) in &gold_m {
            for &(ps, pe) in &pred_m {
                if pe <= gs {
                    continue;
                }
                if ps >= ge {
                    break; // merged ranges are sorted
                }
                covered += ge.min(pe) - gs.max(ps);
            }
        }

        self.sensitive += sensitive as u64;
        self.covered += covered as u64;
        self.over += (predicted - covered) as u64;
    }

    pub fn finish(&self) -> CharCoverage {
        CharCoverage {
            sensitive_bytes: self.sensitive,
            covered_bytes: self.covered,
            over_masked_bytes: self.over,
            ..Default::default()
        }
        .finish()
    }
}

/// The complete quality result for one detector on one dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub exact: Averages,
    pub overlap: Averages,
    /// Pooled recall over the configured critical group, overlap-matched.
    ///
    /// Overlap rather than exact because for a high-risk entity, noticing it at
    /// all is what determines whether the prompt gets blocked.
    pub critical_recall: Prf,
    pub critical_types: Vec<EntityType>,
    pub character_coverage: CharCoverage,
    /// False positives per 1,000 documents, from the overlap alignment. The
    /// usability number: how often a user gets blocked for no reason.
    pub false_positives_per_1k_docs: f64,
    /// Documents with no gold entities on which the detector predicted nothing.
    pub clean_negative_docs: u64,
    pub negative_docs: u64,
    pub documents: u64,
    pub gold_entities: u64,
    pub predicted_entities: u64,
}

/// Micro/macro/per-entity view of one match mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Averages {
    pub micro: Prf,
    pub r#macro: Prf,
    pub per_entity: BTreeMap<EntityType, Prf>,
}

impl Averages {
    pub fn from_counter(c: &Counter) -> Self {
        Self {
            micro: c.micro(),
            r#macro: c.macro_(),
            per_entity: c.per_entity(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityType::{CreditCard, Email, Passport};

    #[test]
    fn prf_arithmetic() {
        let p = Prf::from_counts(8, 2, 4);
        assert!((p.precision - 0.8).abs() < 1e-12);
        assert!((p.recall - 8.0 / 12.0).abs() < 1e-12);
        assert!((p.f1 - 2.0 * 0.8 * (8.0 / 12.0) / (0.8 + 8.0 / 12.0)).abs() < 1e-12);
        assert_eq!(p.support(), 12);
    }

    #[test]
    fn empty_counts_are_perfect_not_undefined() {
        // Correct silence on a document with nothing to find must not be
        // punished as 0.0 precision.
        let p = Prf::from_counts(0, 0, 0);
        assert_eq!((p.precision, p.recall), (1.0, 1.0));
        assert!((p.f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn all_misses_and_all_false_positives() {
        let miss = Prf::from_counts(0, 0, 5);
        assert_eq!((miss.precision, miss.recall, miss.f1), (1.0, 0.0, 0.0));
        let noise = Prf::from_counts(0, 5, 0);
        assert_eq!((noise.precision, noise.recall, noise.f1), (0.0, 1.0, 0.0));
    }

    #[test]
    fn macro_exposes_what_micro_hides() {
        // 100 easy EMAILs all correct, 4 PASSPORTs all missed. Micro looks
        // excellent; macro shows half the taxonomy is broken. This asymmetry is
        // the entire reason both are reported.
        let mut c = Counter::default();
        for _ in 0..100 {
            c.add_tp(Email);
        }
        for _ in 0..4 {
            c.add_fn(Passport);
        }

        let micro = c.micro();
        let macro_ = c.macro_();
        assert!((micro.recall - 100.0 / 104.0).abs() < 1e-12);
        assert!(micro.recall > 0.96, "micro hides the failure");
        assert!((macro_.recall - 0.5).abs() < 1e-12, "macro exposes it");

        let per = c.per_entity();
        assert_eq!(per[&Email].recall, 1.0);
        assert_eq!(per[&Passport].recall, 0.0);
        assert_eq!(per[&Passport].support(), 4);
    }

    #[test]
    fn macro_ignores_types_with_no_support() {
        // A type that only ever produced false positives has support 0; it must
        // not drag the macro recall toward an arbitrary value.
        let mut c = Counter::default();
        c.add_tp(Email);
        c.add_fp(CreditCard);
        assert_eq!(c.macro_().recall, 1.0);
        assert_eq!(c.per_entity()[&CreditCard].support(), 0);
    }

    #[test]
    fn critical_recall_pools_only_the_configured_types() {
        let mut c = Counter::default();
        for _ in 0..50 {
            c.add_tp(Email); // not critical
        }
        c.add_tp(Passport);
        c.add_fn(Passport);
        c.add_fn(CreditCard);

        let crit: BTreeSet<_> = [Passport, CreditCard].into_iter().collect();
        let r = c.recall_over(&crit);
        assert_eq!((r.tp, r.fn_), (1, 2));
        assert!(
            (r.recall - 1.0 / 3.0).abs() < 1e-12,
            "email must not inflate it"
        );
    }

    #[test]
    fn character_coverage_measures_partial_redaction() {
        // The task's example: gold `john.smith@example.com` (22 bytes),
        // predicted `smith@example.com` (17 bytes) starting at offset 5.
        let mut cc = CoverageCounter::default();
        cc.add_document(&[(0, 22)], &[(5, 22)]);
        let c = cc.finish();
        assert_eq!((c.sensitive_bytes, c.covered_bytes), (22, 17));
        assert!((c.sensitive_char_recall - 17.0 / 22.0).abs() < 1e-12);
        assert!((c.exposed_fraction - 5.0 / 22.0).abs() < 1e-12);
        assert_eq!(c.over_masked_bytes, 0);
    }

    #[test]
    fn character_coverage_counts_over_masking() {
        let mut cc = CoverageCounter::default();
        cc.add_document(&[(10, 20)], &[(0, 30)]);
        let c = cc.finish();
        assert_eq!(c.covered_bytes, 10);
        assert_eq!(c.sensitive_char_recall, 1.0);
        assert_eq!(
            c.over_masked_bytes, 20,
            "benign text a redactor would destroy"
        );
    }

    #[test]
    fn character_coverage_deduplicates_overlapping_predictions() {
        // Two predictions covering the same bytes must not report 200% recall.
        let mut cc = CoverageCounter::default();
        cc.add_document(&[(0, 10)], &[(0, 10), (0, 10), (2, 8)]);
        let c = cc.finish();
        assert_eq!(c.covered_bytes, 10);
        assert_eq!(c.sensitive_char_recall, 1.0);
        assert_eq!(c.over_masked_bytes, 0);
    }

    #[test]
    fn character_coverage_handles_no_gold_and_no_predictions() {
        let mut cc = CoverageCounter::default();
        cc.add_document(&[], &[]);
        let c = cc.finish();
        assert_eq!(
            c.sensitive_char_recall, 1.0,
            "nothing sensitive means nothing exposed"
        );

        let mut cc2 = CoverageCounter::default();
        cc2.add_document(&[], &[(0, 5)]);
        assert_eq!(cc2.finish().over_masked_bytes, 5);

        let mut cc3 = CoverageCounter::default();
        cc3.add_document(&[(0, 5)], &[]);
        let c3 = cc3.finish();
        assert_eq!(c3.sensitive_char_recall, 0.0);
        assert_eq!(c3.exposed_fraction, 1.0);
    }

    #[test]
    fn character_coverage_is_type_agnostic_by_design() {
        // A mislabelled prediction still redacts the bytes. Typed F1 punishes
        // it; coverage should not.
        let mut cc = CoverageCounter::default();
        cc.add_document(&[(0, 16)], &[(0, 16)]);
        assert_eq!(cc.finish().sensitive_char_recall, 1.0);
    }
}
