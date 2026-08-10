//! Bounded false-negative / false-positive samples.
//!
//! An aggregate F1 tells you a detector is bad; this tells you *why*. Samples
//! are bounded and quota'd per entity type so one pathological type cannot
//! crowd out the rest of the report.
//!
//! Text is emitted as a window around the span, not the whole document: these
//! are PII corpora, and a 4 KB document dumped into a log is a leak even in a
//! local benchmark.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::entity::EntityType;

/// Bytes of surrounding text kept on each side of a span.
const CONTEXT_BYTES: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    FalseNegative,
    FalsePositive,
}

/// One error, with enough context to act on without opening the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSample {
    pub kind: ErrorKind,
    pub example_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
    /// The span text itself.
    pub text: String,
    /// A window around the span, with the span delimited by `«»`.
    pub context: String,
    /// Detector score. `None` for false negatives — nothing was predicted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// What the detector said in this region instead, if anything. This is the
    /// difference between "missed it" and "found it but called it the wrong
    /// type", which need different fixes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub competing: Vec<Competing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Competing {
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

/// Collects error samples under a global cap and a per-type quota.
#[derive(Debug)]
pub struct ErrorCollector {
    limit: usize,
    per_type_quota: usize,
    samples: Vec<ErrorSample>,
    counts: BTreeMap<(ErrorKind, EntityType), usize>,
    /// Total errors seen, including those not retained, so the report can say
    /// "showing 100 of 4,312" rather than implying there were only 100.
    total: BTreeMap<ErrorKind, u64>,
}

impl ErrorCollector {
    pub fn new(limit: usize) -> Self {
        // Quota keeps a single noisy type from consuming the whole budget.
        // With ~20 types, a quota of limit/4 lets 4+ types be represented while
        // still allowing a dominant one to show depth.
        let per_type_quota = (limit / 4).max(4);
        Self {
            limit,
            per_type_quota,
            samples: Vec::new(),
            counts: BTreeMap::new(),
            total: BTreeMap::new(),
        }
    }

    pub fn disabled() -> Self {
        Self::new(0)
    }

    pub fn is_enabled(&self) -> bool {
        self.limit > 0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        kind: ErrorKind,
        example_id: &str,
        source: Option<&str>,
        tags: &[String],
        text: &str,
        entity_type: EntityType,
        start: usize,
        end: usize,
        score: Option<f32>,
        competing: Vec<Competing>,
    ) {
        *self.total.entry(kind).or_insert(0) += 1;
        if self.samples.len() >= self.limit {
            return;
        }
        let slot = self.counts.entry((kind, entity_type)).or_insert(0);
        if *slot >= self.per_type_quota {
            return;
        }
        *slot += 1;

        self.samples.push(ErrorSample {
            kind,
            example_id: example_id.to_owned(),
            source: source.map(str::to_owned),
            tags: tags.to_vec(),
            entity_type,
            start,
            end,
            text: text.get(start..end).unwrap_or_default().to_owned(),
            context: context_window(text, start, end),
            score,
            competing,
        });
    }

    /// Finish, sorting by entity type then example id so the report groups
    /// naturally and diffs cleanly between runs.
    pub fn finish(mut self) -> ErrorReport {
        self.samples.sort_by(|a, b| {
            a.entity_type
                .cmp(&b.entity_type)
                .then_with(|| a.example_id.cmp(&b.example_id))
                .then(a.start.cmp(&b.start))
        });
        let (mut fns, mut fps) = (Vec::new(), Vec::new());
        for s in self.samples {
            match s.kind {
                ErrorKind::FalseNegative => fns.push(s),
                ErrorKind::FalsePositive => fps.push(s),
            }
        }
        ErrorReport {
            false_negatives_shown: fns.len() as u64,
            false_positives_shown: fps.len() as u64,
            false_negatives_total: *self.total.get(&ErrorKind::FalseNegative).unwrap_or(&0),
            false_positives_total: *self.total.get(&ErrorKind::FalsePositive).unwrap_or(&0),
            false_negatives: fns,
            false_positives: fps,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorReport {
    pub false_negatives_total: u64,
    pub false_positives_total: u64,
    pub false_negatives_shown: u64,
    pub false_positives_shown: u64,
    pub false_negatives: Vec<ErrorSample>,
    pub false_positives: Vec<ErrorSample>,
}

/// Extract a context window with the span marked, snapped to char boundaries.
fn context_window(text: &str, start: usize, end: usize) -> String {
    let lo = floor_boundary(text, start.saturating_sub(CONTEXT_BYTES));
    let hi = ceil_boundary(text, (end + CONTEXT_BYTES).min(text.len()));
    let (Some(before), Some(mid), Some(after)) =
        (text.get(lo..start), text.get(start..end), text.get(end..hi))
    else {
        return String::new();
    };

    let mut s = String::with_capacity(hi - lo + 8);
    if lo > 0 {
        s.push('…');
    }
    s.push_str(&collapse_newlines(before));
    s.push('«');
    s.push_str(&collapse_newlines(mid));
    s.push('»');
    s.push_str(&collapse_newlines(after));
    if hi < text.len() {
        s.push('…');
    }
    s
}

/// Keep the window on one line so error tables stay readable.
fn collapse_newlines(s: &str) -> String {
    s.replace(['\n', '\r'], "⏎")
}

fn floor_boundary(text: &str, mut i: usize) -> usize {
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(text: &str, mut i: usize) -> usize {
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityType::{Email, Passport, Person};

    fn rec(c: &mut ErrorCollector, kind: ErrorKind, t: EntityType, id: &str) {
        c.record(
            kind,
            id,
            None,
            &[],
            "john@example.com here",
            t,
            0,
            16,
            Some(0.5),
            vec![],
        );
    }

    #[test]
    fn respects_the_global_limit_but_reports_the_true_total() {
        let mut c = ErrorCollector::new(3);
        for i in 0..50 {
            rec(&mut c, ErrorKind::FalsePositive, Email, &format!("e{i}"));
        }
        let r = c.finish();
        assert_eq!(r.false_positives_shown, 3);
        assert_eq!(
            r.false_positives_total, 50,
            "must not imply there were only 3"
        );
    }

    #[test]
    fn per_type_quota_stops_one_type_crowding_out_the_rest() {
        let mut c = ErrorCollector::new(20); // quota = 5
        for i in 0..40 {
            rec(&mut c, ErrorKind::FalseNegative, Email, &format!("e{i}"));
        }
        for i in 0..3 {
            rec(&mut c, ErrorKind::FalseNegative, Passport, &format!("p{i}"));
        }
        let r = c.finish();
        let emails = r
            .false_negatives
            .iter()
            .filter(|s| s.entity_type == Email)
            .count();
        let passports = r
            .false_negatives
            .iter()
            .filter(|s| s.entity_type == Passport)
            .count();
        assert_eq!(emails, 5, "quota enforced");
        assert_eq!(passports, 3, "rare type still represented");
    }

    #[test]
    fn output_is_grouped_by_entity_type_then_id() {
        let mut c = ErrorCollector::new(10);
        rec(&mut c, ErrorKind::FalsePositive, Person, "z");
        rec(&mut c, ErrorKind::FalsePositive, Email, "b");
        rec(&mut c, ErrorKind::FalsePositive, Email, "a");
        let r = c.finish();
        let got: Vec<_> = r
            .false_positives
            .iter()
            .map(|s| (s.entity_type, s.example_id.as_str()))
            .collect();
        assert_eq!(got, vec![(Person, "z"), (Email, "a"), (Email, "b")]);
    }

    #[test]
    fn context_marks_the_span_and_truncates_long_documents() {
        let text = format!("{}SECRET{}", "a".repeat(200), "b".repeat(200));
        let ctx = context_window(&text, 200, 206);
        assert!(ctx.starts_with('…') && ctx.ends_with('…'), "{ctx}");
        assert!(ctx.contains("«SECRET»"), "{ctx}");
        assert!(ctx.len() < 130, "window not bounded: {} bytes", ctx.len());
    }

    #[test]
    fn context_window_never_splits_a_multibyte_character() {
        // Cyrillic padding makes every naive byte offset land mid-character.
        let pad = "Ж".repeat(80);
        let text = format!("{pad}john@example.com{pad}");
        let start = pad.len();
        let ctx = context_window(&text, start, start + 16);
        assert!(ctx.contains("«john@example.com»"), "{ctx}");
        assert!(ctx.is_char_boundary(ctx.len()));
    }

    #[test]
    fn context_collapses_newlines_to_stay_on_one_line() {
        let text = "line1\nkey=secret\nline3";
        let ctx = context_window(text, 10, 16);
        assert!(!ctx.contains('\n'), "{ctx}");
        assert!(ctx.contains("«secret»"), "{ctx}");
    }

    #[test]
    fn disabled_collector_records_totals_but_no_samples() {
        let mut c = ErrorCollector::disabled();
        assert!(!c.is_enabled());
        rec(&mut c, ErrorKind::FalsePositive, Email, "e");
        let r = c.finish();
        assert_eq!(r.false_positives_total, 1);
        assert_eq!(r.false_positives_shown, 0);
    }

    #[test]
    fn competing_predictions_distinguish_a_miss_from_a_mislabel() {
        let mut c = ErrorCollector::new(4);
        c.record(
            ErrorKind::FalseNegative,
            "x",
            Some("custom"),
            &["prose".into()],
            "john@example.com",
            Email,
            0,
            16,
            None,
            vec![Competing {
                entity_type: Person,
                start: 0,
                end: 4,
                score: 0.6,
            }],
        );
        let r = c.finish();
        let s = &r.false_negatives[0];
        assert!(
            s.score.is_none(),
            "nothing was predicted for a false negative"
        );
        assert_eq!(s.competing.len(), 1);
        assert_eq!(s.competing[0].entity_type, Person);
        assert_eq!(s.text, "john@example.com");
        assert_eq!(s.source.as_deref(), Some("custom"));
    }
}
