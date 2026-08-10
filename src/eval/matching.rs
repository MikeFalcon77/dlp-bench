//! One-to-one span matching.
//!
//! Both match modes share this implementation so exact and overlap numbers are
//! always computed against the same alignment discipline. The matcher is
//! deterministic and strictly one-to-one: a prediction can satisfy at most one
//! gold span, and a gold span can be satisfied by at most one prediction.
//!
//! # Why one-to-one
//!
//! Without it, two predictions overlapping one gold span both count as true
//! positives and precision is inflated; one prediction overlapping two gold
//! spans satisfies both and recall is inflated. Either way the detector that
//! emits sloppier, larger spans scores better, which is the opposite of what a
//! redaction system wants.
//!
//! # Algorithm
//!
//! 1. Build every candidate `(gold, pred)` pair with the same canonical type
//!    and non-zero byte overlap (exact mode additionally requires identical
//!    boundaries).
//! 2. Sort candidates by descending IoU, then ascending gold start, gold end,
//!    prediction start, prediction end. The tie-break chain is total, so the
//!    result does not depend on input order or sort stability.
//! 3. Walk the sorted list greedily, accepting a pair only if neither side is
//!    already matched.
//!
//! Greedy-by-IoU is not guaranteed to maximise the number of matches the way a
//! Hungarian assignment would, but it is O(n log n), it is what a human reading
//! the error report expects ("the best-overlapping prediction won"), and the
//! pathological cases require gold spans to overlap each other — which the
//! dataset loader rejects.

use crate::span::TypedSpan;

/// How a prediction is allowed to satisfy a gold span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    /// Same canonical type, identical start and end.
    Exact,
    /// Same canonical type, at least one byte of overlap.
    ///
    /// Abutting spans do not overlap: `[0,5)` and `[5,9)` share no byte.
    Overlap,
}

impl MatchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchMode::Exact => "exact",
            MatchMode::Overlap => "overlap",
        }
    }
}

/// Result of aligning predictions to gold spans for one document.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Alignment {
    /// `(gold index, prediction index)` pairs, sorted by gold index.
    pub matched: Vec<(usize, usize)>,
    /// Gold indices with no prediction: false negatives.
    pub unmatched_gold: Vec<usize>,
    /// Prediction indices with no gold: false positives.
    pub unmatched_pred: Vec<usize>,
}

impl Alignment {
    pub fn tp(&self) -> usize {
        self.matched.len()
    }
    pub fn fn_(&self) -> usize {
        self.unmatched_gold.len()
    }
    pub fn fp(&self) -> usize {
        self.unmatched_pred.len()
    }
}

/// Align `pred` to `gold` under `mode`.
pub fn align(gold: &[TypedSpan], pred: &[TypedSpan], mode: MatchMode) -> Alignment {
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();

    for (gi, g) in gold.iter().enumerate() {
        for (pi, p) in pred.iter().enumerate() {
            if g.entity_type != p.entity_type {
                continue;
            }
            let ok = match mode {
                MatchMode::Exact => g.start == p.start && g.end == p.end,
                MatchMode::Overlap => g.overlap_len(*p) > 0,
            };
            if ok {
                candidates.push((g.iou(*p), gi, pi));
            }
        }
    }

    // Descending IoU, then a total tie-break on the spans themselves so the
    // outcome is independent of iteration order.
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| (gold[a.1].start, gold[a.1].end).cmp(&(gold[b.1].start, gold[b.1].end)))
            .then_with(|| (pred[a.2].start, pred[a.2].end).cmp(&(pred[b.2].start, pred[b.2].end)))
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let mut gold_taken = vec![false; gold.len()];
    let mut pred_taken = vec![false; pred.len()];
    let mut matched = Vec::new();

    for (_, gi, pi) in candidates {
        if gold_taken[gi] || pred_taken[pi] {
            continue;
        }
        gold_taken[gi] = true;
        pred_taken[pi] = true;
        matched.push((gi, pi));
    }

    matched.sort_unstable();
    Alignment {
        matched,
        unmatched_gold: (0..gold.len()).filter(|&i| !gold_taken[i]).collect(),
        unmatched_pred: (0..pred.len()).filter(|&i| !pred_taken[i]).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityType::{Email, Person, Phone};

    fn s(t: crate::entity::EntityType, a: usize, b: usize) -> TypedSpan {
        TypedSpan::new(t, a, b)
    }

    #[test]
    fn exact_match() {
        let g = [s(Email, 10, 20)];
        let p = [s(Email, 10, 20)];
        let a = align(&g, &p, MatchMode::Exact);
        assert_eq!(a.matched, vec![(0, 0)]);
        assert_eq!((a.tp(), a.fp(), a.fn_()), (1, 0, 0));
    }

    #[test]
    fn right_span_wrong_type_is_never_a_match() {
        let g = [s(Email, 10, 20)];
        let p = [s(Person, 10, 20)];
        for mode in [MatchMode::Exact, MatchMode::Overlap] {
            let a = align(&g, &p, mode);
            assert_eq!((a.tp(), a.fp(), a.fn_()), (0, 1, 1), "{mode:?}");
        }
    }

    #[test]
    fn partial_overlap_fails_exact_but_passes_overlap() {
        // The task's motivating example: gold `john.smith@example.com`,
        // predicted `smith@example.com`. An exact-entity failure that is still
        // partial DLP protection.
        let g = [s(Email, 0, 22)];
        let p = [s(Email, 5, 22)];

        let e = align(&g, &p, MatchMode::Exact);
        assert_eq!((e.tp(), e.fp(), e.fn_()), (0, 1, 1));

        let o = align(&g, &p, MatchMode::Overlap);
        assert_eq!((o.tp(), o.fp(), o.fn_()), (1, 0, 0));
    }

    #[test]
    fn abutting_spans_do_not_overlap() {
        let g = [s(Email, 0, 5)];
        let p = [s(Email, 5, 9)];
        let a = align(&g, &p, MatchMode::Overlap);
        assert_eq!((a.tp(), a.fp(), a.fn_()), (0, 1, 1));
    }

    #[test]
    fn duplicate_predictions_cost_precision() {
        // Two identical predictions over one gold span: exactly one may match.
        let g = [s(Email, 0, 10)];
        let p = [s(Email, 0, 10), s(Email, 0, 10)];
        for mode in [MatchMode::Exact, MatchMode::Overlap] {
            let a = align(&g, &p, mode);
            assert_eq!((a.tp(), a.fp(), a.fn_()), (1, 1, 0), "{mode:?}");
        }
    }

    #[test]
    fn one_prediction_overlapping_two_gold_spans_satisfies_only_one() {
        // A greedy "cover everything" detector must not get credit twice.
        let g = [s(Email, 0, 10), s(Email, 12, 20)];
        let p = [s(Email, 0, 20)];
        let a = align(&g, &p, MatchMode::Overlap);
        assert_eq!((a.tp(), a.fp(), a.fn_()), (1, 0, 1));
        // The better-IoU gold span (0..10, IoU 0.5) wins over 12..20 (IoU 0.4).
        assert_eq!(a.matched, vec![(0, 0)]);
        assert_eq!(a.unmatched_gold, vec![1]);
    }

    #[test]
    fn two_predictions_inside_one_gold_span_yield_one_tp_and_one_fp() {
        let g = [s(Email, 0, 20)];
        let p = [s(Email, 0, 8), s(Email, 10, 20)];
        let a = align(&g, &p, MatchMode::Overlap);
        assert_eq!((a.tp(), a.fp(), a.fn_()), (1, 1, 0));
        // 10..20 has IoU 0.5, 0..8 has 0.4 — the better one wins.
        assert_eq!(a.matched, vec![(0, 1)]);
        assert_eq!(a.unmatched_pred, vec![0]);
    }

    #[test]
    fn best_overlap_wins_regardless_of_input_order() {
        let g = [s(Email, 10, 20)];
        let good = s(Email, 10, 20);
        let poor = s(Email, 19, 30);

        let a = align(&g, &[poor, good], MatchMode::Overlap);
        let b = align(&g, &[good, poor], MatchMode::Overlap);
        assert_eq!(a.matched, vec![(0, 1)]);
        assert_eq!(b.matched, vec![(0, 0)]);
        assert_eq!(a.tp(), b.tp());
    }

    #[test]
    fn no_predictions_and_no_gold() {
        let none: [TypedSpan; 0] = [];
        let g = [s(Email, 0, 5)];
        let p = [s(Email, 0, 5)];

        let a = align(&g, &none, MatchMode::Overlap);
        assert_eq!((a.tp(), a.fp(), a.fn_()), (0, 0, 1));

        let b = align(&none, &p, MatchMode::Overlap);
        assert_eq!((b.tp(), b.fp(), b.fn_()), (0, 1, 0));

        let c = align(&none, &none, MatchMode::Overlap);
        assert_eq!((c.tp(), c.fp(), c.fn_()), (0, 0, 0));
        assert_eq!(c, Alignment::default());
    }

    #[test]
    fn matching_is_one_to_one_under_heavy_contention() {
        let g = [s(Email, 0, 10), s(Email, 5, 15), s(Email, 20, 30)];
        let p = [
            s(Email, 0, 10),
            s(Email, 5, 15),
            s(Email, 6, 16),
            s(Email, 25, 35),
        ];
        let a = align(&g, &p, MatchMode::Overlap);

        let gs: Vec<_> = a.matched.iter().map(|(g, _)| *g).collect();
        let ps: Vec<_> = a.matched.iter().map(|(_, p)| *p).collect();
        assert_eq!(gs.len(), dedup_len(&gs), "a gold span matched twice");
        assert_eq!(ps.len(), dedup_len(&ps), "a prediction matched twice");
        assert_eq!(a.tp() + a.fn_(), g.len());
        assert_eq!(a.tp() + a.fp(), p.len());
    }

    #[test]
    fn alignment_is_deterministic_across_repeated_runs() {
        let g = [s(Email, 0, 10), s(Phone, 0, 10), s(Email, 4, 14)];
        let p = [s(Email, 2, 12), s(Email, 0, 10), s(Phone, 0, 10)];
        let first = align(&g, &p, MatchMode::Overlap);
        for _ in 0..32 {
            assert_eq!(align(&g, &p, MatchMode::Overlap), first);
        }
    }

    fn dedup_len(v: &[usize]) -> usize {
        let mut v = v.to_vec();
        v.sort_unstable();
        v.dedup();
        v.len()
    }
}
