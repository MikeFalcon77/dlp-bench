//! GLiNER1 `token_level` span decoding.
//!
//! The graph returns `logits[batch, num_words, num_classes, 3]`, where the last
//! axis is `(start, end, inside)` **logits** — sigmoid is not applied inside the
//! graph. Verified empirically against the loaded model rather than assumed:
//! on `"Contact John Smith at john@example.com"` with labels
//! `[person, email address, credit card]` the cells that light up are
//! `start@word1 / end@word2` for class 0 and `start@word4 / end@word8` for
//! class 1 — i.e. `John Smith` and `john@example.com`.
//!
//! A span `[i, j]` of class `c` is admitted when **all three** of
//! `start[i,c]`, `end[j,c]` and every `inside[k,c]` for `k in i..=j` clear the
//! threshold. That is the reference's membership rule; the reported score is
//! the minimum of those probabilities, which is the conservative joint
//! confidence and — usefully — is guaranteed to be `>= threshold`, so the
//! score cannot contradict the gate that admitted it.

use crate::detectors::words::Word;

/// A decoded span in the model's own label space.
#[derive(Debug, Clone, PartialEq)]
pub struct RawSpan {
    pub class_idx: usize,
    /// UTF-8 byte offsets into the original text.
    pub start: usize,
    pub end: usize,
    pub score: f32,
    /// Inclusive word range, kept for tests and diagnostics.
    pub word_start: usize,
    pub word_end: usize,
}

/// Shape of the `logits` tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogitDims {
    pub num_words: usize,
    pub num_classes: usize,
}

impl LogitDims {
    /// Read the shape, rejecting anything that is not `[1, W, C, 3]`.
    pub fn from_shape(shape: &[usize]) -> Option<Self> {
        match shape {
            &[1, num_words, num_classes, 3] => Some(Self {
                num_words,
                num_classes,
            }),
            _ => None,
        }
    }

    fn index(self, word: usize, class: usize, k: usize) -> usize {
        (word * self.num_classes + class) * 3 + k
    }
}

const START: usize = 0;
const END: usize = 1;
const INSIDE: usize = 2;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// How overlapping spans are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlapPolicy {
    /// Suppress across all classes — GLiNER1's own default (`flat_ner=True`).
    ///
    /// Note this differs from the GLiNER2 detector, which suppresses per label
    /// because *its* reference does. The two upstream implementations disagree,
    /// and each is matched to its own model rather than forced to a house
    /// style that neither model was decoded with.
    #[default]
    Flat,
    /// Suppress within a class only, allowing nested entities of other types.
    PerClass,
}

/// Decode `logits` into spans.
///
/// `max_width` bounds the span length in words (12 for these models).
pub fn decode(
    logits: &[f32],
    dims: LogitDims,
    words: &[Word],
    threshold: f32,
    max_width: usize,
    policy: OverlapPolicy,
) -> Vec<RawSpan> {
    let num_words = dims.num_words.min(words.len());
    let mut out: Vec<RawSpan> = Vec::new();

    for class in 0..dims.num_classes {
        for i in 0..num_words {
            let start_p = sigmoid(logits[dims.index(i, class, START)]);
            if start_p < threshold {
                continue;
            }
            // `inside` is accumulated as we extend, so a span is abandoned the
            // moment any interior word drops below the threshold.
            let mut min_inside = f32::INFINITY;
            for j in i..num_words.min(i + max_width) {
                min_inside = min_inside.min(sigmoid(logits[dims.index(j, class, INSIDE)]));
                if min_inside < threshold {
                    break;
                }
                let end_p = sigmoid(logits[dims.index(j, class, END)]);
                if end_p < threshold {
                    continue;
                }
                out.push(RawSpan {
                    class_idx: class,
                    start: words[i].start,
                    end: words[j].end,
                    score: start_p.min(end_p).min(min_inside),
                    word_start: i,
                    word_end: j,
                });
            }
        }
    }

    suppress(out, policy)
}

/// Greedy highest-score-first suppression.
fn suppress(mut spans: Vec<RawSpan>, policy: OverlapPolicy) -> Vec<RawSpan> {
    // Descending score, then ascending position — a total order over distinct
    // cells, so the result does not depend on sort stability.
    spans.sort_unstable_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.start.cmp(&b.start))
            .then(a.end.cmp(&b.end))
            .then(a.class_idx.cmp(&b.class_idx))
    });

    let mut kept: Vec<RawSpan> = Vec::new();
    for s in spans {
        let clashes = kept.iter().any(|k| {
            let overlaps = !(s.end <= k.start || s.start >= k.end);
            overlaps && (policy == OverlapPolicy::Flat || k.class_idx == s.class_idx)
        });
        if !clashes {
            kept.push(s);
        }
    }
    kept.sort_unstable_by_key(|s| (s.start, s.end, s.class_idx));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<Word> {
        let re = regex::Regex::new(r"\w+(?:[-_]\w+)*|\S").unwrap();
        let mut v = Vec::new();
        crate::detectors::words::split_with(&re, text, false, &mut v);
        v
    }

    /// Build a logit tensor from `(word, class, k, logit)` cells; everything
    /// else is a large negative so sigmoid ~ 0.
    fn logits(dims: LogitDims, cells: &[(usize, usize, usize, f32)]) -> Vec<f32> {
        let mut v = vec![-20.0f32; dims.num_words * dims.num_classes * 3];
        for &(w, c, k, x) in cells {
            v[dims.index(w, c, k)] = x;
        }
        v
    }

    /// logit for a target probability
    fn lg(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    #[test]
    fn shape_must_be_batch1_words_classes_3() {
        assert_eq!(
            LogitDims::from_shape(&[1, 9, 3, 3]),
            Some(LogitDims {
                num_words: 9,
                num_classes: 3
            })
        );
        // Anything else is rejected rather than reinterpreted.
        assert_eq!(LogitDims::from_shape(&[1, 9, 3]), None);
        assert_eq!(
            LogitDims::from_shape(&[2, 9, 3, 3]),
            None,
            "batching not supported"
        );
        assert_eq!(
            LogitDims::from_shape(&[1, 9, 3, 2]),
            None,
            "last axis must be 3"
        );
    }

    #[test]
    fn a_single_word_span_needs_start_end_and_inside() {
        let w = words("alpha beta");
        let d = LogitDims {
            num_words: 2,
            num_classes: 1,
        };

        // All three present -> one span.
        let l = logits(
            d,
            &[
                (0, 0, START, lg(0.9)),
                (0, 0, END, lg(0.9)),
                (0, 0, INSIDE, lg(0.9)),
            ],
        );
        let got = decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].word_start, got[0].word_end), (0, 0));

        // Missing `end` -> nothing.
        let l = logits(d, &[(0, 0, START, lg(0.9)), (0, 0, INSIDE, lg(0.9))]);
        assert!(decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat).is_empty());

        // Missing `inside` -> nothing.
        let l = logits(d, &[(0, 0, START, lg(0.9)), (0, 0, END, lg(0.9))]);
        assert!(decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat).is_empty());
    }

    #[test]
    fn a_multi_word_span_requires_inside_on_every_word() {
        let text = "Contact John Smith now";
        let w = words(text);
        let d = LogitDims {
            num_words: 4,
            num_classes: 1,
        };

        let ok = logits(
            d,
            &[
                (1, 0, START, lg(0.9)),
                (2, 0, END, lg(0.8)),
                (1, 0, INSIDE, lg(0.9)),
                (2, 0, INSIDE, lg(0.85)),
            ],
        );
        let got = decode(&ok, d, &w, 0.5, 12, OverlapPolicy::Flat);
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].start..got[0].end], "John Smith");
        // Score is the minimum of start, end and the weakest inside.
        assert!((got[0].score - 0.8).abs() < 1e-4, "score {}", got[0].score);

        // Drop the interior `inside` on the second word and the span dies.
        let broken = logits(
            d,
            &[
                (1, 0, START, lg(0.9)),
                (2, 0, END, lg(0.8)),
                (1, 0, INSIDE, lg(0.9)),
            ],
        );
        assert!(decode(&broken, d, &w, 0.5, 12, OverlapPolicy::Flat).is_empty());
    }

    #[test]
    fn max_width_bounds_the_span_length() {
        let w = words("a b c d e");
        let d = LogitDims {
            num_words: 5,
            num_classes: 1,
        };
        let mut cells = vec![(0, 0, START, lg(0.9)), (4, 0, END, lg(0.9))];
        for i in 0..5 {
            cells.push((i, 0, INSIDE, lg(0.9)));
        }
        let l = logits(d, &cells);
        assert_eq!(decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat).len(), 1);
        // A width cap of 3 cannot reach word 4 from word 0.
        assert!(decode(&l, d, &w, 0.5, 3, OverlapPolicy::Flat).is_empty());
    }

    #[test]
    fn threshold_gates_membership_and_the_score_can_never_contradict_it() {
        let w = words("alpha");
        let d = LogitDims {
            num_words: 1,
            num_classes: 1,
        };
        let l = logits(
            d,
            &[
                (0, 0, START, lg(0.6)),
                (0, 0, END, lg(0.7)),
                (0, 0, INSIDE, lg(0.8)),
            ],
        );

        let got = decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat);
        assert_eq!(got.len(), 1);
        assert!(got[0].score >= 0.5, "score below the gate that admitted it");
        assert!((got[0].score - 0.6).abs() < 1e-4);

        assert!(decode(&l, d, &w, 0.65, 12, OverlapPolicy::Flat).is_empty());
    }

    #[test]
    fn flat_suppression_drops_overlaps_across_classes_and_per_class_keeps_them() {
        let text = "John Smith";
        let w = words(text);
        let d = LogitDims {
            num_words: 2,
            num_classes: 2,
        };
        // Same span, two classes; class 1 slightly more confident.
        let l = logits(
            d,
            &[
                (0, 0, START, lg(0.8)),
                (1, 0, END, lg(0.8)),
                (0, 0, INSIDE, lg(0.8)),
                (1, 0, INSIDE, lg(0.8)),
                (0, 1, START, lg(0.9)),
                (1, 1, END, lg(0.9)),
                (0, 1, INSIDE, lg(0.9)),
                (1, 1, INSIDE, lg(0.9)),
            ],
        );

        let flat = decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat);
        assert_eq!(flat.len(), 1, "flat mode keeps only the best class");
        assert_eq!(flat[0].class_idx, 1);

        let nested = decode(&l, d, &w, 0.5, 12, OverlapPolicy::PerClass);
        assert_eq!(nested.len(), 2);
    }

    #[test]
    fn suppression_keeps_the_highest_scoring_of_two_overlapping_spans() {
        let text = "Contact John Smith now";
        let w = words(text);
        let d = LogitDims {
            num_words: 4,
            num_classes: 1,
        };
        let l = logits(
            d,
            &[
                (1, 0, START, lg(0.95)),
                (1, 0, END, lg(0.6)), // "John"
                (2, 0, END, lg(0.9)), // "John Smith" — better
                (1, 0, INSIDE, lg(0.95)),
                (2, 0, INSIDE, lg(0.95)),
            ],
        );
        let got = decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat);
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].start..got[0].end], "John Smith");
    }

    #[test]
    fn output_is_deterministic_and_position_sorted() {
        let text = "a b c d";
        let w = words(text);
        let d = LogitDims {
            num_words: 4,
            num_classes: 2,
        };
        let mut cells = Vec::new();
        for i in 0..4 {
            for c in 0..2 {
                cells.push((i, c, START, lg(0.7)));
                cells.push((i, c, END, lg(0.7)));
                cells.push((i, c, INSIDE, lg(0.7)));
            }
        }
        let l = logits(d, &cells);
        let first = decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat);
        for _ in 0..16 {
            assert_eq!(decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat), first);
        }
        assert!(
            first
                .windows(2)
                .all(|p| (p[0].start, p[0].end) <= (p[1].start, p[1].end))
        );
    }

    #[test]
    fn byte_offsets_are_exact_across_multibyte_text() {
        let text = "Свяжитесь с Михаил Ястребцов";
        let w = words(text);
        let names: Vec<_> = w.iter().map(|x| x.lower.to_string()).collect();
        assert_eq!(names, vec!["Свяжитесь", "с", "Михаил", "Ястребцов"]);

        let d = LogitDims {
            num_words: 4,
            num_classes: 1,
        };
        let l = logits(
            d,
            &[
                (2, 0, START, lg(0.9)),
                (3, 0, END, lg(0.9)),
                (2, 0, INSIDE, lg(0.9)),
                (3, 0, INSIDE, lg(0.9)),
            ],
        );
        let got = decode(&l, d, &w, 0.5, 12, OverlapPolicy::Flat);
        assert_eq!(&text[got[0].start..got[0].end], "Михаил Ястребцов");
        assert!(text.is_char_boundary(got[0].start) && text.is_char_boundary(got[0].end));
    }

    #[test]
    fn more_words_claimed_than_supplied_does_not_panic() {
        let w = words("a b");
        let claimed = LogitDims {
            num_words: 10,
            num_classes: 1,
        };
        let l = logits(
            claimed,
            &[
                (0, 0, START, lg(0.9)),
                (0, 0, END, lg(0.9)),
                (0, 0, INSIDE, lg(0.9)),
            ],
        );
        assert_eq!(
            decode(&l, claimed, &w, 0.5, 12, OverlapPolicy::Flat).len(),
            1
        );
    }
}
