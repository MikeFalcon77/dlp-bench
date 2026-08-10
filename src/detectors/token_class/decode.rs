//! BIO decoding for token-classification models.
//!
//! Three things here are decisions rather than mechanics, and each changes the
//! measured score:
//!
//! 1. **Sub-token offsets are byte offsets into the original text.** The Rust
//!    `tokenizers` crate reports them that way (the Python binding converts to
//!    char offsets, which is where most ported code goes wrong). Verified by
//!    test against non-ASCII input rather than assumed.
//! 2. **Score is the softmax probability of the winning label**, averaged over
//!    the span's tokens. Averaging rather than taking the minimum keeps a
//!    single hesitant interior token from sinking an otherwise confident span.
//! 3. **Adjacent spans of the same canonical type are merged.** The ai4privacy
//!    taxonomy decomposes composites — a person is `FIRSTNAME` + `LASTNAME`, an
//!    address is `BUILDINGNUMBER` + `STREET` + `CITY`. Without merging, one
//!    gold `PERSON` is matched by one prediction and *counted against* by the
//!    rest, which reports a schema artefact as a precision failure.

use crate::entity::EntityType;

/// One BIO-decoded span, in the model's own label vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct TaggedSpan {
    pub entity_type: EntityType,
    /// UTF-8 byte offsets into the original text.
    pub start: usize,
    pub end: usize,
    pub score: f32,
    /// Number of sub-tokens the span covers, used for score averaging.
    pub tokens: usize,
}

/// A decoded label for one sub-token.
#[derive(Debug, Clone, Copy)]
pub struct TokenTag<'a> {
    /// `O`, or `B-TYPE` / `I-TYPE`.
    pub label: &'a str,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

/// Split a BIO label into its prefix and base type.
///
/// Returns `None` for `O` and for anything that is not BIO-shaped, which is
/// itself worth surfacing rather than guessing at.
pub fn split_bio(label: &str) -> Option<(bool, &str)> {
    if label == "O" {
        return None;
    }
    match label.split_once('-') {
        Some(("B", t)) => Some((true, t)),
        Some(("I", t)) => Some((false, t)),
        _ => None,
    }
}

/// Aggregate per-token BIO tags into spans.
///
/// `map` turns a model label base type into a canonical type; `None` drops the
/// span, and the caller counts it as unmapped.
pub fn decode_bio(
    tags: &[TokenTag<'_>],
    map: &dyn Fn(&str) -> Option<EntityType>,
    threshold: f32,
) -> Vec<TaggedSpan> {
    let mut out: Vec<TaggedSpan> = Vec::new();
    // (base type, canonical type, start, end, score sum, token count)
    let mut open: Option<(String, EntityType, usize, usize, f32, usize)> = None;

    let flush = |open: &mut Option<(String, EntityType, usize, usize, f32, usize)>,
                 out: &mut Vec<TaggedSpan>| {
        if let Some((_, ty, start, end, sum, n)) = open.take()
            && n > 0
        {
            out.push(TaggedSpan {
                entity_type: ty,
                start,
                end,
                score: sum / n as f32,
                tokens: n,
            });
        }
    };

    for t in tags {
        let Some((is_begin, base)) = split_bio(t.label) else {
            flush(&mut open, &mut out);
            continue;
        };
        let Some(canonical) = map(base) else {
            // Unmapped model type: close any open span rather than letting it
            // swallow the tokens.
            flush(&mut open, &mut out);
            continue;
        };

        let continues = matches!(&open, Some((b, _, _, _, _, _)) if b == base) && !is_begin;
        if continues {
            let cur = open.as_mut().expect("checked by `continues`");
            cur.3 = t.end;
            cur.4 += t.score;
            cur.5 += 1;
        } else {
            flush(&mut open, &mut out);
            open = Some((base.to_owned(), canonical, t.start, t.end, t.score, 1));
        }
    }
    flush(&mut open, &mut out);

    // Threshold on the averaged span score, not per token: a span is accepted
    // or rejected as a unit.
    out.retain(|s| s.score >= threshold);
    out
}

/// Merge adjacent spans of the same canonical type.
///
/// Two spans merge when only whitespace or a joining punctuation character
/// separates them. `text` is needed to inspect the gap, which is the whole
/// point: `"John Smith"` merges, but `"John ... Smith"` in two different
/// sentences must not.
///
/// `max_gap_bytes` bounds how far apart the pieces may be. Small on purpose —
/// a large value would chain unrelated entities across a document.
pub fn merge_adjacent(
    mut spans: Vec<TaggedSpan>,
    text: &str,
    max_gap_bytes: usize,
) -> Vec<TaggedSpan> {
    if spans.len() < 2 {
        return spans;
    }
    spans.sort_by_key(|s| (s.start, s.end));

    let mut out: Vec<TaggedSpan> = Vec::with_capacity(spans.len());
    for s in spans {
        let Some(prev) = out.last_mut() else {
            out.push(s);
            continue;
        };
        if prev.entity_type != s.entity_type || s.start < prev.end {
            // Overlapping same-type spans also merge — a model that emits
            // B-CITY immediately inside a STREET span should not double-count.
            if prev.entity_type == s.entity_type && s.start < prev.end {
                prev.end = prev.end.max(s.end);
                prev.score = (prev.score * prev.tokens as f32 + s.score * s.tokens as f32)
                    / (prev.tokens + s.tokens) as f32;
                prev.tokens += s.tokens;
                continue;
            }
            out.push(s);
            continue;
        }

        let gap = &text[prev.end..s.start];
        let joinable = gap.len() <= max_gap_bytes
            && !gap.is_empty()
            && gap.chars().all(|c| c.is_whitespace() || matches!(c, ',' | '-' | '.' | '\''))
            // A line break separates records far more often than it joins them.
            && !gap.contains('\n');

        if gap.is_empty() || joinable {
            prev.end = s.end;
            prev.score = (prev.score * prev.tokens as f32 + s.score * s.tokens as f32)
                / (prev.tokens + s.tokens) as f32;
            prev.tokens += s.tokens;
        } else {
            out.push(s);
        }
    }
    out
}

/// Tighten a span to the entity itself.
///
/// Two adjustments, both required for the spans to be comparable with the other
/// detectors:
///
/// * **Leading/trailing whitespace.** SentencePiece attributes the word-leading
///   space to the token, so `▁Müller` reports offsets covering `" Müller"`.
/// * **Trailing punctuation.** The model routinely tags a trailing separator as
///   part of the entity — `,` comes back as `I-IBAN`, `.` as `I-PHONENUMBER`.
///   The regex recognizers already strip these, so applying the same
///   [`crate::span::TRAILING_PUNCT`] rule here keeps exact-match comparisons
///   about the detector rather than about boundary conventions.
pub fn trim_spans(spans: Vec<TaggedSpan>, text: &str) -> Vec<TaggedSpan> {
    spans
        .into_iter()
        .filter_map(|mut s| {
            let (start, end) = crate::span::trim_span(text, s.start, s.end)?;
            let (start, end) =
                crate::span::trim_trailing(text, start, end, crate::span::TRAILING_PUNCT);
            // Stripping punctuation can expose whitespace beneath it.
            let (start, end) = crate::span::trim_span(text, start, end)?;
            s.start = start;
            s.end = end;
            Some(s)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityType::{Address, Email, Person};

    fn map(base: &str) -> Option<EntityType> {
        match base {
            "FIRSTNAME" | "LASTNAME" | "MIDDLENAME" => Some(Person),
            "EMAIL" => Some(Email),
            "STREET" | "CITY" | "BUILDINGNUMBER" | "ZIPCODE" => Some(Address),
            _ => None,
        }
    }

    fn tag<'a>(label: &'a str, score: f32, start: usize, end: usize) -> TokenTag<'a> {
        TokenTag {
            label,
            score,
            start,
            end,
        }
    }

    #[test]
    fn bio_prefixes_are_parsed_and_junk_is_rejected() {
        assert_eq!(split_bio("B-EMAIL"), Some((true, "EMAIL")));
        assert_eq!(split_bio("I-EMAIL"), Some((false, "EMAIL")));
        assert_eq!(split_bio("O"), None);
        assert_eq!(split_bio("EMAIL"), None, "unprefixed labels are not BIO");
        assert_eq!(split_bio("X-EMAIL"), None);
        // Hyphenated type names still work, because only the first '-' splits.
        assert_eq!(split_bio("B-SOME-TYPE"), Some((true, "SOME-TYPE")));
    }

    #[test]
    fn a_single_begin_token_is_a_span() {
        let tags = [tag("B-EMAIL", 0.99, 5, 21)];
        let got = decode_bio(&tags, &map, 0.5);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].start, got[0].end), (5, 21));
        assert_eq!(got[0].entity_type, Email);
    }

    #[test]
    fn inside_tokens_extend_the_span_and_scores_average() {
        let tags = [
            tag("B-EMAIL", 1.0, 0, 4),
            tag("I-EMAIL", 0.8, 4, 10),
            tag("I-EMAIL", 0.6, 10, 16),
        ];
        let got = decode_bio(&tags, &map, 0.5);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].start, got[0].end), (0, 16));
        assert_eq!(got[0].tokens, 3);
        assert!((got[0].score - 0.8).abs() < 1e-6, "score {}", got[0].score);
    }

    #[test]
    fn a_new_begin_starts_a_new_span_even_for_the_same_type() {
        // Two adjacent people must not become one at the BIO level; merging is
        // a separate, explicit decision.
        let tags = [
            tag("B-FIRSTNAME", 0.9, 0, 4),
            tag("B-FIRSTNAME", 0.9, 5, 10),
        ];
        let got = decode_bio(&tags, &map, 0.5);
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn o_tokens_and_unmapped_types_close_the_open_span() {
        let tags = [
            tag("B-EMAIL", 0.9, 0, 4),
            tag("O", 0.9, 4, 5),
            tag("I-EMAIL", 0.9, 5, 9),
        ];
        let got = decode_bio(&tags, &map, 0.5);
        assert_eq!(got.len(), 2, "an O must break the span");

        let tags = [
            tag("B-EMAIL", 0.9, 0, 4),
            tag("B-JOBTITLE", 0.9, 4, 9), // unmapped
            tag("I-EMAIL", 0.9, 9, 12),
        ];
        let got = decode_bio(&tags, &map, 0.5);
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|s| s.entity_type == Email));
    }

    #[test]
    fn a_dangling_inside_tag_still_opens_a_span() {
        // Real models emit I- without a preceding B-. Dropping those loses
        // recall for no reason.
        let tags = [tag("I-EMAIL", 0.9, 0, 8)];
        let got = decode_bio(&tags, &map, 0.5);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].start, got[0].end), (0, 8));
    }

    #[test]
    fn threshold_applies_to_the_averaged_span_score() {
        let tags = [tag("B-EMAIL", 0.9, 0, 4), tag("I-EMAIL", 0.1, 4, 8)];
        // mean 0.5
        assert_eq!(decode_bio(&tags, &map, 0.5).len(), 1);
        assert_eq!(decode_bio(&tags, &map, 0.6).len(), 0);
    }

    #[test]
    fn adjacent_pieces_of_a_composite_entity_merge() {
        // The case that would otherwise report 1 TP + 1 FP against one gold.
        let text = "Contact John Smith today";
        let spans = vec![
            TaggedSpan {
                entity_type: Person,
                start: 8,
                end: 12,
                score: 0.9,
                tokens: 1,
            },
            TaggedSpan {
                entity_type: Person,
                start: 13,
                end: 18,
                score: 0.7,
                tokens: 1,
            },
        ];
        let got = merge_adjacent(spans, text, 3);
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].start..got[0].end], "John Smith");
        assert!((got[0].score - 0.8).abs() < 1e-6);
        assert_eq!(got[0].tokens, 2);
    }

    #[test]
    fn a_multi_part_address_merges_into_one_span() {
        let text = "Ship to 12 Hauptstrasse, Munich now";
        let spans = vec![
            TaggedSpan {
                entity_type: Address,
                start: 8,
                end: 10,
                score: 0.9,
                tokens: 1,
            },
            TaggedSpan {
                entity_type: Address,
                start: 11,
                end: 23,
                score: 0.9,
                tokens: 1,
            },
            TaggedSpan {
                entity_type: Address,
                start: 25,
                end: 31,
                score: 0.9,
                tokens: 1,
            },
        ];
        let got = merge_adjacent(spans, text, 3);
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].start..got[0].end], "12 Hauptstrasse, Munich");
    }

    #[test]
    fn different_types_and_distant_spans_never_merge() {
        let text = "John Smith wrote to a@b.co";
        let spans = vec![
            TaggedSpan {
                entity_type: Person,
                start: 0,
                end: 10,
                score: 0.9,
                tokens: 2,
            },
            TaggedSpan {
                entity_type: Email,
                start: 20,
                end: 26,
                score: 0.9,
                tokens: 1,
            },
        ];
        assert_eq!(merge_adjacent(spans, text, 3).len(), 2, "different types");

        let far = "John wrote a long sentence and then Smith";
        let spans = vec![
            TaggedSpan {
                entity_type: Person,
                start: 0,
                end: 4,
                score: 0.9,
                tokens: 1,
            },
            TaggedSpan {
                entity_type: Person,
                start: 35,
                end: 40,
                score: 0.9,
                tokens: 1,
            },
        ];
        assert_eq!(merge_adjacent(spans, far, 3).len(), 2, "gap too wide");
    }

    #[test]
    fn a_newline_prevents_merging() {
        // Two records in a config or a YAML list, not one entity.
        let text = "name: John\nname: Smith";
        let spans = vec![
            TaggedSpan {
                entity_type: Person,
                start: 6,
                end: 10,
                score: 0.9,
                tokens: 1,
            },
            TaggedSpan {
                entity_type: Person,
                start: 17,
                end: 22,
                score: 0.9,
                tokens: 1,
            },
        ];
        assert_eq!(merge_adjacent(spans, text, 8).len(), 2);
    }

    #[test]
    fn overlapping_same_type_spans_collapse() {
        let text = "12 Hauptstrasse";
        let spans = vec![
            TaggedSpan {
                entity_type: Address,
                start: 0,
                end: 15,
                score: 0.9,
                tokens: 3,
            },
            TaggedSpan {
                entity_type: Address,
                start: 3,
                end: 15,
                score: 0.7,
                tokens: 2,
            },
        ];
        let got = merge_adjacent(spans, text, 3);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].start, got[0].end), (0, 15));
    }

    #[test]
    fn merging_is_byte_exact_across_multibyte_text() {
        let text = "Свяжитесь с Михаил Ястребцов сегодня";
        let a = text.find("Михаил").unwrap();
        let b = text.find("Ястребцов").unwrap();
        let spans = vec![
            TaggedSpan {
                entity_type: Person,
                start: a,
                end: a + "Михаил".len(),
                score: 0.9,
                tokens: 2,
            },
            TaggedSpan {
                entity_type: Person,
                start: b,
                end: b + "Ястребцов".len(),
                score: 0.9,
                tokens: 3,
            },
        ];
        let got = merge_adjacent(spans, text, 3);
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].start..got[0].end], "Михаил Ястребцов");
        assert!(text.is_char_boundary(got[0].start) && text.is_char_boundary(got[0].end));
    }

    #[test]
    fn trimming_drops_whitespace_only_spans_and_tightens_the_rest() {
        let text = "a  John  b";
        let spans = vec![
            TaggedSpan {
                entity_type: Person,
                start: 1,
                end: 9,
                score: 0.9,
                tokens: 1,
            },
            TaggedSpan {
                entity_type: Person,
                start: 1,
                end: 3,
                score: 0.9,
                tokens: 1,
            },
        ];
        let got = trim_spans(spans, text);
        assert_eq!(got.len(), 1);
        assert_eq!(&text[got[0].start..got[0].end], "John");
    }

    #[test]
    fn trimming_strips_trailing_punctuation_the_model_tagged_as_part_of_the_entity() {
        // Observed for real: the model returns `,` as I-IBAN and `.` as
        // I-PHONENUMBER. Both must be excluded to match the gold convention.
        let text = "IBAN DE89370400440532013000, and phone +1-415-555-1234.";
        let iban_end = text.find(" and").unwrap();
        let spans = vec![
            TaggedSpan {
                entity_type: Email,
                start: 5,
                end: iban_end,
                score: 0.9,
                tokens: 6,
            },
            TaggedSpan {
                entity_type: Person,
                start: 39,
                end: text.len(),
                score: 0.9,
                tokens: 4,
            },
        ];
        let got = trim_spans(spans, text);
        assert_eq!(&text[got[0].start..got[0].end], "DE89370400440532013000");
        assert_eq!(&text[got[1].start..got[1].end], "+1-415-555-1234");
    }

    #[test]
    fn trimming_never_splits_a_multibyte_character() {
        let text = "Свяжитесь с Михаил Ястребцов.";
        let start = text.find("Михаил").unwrap();
        let spans = vec![TaggedSpan {
            entity_type: Person,
            start: start - 1, // leading space, as SentencePiece reports it
            end: text.len(),  // includes the trailing '.'
            score: 0.9,
            tokens: 4,
        }];
        let got = trim_spans(spans, text);
        assert_eq!(&text[got[0].start..got[0].end], "Михаил Ястребцов");
        assert!(text.is_char_boundary(got[0].start) && text.is_char_boundary(got[0].end));
    }

    #[test]
    fn empty_input_is_safe() {
        assert!(decode_bio(&[], &map, 0.5).is_empty());
        assert!(merge_adjacent(vec![], "", 3).is_empty());
        assert!(trim_spans(vec![], "").is_empty());
    }
}
