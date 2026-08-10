//! Span convention and offset normalisation.
//!
//! **One rule: every offset that crosses a module boundary in this crate is a
//! UTF-8 byte offset**, such that `&text[start..end]` is the span. Everything
//! else — Python character indices, UTF-16 code units, tokenizer offsets — is
//! converted explicitly at the edge, by the functions here, and never guessed.
//!
//! This module exists because offset bugs are invisible on ASCII. Every
//! conversion has a test using Cyrillic, accented Latin and CJK text.

use crate::entity::{Detection, EntityType};

/// Why a span was rejected. Every field is included because these surface in
/// dataset-load errors that a human has to act on.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpanError {
    #[error("inverted span: start {start} > end {end}")]
    Inverted { start: usize, end: usize },

    #[error("span end {end} exceeds text length {len}")]
    OutOfBounds { end: usize, len: usize },

    #[error("span start {start} is not a UTF-8 character boundary")]
    StartNotCharBoundary { start: usize },

    #[error("span end {end} is not a UTF-8 character boundary")]
    EndNotCharBoundary { end: usize },

    #[error("empty span at {start}")]
    Empty { start: usize },
}

/// Validate a byte range against `text`.
///
/// Used by dataset loading, by every detector before it returns, and by the
/// evaluator. Empty spans are permitted here (`allow_empty`) because a
/// zero-length gold span is a dataset bug worth naming separately from a
/// detector legitimately producing nothing.
pub fn validate(text: &str, start: usize, end: usize, allow_empty: bool) -> Result<(), SpanError> {
    if start > end {
        return Err(SpanError::Inverted { start, end });
    }
    if end > text.len() {
        return Err(SpanError::OutOfBounds {
            end,
            len: text.len(),
        });
    }
    if !text.is_char_boundary(start) {
        return Err(SpanError::StartNotCharBoundary { start });
    }
    if !text.is_char_boundary(end) {
        return Err(SpanError::EndNotCharBoundary { end });
    }
    if !allow_empty && start == end {
        return Err(SpanError::Empty { start });
    }
    Ok(())
}

/// Slice `text` by a validated byte range.
pub fn slice(text: &str, start: usize, end: usize) -> Result<&str, SpanError> {
    validate(text, start, end, true)?;
    Ok(&text[start..end])
}

/// Validate every detection a detector is about to return.
///
/// Detectors call this rather than trusting themselves; a span that fails here
/// is a bug in the detector, not in the input, so it is reported with the
/// detector's own name attached by the caller.
pub fn validate_all(text: &str, detections: &[Detection]) -> Result<(), (usize, SpanError)> {
    for (i, d) in detections.iter().enumerate() {
        validate(text, d.start, d.end, false).map_err(|e| (i, e))?;
    }
    Ok(())
}

/// Build a char-index -> byte-offset table for `text`.
///
/// Length is `text.chars().count() + 1`; the final element is `text.len()`, so
/// a half-open char range `[a, b)` maps to bytes `[table[a], table[b])`.
///
/// Needed because the GLiNER2 reference implementation, and most Python-derived
/// datasets, index by Unicode scalar value.
pub fn char_to_byte_table(text: &str) -> Vec<usize> {
    let mut table = Vec::with_capacity(text.len() + 1);
    table.extend(text.char_indices().map(|(i, _)| i));
    table.push(text.len());
    table
}

/// Convert a single char offset to a byte offset. `None` if out of range.
///
/// O(n). For converting many offsets over one text, build a
/// [`char_to_byte_table`] once instead.
pub fn char_to_byte(text: &str, char_idx: usize) -> Option<usize> {
    if char_idx == 0 {
        return Some(0);
    }
    let mut n = 0usize;
    for (byte_idx, _) in text.char_indices() {
        if n == char_idx {
            return Some(byte_idx);
        }
        n += 1;
    }
    (n == char_idx).then_some(text.len())
}

/// Convert a UTF-16 code-unit offset to a byte offset. `None` if out of range
/// or if the offset splits a surrogate pair.
///
/// Present because JS/browser-sourced annotations and some tokenizer exports
/// use UTF-16 offsets, and an unconverted one is silently wrong for any text
/// containing an emoji or a CJK ideograph outside the BMP.
pub fn utf16_to_byte(text: &str, utf16_idx: usize) -> Option<usize> {
    if utf16_idx == 0 {
        return Some(0);
    }
    let mut units = 0usize;
    for (byte_idx, ch) in text.char_indices() {
        if units == utf16_idx {
            return Some(byte_idx);
        }
        if units > utf16_idx {
            return None; // landed inside a surrogate pair
        }
        units += ch.len_utf16();
    }
    (units == utf16_idx).then_some(text.len())
}

/// Trim whitespace from a byte span, returning the tightened range.
///
/// Returns `None` if the span is entirely whitespace. Used when a model emits
/// word-aligned spans that include trailing punctuation spacing.
pub fn trim_span(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let raw = text.get(start..end)?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lead = raw.len() - raw.trim_start().len();
    Some((start + lead, start + lead + trimmed.len()))
}

/// Punctuation that is never part of an entity but frequently abuts one in
/// prose, JSON, YAML, logs and SQL.
///
/// Shared so every detector uses the same span-boundary convention: the
/// deterministic recognizers strip it from their regex matches and the token
/// classifier strips it from BIO spans (the model happily tags a trailing `,`
/// as `I-IBAN`). Without a common rule the two would disagree by one byte on
/// every entity and exact-match F1 would measure the convention, not the
/// detector.
pub const TRAILING_PUNCT: &[char] = &[
    '.', ',', ';', ':', '!', '?', ')', ']', '}', '"', '\'', '>', '`', '\\',
];

/// Trim trailing characters in `chars` from a byte span, never crossing a
/// char boundary. Returns the tightened range.
pub fn trim_trailing(text: &str, start: usize, mut end: usize, chars: &[char]) -> (usize, usize) {
    while end > start {
        let Some(prev) = text[start..end].chars().next_back() else {
            break;
        };
        if chars.contains(&prev) {
            end -= prev.len_utf8();
        } else {
            break;
        }
    }
    (start, end)
}

/// Merge overlapping/adjacent byte ranges into a sorted, disjoint cover.
///
/// Used by the character-coverage metric and by hybrid span merging. Adjacent
/// (touching) ranges are merged because for redaction purposes `[0,5)` and
/// `[5,9)` cover a contiguous region.
pub fn merge_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.is_empty() {
        return ranges;
    }
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (s, e) in ranges {
        if s >= e {
            continue;
        }
        match out.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => out.push((s, e)),
        }
    }
    out
}

/// Total covered bytes in a set of ranges, merging overlaps first.
pub fn covered_bytes(ranges: Vec<(usize, usize)>) -> usize {
    merge_ranges(ranges).iter().map(|(s, e)| e - s).sum()
}

/// Deterministic ordering for detector output: by position, then by type.
pub fn sort_detections(detections: &mut [Detection]) {
    detections.sort_by_key(Detection::sort_key);
}

/// Drop exact duplicates (same type and same range), keeping the highest score.
///
/// Detectors composed of several recognizers can legitimately produce the same
/// span twice; the evaluator must not see it as two predictions.
pub fn dedup_detections(detections: &mut Vec<Detection>) {
    detections.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.end.cmp(&b.end))
            .then(a.entity_type.cmp(&b.entity_type))
            .then(b.score.total_cmp(&a.score))
    });
    detections
        .dedup_by(|a, b| a.start == b.start && a.end == b.end && a.entity_type == b.entity_type);
}

/// A gold or predicted span reduced to what matching needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypedSpan {
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
}

impl TypedSpan {
    pub fn new(entity_type: EntityType, start: usize, end: usize) -> Self {
        Self {
            entity_type,
            start,
            end,
        }
    }

    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }

    pub fn overlap_len(self, other: TypedSpan) -> usize {
        let lo = self.start.max(other.start);
        let hi = self.end.min(other.end);
        hi.saturating_sub(lo)
    }

    /// Intersection over union on byte ranges. `0.0` when disjoint.
    pub fn iou(self, other: TypedSpan) -> f64 {
        let inter = self.overlap_len(other);
        if inter == 0 {
            return 0.0;
        }
        let union = self.len() + other.len() - inter;
        if union == 0 {
            0.0
        } else {
            inter as f64 / union as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::DetectionSource;

    // Names chosen to exercise 1-, 2- and 3-byte UTF-8 sequences. An
    // ASCII-only suite would pass with every offset function subtly broken.
    const LATIN: &str = "Mikhail Yastrebtsov";
    const CYRILLIC: &str = "Михаил Ястребцов";
    const GERMAN: &str = "Jörg Müller";
    const SPANISH: &str = "José García";
    const FRENCH: &str = "François Dupont";
    const CHINESE: &str = "王小明";

    #[test]
    fn byte_spans_slice_correctly_for_non_ascii_names() {
        // Cyrillic: 2 bytes per letter.
        let text = format!("Автор: {CYRILLIC}, email: a@b.co");
        let start = text.find(CYRILLIC).unwrap();
        let end = start + CYRILLIC.len();
        validate(&text, start, end, false).unwrap();
        assert_eq!(slice(&text, start, end).unwrap(), CYRILLIC);
        assert_eq!(
            CYRILLIC.len(),
            31,
            "16 chars, 15 of them 2-byte, plus a space"
        );
        assert_eq!(CYRILLIC.chars().count(), 16);
    }

    #[test]
    fn every_fixture_name_round_trips_through_char_and_utf16_conversion() {
        for name in [LATIN, CYRILLIC, GERMAN, SPANISH, FRENCH, CHINESE] {
            let text = format!("Contact {name} now");
            let byte_start = text.find(name).unwrap();
            let byte_end = byte_start + name.len();

            // char offsets -> bytes
            let char_start = text[..byte_start].chars().count();
            let char_end = char_start + name.chars().count();
            assert_eq!(char_to_byte(&text, char_start), Some(byte_start), "{name}");
            assert_eq!(char_to_byte(&text, char_end), Some(byte_end), "{name}");

            // the table must agree with the scalar function
            let table = char_to_byte_table(&text);
            assert_eq!(table[char_start], byte_start, "{name}");
            assert_eq!(table[char_end], byte_end, "{name}");
            assert_eq!(*table.last().unwrap(), text.len());
            assert_eq!(table.len(), text.chars().count() + 1);

            // utf16 offsets -> bytes
            let u16_start = text[..byte_start].encode_utf16().count();
            let u16_end = u16_start + name.encode_utf16().count();
            assert_eq!(utf16_to_byte(&text, u16_start), Some(byte_start), "{name}");
            assert_eq!(utf16_to_byte(&text, u16_end), Some(byte_end), "{name}");

            assert_eq!(&text[byte_start..byte_end], name);
        }
    }

    #[test]
    fn char_and_utf16_offsets_diverge_outside_the_bmp() {
        // The exact case that makes an unconverted UTF-16 offset silently wrong.
        let text = "hi 👋 there";
        let emoji_byte = text.find('👋').unwrap();
        assert_eq!(emoji_byte, 3);

        // 'there' begins at char 5 but at UTF-16 unit 6 — the emoji occupies
        // two units. Feeding a char offset to a UTF-16 consumer (or vice versa)
        // silently lands one character off.
        let there_byte = text.find("there").unwrap();
        assert_eq!(char_to_byte(text, 5), Some(there_byte));
        assert_eq!(
            utf16_to_byte(text, 5),
            Some(emoji_byte + 4),
            "the space, not 'there'"
        );
        assert_eq!(utf16_to_byte(text, 6), Some(there_byte));
        // Unit 4 lands between the emoji's surrogates and has no byte offset.
        assert_eq!(
            utf16_to_byte(text, 4),
            None,
            "offset splits the surrogate pair"
        );
    }

    #[test]
    fn validate_rejects_interior_byte_offsets() {
        let text = CYRILLIC; // every char is 2 bytes, so odd offsets are interior
        assert_eq!(
            validate(text, 1, 4, false),
            Err(SpanError::StartNotCharBoundary { start: 1 })
        );
        assert_eq!(
            validate(text, 0, 3, false),
            Err(SpanError::EndNotCharBoundary { end: 3 })
        );
        validate(text, 0, 4, false).unwrap();
    }

    #[test]
    fn validate_rejects_inverted_out_of_bounds_and_empty() {
        let text = "abcdef";
        assert_eq!(
            validate(text, 4, 2, false),
            Err(SpanError::Inverted { start: 4, end: 2 })
        );
        assert_eq!(
            validate(text, 0, 99, false),
            Err(SpanError::OutOfBounds { end: 99, len: 6 })
        );
        assert_eq!(
            validate(text, 3, 3, false),
            Err(SpanError::Empty { start: 3 })
        );
        validate(text, 3, 3, true).unwrap();
        validate(text, 6, 6, true).unwrap();
    }

    #[test]
    fn conversion_helpers_reject_out_of_range() {
        assert_eq!(char_to_byte(CHINESE, 3), Some(CHINESE.len()));
        assert_eq!(char_to_byte(CHINESE, 4), None);
        assert_eq!(utf16_to_byte(CHINESE, 3), Some(CHINESE.len()));
        assert_eq!(utf16_to_byte(CHINESE, 4), None);
        assert_eq!(char_to_byte("", 0), Some(0));
        assert_eq!(char_to_byte("", 1), None);
    }

    #[test]
    fn trim_span_tightens_and_rejects_whitespace_only() {
        let text = "  José García  ";
        assert_eq!(trim_span(text, 0, text.len()), Some((2, 2 + SPANISH.len())));
        assert_eq!(&text[2..2 + SPANISH.len()], SPANISH);
        assert_eq!(trim_span("   ", 0, 3), None);
    }

    #[test]
    fn merge_ranges_merges_overlapping_and_touching() {
        assert_eq!(merge_ranges(vec![(0, 5), (5, 9)]), vec![(0, 9)]);
        assert_eq!(merge_ranges(vec![(0, 5), (3, 9)]), vec![(0, 9)]);
        assert_eq!(merge_ranges(vec![(6, 9), (0, 5)]), vec![(0, 5), (6, 9)]);
        assert_eq!(merge_ranges(vec![(0, 10), (2, 4)]), vec![(0, 10)]);
        assert_eq!(merge_ranges(vec![(3, 3)]), vec![], "empty ranges dropped");
        assert_eq!(covered_bytes(vec![(0, 5), (3, 9), (20, 22)]), 11);
    }

    #[test]
    fn dedup_keeps_highest_score_and_preserves_distinct_types() {
        let src = DetectionSource::recognizer("x");
        let mut v = vec![
            Detection::new(EntityType::Email, 0, 5, 0.4, src.clone()),
            Detection::new(EntityType::Email, 0, 5, 0.9, src.clone()),
            Detection::new(EntityType::Url, 0, 5, 0.7, src.clone()),
            Detection::new(EntityType::Email, 6, 9, 0.5, src.clone()),
        ];
        dedup_detections(&mut v);
        assert_eq!(v.len(), 3);
        let email = v
            .iter()
            .find(|d| d.entity_type == EntityType::Email && d.start == 0)
            .unwrap();
        assert_eq!(email.score, 0.9, "duplicate collapsed to the higher score");
        assert!(v.iter().any(|d| d.entity_type == EntityType::Url));
    }

    #[test]
    fn validate_all_reports_the_offending_index() {
        let src = DetectionSource::recognizer("x");
        // "Jörg": J=0, ö=1..3, r=3, g=4. Ending at 2 lands inside 'ö'.
        let text = GERMAN;
        let bad = vec![
            Detection::new(EntityType::Person, 0, 4, 1.0, src.clone()),
            Detection::new(EntityType::Person, 0, 2, 1.0, src.clone()),
        ];
        let (idx, err) = validate_all(text, &bad).unwrap_err();
        assert_eq!(idx, 1);
        assert_eq!(err, SpanError::EndNotCharBoundary { end: 2 });
    }

    #[test]
    fn iou_is_symmetric_and_bounded() {
        let a = TypedSpan::new(EntityType::Email, 0, 10);
        let b = TypedSpan::new(EntityType::Email, 5, 15);
        assert!((a.iou(b) - b.iou(a)).abs() < f64::EPSILON);
        assert!((a.iou(b) - 5.0 / 15.0).abs() < 1e-12);
        assert_eq!(a.iou(TypedSpan::new(EntityType::Email, 10, 20)), 0.0);
        assert_eq!(a.iou(a), 1.0);
    }
}
