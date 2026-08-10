//! The recognizer abstraction.
//!
//! `candidate pattern -> optional validator -> canonical Detection`.
//!
//! Splitting it this way means the expensive-to-get-right part (validation) is
//! unit-testable without any regex, and adding a recognizer never touches a
//! shared monolithic function.

use crate::entity::{Detection, DetectionSource, EntityType};

/// A single-purpose deterministic recognizer.
pub trait Recognizer: Send + Sync {
    /// Stable name, used as detection provenance and in error reports.
    fn name(&self) -> &'static str;

    /// The canonical type this recognizer produces. Recognizers that can emit
    /// more than one type (the connection-string one emits
    /// `CONNECTION_STRING` and sometimes `PASSWORD`) report their primary type
    /// here and set the real type per detection.
    fn entity_type(&self) -> EntityType;

    /// Find candidates in `text`. Offsets are UTF-8 byte offsets.
    ///
    /// Implementations return raw findings; deduplication, sorting and span
    /// validation happen once in [`super::DeterministicDetector::detect`].
    fn recognize(&self, text: &str) -> Vec<Detection>;
}

/// Confidence assigned when a structural validator confirmed the match
/// (Luhn, IBAN checksum, `Ipv4Addr::from_str`, a decodable JWT header).
pub const SCORE_VALIDATED: f32 = 1.0;

/// Confidence for a high-confidence vendor-specific pattern with no separate
/// validator — e.g. a PEM block delimiter pair. Distinguishable from
/// `SCORE_VALIDATED` so the hybrid detector can prefer verified matches.
pub const SCORE_STRUCTURAL: f32 = 0.95;

/// Confidence for a pattern-only match. Kept below the others so a threshold
/// sweep over a hybrid detector can trade these away first.
pub const SCORE_PATTERN: f32 = 0.80;

/// Helper for building a detection with this recognizer's provenance.
pub fn detection(
    name: &'static str,
    entity_type: EntityType,
    start: usize,
    end: usize,
    score: f32,
) -> Detection {
    Detection::new(
        entity_type,
        start,
        end,
        score,
        DetectionSource::recognizer(name),
    )
}

/// The span-boundary convention, shared with the ONNX detectors.
///
/// `Contact john@example.com.` must yield `john@example.com`, not a span with a
/// trailing period. Both the regex recognizers and the token classifier apply
/// the same rule — otherwise the two would disagree by one byte on every
/// entity and exact-match F1 would be measuring the convention rather than the
/// detector. Lives in [`crate::span`] now that there is a second user.
pub use crate::span::{TRAILING_PUNCT, trim_trailing};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_trailing_strips_prose_punctuation() {
        let t = "see john@example.com.";
        let (s, e) = trim_trailing(t, 4, t.len(), TRAILING_PUNCT);
        assert_eq!(&t[s..e], "john@example.com");
    }

    #[test]
    fn trim_trailing_strips_a_run_and_stops_at_content() {
        let t = r#"{"email": "a@b.co"},"#;
        let start = t.find("a@b.co").unwrap();
        let (s, e) = trim_trailing(t, start, t.len(), TRAILING_PUNCT);
        assert_eq!(&t[s..e], "a@b.co");
    }

    #[test]
    fn trim_trailing_never_splits_a_multibyte_char_and_cannot_underflow() {
        let t = "Ястребцов…";
        let (s, e) = trim_trailing(t, 0, t.len(), &['…']);
        assert_eq!(&t[s..e], "Ястребцов");
        assert!(t.is_char_boundary(e));

        // An all-punctuation span collapses to empty rather than panicking.
        let p = "...";
        let (s, e) = trim_trailing(p, 0, 3, TRAILING_PUNCT);
        assert_eq!((s, e), (0, 0));
    }
}
