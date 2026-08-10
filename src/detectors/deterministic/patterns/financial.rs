//! Credit card and IBAN recognizers.
//!
//! Both are pattern-plus-checksum. Neither emits anything on a checksum
//! failure, which is what keeps them usable on corpora full of build ids,
//! hashes and integer arrays.

use std::sync::LazyLock;

use regex::Regex;

use super::super::recognizer::{Recognizer, SCORE_VALIDATED, detection};
use super::super::validators as v;
use crate::entity::{Detection, EntityType};

// ---------------------------------------------------------------------------
// Credit card
// ---------------------------------------------------------------------------

/// Maximal run of digits joined by at most one space or dash.
///
/// The `regex` crate has no look-around, so rather than asserting "not preceded
/// by a digit" in the pattern we match *maximal* runs: `find_iter` starts each
/// scan after the previous match, and the run itself is greedy, so a 16-digit
/// window inside a 25-digit build id can never be matched on its own — the
/// whole 25-digit run is, and then fails the length check.
static CARD_RUN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[0-9](?:[ \-]?[0-9])*").expect("card run regex"));

pub struct CreditCardRecognizer;

impl Recognizer for CreditCardRecognizer {
    fn name(&self) -> &'static str {
        "credit_card"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::CreditCard
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        CARD_RUN_RE
            .find_iter(text)
            .filter_map(|m| {
                let candidate = m.as_str();
                // Separators must be consistent: `4111-1111 1111-1111` is
                // almost certainly two unrelated numbers that happen to abut.
                if candidate.contains(' ') && candidate.contains('-') {
                    return None;
                }
                // Both gates. Luhn alone accepts ~1 in 10 random digit runs of
                // the right length; requiring a recognised BIN family *and*
                // that family's length is what makes this safe over logs.
                if !v::luhn_valid(candidate) || v::card_network(candidate).is_none() {
                    return None;
                }
                Some(detection(
                    self.name(),
                    EntityType::CreditCard,
                    m.start(),
                    m.end(),
                    SCORE_VALIDATED,
                ))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IBAN
// ---------------------------------------------------------------------------

static IBAN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Z]{2}[0-9]{2}(?:[ ]?[A-Z0-9]{1,4}){2,8}\b").expect("iban regex")
});

pub struct IbanRecognizer;

impl Recognizer for IbanRecognizer {
    fn name(&self) -> &'static str {
        "iban"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::Iban
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        IBAN_RE
            .find_iter(text)
            .filter_map(|m| {
                let raw = m.as_str();
                // The regex is greedy across spaces and will happily absorb the
                // following word. Retry on progressively shorter prefixes,
                // longest first, and keep the first that checksums.
                let (end, _) = shrink_to_valid(raw)?;
                Some(detection(
                    self.name(),
                    EntityType::Iban,
                    m.start(),
                    m.start() + end,
                    SCORE_VALIDATED,
                ))
            })
            .collect()
    }
}

/// Find the longest prefix of `raw` (ending on a group boundary) that is a
/// valid IBAN. Returns its byte length.
fn shrink_to_valid(raw: &str) -> Option<(usize, &str)> {
    // Candidate cut points: end of string, and before each space.
    let mut cuts: Vec<usize> = raw
        .char_indices()
        .filter(|(_, c)| *c == ' ')
        .map(|(i, _)| i)
        .collect();
    cuts.push(raw.len());
    cuts.sort_unstable_by(|a, b| b.cmp(a)); // longest first

    for cut in cuts {
        let prefix = raw[..cut].trim_end();
        if prefix.len() >= 15 && v::iban_valid(prefix) {
            return Some((prefix.len(), prefix));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found<'a>(r: &dyn Recognizer, text: &'a str) -> Vec<&'a str> {
        r.recognize(text)
            .into_iter()
            .map(|d| &text[d.start..d.end])
            .collect()
    }

    #[test]
    fn card_positive_formats() {
        for (text, want) in [
            ("card 4111111111111111 ok", "4111111111111111"),
            ("card 4111 1111 1111 1111 ok", "4111 1111 1111 1111"),
            ("card 4111-1111-1111-1111 ok", "4111-1111-1111-1111"),
            ("amex 3714 496353 98431 ok", "3714 496353 98431"),
            ("mc 5500005555555559 ok", "5500005555555559"),
        ] {
            assert_eq!(
                found(&CreditCardRecognizer, text),
                vec![want],
                "in {text:?}"
            );
        }
    }

    #[test]
    fn card_requires_both_luhn_and_a_recognised_bin() {
        // Luhn-valid but no BIN family with that length.
        assert!(found(&CreditCardRecognizer, "1234567812345670").is_empty());
        // Correct Visa BIN, broken checksum.
        assert!(found(&CreditCardRecognizer, "4111111111111112").is_empty());
    }

    #[test]
    fn card_negative_cases_from_the_developer_corpus() {
        for t in [
            "Request ID: 550e8400-e29b-41d4-a716-446655440000",
            "SHA256: 9f86d081884c7d659a2feaa0c55ad0159abc12345678901234567890abcdef",
            "build 1234567890123456789012345",
            "0000000000000000",
            "1111111111111111",
            "array [1111, 2222, 3333, 4444]",
            "timestamp 1754630400000",
        ] {
            assert!(
                found(&CreditCardRecognizer, t).is_empty(),
                "false positive in {t:?}: {:?}",
                found(&CreditCardRecognizer, t)
            );
        }
    }

    #[test]
    fn card_rejects_mixed_separators() {
        assert!(found(&CreditCardRecognizer, "4111-1111 1111-1111").is_empty());
    }

    #[test]
    fn card_does_not_match_a_digit_run_inside_a_longer_number() {
        // The lookarounds must prevent a 16-digit window inside a 20-digit id.
        assert!(found(&CreditCardRecognizer, "id 41111111111111114111").is_empty());
    }

    #[test]
    fn iban_positive_formats_and_exact_spans() {
        let t = "Transfer to GB82 WEST 1234 5698 7654 32 today";
        assert_eq!(
            found(&IbanRecognizer, t),
            vec!["GB82 WEST 1234 5698 7654 32"]
        );

        let t2 = "IBAN: DE89370400440532013000.";
        assert_eq!(found(&IbanRecognizer, t2), vec!["DE89370400440532013000"]);

        let t3 = "iban=FR1420041010050500013M02606&x=1";
        assert_eq!(
            found(&IbanRecognizer, t3),
            vec!["FR1420041010050500013M02606"]
        );
    }

    #[test]
    fn iban_span_does_not_absorb_the_following_word() {
        // The greedy group pattern would otherwise swallow " today" fragments.
        let t = "GB82 WEST 1234 5698 7654 32 PLEASE CONFIRM";
        let got = found(&IbanRecognizer, t);
        assert_eq!(got, vec!["GB82 WEST 1234 5698 7654 32"]);
    }

    #[test]
    fn iban_negative_cases() {
        for t in [
            "GB82WEST12345698765433", // checksum broken
            "ZZ89370400440532013000", // unknown country
            "DE8937040044053201300",  // wrong length for DE
            "AB12 CDEF GHIJ",         // not a country code / too short
            "US12 3456 7890 1234",    // US has no IBAN
        ] {
            assert!(
                found(&IbanRecognizer, t).is_empty(),
                "false positive in {t:?}: {:?}",
                found(&IbanRecognizer, t)
            );
        }
    }

    #[test]
    fn recognizers_report_validated_confidence() {
        let d = &CreditCardRecognizer.recognize("4111111111111111")[0];
        assert_eq!(d.score, SCORE_VALIDATED);
        assert_eq!(d.source.name(), "credit_card");
    }
}
