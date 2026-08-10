//! Email, URL, IP and phone recognizers.

use std::sync::LazyLock;

use regex::Regex;

use super::super::recognizer::{
    Recognizer, SCORE_PATTERN, SCORE_STRUCTURAL, SCORE_VALIDATED, TRAILING_PUNCT, detection,
    trim_trailing,
};
use super::super::validators as v;
use crate::entity::{Detection, EntityType};

// ---------------------------------------------------------------------------
// Email
// ---------------------------------------------------------------------------

static EMAIL_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Deliberately not RFC 5322. That grammar accepts quoted local parts and
    // comments, which in a DLP corpus matches far more noise than it gains.
    Regex::new(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?)*\.[a-z]{2,24}\b")
        .expect("email regex")
});

pub struct EmailRecognizer;

impl Recognizer for EmailRecognizer {
    fn name(&self) -> &'static str {
        "email"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::Email
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        EMAIL_RE
            .find_iter(text)
            .filter_map(|m| {
                let (s, e) = trim_trailing(text, m.start(), m.end(), TRAILING_PUNCT);
                let candidate = &text[s..e];
                // A doubled dot is not a deliverable local part.
                if candidate.contains("..") {
                    return None;
                }
                // `\b` starts the match after any leading dots, so the guard
                // has to look at the byte *before* the match, not inside it.
                // Without this, `..dots@example.com` yields `dots@example.com`.
                if text[..s].ends_with('.') {
                    return None;
                }
                Some(detection(
                    self.name(),
                    EntityType::Email,
                    s,
                    e,
                    SCORE_VALIDATED,
                ))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// URL
// ---------------------------------------------------------------------------

static URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:https?|ftp|ftps|ws|wss)://[^\s<>\x22'`\\]+").expect("url regex")
});

pub struct UrlRecognizer;

impl Recognizer for UrlRecognizer {
    fn name(&self) -> &'static str {
        "url"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::Url
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        URL_RE
            .find_iter(text)
            .filter_map(|m| {
                let (s, e) = trim_trailing(text, m.start(), m.end(), TRAILING_PUNCT);
                // Structural check rather than a bigger regex: there must be a
                // non-empty authority after `://`.
                let candidate = &text[s..e];
                let authority = candidate.split_once("://")?.1;
                let host = authority
                    .split(['/', '?', '#'])
                    .next()
                    .unwrap_or_default()
                    .rsplit('@')
                    .next()
                    .unwrap_or_default();
                if host.is_empty() {
                    return None;
                }
                Some(detection(
                    self.name(),
                    EntityType::Url,
                    s,
                    e,
                    SCORE_STRUCTURAL,
                ))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// IP addresses
// ---------------------------------------------------------------------------

static IPV4_CANDIDATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{1,3}(?:\.\d{1,3}){3}\b").expect("ipv4 regex"));

// Loose candidate shape; `Ipv6Addr::from_str` does the real work.
static IPV6_CANDIDATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?:%[0-9a-z]+)?").expect("ipv6 regex")
});

/// How the detector treats private, loopback and link-local addresses.
///
/// Explicit rather than accidental: `10.20.30.40` in a log line is PII under
/// some policies and pure noise under others, and the benchmark records which
/// setting produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrivateIpPolicy {
    /// Do not report private/loopback/link-local addresses. The default:
    /// `127.0.0.1:8080` appears constantly in developer prompts and flagging it
    /// destroys precision for no privacy gain.
    #[default]
    Ignore,
    /// Report them like any other address.
    Report,
}

pub struct IpRecognizer {
    pub policy: PrivateIpPolicy,
}

impl Recognizer for IpRecognizer {
    fn name(&self) -> &'static str {
        "ip_address"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::IpAddress
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        let mut out = Vec::new();

        for m in IPV4_CANDIDATE.find_iter(text) {
            // Parse rather than trust the regex: this rejects 999.1.1.1,
            // 256.0.0.1 and 01.2.3.4, all of which the pattern accepts.
            if !v::ipv4_valid(m.as_str()) {
                continue;
            }
            if self.policy == PrivateIpPolicy::Ignore && v::ip_is_private(m.as_str()) {
                continue;
            }
            out.push(detection(
                self.name(),
                EntityType::IpAddress,
                m.start(),
                m.end(),
                SCORE_VALIDATED,
            ));
        }

        for m in IPV6_CANDIDATE.find_iter(text) {
            let raw = m.as_str();
            // Strip a zone index before parsing; `fe80::1%eth0` is a valid
            // address but `Ipv6Addr::from_str` rejects the suffix.
            let bare = raw.split('%').next().unwrap_or(raw);
            if bare.len() < 3 || !v::ipv6_valid(bare) {
                continue;
            }
            if self.policy == PrivateIpPolicy::Ignore && v::ip_is_private(bare) {
                continue;
            }
            out.push(detection(
                self.name(),
                EntityType::IpAddress,
                m.start(),
                m.start() + bare.len(),
                SCORE_VALIDATED,
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Phone
// ---------------------------------------------------------------------------

static PHONE_RE: LazyLock<Regex> = LazyLock::new(|| {
    // E.164-ish and common national formats. Requires either a leading `+` or
    // grouping punctuation: a bare run of ten digits is far more often an id,
    // a timestamp or a build number than a phone number.
    Regex::new(
        r"(?x)
        (?:
            \+\d{1,3}[\s.\-]?(?:\(\d{1,4}\)[\s.\-]?)?\d{1,4}(?:[\s.\-]\d{2,4}){1,4}
          | \(\d{3}\)[\s.\-]?\d{3}[\s.\-]?\d{4}
          | \b\d{3}[.\-]\d{3}[.\-]\d{4}\b
          | \b\d{3}\s\d{3}\s\d{4}\b
        )",
    )
    .expect("phone regex")
});

pub struct PhoneRecognizer;

impl Recognizer for PhoneRecognizer {
    fn name(&self) -> &'static str {
        "phone"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::Phone
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        PHONE_RE
            .find_iter(text)
            .filter_map(|m| {
                let (s, e) = trim_trailing(text, m.start(), m.end(), &['.', ',', ';', ':', '-']);
                let candidate = &text[s..e];
                let digits = candidate.chars().filter(char::is_ascii_digit).count();
                // E.164 allows 7..15 digits including the country code.
                if !(7..=15).contains(&digits) {
                    return None;
                }
                // Reject dotted quads that already parsed as an IP; an
                // unqualified `192.168.1.1` must not also be a phone number.
                if v::ipv4_valid(candidate) {
                    return None;
                }
                Some(detection(
                    self.name(),
                    EntityType::Phone,
                    s,
                    e,
                    SCORE_PATTERN,
                ))
            })
            .collect()
    }
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

    // --- email ------------------------------------------------------------

    #[test]
    fn email_positive_cases_across_formats() {
        assert_eq!(
            found(&EmailRecognizer, "Contact john@example.com."),
            vec!["john@example.com"]
        );
        assert_eq!(
            found(
                &EmailRecognizer,
                r#"{"email": "a.b+tag@sub.example.co.uk"}"#
            ),
            vec!["a.b+tag@sub.example.co.uk"]
        );
        assert_eq!(
            found(&EmailRecognizer, "let email = \"customer@example.com\";"),
            vec!["customer@example.com"]
        );
        assert_eq!(
            found(
                &EmailRecognizer,
                "2026-08-08 user=john@example.com ip=10.20.30.40 login failed"
            ),
            vec!["john@example.com"]
        );
        assert_eq!(
            found(&EmailRecognizer, "WHERE email = 'john@example.com';"),
            vec!["john@example.com"]
        );
    }

    #[test]
    fn email_negative_cases() {
        // Each of these is matched by a careless email regex.
        for t in [
            "version 1.2@3.4",
            "@handle mentioned it",
            "path/to/file.tar.gz",
            "a@b",   // no TLD
            "a@b.c", // 1-char TLD
            "..dots@example.com",
            "npm i @scope/pkg@1.2.3",
        ] {
            assert!(
                found(&EmailRecognizer, t).is_empty(),
                "false positive in {t:?}: {:?}",
                found(&EmailRecognizer, t)
            );
        }
    }

    #[test]
    fn email_span_is_exact_next_to_punctuation() {
        let t = "Reach me at (john@example.com), thanks.";
        let d = &EmailRecognizer.recognize(t)[0];
        assert_eq!(&t[d.start..d.end], "john@example.com");
    }

    #[test]
    fn email_offsets_are_correct_after_multibyte_text() {
        let t = "Свяжитесь: john@example.com";
        let d = &EmailRecognizer.recognize(t)[0];
        assert_eq!(&t[d.start..d.end], "john@example.com");
        assert_eq!(d.start, t.find("john").unwrap());
    }

    // --- url ---------------------------------------------------------------

    #[test]
    fn url_positive_and_negative() {
        assert_eq!(
            found(&UrlRecognizer, "See https://example.com/a?b=1#c."),
            vec!["https://example.com/a?b=1#c"]
        );
        assert_eq!(
            found(&UrlRecognizer, "ws://localhost:8080/socket"),
            vec!["ws://localhost:8080/socket"]
        );
        for t in ["example.com", "not a url", "://missing-scheme", "https://"] {
            assert!(
                found(&UrlRecognizer, t).is_empty(),
                "false positive in {t:?}"
            );
        }
    }

    // --- ip -----------------------------------------------------------------

    #[test]
    fn ipv4_validated_not_regexed() {
        let r = IpRecognizer {
            policy: PrivateIpPolicy::Report,
        };
        assert_eq!(found(&r, "from 192.0.2.44 ok"), vec!["192.0.2.44"]);
        // All matched by `\d{1,3}(\.\d{1,3}){3}`, all rejected by the parser.
        for t in ["999.999.999.999", "256.1.1.1", "01.2.3.4"] {
            assert!(found(&r, t).is_empty(), "false positive in {t:?}");
        }
        // A version number is not an address.
        assert!(found(&r, "Version 1.23.4567 was deployed.").is_empty());
    }

    #[test]
    fn private_ip_policy_is_explicit_in_both_directions() {
        let text = "connect 127.0.0.1:8080 and 10.20.30.40 and 8.8.8.8";
        let ignore = IpRecognizer {
            policy: PrivateIpPolicy::Ignore,
        };
        let report = IpRecognizer {
            policy: PrivateIpPolicy::Report,
        };
        assert_eq!(found(&ignore, text), vec!["8.8.8.8"]);
        assert_eq!(
            found(&report, text),
            vec!["127.0.0.1", "10.20.30.40", "8.8.8.8"]
        );
    }

    #[test]
    fn ipv6_is_recognised_and_zone_index_excluded_from_the_span() {
        let r = IpRecognizer {
            policy: PrivateIpPolicy::Report,
        };
        assert_eq!(
            found(&r, "addr 2606:4700::1111 here"),
            vec!["2606:4700::1111"]
        );
        let t = "iface fe80::1%eth0 up";
        let d = &r.recognize(t)[0];
        assert_eq!(
            &t[d.start..d.end],
            "fe80::1",
            "zone index is not part of the address"
        );
    }

    #[test]
    fn timestamps_are_not_ipv6() {
        let r = IpRecognizer {
            policy: PrivateIpPolicy::Report,
        };
        for t in ["12:34:56", "2026-08-08 10:20:30 INFO", "ratio 1:2:3"] {
            assert!(
                found(&r, t).is_empty(),
                "false positive in {t:?}: {:?}",
                found(&r, t)
            );
        }
    }

    // --- phone ---------------------------------------------------------------

    #[test]
    fn phone_positive_formats() {
        for (text, want) in [
            ("call +1-415-555-1234 now", "+1-415-555-1234"),
            ("call +1 415 555 1234 now", "+1 415 555 1234"),
            ("call 415.555.1234 now", "415.555.1234"),
            ("call (415) 555-1234 now", "(415) 555-1234"),
            ("phone: +44 20 7946 0958.", "+44 20 7946 0958"),
        ] {
            assert_eq!(found(&PhoneRecognizer, text), vec![want], "in {text:?}");
        }
    }

    #[test]
    fn phone_negative_cases_from_the_developer_corpus() {
        for t in [
            "Fix issue #4411.",
            "Version 1.23.4567 was deployed.",
            "SHA256: 9f86d081884c7d659a2feaa0c55ad015",
            "127.0.0.1:8080",
            "192.168.1.1",
            "array [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "build 20260808",
            "port 8080",
            "id 1234567890",
        ] {
            assert!(
                found(&PhoneRecognizer, t).is_empty(),
                "false positive in {t:?}: {:?}",
                found(&PhoneRecognizer, t)
            );
        }
    }

    #[test]
    fn phone_rejects_digit_counts_outside_e164() {
        assert!(
            found(&PhoneRecognizer, "+1 2 3").is_empty(),
            "too few digits"
        );
        assert!(
            found(&PhoneRecognizer, "+1 2345 6789 0123 4567 8901").is_empty(),
            "more than 15 digits"
        );
    }
}
