//! Deterministic (rule-based) PII detector.
//!
//! Not a straw-man baseline. On the entity types where structure exists — cards,
//! IBANs, IPs, JWTs, vendor keys — a validated recognizer beats any model on
//! both precision and latency, and a production DLP agent should run these
//! regardless of which model it also runs. The benchmark's job is to show
//! exactly where that stops being true.
//!
//! Architecture: `pattern -> validator -> canonical Detection`, one recognizer
//! per family, composed here. There is no monolithic regex function.

pub mod patterns;
pub mod recognizer;
pub mod validators;

use crate::detector::{Detector, finish};
use crate::entity::Detection;
use crate::error::DetectorError;

pub use patterns::contact::PrivateIpPolicy;
pub use recognizer::Recognizer;

/// Configuration for the deterministic detector.
#[derive(Debug, Clone, Default)]
pub struct DeterministicConfig {
    /// Whether private/loopback/link-local IPs count as PII. Policy-dependent,
    /// so it is explicit and recorded in the run metadata.
    pub private_ip_policy: PrivateIpPolicy,
}

pub struct DeterministicDetector {
    recognizers: Vec<Box<dyn Recognizer>>,
    config: DeterministicConfig,
}

impl DeterministicDetector {
    pub fn new(config: DeterministicConfig) -> Self {
        let recognizers: Vec<Box<dyn Recognizer>> = vec![
            Box::new(patterns::contact::EmailRecognizer),
            Box::new(patterns::contact::UrlRecognizer),
            Box::new(patterns::contact::IpRecognizer {
                policy: config.private_ip_policy,
            }),
            Box::new(patterns::contact::PhoneRecognizer),
            Box::new(patterns::financial::CreditCardRecognizer),
            Box::new(patterns::financial::IbanRecognizer),
            Box::new(patterns::secrets::JwtRecognizer),
            Box::new(patterns::secrets::AwsAccessKeyRecognizer),
            Box::new(patterns::secrets::GithubTokenRecognizer),
            Box::new(patterns::secrets::VendorKeyRecognizer),
            Box::new(patterns::secrets::BearerTokenRecognizer),
            Box::new(patterns::secrets::PemPrivateKeyRecognizer),
            Box::new(patterns::secrets::ConnectionStringRecognizer),
            Box::new(patterns::secrets::KeyValueSecretRecognizer),
        ];
        Self {
            recognizers,
            config,
        }
    }

    pub fn config(&self) -> &DeterministicConfig {
        &self.config
    }

    pub fn recognizer_names(&self) -> Vec<&'static str> {
        self.recognizers.iter().map(|r| r.name()).collect()
    }
}

impl Default for DeterministicDetector {
    fn default() -> Self {
        Self::new(DeterministicConfig::default())
    }
}

impl Detector for DeterministicDetector {
    fn name(&self) -> &str {
        "deterministic"
    }

    fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
        let mut out = Vec::new();
        for r in &self.recognizers {
            out.extend(r.recognize(text));
        }
        // Recognizers may legitimately overlap: a connection string contains a
        // password, a URL may contain an email. Only exact duplicates are
        // collapsed; genuine nesting is preserved and left for the hybrid
        // detector's merge policy to resolve.
        finish(text, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityType::{self, *};

    fn detect(text: &str) -> Vec<Detection> {
        DeterministicDetector::default()
            .detect(text)
            .expect("detection must not fail")
    }

    fn types(text: &str) -> Vec<EntityType> {
        let mut t: Vec<_> = detect(text).into_iter().map(|d| d.entity_type).collect();
        t.sort_unstable();
        t.dedup();
        t
    }

    #[test]
    fn every_recognizer_has_a_unique_name() {
        let names = DeterministicDetector::default().recognizer_names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            names.len(),
            sorted.len(),
            "duplicate recognizer name in {names:?}"
        );
    }

    #[test]
    fn natural_language_prompt() {
        let t = "Please contact John Smith at john@example.com or +1-415-555-1234.";
        assert_eq!(types(t), vec![Email, Phone]);
        // PERSON is out of reach for rules — that is the gap the model fills,
        // and the benchmark should show it rather than fake it.
        assert!(!types(t).contains(&Person));
    }

    #[test]
    fn config_file_prompt() {
        let t = "Can you debug this config?\n\nAWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n\
                 DATABASE_URL=postgres://admin:s3cret99@prod-db.internal/main\n";
        let ty = types(t);
        assert!(ty.contains(&ApiKey), "{ty:?}");
        assert!(ty.contains(&ConnectionString), "{ty:?}");
        assert!(ty.contains(&Password), "{ty:?}");
    }

    #[test]
    fn json_yaml_sql_and_log_shapes() {
        let json = r#"{"customer": "John Smith", "email": "john@example.com"}"#;
        assert!(types(json).contains(&Email));

        let yaml = "customer:\n  name: John Smith\n  phone: +1-415-555-1234\n";
        assert!(types(yaml).contains(&Phone));

        let sql = "SELECT * FROM customers WHERE email = 'john@example.com';";
        let d = detect(sql);
        assert_eq!(
            &sql[d[0].start..d[0].end],
            "john@example.com",
            "quote must not be included"
        );

        let log = "2026-08-08 user=john@example.com ip=8.8.8.8 login failed";
        assert_eq!(types(log), vec![Email, IpAddress]);
    }

    #[test]
    fn source_code_shape() {
        let code = "let email = \"customer@example.com\";\nlet key = \"sk-proj-abcdefghijklmnopqrstuvwxyz01\";";
        let ty = types(code);
        assert!(ty.contains(&Email), "{ty:?}");
        assert!(ty.contains(&ApiKey), "{ty:?}");
    }

    /// The single most important test in this module: a detector that flags
    /// these is unusable in front of a coding assistant.
    #[test]
    fn negative_corpus_produces_no_detections() {
        let negatives = [
            "Fix issue #4411.",
            "Version 1.23.4567 was deployed.",
            "Request ID: 550e8400-e29b-41d4-a716-446655440000",
            "SHA256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "127.0.0.1:8080",
            "Listening on port 8080, timeout 30000ms.",
            "const nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];",
            "Build 20260808.3 finished in 1.234s.",
            "See the docs at YOUR_API_KEY placeholder.",
            "git commit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b",
            "Acme Corporation and Globex Inc. announced a merger.",
            "Set MAX_RETRIES=5 and TIMEOUT=30",
            "npm install @scope/package@1.2.3",
            "Colours: #ff00aa, #00ffcc",
        ];
        for t in negatives {
            let got = detect(t);
            assert!(
                got.is_empty(),
                "false positive in {t:?}: {:?}",
                got.iter()
                    .map(|d| (d.entity_type, &t[d.start..d.end]))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// The limit of a context-free detector, stated as a test rather than left
    /// as a surprise: a Luhn-valid PAN in a sentence about array indices is
    /// indistinguishable from a real one without semantics. Rules will flag it;
    /// only a model has a chance of not doing so. This is one of the concrete
    /// things the benchmark is meant to measure.
    #[test]
    fn a_valid_pan_in_innocent_prose_is_detected_and_that_is_expected() {
        let t = "The array index is 4111111111111111 elements long.";
        let got = detect(t);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].entity_type, CreditCard);
    }

    #[test]
    fn private_ip_policy_changes_the_negative_verdict_deliberately() {
        let t = "service bound to 127.0.0.1:8080 and 10.20.30.40";
        assert!(detect(t).is_empty(), "default policy ignores private IPs");

        let reporting = DeterministicDetector::new(DeterministicConfig {
            private_ip_policy: PrivateIpPolicy::Report,
        });
        assert_eq!(reporting.detect(t).unwrap().len(), 2);
    }

    #[test]
    fn all_spans_validate_and_are_sorted() {
        let t = "Mail john@example.com, card 4111 1111 1111 1111, iban DE89370400440532013000, \
                 ip 8.8.8.8, url https://example.com/x";
        let ds = detect(t);
        assert!(ds.len() >= 5, "got {ds:?}");
        crate::span::validate_all(t, &ds).expect("all spans must validate");
        assert!(
            ds.windows(2).all(|w| w[0].sort_key() <= w[1].sort_key()),
            "output not sorted"
        );
    }

    #[test]
    fn unicode_document_offsets_are_byte_exact() {
        let t = "Свяжитесь с Михаил Ястребцов: john@example.com, карта 4111 1111 1111 1111";
        let ds = detect(t);
        crate::span::validate_all(t, &ds).unwrap();
        let email = ds.iter().find(|d| d.entity_type == Email).unwrap();
        assert_eq!(&t[email.start..email.end], "john@example.com");
        let card = ds.iter().find(|d| d.entity_type == CreditCard).unwrap();
        assert_eq!(&t[card.start..card.end], "4111 1111 1111 1111");
    }

    #[test]
    fn empty_and_whitespace_documents_are_safe() {
        assert!(detect("").is_empty());
        assert!(detect("   \n\t  ").is_empty());
    }

    #[test]
    fn detector_reports_no_threshold() {
        // A sweep over this detector would be a flat line; the trait says so
        // explicitly rather than producing one.
        let d = DeterministicDetector::default();
        assert_eq!(d.threshold(), None);
        assert!(d.with_threshold(0.5).is_none());
    }

    #[test]
    fn repeated_detections_are_deduplicated_but_nesting_is_preserved() {
        // The connection string and its password overlap by design.
        let t = "url=postgres://admin:s3cret99@db.internal/main";
        let ds = detect(t);
        let conn = ds
            .iter()
            .filter(|d| d.entity_type == ConnectionString)
            .count();
        let pw = ds.iter().filter(|d| d.entity_type == Password).count();
        assert_eq!((conn, pw), (1, 1), "nested detections both survive: {ds:?}");
    }
}
