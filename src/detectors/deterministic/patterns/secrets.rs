//! Credential and secret recognizers.
//!
//! Vendor-specific rules are aggressive because their prefixes are unambiguous.
//! Generic rules are deliberately conservative: a loose "high-entropy string"
//! rule flags every hash, UUID and minified bundle in a developer's prompt, and
//! a DLP tool that blocks those gets switched off.

use std::sync::LazyLock;

use regex::Regex;

use super::super::recognizer::{
    Recognizer, SCORE_PATTERN, SCORE_STRUCTURAL, SCORE_VALIDATED, TRAILING_PUNCT, detection,
    trim_trailing,
};
use super::super::validators as v;
use crate::entity::{Detection, EntityType};

// ---------------------------------------------------------------------------
// JWT
// ---------------------------------------------------------------------------

static JWT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{2,}\.[A-Za-z0-9_\-]*").expect("jwt regex")
});

pub struct JwtRecognizer;

impl Recognizer for JwtRecognizer {
    fn name(&self) -> &'static str {
        "jwt"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::AccessToken
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        JWT_RE
            .find_iter(text)
            .filter(|m| v::jwt_valid(m.as_str()))
            .map(|m| {
                detection(
                    self.name(),
                    EntityType::AccessToken,
                    m.start(),
                    m.end(),
                    SCORE_VALIDATED,
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Vendor keys and tokens
// ---------------------------------------------------------------------------

static AWS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:AKIA|ASIA|ABIA|ACCA|AIDA|AROA|AIPA)[A-Z0-9]{16}\b").expect("aws regex")
});

static GITHUB_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b")
        .expect("github regex")
});

/// Vendor prefixes whose mere presence is enough. Each entry is
/// `(recognizer name, regex, canonical type)`.
static VENDOR_KEYS: LazyLock<Vec<(&'static str, Regex, EntityType)>> = LazyLock::new(|| {
    vec![
        (
            "openai_key",
            Regex::new(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b").unwrap(),
            EntityType::ApiKey,
        ),
        (
            "anthropic_key",
            Regex::new(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b").unwrap(),
            EntityType::ApiKey,
        ),
        (
            "slack_token",
            Regex::new(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b").unwrap(),
            EntityType::AccessToken,
        ),
        (
            "google_api_key",
            Regex::new(r"\bAIza[A-Za-z0-9_\-]{35}\b").unwrap(),
            EntityType::ApiKey,
        ),
        (
            "stripe_key",
            Regex::new(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b").unwrap(),
            EntityType::ApiKey,
        ),
        (
            "twilio_sid",
            Regex::new(r"\bAC[a-f0-9]{32}\b").unwrap(),
            EntityType::ApiKey,
        ),
        (
            "sendgrid_key",
            Regex::new(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}\b").unwrap(),
            EntityType::ApiKey,
        ),
        (
            "npm_token",
            Regex::new(r"\bnpm_[A-Za-z0-9]{36}\b").unwrap(),
            EntityType::AccessToken,
        ),
    ]
});

pub struct AwsAccessKeyRecognizer;

impl Recognizer for AwsAccessKeyRecognizer {
    fn name(&self) -> &'static str {
        "aws_access_key"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::ApiKey
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        AWS_RE
            .find_iter(text)
            .filter(|m| v::aws_access_key_valid(m.as_str()))
            .map(|m| {
                detection(
                    self.name(),
                    EntityType::ApiKey,
                    m.start(),
                    m.end(),
                    SCORE_VALIDATED,
                )
            })
            .collect()
    }
}

pub struct GithubTokenRecognizer;

impl Recognizer for GithubTokenRecognizer {
    fn name(&self) -> &'static str {
        "github_token"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::AccessToken
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        GITHUB_RE
            .find_iter(text)
            .filter(|m| v::github_token_valid(m.as_str()))
            .map(|m| {
                detection(
                    self.name(),
                    EntityType::AccessToken,
                    m.start(),
                    m.end(),
                    SCORE_VALIDATED,
                )
            })
            .collect()
    }
}

pub struct VendorKeyRecognizer;

impl Recognizer for VendorKeyRecognizer {
    fn name(&self) -> &'static str {
        "vendor_key"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::ApiKey
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        let mut out = Vec::new();
        for (name, re, ty) in VENDOR_KEYS.iter() {
            for m in re.find_iter(text) {
                out.push(detection(name, *ty, m.start(), m.end(), SCORE_VALIDATED));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Bearer token
// ---------------------------------------------------------------------------

static BEARER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bbearer\s+([A-Za-z0-9_\-.=+/]{16,})").expect("bearer regex")
});

pub struct BearerTokenRecognizer;

impl Recognizer for BearerTokenRecognizer {
    fn name(&self) -> &'static str {
        "bearer_token"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::AccessToken
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        BEARER_RE
            .captures_iter(text)
            .filter_map(|c| {
                let g = c.get(1)?;
                let (s, e) = trim_trailing(text, g.start(), g.end(), TRAILING_PUNCT);
                let token = &text[s..e];
                // The span is the token, not the `Bearer ` prefix: that is what
                // a redactor needs to replace.
                if token.len() < 16 {
                    return None;
                }
                // Documentation placeholders are the dominant false positive
                // for this rule.
                if is_placeholder(token) {
                    return None;
                }
                Some(detection(
                    self.name(),
                    EntityType::AccessToken,
                    s,
                    e,
                    SCORE_STRUCTURAL,
                ))
            })
            .collect()
    }
}

/// Common documentation stand-ins that must not be reported.
fn is_placeholder(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    const NEEDLES: &[&str] = &[
        "your_token",
        "your-token",
        "yourtoken",
        "<token>",
        "xxxxx",
        "example",
        "placeholder",
        "changeme",
        "redacted",
        "replace_me",
        "todo",
        "dummy",
        "<your",
        "insert_",
        "abc123",
        "test_token",
        "fake",
    ];
    NEEDLES.iter().any(|n| lower.contains(n)) || s.chars().all(|c| c == 'x' || c == 'X')
}

// ---------------------------------------------------------------------------
// PEM private key
// ---------------------------------------------------------------------------

static PEM_BEGIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-----BEGIN ((?:[A-Z0-9]+ )*PRIVATE KEY)-----").expect("pem regex")
});

/// Cap on how far past a BEGIN marker we look for its END.
///
/// Without a bound, an unterminated `-----BEGIN PRIVATE KEY-----` would emit a
/// span covering the entire rest of the document — exactly the runaway the task
/// warns about.
const PEM_MAX_BLOCK_BYTES: usize = 16 * 1024;

pub struct PemPrivateKeyRecognizer;

impl Recognizer for PemPrivateKeyRecognizer {
    fn name(&self) -> &'static str {
        "pem_private_key"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::PrivateKey
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        let mut out = Vec::new();
        for c in PEM_BEGIN.captures_iter(text) {
            let whole = c.get(0).expect("group 0 always present");
            let kind = c.get(1).expect("group 1 in the pattern").as_str();
            let end_marker = format!("-----END {kind}-----");

            let search_from = whole.end();
            let window_end = (search_from + PEM_MAX_BLOCK_BYTES).min(text.len());
            let window_end = ceil_boundary(text, window_end);

            let Some(rel) = text[search_from..window_end].find(&end_marker) else {
                // Unterminated block: report only the marker itself. Still a
                // strong signal, but it cannot swallow the document.
                out.push(detection(
                    self.name(),
                    EntityType::PrivateKey,
                    whole.start(),
                    whole.end(),
                    SCORE_PATTERN,
                ));
                continue;
            };

            out.push(detection(
                self.name(),
                EntityType::PrivateKey,
                whole.start(),
                search_from + rel + end_marker.len(),
                SCORE_VALIDATED,
            ));
        }
        out
    }
}

fn ceil_boundary(text: &str, mut i: usize) -> usize {
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Database connection string
// ---------------------------------------------------------------------------

static CONN_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Terminates on whitespace, quotes and shell metacharacters so a URI at the
    // start of a config file cannot absorb the rest of it.
    Regex::new(
        r#"(?i)\b(?:postgres|postgresql|mysql|mariadb|mongodb\+srv|mongodb|redis|rediss|amqp|amqps|mssql|sqlserver|clickhouse|cassandra)://[^\s"'`<>\\]+"#,
    )
    .expect("connection string regex")
});

pub struct ConnectionStringRecognizer;

impl Recognizer for ConnectionStringRecognizer {
    fn name(&self) -> &'static str {
        "connection_string"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::ConnectionString
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        let mut out = Vec::new();
        for m in CONN_RE.find_iter(text) {
            let (s, e) = trim_trailing(text, m.start(), m.end(), TRAILING_PUNCT);
            let uri = &text[s..e];
            let Some(rest) = uri.split_once("://").map(|x| x.1) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            out.push(detection(
                self.name(),
                EntityType::ConnectionString,
                s,
                e,
                SCORE_STRUCTURAL,
            ));

            // An embedded password is separately actionable: a redactor may
            // want to mask just the credential and keep the host for debugging.
            //
            // `rsplit_once` because the userinfo/host separator is the *last*
            // `@` — passwords routinely contain one, and splitting on the first
            // truncates `s3cr3tp@ss` to `s3cr3tp`.
            if let Some((userinfo, _)) = rest.rsplit_once('@')
                && let Some((user, pass)) = userinfo.split_once(':')
                && !pass.is_empty()
                && !is_placeholder(pass)
            {
                let scheme_end = uri.find("://").map_or(0, |i| i + 3);
                let pass_start = s + scheme_end + user.len() + 1;
                let pass_end = pass_start + pass.len();
                if text.get(pass_start..pass_end) == Some(pass) {
                    out.push(detection(
                        "connection_string_password",
                        EntityType::Password,
                        pass_start,
                        pass_end,
                        SCORE_STRUCTURAL,
                    ));
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// key=value secrets
// ---------------------------------------------------------------------------

static KV_SECRET_RE: LazyLock<Regex> = LazyLock::new(|| {
    // Config/env-style assignment where the *key* names a secret. Anchoring on
    // the key rather than the value shape is what keeps this precise.
    Regex::new(
        r#"(?im)^[ \t]*(?:export[ \t]+)?["']?([A-Za-z0-9_.\-]*(?:password|passwd|pwd|secret|api[_.\-]?key|apikey|access[_.\-]?token|auth[_.\-]?token|private[_.\-]?key|client[_.\-]?secret|token)) *["']?[:=] *["']?([^\s"'#,}\]]{6,})["']?"#,
    )
    .expect("kv secret regex")
});

pub struct KeyValueSecretRecognizer;

impl Recognizer for KeyValueSecretRecognizer {
    fn name(&self) -> &'static str {
        "kv_secret"
    }
    fn entity_type(&self) -> EntityType {
        EntityType::Secret
    }
    fn recognize(&self, text: &str) -> Vec<Detection> {
        KV_SECRET_RE
            .captures_iter(text)
            .filter_map(|c| {
                let key = c.get(1)?.as_str().to_ascii_lowercase();
                let val = c.get(2)?;
                let value = val.as_str();
                if is_placeholder(value) {
                    return None;
                }
                // A value that is itself a reference (`${VAR}`, `$VAR`,
                // `!secret name`) is not the secret.
                if value.starts_with('$') || value.starts_with("!{") {
                    return None;
                }
                let ty =
                    if key.contains("password") || key.contains("passwd") || key.contains("pwd") {
                        EntityType::Password
                    } else if key.contains("api") && key.contains("key") || key.contains("apikey") {
                        EntityType::ApiKey
                    } else if key.contains("token") {
                        EntityType::AccessToken
                    } else if key.contains("private") && key.contains("key") {
                        EntityType::PrivateKey
                    } else {
                        EntityType::Secret
                    };
                Some(detection(
                    self.name(),
                    ty,
                    val.start(),
                    val.end(),
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

    const JWT: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";

    #[test]
    fn jwt_recognised_and_version_strings_are_not() {
        assert_eq!(found(&JwtRecognizer, &format!("token: {JWT}")), vec![JWT]);
        for t in [
            "1.23.4567",
            "a.b.c",
            "eyJhbGciOiJIUzI1NiJ9",
            "com.example.app",
        ] {
            assert!(
                found(&JwtRecognizer, t).is_empty(),
                "false positive in {t:?}"
            );
        }
    }

    #[test]
    fn aws_and_github_tokens() {
        assert_eq!(
            found(
                &AwsAccessKeyRecognizer,
                "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
            ),
            vec!["AKIAIOSFODNN7EXAMPLE"]
        );
        assert!(found(&AwsAccessKeyRecognizer, "AKIASHORT").is_empty());

        let gh = format!("ghp_{}", "a1B2c3D4e5".repeat(4));
        assert_eq!(
            found(&GithubTokenRecognizer, &format!("token {gh}")),
            vec![gh.as_str()]
        );
        assert!(found(&GithubTokenRecognizer, "ghp_short").is_empty());
    }

    #[test]
    fn vendor_keys_are_recognised_with_the_right_canonical_type() {
        let text = "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz012345 and xoxb-1234567890-abcdefghij";
        let ds = VendorKeyRecognizer.recognize(text);
        assert_eq!(ds.len(), 2, "got {:?}", found(&VendorKeyRecognizer, text));
        assert!(ds.iter().any(|d| d.entity_type == EntityType::ApiKey));
        assert!(ds.iter().any(|d| d.entity_type == EntityType::AccessToken));

        // Google API keys are `AIza` plus exactly 35 characters.
        let google = format!("key=AIza{}", "Sy0aB1cD2eF3gH4iJ5kL6mN7oP8qR9sT0uV");
        assert_eq!(google.len(), 4 + 4 + 35);
        assert_eq!(found(&VendorKeyRecognizer, &google).len(), 1);
    }

    #[test]
    fn bearer_span_covers_the_token_not_the_scheme() {
        let t = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345";
        let d = &BearerTokenRecognizer.recognize(t)[0];
        assert_eq!(&t[d.start..d.end], "abcdefghijklmnopqrstuvwxyz012345");
        assert!(!t[d.start..d.end].contains("Bearer"));
    }

    #[test]
    fn bearer_ignores_documentation_placeholders() {
        for t in [
            "Bearer YOUR_TOKEN_HERE_XXXX",
            "Bearer <your-token-goes-here>",
            "Bearer xxxxxxxxxxxxxxxxxxxx",
            "Bearer replace_me_with_token",
        ] {
            assert!(
                found(&BearerTokenRecognizer, t).is_empty(),
                "false positive in {t:?}: {:?}",
                found(&BearerTokenRecognizer, t)
            );
        }
    }

    #[test]
    fn pem_block_span_covers_begin_through_end() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...\nAB==\n-----END RSA PRIVATE KEY-----";
        let text = format!("key:\n{pem}\ndone");
        let ds = PemPrivateKeyRecognizer.recognize(&text);
        assert_eq!(ds.len(), 1);
        assert_eq!(&text[ds[0].start..ds[0].end], pem);
        assert_eq!(ds[0].score, SCORE_VALIDATED);
    }

    #[test]
    fn unterminated_pem_reports_only_the_marker_and_cannot_swallow_the_document() {
        // The runaway-span case: no END marker, 100 KB of following text.
        let text = format!("-----BEGIN PRIVATE KEY-----\n{}", "A".repeat(100_000));
        let ds = PemPrivateKeyRecognizer.recognize(&text);
        assert_eq!(ds.len(), 1);
        assert_eq!(&text[ds[0].start..ds[0].end], "-----BEGIN PRIVATE KEY-----");
        assert_eq!(
            ds[0].score, SCORE_PATTERN,
            "lower confidence without an END marker"
        );
        assert!(ds[0].len() < 40, "span must not run away");
    }

    #[test]
    fn pem_handles_multiple_blocks_and_matching_key_kinds() {
        let text = "-----BEGIN EC PRIVATE KEY-----\nx\n-----END EC PRIVATE KEY-----\n\
                    -----BEGIN OPENSSH PRIVATE KEY-----\ny\n-----END OPENSSH PRIVATE KEY-----";
        assert_eq!(PemPrivateKeyRecognizer.recognize(text).len(), 2);
        // A public-key block is not a private key.
        assert!(
            PemPrivateKeyRecognizer
                .recognize("-----BEGIN PUBLIC KEY-----\nx\n-----END PUBLIC KEY-----")
                .is_empty()
        );
    }

    #[test]
    fn connection_string_span_stops_at_the_line_and_extracts_the_password() {
        let text = "DATABASE_URL=postgres://admin:s3cr3tp@ss@prod-db.internal/main\nOTHER=1";
        let ds = ConnectionStringRecognizer.recognize(text);
        let conn = ds
            .iter()
            .find(|d| d.entity_type == EntityType::ConnectionString)
            .unwrap();
        assert_eq!(
            &text[conn.start..conn.end],
            "postgres://admin:s3cr3tp@ss@prod-db.internal/main"
        );
        assert!(
            !text[conn.start..conn.end].contains("OTHER"),
            "must not cross the newline"
        );

        let pw = ds
            .iter()
            .find(|d| d.entity_type == EntityType::Password)
            .unwrap();
        assert_eq!(&text[pw.start..pw.end], "s3cr3tp@ss");
    }

    #[test]
    fn connection_string_schemes_all_recognised() {
        for uri in [
            "postgres://u:p@h/db",
            "postgresql://u:p@h/db",
            "mysql://u:p@h:3306/db",
            "mongodb://u:p@h/db",
            "mongodb+srv://u:p@cluster.example.net/db",
            "redis://u:p@h:6379/0",
        ] {
            let text = format!("url = \"{uri}\"");
            assert_eq!(
                found(&ConnectionStringRecognizer, &text)
                    .into_iter()
                    .find(|s| s.starts_with(uri.split(':').next().unwrap())),
                Some(uri),
                "in {text:?}"
            );
        }
    }

    #[test]
    fn connection_string_without_credentials_is_still_reported_but_yields_no_password() {
        let text = "redis://cache.internal:6379/0";
        let ds = ConnectionStringRecognizer.recognize(text);
        assert_eq!(ds.len(), 1);
        assert_eq!(ds[0].entity_type, EntityType::ConnectionString);
    }

    #[test]
    fn kv_secrets_are_typed_from_the_key_name() {
        // Values deliberately avoid the placeholder needles (`abc123`,
        // `example`, ...) — those are covered by their own test below.
        let text = "\
DB_PASSWORD=hunter2xyz
API_KEY: k9Qd2mZ4pR7wL1tY6vB3
access_token = \"tok_9f8e7d6c5b4a3\"
CLIENT_SECRET=s3cr3tvalue99
";
        let ds = KeyValueSecretRecognizer.recognize(text);
        let types: Vec<_> = ds.iter().map(|d| d.entity_type).collect();
        assert!(types.contains(&EntityType::Password), "{types:?}");
        assert!(types.contains(&EntityType::ApiKey), "{types:?}");
        assert!(types.contains(&EntityType::AccessToken), "{types:?}");
        assert!(types.contains(&EntityType::Secret), "{types:?}");

        let pw = ds
            .iter()
            .find(|d| d.entity_type == EntityType::Password)
            .unwrap();
        assert_eq!(
            &text[pw.start..pw.end],
            "hunter2xyz",
            "span is the value, not the key"
        );
    }

    #[test]
    fn kv_secrets_ignore_placeholders_and_variable_references() {
        let text = "\
PASSWORD=changeme
API_KEY=<your-api-key>
DB_PASSWORD=${DB_PASSWORD}
SECRET=$SECRET_FROM_VAULT
TOKEN=your_token_here
";
        assert!(
            KeyValueSecretRecognizer.recognize(text).is_empty(),
            "false positives: {:?}",
            found(&KeyValueSecretRecognizer, text)
        );
    }

    #[test]
    fn kv_secrets_do_not_fire_on_unrelated_keys() {
        let text = "USERNAME=admin\nPORT=8080\nTIMEOUT=30000\nVERSION=1.23.4567\n";
        assert!(
            KeyValueSecretRecognizer.recognize(text).is_empty(),
            "false positives: {:?}",
            found(&KeyValueSecretRecognizer, text)
        );
    }

    #[test]
    fn placeholder_detection_covers_the_common_stand_ins() {
        assert!(is_placeholder("YOUR_TOKEN"));
        assert!(is_placeholder("xxxxxxxx"));
        assert!(is_placeholder("<your-key>"));
        assert!(is_placeholder("changeme"));
        assert!(!is_placeholder("s3cr3tp@ss"));
        assert!(!is_placeholder("hunter2xyz"));
    }
}
