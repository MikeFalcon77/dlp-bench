//! Canonical PII taxonomy and the single detection representation every
//! detector produces and the evaluator consumes.
//!
//! Model-specific label names never appear here or downstream of
//! [`crate::labels`]. That is the whole point: two detectors are only
//! comparable if their outputs share one vocabulary.

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Canonical entity category.
///
/// Stable `SCREAMING_SNAKE_CASE` string form, usable as a map key, and ordered
/// so reports have deterministic layout regardless of `HashMap` iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[non_exhaustive]
pub enum EntityType {
    // --- identity -------------------------------------------------------
    Person,
    Email,
    Phone,
    Address,
    DateOfBirth,

    // --- government identifiers -----------------------------------------
    NationalId,
    Passport,
    DriversLicense,

    // --- financial --------------------------------------------------------
    CreditCard,
    BankAccount,
    Iban,

    // --- digital identity -------------------------------------------------
    IpAddress,
    Username,

    // --- credentials ------------------------------------------------------
    ApiKey,
    AccessToken,
    Password,
    Secret,

    // --- other sensitive material ----------------------------------------
    Url,
    PrivateKey,
    ConnectionString,
}

impl EntityType {
    /// Every variant, in declaration order. Used for per-entity report layout
    /// and to keep the parser and the taxonomy from drifting apart.
    pub const ALL: &'static [EntityType] = &[
        EntityType::Person,
        EntityType::Email,
        EntityType::Phone,
        EntityType::Address,
        EntityType::DateOfBirth,
        EntityType::NationalId,
        EntityType::Passport,
        EntityType::DriversLicense,
        EntityType::CreditCard,
        EntityType::BankAccount,
        EntityType::Iban,
        EntityType::IpAddress,
        EntityType::Username,
        EntityType::ApiKey,
        EntityType::AccessToken,
        EntityType::Password,
        EntityType::Secret,
        EntityType::Url,
        EntityType::PrivateKey,
        EntityType::ConnectionString,
    ];

    /// High-risk group whose recall is reported separately.
    ///
    /// Rationale: macro-averaged F1 still lets a catastrophic `PASSPORT` recall
    /// hide behind an easy `EMAIL`. For DLP, missing one passport number is
    /// materially worse than missing one URL, so the group gets its own number.
    /// Configurable at the call site via [`crate::eval::CriticalSet`]; this is
    /// only the default.
    pub const DEFAULT_CRITICAL: &'static [EntityType] = &[
        EntityType::NationalId,
        EntityType::Passport,
        EntityType::DriversLicense,
        EntityType::CreditCard,
        EntityType::BankAccount,
        EntityType::Iban,
        EntityType::ApiKey,
        EntityType::AccessToken,
        EntityType::Password,
        EntityType::Secret,
        EntityType::PrivateKey,
        EntityType::ConnectionString,
    ];

    /// Stable wire/report name. Must match the serde representation.
    pub const fn as_str(self) -> &'static str {
        match self {
            EntityType::Person => "PERSON",
            EntityType::Email => "EMAIL",
            EntityType::Phone => "PHONE",
            EntityType::Address => "ADDRESS",
            EntityType::DateOfBirth => "DATE_OF_BIRTH",
            EntityType::NationalId => "NATIONAL_ID",
            EntityType::Passport => "PASSPORT",
            EntityType::DriversLicense => "DRIVERS_LICENSE",
            EntityType::CreditCard => "CREDIT_CARD",
            EntityType::BankAccount => "BANK_ACCOUNT",
            EntityType::Iban => "IBAN",
            EntityType::IpAddress => "IP_ADDRESS",
            EntityType::Username => "USERNAME",
            EntityType::ApiKey => "API_KEY",
            EntityType::AccessToken => "ACCESS_TOKEN",
            EntityType::Password => "PASSWORD",
            EntityType::Secret => "SECRET",
            EntityType::Url => "URL",
            EntityType::PrivateKey => "PRIVATE_KEY",
            EntityType::ConnectionString => "CONNECTION_STRING",
        }
    }

    /// True if this type is in the default critical set.
    pub fn is_default_critical(self) -> bool {
        Self::DEFAULT_CRITICAL.contains(&self)
    }
}

impl fmt::Display for EntityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returned when a dataset or CLI argument names a type outside the taxonomy.
///
/// Deliberately an error rather than a fallback variant: silently bucketing an
/// unknown label into `SECRET` would corrupt every metric downstream.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unknown entity type: {0:?}")]
pub struct UnknownEntityType(pub String);

impl FromStr for EntityType {
    type Err = UnknownEntityType;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        EntityType::ALL
            .iter()
            .copied()
            .find(|t| t.as_str() == s)
            .ok_or_else(|| UnknownEntityType(s.to_owned()))
    }
}

/// Where a detection came from. Carried from Phase 1 because the hybrid
/// detector's merge rules and the false-positive reports both need it, and
/// retrofitting provenance later would touch every detector.
/// `Cow` rather than `&'static str` so the type can round-trip through serde:
/// recognizer names are static, but a deserialized report has owned strings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum DetectionSource {
    /// Produced by a named deterministic recognizer, e.g. `"credit_card"`.
    Deterministic(Cow<'static, str>),
    /// Produced by a model, named by detector id, e.g. `"gliner2-pii"`.
    Model(Cow<'static, str>),
}

impl DetectionSource {
    /// A deterministic recognizer, named by a `&'static str`.
    pub const fn recognizer(name: &'static str) -> Self {
        DetectionSource::Deterministic(Cow::Borrowed(name))
    }

    /// A model, named by detector id.
    pub fn model(name: impl Into<Cow<'static, str>>) -> Self {
        DetectionSource::Model(name.into())
    }

    pub fn name(&self) -> &str {
        match self {
            DetectionSource::Deterministic(n) | DetectionSource::Model(n) => n,
        }
    }
}

/// One detected span.
///
/// `start`/`end` are **UTF-8 byte offsets** into the text that was passed to
/// the detector, such that `&text[start..end]` is the matched substring. See
/// [`crate::span`] for the validation every detector output goes through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    pub entity_type: EntityType,
    pub start: usize,
    pub end: usize,
    /// Confidence in `[0, 1]`. Deterministic recognizers report `1.0` when a
    /// structural validator passed and a lower fixed value when the match rests
    /// on a pattern alone.
    pub score: f32,
    pub source: DetectionSource,
}

impl Detection {
    pub fn new(
        entity_type: EntityType,
        start: usize,
        end: usize,
        score: f32,
        source: DetectionSource,
    ) -> Self {
        Self {
            entity_type,
            start,
            end,
            score,
            source,
        }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Byte overlap with another span. Zero when disjoint or merely touching.
    pub fn overlap_len(&self, other_start: usize, other_end: usize) -> usize {
        let lo = self.start.max(other_start);
        let hi = self.end.min(other_end);
        hi.saturating_sub(lo)
    }

    /// Sort key giving deterministic output ordering across all detectors.
    pub fn sort_key(&self) -> (usize, usize, EntityType) {
        (self.start, self.end, self.entity_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_covers_every_variant_and_round_trips() {
        // If a variant is added without updating ALL, the serde round-trip
        // below is the thing that keeps the parser honest.
        for &t in EntityType::ALL {
            let s = t.as_str();
            assert_eq!(
                EntityType::from_str(s).unwrap(),
                t,
                "from_str failed for {s}"
            );

            let json = serde_json::to_string(&t).unwrap();
            assert_eq!(
                json,
                format!("\"{s}\""),
                "serde name differs from as_str for {t:?}"
            );
            let back: EntityType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, t);
        }
    }

    #[test]
    fn all_has_no_duplicates() {
        let mut seen = std::collections::BTreeSet::new();
        for &t in EntityType::ALL {
            assert!(seen.insert(t), "duplicate in ALL: {t:?}");
        }
    }

    #[test]
    fn unknown_label_is_an_error_not_a_fallback() {
        let err = EntityType::from_str("SSN").unwrap_err();
        assert_eq!(err, UnknownEntityType("SSN".to_owned()));
        // Case matters — no lenient matching that could mask a mapping bug.
        assert!(EntityType::from_str("email").is_err());
    }

    #[test]
    fn critical_set_is_a_subset_of_taxonomy() {
        for &t in EntityType::DEFAULT_CRITICAL {
            assert!(EntityType::ALL.contains(&t), "{t:?} missing from ALL");
        }
        assert!(EntityType::CreditCard.is_default_critical());
        assert!(!EntityType::Url.is_default_critical());
    }

    #[test]
    fn overlap_len_is_zero_for_touching_spans() {
        let d = Detection::new(
            EntityType::Email,
            10,
            20,
            1.0,
            DetectionSource::recognizer("email"),
        );
        assert_eq!(d.overlap_len(20, 30), 0, "abutting spans must not overlap");
        assert_eq!(d.overlap_len(0, 10), 0);
        assert_eq!(d.overlap_len(15, 25), 5);
        assert_eq!(d.overlap_len(0, 100), 10);
    }
}
