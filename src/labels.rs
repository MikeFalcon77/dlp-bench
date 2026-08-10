//! Model label -> canonical taxonomy mappings.
//!
//! Raw model labels are never compared to each other or to dataset labels.
//! Each model gets an explicit, testable table here, and anything not in the
//! table is *reported*, not silently dropped and not coerced into a neighbour.
//!
//! A mapping table is also a schema: for zero-shot models like GLiNER2 the
//! labels we ask for are exactly the keys of the table, so the two cannot drift.

use std::collections::BTreeMap;

use crate::entity::EntityType;

/// A model's label vocabulary and its mapping to the canonical taxonomy.
#[derive(Debug, Clone)]
pub struct LabelMap {
    model: &'static str,
    entries: &'static [(&'static str, EntityType)],
}

impl LabelMap {
    pub const fn new(model: &'static str, entries: &'static [(&'static str, EntityType)]) -> Self {
        Self { model, entries }
    }

    pub fn model(&self) -> &'static str {
        self.model
    }

    /// The labels to request from a zero-shot model, in table order.
    ///
    /// Order is stable and meaningful: GLiNER2's decode loop iterates labels in
    /// the order they were requested, so a stable order means reproducible
    /// output ordering.
    pub fn labels(&self) -> Vec<&'static str> {
        self.entries.iter().map(|(l, _)| *l).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Canonical type for a model label. `None` for unmapped labels.
    pub fn map(&self, label: &str) -> Option<EntityType> {
        self.entries
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, t)| *t)
    }

    /// Canonical type by index into [`labels`]. This is the hot path — GLiNER2
    /// decodes by label index, not by string.
    pub fn map_index(&self, index: usize) -> Option<EntityType> {
        self.entries.get(index).map(|(_, t)| *t)
    }

    pub fn label_at(&self, index: usize) -> Option<&'static str> {
        self.entries.get(index).map(|(l, _)| *l)
    }

    /// Canonical types this schema can produce, deduplicated.
    ///
    /// Recall for a type absent from this set is structurally zero, which the
    /// report must state rather than let the reader infer that the model is bad
    /// at something it was never asked about.
    pub fn covered_types(&self) -> Vec<EntityType> {
        let mut v: Vec<_> = self.entries.iter().map(|(_, t)| *t).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Restrict the schema to a subset of labels, for the label-count sweep.
    ///
    /// GLiNER2's cost is dominated by the label prompt, so measuring latency
    /// against schema size is a first-class benchmark dimension.
    pub fn subset(&self, keep: &[&str]) -> Vec<(&'static str, EntityType)> {
        self.entries
            .iter()
            .filter(|(l, _)| keep.contains(l))
            .copied()
            .collect()
    }
}

/// Tracks labels a model emitted that the table does not cover.
///
/// Surfaced in the report. Silence here would mean a model is finding entities
/// we then throw away, which looks identical to the model missing them.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct UnmappedLabels(BTreeMap<String, u64>);

impl UnmappedLabels {
    pub fn record(&mut self, label: &str) {
        *self.0.entry(label.to_owned()).or_insert(0) += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.0.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

// ---------------------------------------------------------------------------
// Fastino GLiNER2 PII
// ---------------------------------------------------------------------------

/// The Phase-1 GLiNER2 schema.
///
/// Restrained on purpose. The model supports 42 labels, but several are
/// overlapping views of the same span (`person` / `full_name` / `first_name` /
/// `last_name`; `card_number` / `payment_card`; `bank_account` /
/// `account_number`). Requesting all of them without a normalisation policy
/// produces N predictions over one gold span, which the one-to-one matcher then
/// counts as 1 TP and N-1 FP — a precision artefact of the schema, not a
/// property of the model.
///
/// So we pick one label per canonical type. The full 42-label schema is
/// available as [`GLINER2_PII_FULL`] for the label-count perf sweep, where
/// quality is not being measured.
pub static GLINER2_PII: LabelMap = LabelMap::new(
    "gliner2-pii",
    &[
        ("person", EntityType::Person),
        ("email", EntityType::Email),
        ("phone_number", EntityType::Phone),
        ("address", EntityType::Address),
        ("date_of_birth", EntityType::DateOfBirth),
        ("national_id_number", EntityType::NationalId),
        ("passport_number", EntityType::Passport),
        ("drivers_license_number", EntityType::DriversLicense),
        ("card_number", EntityType::CreditCard),
        ("bank_account", EntityType::BankAccount),
        ("iban", EntityType::Iban),
        ("ip_address", EntityType::IpAddress),
        ("username", EntityType::Username),
        ("api_key", EntityType::ApiKey),
        ("access_token", EntityType::AccessToken),
        ("password", EntityType::Password),
        ("secret", EntityType::Secret),
    ],
);

/// All 42 labels the Fastino model advertises, for perf-vs-schema-size only.
///
/// Deliberately NOT the default quality schema — see [`GLINER2_PII`]. Labels
/// with no canonical home are mapped to their nearest type; that is acceptable
/// here because this schema is never used for quality measurement.
pub static GLINER2_PII_FULL: LabelMap = LabelMap::new(
    "gliner2-pii-full",
    &[
        ("person", EntityType::Person),
        ("full_name", EntityType::Person),
        ("first_name", EntityType::Person),
        ("middle_name", EntityType::Person),
        ("last_name", EntityType::Person),
        ("date_of_birth", EntityType::DateOfBirth),
        ("email", EntityType::Email),
        ("phone_number", EntityType::Phone),
        ("address", EntityType::Address),
        ("street_address", EntityType::Address),
        ("city", EntityType::Address),
        ("state_or_region", EntityType::Address),
        ("postal_code", EntityType::Address),
        ("country", EntityType::Address),
        ("government_id", EntityType::NationalId),
        ("national_id_number", EntityType::NationalId),
        ("passport_number", EntityType::Passport),
        ("drivers_license_number", EntityType::DriversLicense),
        ("license_number", EntityType::DriversLicense),
        ("tax_id", EntityType::NationalId),
        ("tax_number", EntityType::NationalId),
        ("bank_account", EntityType::BankAccount),
        ("account_number", EntityType::BankAccount),
        ("routing_number", EntityType::BankAccount),
        ("iban", EntityType::Iban),
        ("payment_card", EntityType::CreditCard),
        ("card_number", EntityType::CreditCard),
        ("card_expiry", EntityType::CreditCard),
        ("card_cvv", EntityType::CreditCard),
        ("username", EntityType::Username),
        ("ip_address", EntityType::IpAddress),
        ("account_id", EntityType::Username),
        ("sensitive_account_id", EntityType::Username),
        ("password", EntityType::Password),
        ("secret", EntityType::Secret),
        ("api_key", EntityType::ApiKey),
        ("access_token", EntityType::AccessToken),
        ("recovery_code", EntityType::Secret),
        ("sensitive_date", EntityType::DateOfBirth),
        ("document_date", EntityType::DateOfBirth),
        ("expiration_date", EntityType::DateOfBirth),
        ("transaction_date", EntityType::DateOfBirth),
    ],
);

/// A minimal schema, for the low end of the label-count perf sweep.
pub static GLINER2_PII_MINIMAL: LabelMap = LabelMap::new(
    "gliner2-pii-minimal",
    &[
        ("person", EntityType::Person),
        ("email", EntityType::Email),
        ("phone_number", EntityType::Phone),
        ("card_number", EntityType::CreditCard),
        ("api_key", EntityType::ApiKey),
    ],
);

// ---------------------------------------------------------------------------
// ai4privacy / OpenPII token classifier
// ---------------------------------------------------------------------------

/// The 56 entity types of the ai4privacy v2 taxonomy, mapped to ours.
///
/// Unlike the GLiNER2 schemas this is not a request — the model has a fixed
/// 111-label BIO head and will emit whatever it was trained on. The table's job
/// is therefore the opposite one: decide which of *its* types correspond to
/// something in our taxonomy, and make the rest visible as unmapped rather than
/// silently discarded.
///
/// Two consequences of this taxonomy that the report must surface:
///
/// * **It has no developer-secret types at all** — no API key, access token,
///   private key or connection string. Recall for those is structurally zero,
///   which matters a great deal for AI-DLP, where pasted configs and keys are
///   a primary leak channel.
/// * **It decomposes composite entities.** A person is `FIRSTNAME` +
///   `LASTNAME`, an address is `BUILDINGNUMBER` + `STREET` + `CITY` + …. Each
///   piece maps to the same canonical type, so without the adjacent-span merge
///   in [`crate::detectors::token_class`] one gold `PERSON` would be matched by
///   one prediction and counted against by the rest.
pub static AI4PRIVACY: LabelMap = LabelMap::new(
    "ai4privacy-mdeberta",
    &[
        // --- person ---------------------------------------------------------
        ("FIRSTNAME", EntityType::Person),
        ("MIDDLENAME", EntityType::Person),
        ("LASTNAME", EntityType::Person),
        // --- contact --------------------------------------------------------
        ("EMAIL", EntityType::Email),
        ("PHONENUMBER", EntityType::Phone),
        // --- address (composite; merged downstream) --------------------------
        ("BUILDINGNUMBER", EntityType::Address),
        ("STREET", EntityType::Address),
        ("SECONDARYADDRESS", EntityType::Address),
        ("CITY", EntityType::Address),
        ("STATE", EntityType::Address),
        ("COUNTY", EntityType::Address),
        ("ZIPCODE", EntityType::Address),
        ("NEARBYGPSCOORDINATE", EntityType::Address),
        // --- government identifiers -----------------------------------------
        ("DOB", EntityType::DateOfBirth),
        ("SSN", EntityType::NationalId),
        // --- financial --------------------------------------------------------
        ("CREDITCARDNUMBER", EntityType::CreditCard),
        ("CREDITCARDCVV", EntityType::CreditCard),
        ("IBAN", EntityType::Iban),
        ("ACCOUNTNUMBER", EntityType::BankAccount),
        // Crypto wallets are account identifiers for DLP purposes; leaking one
        // is a financial disclosure even though it is not a bank account.
        ("BITCOINADDRESS", EntityType::BankAccount),
        ("ETHEREUMADDRESS", EntityType::BankAccount),
        ("LITECOINADDRESS", EntityType::BankAccount),
        // --- digital identity -------------------------------------------------
        ("IP", EntityType::IpAddress),
        ("IPV4", EntityType::IpAddress),
        ("IPV6", EntityType::IpAddress),
        ("USERNAME", EntityType::Username),
        ("URL", EntityType::Url),
        // --- credentials ------------------------------------------------------
        ("PASSWORD", EntityType::Password),
        ("PIN", EntityType::Password),
    ],
);

/// ai4privacy types deliberately left unmapped, with the reason.
///
/// Present as data rather than as an omission so that "the model predicted
/// something we ignored" is auditable. Anything here is counted by
/// [`UnmappedLabels`] at runtime and reported.
pub const AI4PRIVACY_UNMAPPED: &[(&str, &str)] = &[
    (
        "ACCOUNTNAME",
        "name on an account; ambiguous between PERSON and an identifier",
    ),
    (
        "AGE",
        "demographic attribute, not an identifier in this taxonomy",
    ),
    ("AMOUNT", "not PII"),
    ("BIC", "identifies a bank, not a person's account"),
    ("COMPANYNAME", "not personal data"),
    ("CREDITCARDISSUER", "the brand, not the card"),
    ("CURRENCY", "not PII"),
    ("CURRENCYCODE", "not PII"),
    ("CURRENCYNAME", "not PII"),
    ("CURRENCYSYMBOL", "not PII"),
    ("DATE", "only DOB is sensitive in this taxonomy"),
    ("EYECOLOR", "demographic attribute"),
    ("GENDER", "demographic attribute"),
    ("HEIGHT", "demographic attribute"),
    ("JOBAREA", "not personal data"),
    ("JOBTITLE", "not personal data"),
    ("JOBTYPE", "not personal data"),
    ("MAC", "device identifier; no canonical slot"),
    ("MASKEDNUMBER", "already redacted, so not sensitive"),
    ("ORDINALDIRECTION", "not PII"),
    ("PHONEIMEI", "device identifier; no canonical slot"),
    (
        "PREFIX",
        "a title such as Mr./Dr.; not an identifier on its own",
    ),
    ("SEX", "demographic attribute"),
    ("TIME", "not PII"),
    ("USERAGENT", "device fingerprint; no canonical slot"),
    ("VEHICLEVIN", "vehicle identifier; no canonical slot"),
    ("VEHICLEVRM", "vehicle identifier; no canonical slot"),
];

/// Canonical types this model can never produce.
///
/// Reported alongside its metrics: a zero recall here is a property of the
/// training taxonomy, not of the model's ability, and the two must not be
/// confused when comparing detectors.
pub fn ai4privacy_unreachable_types() -> Vec<EntityType> {
    unreachable_types(&AI4PRIVACY)
}

// ---------------------------------------------------------------------------
// GLiNER1 PII (knowledgator gliner-pii-{edge,small}-v1.0)
// ---------------------------------------------------------------------------

/// Zero-shot schema for the knowledgator GLiNER-PII checkpoints.
///
/// Labels are natural-language phrases taken from the model card's own
/// vocabulary (`"email address"`, `"credit card"`, `"dob"`), not snake_case
/// identifiers — a different convention from GLiNER2, and one the model was
/// actually prompted with during training.
///
/// Deliberately one label per canonical type and the same 17 types as
/// [`GLINER2_PII`], so the two GLiNER generations are compared on an identical
/// target set rather than on schemas of different breadth. The card's
/// composite groups (`"first name"` / `"last name"`, `"location city"` /
/// `"location street"`) are omitted for the same reason they are in GLiNER2:
/// several labels over one gold span manufacture false positives.
pub static GLINER1_PII: LabelMap = LabelMap::new(
    "gliner1-pii",
    &[
        ("name", EntityType::Person),
        ("email address", EntityType::Email),
        ("phone number", EntityType::Phone),
        ("location address", EntityType::Address),
        ("dob", EntityType::DateOfBirth),
        ("ssn", EntityType::NationalId),
        ("passport number", EntityType::Passport),
        ("driver license", EntityType::DriversLicense),
        ("credit card", EntityType::CreditCard),
        ("bank account", EntityType::BankAccount),
        ("iban", EntityType::Iban),
        ("ip address", EntityType::IpAddress),
        ("username", EntityType::Username),
        ("api key", EntityType::ApiKey),
        ("access token", EntityType::AccessToken),
        ("password", EntityType::Password),
        ("secret", EntityType::Secret),
    ],
);

// ---------------------------------------------------------------------------
// MiniLM-L6 PII (Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2)
// ---------------------------------------------------------------------------

/// 55 entity types over a 6-layer x 384 BERT.
///
/// Notably **this taxonomy reaches developer secrets** — `api_key`, `password`,
/// `pin`, `cvv` — which the ai4privacy one does not at all. That makes it the
/// more interesting small-model candidate for AI-DLP despite being English-only
/// (wordpiece vocab 30522).
///
/// Like ai4privacy it decomposes composites (`first_name` + `last_name`,
/// `street_address` + `city` + `postcode`), so the adjacent-span merge in
/// [`crate::detectors::token_class`] applies here too.
pub static MINILM_PII: LabelMap = LabelMap::new(
    "minilm-l6-pii",
    &[
        // --- person ---------------------------------------------------------
        ("first_name", EntityType::Person),
        ("last_name", EntityType::Person),
        // --- contact --------------------------------------------------------
        ("email", EntityType::Email),
        ("phone_number", EntityType::Phone),
        ("fax_number", EntityType::Phone),
        // --- address (composite; merged downstream) --------------------------
        ("street_address", EntityType::Address),
        ("city", EntityType::Address),
        ("state", EntityType::Address),
        ("county", EntityType::Address),
        ("country", EntityType::Address),
        ("postcode", EntityType::Address),
        ("coordinate", EntityType::Address),
        // --- government identifiers -----------------------------------------
        ("date_of_birth", EntityType::DateOfBirth),
        ("ssn", EntityType::NationalId),
        ("national_id", EntityType::NationalId),
        ("tax_id", EntityType::NationalId),
        // --- financial --------------------------------------------------------
        ("credit_debit_card", EntityType::CreditCard),
        ("cvv", EntityType::CreditCard),
        ("account_number", EntityType::BankAccount),
        ("bank_routing_number", EntityType::BankAccount),
        // --- digital identity -------------------------------------------------
        ("ipv4", EntityType::IpAddress),
        ("ipv6", EntityType::IpAddress),
        ("user_name", EntityType::Username),
        ("url", EntityType::Url),
        // --- credentials ------------------------------------------------------
        ("api_key", EntityType::ApiKey),
        ("password", EntityType::Password),
        ("pin", EntityType::Password),
    ],
);

/// MiniLM types deliberately left unmapped, with the reason.
pub const MINILM_PII_UNMAPPED: &[(&str, &str)] = &[
    (
        "age",
        "demographic attribute, not an identifier in this taxonomy",
    ),
    ("biometric_identifier", "no canonical slot"),
    (
        "blood_type",
        "special-category health data, but no canonical slot",
    ),
    (
        "certificate_license_number",
        "too broad to equate with DRIVERS_LICENSE",
    ),
    ("company_name", "not personal data"),
    (
        "customer_id",
        "internal identifier; ambiguous without system context",
    ),
    ("date", "only DOB is sensitive in this taxonomy"),
    ("date_time", "only DOB is sensitive in this taxonomy"),
    ("device_identifier", "device identifier; no canonical slot"),
    ("education_level", "demographic attribute"),
    (
        "employee_id",
        "internal identifier; ambiguous without system context",
    ),
    ("employment_status", "demographic attribute"),
    ("gender", "demographic attribute"),
    (
        "health_plan_beneficiary_number",
        "health identifier; no canonical slot",
    ),
    ("http_cookie", "session artefact; no canonical slot"),
    ("language", "not PII"),
    ("license_plate", "vehicle identifier; no canonical slot"),
    ("mac_address", "device identifier; no canonical slot"),
    (
        "medical_record_number",
        "health identifier; no canonical slot",
    ),
    ("occupation", "not an identifier"),
    (
        "political_view",
        "special-category data, but no canonical slot",
    ),
    (
        "race_ethnicity",
        "special-category data, but no canonical slot",
    ),
    (
        "religious_belief",
        "special-category data, but no canonical slot",
    ),
    ("sexuality", "special-category data, but no canonical slot"),
    ("swift_bic", "identifies a bank, not a person's account"),
    ("time", "not PII"),
    (
        "unique_id",
        "internal identifier; ambiguous without system context",
    ),
    (
        "vehicle_identifier",
        "vehicle identifier; no canonical slot",
    ),
];

// ---------------------------------------------------------------------------
// DeBERTa-v3-xsmall PII (megaloxorafitis/pii-ner-deberta-v3-xsmall)
// ---------------------------------------------------------------------------

/// Only 9 entity types — by far the narrowest taxonomy in the benchmark.
///
/// Note the label ids are not in the usual order: `O` is index 13, not 0, and
/// five types have a `B-` tag with no `I-`. Both are handled by reading
/// `id2label` from the model rather than assuming a layout.
///
/// **`ID` is a judgement call.** The label is generic — it could mean a
/// national id, a passport, a driver's licence or an account number. It is
/// mapped to [`EntityType::NationalId`] as the nearest high-risk slot, which
/// slightly flatters the model (it can then match our one SSN) and slightly
/// penalises it (a passport or licence it finds is scored as a wrong-type
/// false positive). The alternative — leaving it unmapped — would discard the
/// model's only government-identifier signal. Recorded here so the choice is
/// arguable rather than invisible.
pub static DEBERTA_XSMALL_PII: LabelMap = LabelMap::new(
    "deberta-xsmall-pii",
    &[
        ("NAME", EntityType::Person),
        ("EMAIL", EntityType::Email),
        ("PHONE", EntityType::Phone),
        ("ADDRESS", EntityType::Address),
        ("DOB", EntityType::DateOfBirth),
        ("ID", EntityType::NationalId),
        ("URL", EntityType::Url),
        ("USERNAME", EntityType::Username),
    ],
);

pub const DEBERTA_XSMALL_PII_UNMAPPED: &[(&str, &str)] =
    &[("ORG", "an organisation is not personal data")];

// ---------------------------------------------------------------------------
// Presidio research synthetic corpus (dataset labels, not a model's)
// ---------------------------------------------------------------------------

/// Entity types of `presidio-research`'s `synth_dataset_v2`.
///
/// Unlike every other table here this maps a *dataset's* vocabulary rather than
/// a model's, but the discipline is identical: nothing is silently dropped, and
/// a type we choose not to score is listed with a reason.
///
/// The choices that materially affect scores:
///
/// * **`GPE` -> `ADDRESS`.** A geopolitical entity — a city or country — is
///   mapped the same way `CITY`/`STATE`/`COUNTRY` are in the ai4privacy table,
///   for consistency. It is 411 of 2 863 spans, so the decision is not
///   cosmetic: excluding it instead would turn every address-aware detector's
///   hits on bare place names into false positives.
/// * **`DOMAIN_NAME` excluded.** Our `URL` type is scheme-qualified
///   (`https://…`) and the deterministic recognizer requires a scheme, so
///   scoring bare domains against it would manufacture 37 guaranteed false
///   negatives from a definitional mismatch rather than a detector weakness.
pub static PRESIDIO: LabelMap = LabelMap::new(
    "presidio-synth",
    &[
        ("PERSON", EntityType::Person),
        ("EMAIL_ADDRESS", EntityType::Email),
        ("PHONE_NUMBER", EntityType::Phone),
        ("STREET_ADDRESS", EntityType::Address),
        ("GPE", EntityType::Address),
        ("ZIP_CODE", EntityType::Address),
        ("CREDIT_CARD", EntityType::CreditCard),
        ("IBAN_CODE", EntityType::Iban),
        ("US_SSN", EntityType::NationalId),
        ("US_DRIVER_LICENSE", EntityType::DriversLicense),
        ("IP_ADDRESS", EntityType::IpAddress),
    ],
);

/// Presidio types deliberately not scored, with the reason.
pub const PRESIDIO_UNMAPPED: &[(&str, &str)] = &[
    ("ORGANIZATION", "an organisation is not personal data"),
    (
        "DATE_TIME",
        "only a date of birth is sensitive in this taxonomy",
    ),
    (
        "TITLE",
        "a title such as Mr./Dr. is not an identifier on its own",
    ),
    ("AGE", "demographic attribute, not an identifier"),
    (
        "NRP",
        "nationality/religion/politics: special-category data, no canonical slot",
    ),
    (
        "DOMAIN_NAME",
        "our URL type is scheme-qualified; a bare domain is not one",
    ),
];

/// Canonical types a label map can never produce.
///
/// Reported alongside a detector's metrics: a zero recall for one of these is
/// a property of the training taxonomy, not of the model's ability, and the
/// two must not be confused when comparing detectors.
pub fn unreachable_types(map: &LabelMap) -> Vec<EntityType> {
    let covered = map.covered_types();
    EntityType::ALL
        .iter()
        .copied()
        .filter(|t| !covered.contains(t))
        .collect()
}

/// Schemas addressable by name from the CLI.
pub fn schema_by_name(name: &str) -> Option<&'static LabelMap> {
    match name {
        "default" | "gliner2-pii" => Some(&GLINER2_PII),
        "full" | "gliner2-pii-full" => Some(&GLINER2_PII_FULL),
        "minimal" | "gliner2-pii-minimal" => Some(&GLINER2_PII_MINIMAL),
        _ => None,
    }
}

pub const SCHEMA_NAMES: &[&str] = &["minimal", "default", "full"];

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_no_duplicate_labels(m: &LabelMap) {
        let mut seen = std::collections::BTreeSet::new();
        for l in m.labels() {
            assert!(seen.insert(l), "duplicate label {l:?} in {}", m.model());
        }
    }

    #[test]
    fn every_schema_label_maps_and_is_unique() {
        for m in [&GLINER2_PII, &GLINER2_PII_FULL, &GLINER2_PII_MINIMAL] {
            assert_no_duplicate_labels(m);
            for (i, l) in m.labels().iter().enumerate() {
                assert!(m.map(l).is_some(), "{l:?} unmapped in {}", m.model());
                assert_eq!(
                    m.map(l),
                    m.map_index(i),
                    "index/name mapping disagree for {l:?}"
                );
            }
        }
    }

    #[test]
    fn default_schema_has_no_overlapping_person_labels() {
        // The exact trap called out in the task spec: requesting person,
        // full_name, first_name and last_name together without a normalisation
        // policy manufactures false positives.
        let labels = GLINER2_PII.labels();
        let person_ish = [
            "person",
            "full_name",
            "first_name",
            "middle_name",
            "last_name",
        ];
        let present: Vec<_> = person_ish.iter().filter(|l| labels.contains(l)).collect();
        assert_eq!(
            present.len(),
            1,
            "default schema must pick exactly one person label"
        );
    }

    #[test]
    fn default_schema_is_one_label_per_canonical_type() {
        let mut counts = std::collections::BTreeMap::new();
        for l in GLINER2_PII.labels() {
            *counts.entry(GLINER2_PII.map(l).unwrap()).or_insert(0) += 1;
        }
        for (t, n) in counts {
            assert_eq!(n, 1, "{t} has {n} labels in the default schema");
        }
    }

    #[test]
    fn unknown_label_does_not_become_another_type() {
        assert_eq!(GLINER2_PII.map("ssn"), None);
        assert_eq!(
            GLINER2_PII.map("credit_card"),
            None,
            "not the label this model uses"
        );
        assert_eq!(GLINER2_PII.map("EMAIL"), None, "case-sensitive by design");
        // full_name exists in the full schema but not the default one
        assert_eq!(GLINER2_PII.map("full_name"), None);
        assert_eq!(GLINER2_PII_FULL.map("full_name"), Some(EntityType::Person));
    }

    #[test]
    fn unmapped_labels_are_counted_not_swallowed() {
        let mut u = UnmappedLabels::default();
        assert!(u.is_empty());
        u.record("ssn");
        u.record("ssn");
        u.record("vin");
        let got: Vec<_> = u.iter().collect();
        assert_eq!(got, vec![("ssn", 2), ("vin", 1)]);
    }

    #[test]
    fn schema_sizes_are_distinct_for_the_perf_sweep() {
        assert_eq!(GLINER2_PII_MINIMAL.len(), 5);
        assert_eq!(GLINER2_PII.len(), 17);
        assert_eq!(
            GLINER2_PII_FULL.len(),
            42,
            "matches the 42 labels the model card lists"
        );
    }

    #[test]
    fn schema_lookup_covers_every_advertised_name() {
        for n in SCHEMA_NAMES {
            assert!(schema_by_name(n).is_some(), "{n} not resolvable");
        }
        assert!(schema_by_name("nope").is_none());
    }

    /// The 56 ai4privacy types must be fully accounted for: every one is
    /// either mapped or explicitly listed as unmapped-with-a-reason. A type
    /// that is in neither list would be silently dropped at runtime.
    #[test]
    fn every_ai4privacy_type_is_either_mapped_or_explicitly_excluded() {
        // Straight from the model's config.json id2label.
        const ALL: &[&str] = &[
            "ACCOUNTNAME",
            "ACCOUNTNUMBER",
            "AGE",
            "AMOUNT",
            "BIC",
            "BITCOINADDRESS",
            "BUILDINGNUMBER",
            "CITY",
            "COMPANYNAME",
            "COUNTY",
            "CREDITCARDCVV",
            "CREDITCARDISSUER",
            "CREDITCARDNUMBER",
            "CURRENCY",
            "CURRENCYCODE",
            "CURRENCYNAME",
            "CURRENCYSYMBOL",
            "DATE",
            "DOB",
            "EMAIL",
            "ETHEREUMADDRESS",
            "EYECOLOR",
            "FIRSTNAME",
            "GENDER",
            "HEIGHT",
            "IBAN",
            "IP",
            "IPV4",
            "IPV6",
            "JOBAREA",
            "JOBTITLE",
            "JOBTYPE",
            "LASTNAME",
            "LITECOINADDRESS",
            "MAC",
            "MASKEDNUMBER",
            "MIDDLENAME",
            "NEARBYGPSCOORDINATE",
            "ORDINALDIRECTION",
            "PASSWORD",
            "PHONEIMEI",
            "PHONENUMBER",
            "PIN",
            "PREFIX",
            "SECONDARYADDRESS",
            "SEX",
            "SSN",
            "STATE",
            "STREET",
            "TIME",
            "URL",
            "USERAGENT",
            "USERNAME",
            "VEHICLEVIN",
            "VEHICLEVRM",
            "ZIPCODE",
        ];
        assert_eq!(ALL.len(), 56);

        let excluded: Vec<&str> = AI4PRIVACY_UNMAPPED.iter().map(|(t, _)| *t).collect();
        for t in ALL {
            let mapped = AI4PRIVACY.map(t).is_some();
            let listed = excluded.contains(t);
            assert!(
                mapped || listed,
                "{t} is neither mapped nor explicitly excluded"
            );
            assert!(!(mapped && listed), "{t} is both mapped and excluded");
        }
        assert_eq!(AI4PRIVACY.len() + AI4PRIVACY_UNMAPPED.len(), 56);

        // Every exclusion carries a reason a reviewer can argue with.
        for (t, why) in AI4PRIVACY_UNMAPPED {
            assert!(!why.is_empty(), "{t} excluded without a reason");
        }
    }

    /// The gap that decides whether this model can serve an AI-DLP use case.
    #[test]
    fn ai4privacy_cannot_reach_the_developer_secret_types() {
        let unreachable = ai4privacy_unreachable_types();
        for t in [
            EntityType::ApiKey,
            EntityType::AccessToken,
            EntityType::Secret,
            EntityType::PrivateKey,
            EntityType::ConnectionString,
            EntityType::Passport,
            EntityType::DriversLicense,
        ] {
            assert!(unreachable.contains(&t), "{t} unexpectedly reachable");
        }
        // ...but it does cover the consumer-PII core.
        for t in [
            EntityType::Person,
            EntityType::Email,
            EntityType::Phone,
            EntityType::Address,
            EntityType::CreditCard,
            EntityType::Iban,
        ] {
            assert!(!unreachable.contains(&t), "{t} should be reachable");
        }
    }

    #[test]
    fn ai4privacy_composite_types_share_a_canonical_type() {
        // This is exactly why the adjacent-span merge exists.
        for t in ["FIRSTNAME", "MIDDLENAME", "LASTNAME"] {
            assert_eq!(AI4PRIVACY.map(t), Some(EntityType::Person), "{t}");
        }
        for t in ["BUILDINGNUMBER", "STREET", "CITY", "STATE", "ZIPCODE"] {
            assert_eq!(AI4PRIVACY.map(t), Some(EntityType::Address), "{t}");
        }
        for t in ["IP", "IPV4", "IPV6"] {
            assert_eq!(AI4PRIVACY.map(t), Some(EntityType::IpAddress), "{t}");
        }
    }

    /// Same contract as the ai4privacy test: every label the model can emit is
    /// either mapped or explicitly excluded with a reason. A label in neither
    /// list is silently discarded at runtime.
    /// The two GLiNER generations must target the same canonical types, or
    /// their scores are not comparable.
    #[test]
    fn gliner1_and_gliner2_schemas_cover_identical_canonical_types() {
        assert_eq!(GLINER1_PII.covered_types(), GLINER2_PII.covered_types());
        assert_eq!(GLINER1_PII.len(), GLINER2_PII.len());
        // One label per canonical type, as in the GLiNER2 schema.
        let mut counts = std::collections::BTreeMap::new();
        for l in GLINER1_PII.labels() {
            *counts.entry(GLINER1_PII.map(l).unwrap()).or_insert(0) += 1;
        }
        for (t, n) in counts {
            assert_eq!(n, 1, "{t} has {n} labels");
        }
        // Natural-language labels, not snake_case — the model's own convention.
        assert!(GLINER1_PII.labels().iter().any(|l| l.contains(' ')));
        assert!(!GLINER1_PII.labels().iter().any(|l| l.contains('_')));
    }

    #[test]
    fn every_minilm_type_is_either_mapped_or_explicitly_excluded() {
        // Straight from the model's config.json id2label.
        const ALL: &[&str] = &[
            "account_number",
            "age",
            "api_key",
            "bank_routing_number",
            "biometric_identifier",
            "blood_type",
            "certificate_license_number",
            "city",
            "company_name",
            "coordinate",
            "country",
            "county",
            "credit_debit_card",
            "customer_id",
            "cvv",
            "date",
            "date_of_birth",
            "date_time",
            "device_identifier",
            "education_level",
            "email",
            "employee_id",
            "employment_status",
            "fax_number",
            "first_name",
            "gender",
            "health_plan_beneficiary_number",
            "http_cookie",
            "ipv4",
            "ipv6",
            "language",
            "last_name",
            "license_plate",
            "mac_address",
            "medical_record_number",
            "national_id",
            "occupation",
            "password",
            "phone_number",
            "pin",
            "political_view",
            "postcode",
            "race_ethnicity",
            "religious_belief",
            "sexuality",
            "ssn",
            "state",
            "street_address",
            "swift_bic",
            "tax_id",
            "time",
            "unique_id",
            "url",
            "user_name",
            "vehicle_identifier",
        ];
        assert_eq!(ALL.len(), 55);

        let excluded: Vec<&str> = MINILM_PII_UNMAPPED.iter().map(|(t, _)| *t).collect();
        for t in ALL {
            let mapped = MINILM_PII.map(t).is_some();
            let listed = excluded.contains(t);
            assert!(
                mapped || listed,
                "{t} is neither mapped nor explicitly excluded"
            );
            assert!(!(mapped && listed), "{t} is both mapped and excluded");
        }
        assert_eq!(MINILM_PII.len() + MINILM_PII_UNMAPPED.len(), 55);
        for (t, why) in MINILM_PII_UNMAPPED {
            assert!(!why.is_empty(), "{t} excluded without a reason");
        }
    }

    /// The reason MiniLM is the more interesting small model: unlike
    /// ai4privacy it can actually reach some developer secrets.
    #[test]
    fn minilm_reaches_credentials_that_ai4privacy_cannot() {
        assert_eq!(MINILM_PII.map("api_key"), Some(EntityType::ApiKey));
        assert_eq!(MINILM_PII.map("password"), Some(EntityType::Password));
        assert_eq!(MINILM_PII.map("pin"), Some(EntityType::Password));

        let minilm_gap = unreachable_types(&MINILM_PII);
        let ai4_gap = unreachable_types(&AI4PRIVACY);
        assert!(!minilm_gap.contains(&EntityType::ApiKey));
        assert!(ai4_gap.contains(&EntityType::ApiKey));
        // Both still miss these entirely.
        for t in [
            EntityType::AccessToken,
            EntityType::PrivateKey,
            EntityType::ConnectionString,
        ] {
            assert!(minilm_gap.contains(&t), "{t}");
            assert!(ai4_gap.contains(&t), "{t}");
        }
    }

    #[test]
    fn every_deberta_xsmall_type_is_either_mapped_or_explicitly_excluded() {
        const ALL: &[&str] = &[
            "ADDRESS", "DOB", "EMAIL", "ID", "NAME", "ORG", "PHONE", "URL", "USERNAME",
        ];
        let excluded: Vec<&str> = DEBERTA_XSMALL_PII_UNMAPPED
            .iter()
            .map(|(t, _)| *t)
            .collect();
        for t in ALL {
            let mapped = DEBERTA_XSMALL_PII.map(t).is_some();
            assert!(mapped || excluded.contains(t), "{t} unaccounted for");
        }
        assert_eq!(
            DEBERTA_XSMALL_PII.len() + DEBERTA_XSMALL_PII_UNMAPPED.len(),
            9
        );
        // The documented judgement call.
        assert_eq!(DEBERTA_XSMALL_PII.map("ID"), Some(EntityType::NationalId));
        // Its taxonomy is by far the narrowest: no cards, no IBAN, no secrets.
        let gap = unreachable_types(&DEBERTA_XSMALL_PII);
        for t in [
            EntityType::CreditCard,
            EntityType::Iban,
            EntityType::ApiKey,
            EntityType::Password,
            EntityType::Passport,
            EntityType::BankAccount,
        ] {
            assert!(gap.contains(&t), "{t} should be unreachable");
        }
        assert!(gap.len() >= 12, "expected a wide gap, got {}", gap.len());
    }

    #[test]
    fn subset_preserves_table_order() {
        let s = GLINER2_PII.subset(&["api_key", "email", "person"]);
        let names: Vec<_> = s.iter().map(|(l, _)| *l).collect();
        assert_eq!(names, vec!["person", "email", "api_key"]);
    }
}
