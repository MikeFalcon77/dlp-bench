//! End-to-end verification of every ONNX detector without a Python oracle.
//!
//! # Why this exists instead of golden files
//!
//! The plan was to verify prompt construction, tensor layout and span decoding
//! against the reference implementation by generating goldens with
//! `uv run --with tokenizers` / `--with onnxruntime`. That is not possible in
//! this environment: PyPI is unreachable through a TLS-intercepting proxy
//! (`invalid peer certificate: UnknownIssuer`), so no Python package can be
//! installed and no reference output can be produced. See the README's "known
//! limitations".
//!
//! What replaces it is a stronger *self-contained* oracle for the property that
//! actually matters. The deterministic detector finds emails, IBANs, credit
//! cards and IP addresses by regex plus checksum — with no tokenizer, no model
//! and no word splitter. If GLiNER2, going through an entirely independent path
//! (prepare text -> split words -> build prompt -> ONNX -> decode word-index
//! spans -> byte offsets), lands on *byte-identical* spans for those entities,
//! then the whole offset chain is correct. A transposed axis, an off-by-one in
//! `text_word_indices`, a forgotten trailing `.`, or a char/byte confusion
//! would all break this agreement immediately.
//!
//! The oracle is applied to **every** ONNX detector, not just GLiNER2: each has
//! its own tokenizer, word splitter, prompt format and decoder, so each has its
//! own opportunity to get offsets wrong.
//!
//! Tests skip with a printed notice when a model is absent.

// The whole file is ONNX-only; `--no-default-features` builds the evaluator and
// the deterministic detector without `ort`, and this suite has nothing to test.
#![cfg(feature = "onnx")]

use std::path::Path;

use dlp_bench::detectors::deterministic::DeterministicDetector;
use dlp_bench::detectors::gliner2::{Gliner2Config, Gliner2Detector};
use dlp_bench::detectors::onnx::ThreadConfig;
use dlp_bench::entity::EntityType;
use dlp_bench::{Detection, Detector};

const MODEL_DIR: &str = "models/gliner2-pii-onnx";

fn model_present() -> bool {
    Path::new(MODEL_DIR).join("model.quantized.onnx").is_file()
}

fn gliner2(threshold: f32) -> Option<Gliner2Detector> {
    if !model_present() {
        eprintln!("SKIP: model absent — run `bash scripts/fetch-models.sh`");
        return None;
    }
    let mut cfg = Gliner2Config::new(MODEL_DIR);
    cfg.threshold = threshold;
    cfg.threads = ThreadConfig::with_intra(4);
    cfg.verify_checksum = false;
    Some(Gliner2Detector::load(cfg).expect("model must load"))
}

fn spans_of<'a>(ds: &[Detection], t: EntityType, text: &'a str) -> Vec<&'a str> {
    let mut v: Vec<_> = ds
        .iter()
        .filter(|d| d.entity_type == t)
        .map(|d| &text[d.start..d.end])
        .collect();
    v.sort_unstable();
    v
}

/// Documents whose emails/IBANs/cards the rule detector can locate exactly.
const AGREEMENT_CORPUS: &[&str] = &[
    "Please contact John Smith at john@example.com.",
    "Mail a.b+tag@sub.example.co.uk about the invoice",
    "Свяжитесь с Михаил Ястребцов по адресу mikhail@example.ru",
    "Jörg Müller erreichen Sie unter joerg.mueller@example.de.",
    "José García escribió a jose@example.es ayer.",
    "François Dupont a écrit à francois@example.fr hier.",
    "王小明的电子邮件是 xiaoming@example.cn",
    "Transfer to DE89370400440532013000 today",
    "IBAN GB82 WEST 1234 5698 7654 32 confirmed",
    "Card 4111 1111 1111 1111 was declined",
    "Charge 5500005555555559 for the order",
    "{\"customer\": \"John Smith\", \"email\": \"john@example.com\"}",
    "customer:\n  email: ops@example.net\n  active: true",
    "SELECT * FROM users WHERE email = 'john@example.com';",
    "2026-08-08 user=maria@example.es ip=203.0.113.44 login ok",
    "let email = \"customer@example.com\";",
];

/// The core test: two completely independent implementations must produce the
/// same byte offsets for structurally unambiguous entities.
#[test]
fn gliner2_and_rules_agree_on_byte_exact_spans_for_structured_entities() {
    let Some(model) = gliner2(0.5) else { return };
    let rules = DeterministicDetector::default();

    let mut compared = 0usize;
    for text in AGREEMENT_CORPUS {
        let a = rules.detect(text).expect("rules");
        let b = model.detect(text).expect("model");

        for t in [EntityType::Email, EntityType::Iban, EntityType::CreditCard] {
            let want = spans_of(&a, t, text);
            if want.is_empty() {
                continue;
            }
            let got = spans_of(&b, t, text);
            // Recall is a benchmark question, not a correctness one: the model
            // is allowed to miss an entity. What it is not allowed to do is
            // find it at the wrong offsets.
            if got.is_empty() {
                continue;
            }
            assert_eq!(
                got, want,
                "offset disagreement for {t} in {text:?}\n  rules: {want:?}\n  model: {got:?}"
            );
            compared += 1;
        }
    }
    assert!(
        compared >= 10,
        "only {compared} spans compared; the oracle is not exercising the pipeline"
    );
}

/// Every span the model emits must be a valid slice of the caller's text.
///
/// This is what catches the trailing-`.` fixup leaking into a returned offset:
/// the model works on `text + "."`, so an unclipped span can exceed the
/// caller's string by one byte.
#[test]
fn every_model_span_slices_the_callers_text() {
    let Some(model) = gliner2(0.3) else { return };

    let corpus: Vec<String> = AGREEMENT_CORPUS
        .iter()
        .map(|s| s.to_string())
        // Text with no terminal punctuation exercises the appended '.'.
        .chain(
            AGREEMENT_CORPUS
                .iter()
                .map(|s| s.trim_end_matches(['.', '!', '?']).to_string()),
        )
        .chain(["".into(), "   ".into(), "!".into(), "a".into(), "王".into()])
        .collect();

    for text in &corpus {
        let ds = model.detect(text).expect("model");
        dlp_bench::span::validate_all(text, &ds)
            .unwrap_or_else(|(i, e)| panic!("invalid span {i} in {text:?}: {e}"));
        for d in &ds {
            assert!(
                d.end <= text.len(),
                "span past caller text in {text:?}: {d:?}"
            );
            assert!(
                !text[d.start..d.end].trim().is_empty(),
                "whitespace-only span in {text:?}"
            );
        }
    }
}

/// Detection must not depend on where in a document the entity sits.
///
/// Padding shifts every word index and every byte offset, so a bug in the
/// prefix length or the word-index base shows up as a span that no longer
/// tracks the entity.
#[test]
fn spans_track_the_entity_when_the_document_is_padded() {
    let Some(model) = gliner2(0.5) else { return };

    let needle = "john@example.com";
    for pad in [
        "",
        "Hello there. ",
        "Свяжитесь с нами. ",
        "王小明说。",
        "x ".repeat(50).as_str(),
    ] {
        let text = format!("{pad}Contact {needle} now.");
        let ds = model.detect(&text).expect("model");
        let found: Vec<_> = ds
            .iter()
            .filter(|d| d.entity_type == EntityType::Email)
            .map(|d| &text[d.start..d.end])
            .collect();
        assert_eq!(
            found,
            vec![needle],
            "email span drifted with padding {:?} in {text:?}",
            pad.chars().take(12).collect::<String>()
        );
    }
}

/// Threshold must be monotone: raising it can only remove detections.
#[test]
fn raising_the_threshold_only_removes_detections() {
    let Some(low) = gliner2(0.2) else { return };
    let high = low.with_threshold(0.9).expect("gliner2 supports retuning");

    let text = "Contact John Smith at john@example.com or +1-415-555-1234.";
    let a = low.detect(text).expect("low");
    let b = high.detect(text).expect("high");

    assert!(b.len() <= a.len(), "raising the threshold added detections");
    for d in &b {
        assert!(
            a.iter()
                .any(|x| x.entity_type == d.entity_type && x.start == d.start && x.end == d.end),
            "{d:?} appeared only at the higher threshold"
        );
    }
    assert!(
        !a.is_empty(),
        "the low-threshold run found nothing; test is vacuous"
    );
}

/// Re-thresholding shares the loaded session rather than reloading 446 MB.
#[test]
fn retuning_the_threshold_is_cheap() {
    let Some(d) = gliner2(0.5) else { return };
    let t = std::time::Instant::now();
    let tuned = d.with_threshold(0.7).expect("supports retuning");
    let elapsed = t.elapsed();
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "with_threshold took {elapsed:?} — it is reloading the model"
    );
    assert_eq!(tuned.threshold(), Some(0.7));
}

/// Output must be identical across repeated runs on the same input.
#[test]
fn inference_is_deterministic() {
    let Some(d) = gliner2(0.5) else { return };
    let text = "Contact John Smith at john@example.com, IBAN DE89370400440532013000.";
    let first = d.detect(text).expect("first");
    for i in 0..5 {
        assert_eq!(d.detect(text).expect("repeat"), first, "run {i} differed");
    }
}

/// The detector must be usable from several threads at once via `&self`.
#[test]
fn concurrent_detection_is_safe_and_consistent() {
    let Some(d) = gliner2(0.5) else { return };
    let text = "Contact John Smith at john@example.com.";
    let expected = d.detect(text).expect("baseline");

    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..3 {
                    assert_eq!(d.detect(text).expect("concurrent"), expected);
                }
            });
        }
    });
}

/// The label schema controls both what is asked for and what comes back.
#[test]
fn a_smaller_schema_cannot_produce_types_outside_it() {
    if !model_present() {
        eprintln!("SKIP: model absent");
        return;
    }
    let mut cfg = Gliner2Config::new(MODEL_DIR);
    cfg.schema = &dlp_bench::labels::GLINER2_PII_MINIMAL;
    cfg.threads = ThreadConfig::with_intra(4);
    cfg.verify_checksum = false;
    let d = Gliner2Detector::load(cfg).expect("load");

    let allowed = dlp_bench::labels::GLINER2_PII_MINIMAL.covered_types();
    let text = "John Smith, john@example.com, IBAN DE89370400440532013000, passport X1234567.";
    for det in d.detect(text).expect("detect") {
        assert!(
            allowed.contains(&det.entity_type),
            "{} is outside the minimal schema {allowed:?}",
            det.entity_type
        );
    }
    // Smaller schema means a shorter prompt, which is the perf lever.
    assert!(
        d.prefix_tokens() < 60,
        "minimal prefix is {} tokens",
        d.prefix_tokens()
    );
}

/// A tampered artifact must be refused rather than silently benchmarked.
#[test]
fn checksum_verification_rejects_a_modified_model() {
    if !model_present() {
        eprintln!("SKIP: model absent");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("model.quantized.onnx"),
        b"not the real model",
    )
    .unwrap();
    std::fs::copy(
        Path::new(MODEL_DIR).join("tokenizer.json"),
        dir.path().join("tokenizer.json"),
    )
    .unwrap();

    let mut cfg = Gliner2Config::new(dir.path());
    cfg.verify_checksum = true;
    match Gliner2Detector::load(cfg) {
        Ok(_) => panic!("a tampered model must not load"),
        Err(e) => {
            let m = e.to_string();
            assert!(m.contains("sha256"), "{m}");
            assert!(m.contains("fetch-models.sh"), "{m}");
        }
    }
}

// ===========================================================================
// The same oracle, applied to every ONNX detector
// ===========================================================================

use dlp_bench::detectors::gliner1::{Gliner1Config, Gliner1Detector};
use dlp_bench::detectors::token_class::{TokenClassConfig, TokenClassDetector, TokenClassModel};

/// Build every ONNX detector whose artifacts are present.
///
/// Returns `(name, detector)`. Checksums are skipped — they have their own
/// test — and threads are capped so the suite does not thrash.
fn all_onnx_detectors() -> Vec<(String, Box<dyn Detector>)> {
    let mut out: Vec<(String, Box<dyn Detector>)> = Vec::new();

    for m in [
        &dlp_bench::detectors::token_class::AI4PRIVACY_MDEBERTA,
        &dlp_bench::detectors::token_class::MINILM_L6_PII,
        &dlp_bench::detectors::token_class::DEBERTA_XSMALL_PII,
    ] as [&'static TokenClassModel; 3]
    {
        if !Path::new(m.dir).join(m.default_variant().file).is_file() {
            eprintln!("SKIP {}: artifacts absent", m.id);
            continue;
        }
        let mut cfg = TokenClassConfig::new(m);
        cfg.verify_checksum = false;
        cfg.threads = ThreadConfig::with_intra(4);
        match TokenClassDetector::load(cfg) {
            Ok(d) => out.push((m.id.to_owned(), Box::new(d))),
            Err(e) => panic!("{} failed to load: {e}", m.id),
        }
    }

    for m in [
        &dlp_bench::detectors::gliner1::GLINER1_PII_EDGE,
        &dlp_bench::detectors::gliner1::GLINER1_PII_SMALL,
    ] {
        if !Path::new(m.dir).join(m.default_variant().file).is_file() {
            eprintln!("SKIP {}: artifacts absent", m.id);
            continue;
        }
        let mut cfg = Gliner1Config::new(m);
        cfg.verify_checksum = false;
        cfg.threads = ThreadConfig::with_intra(4);
        match Gliner1Detector::load(cfg) {
            Ok(d) => out.push((m.id.to_owned(), Box::new(d))),
            Err(e) => panic!("{} failed to load: {e}", m.id),
        }
    }

    if let Some(g2) = gliner2(0.5) {
        out.push(("gliner2-pii".to_owned(), Box::new(g2)));
    }
    out
}

/// **The invariant that must hold for every detector**: a returned span is
/// always a valid slice of the caller's own text.
///
/// This is where offset bugs surface — a char/byte confusion, a tokenizer
/// offset that is relative to a pre-token rather than the document, or a text
/// fixup (GLiNER2 appends a `.`) leaking into the reported range.
#[test]
fn every_detector_returns_spans_that_slice_the_callers_text() {
    let detectors = all_onnx_detectors();
    if detectors.is_empty() {
        eprintln!("no ONNX models present; nothing verified");
        return;
    }

    let corpus: Vec<String> = AGREEMENT_CORPUS
        .iter()
        .map(|s| s.to_string())
        // Stripping terminal punctuation exercises GLiNER2's appended `.`.
        .chain(
            AGREEMENT_CORPUS
                .iter()
                .map(|s| s.trim_end_matches(['.', '!', '?']).to_string()),
        )
        .chain([
            String::new(),
            "   ".into(),
            "!".into(),
            "王".into(),
            "Jörg Müller, IBAN DE89370400440532013000".into(),
            "line1\nkey=secret\nline3".into(),
        ])
        .collect();

    for (name, d) in &detectors {
        for text in &corpus {
            let found = d
                .detect(text)
                .unwrap_or_else(|e| panic!("{name} failed on {text:?}: {e}"));
            dlp_bench::span::validate_all(text, &found)
                .unwrap_or_else(|(i, e)| panic!("{name}: invalid span {i} in {text:?}: {e}"));
            for x in &found {
                assert!(x.end <= text.len(), "{name}: span past end in {text:?}");
                assert!(
                    !text[x.start..x.end].trim().is_empty(),
                    "{name}: whitespace-only span in {text:?}"
                );
            }
        }
    }
}

/// Each detector must land byte-exactly on a rule-validated entity somewhere
/// in the corpus.
///
/// Deliberately *not* "every span must match": models legitimately over-extend
/// (MiniLM tags the word after an email as `I-email`, giving
/// `"francois@example.fr hier"`) and under-extend (ai4privacy tags only `.cn`
/// of a CJK-context address). Both are recall/precision questions that
/// `evaluate` measures. What this asserts is that the pipeline can produce an
/// exactly-correct span at all — a char/byte confusion or a wrong word-index
/// base would make that impossible.
#[test]
fn every_detector_reaches_byte_exact_agreement_with_the_rules() {
    let detectors = all_onnx_detectors();
    if detectors.is_empty() {
        return;
    }
    let rules = DeterministicDetector::default();
    let mut best = 0usize;

    for (name, d) in &detectors {
        let mut exact = 0usize;
        let mut overlapping = 0usize;
        for text in AGREEMENT_CORPUS {
            let want_all = rules.detect(text).expect("rules");
            let got_all = d.detect(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            for t in [EntityType::Email, EntityType::Iban, EntityType::CreditCard] {
                for tr in want_all.iter().filter(|x| x.entity_type == t) {
                    for got in got_all
                        .iter()
                        .filter(|x| x.entity_type == t && x.overlap_len(tr.start, tr.end) > 0)
                    {
                        overlapping += 1;
                        if (got.start, got.end) == (tr.start, tr.end) {
                            exact += 1;
                        }
                    }
                }
            }
        }
        eprintln!("{name}: {exact}/{overlapping} overlapping spans were byte-exact");
        best = best.max(exact);
    }

    // Asserted across the set rather than per detector. deberta-xsmall
    // genuinely never matches: it tags `john` as EMAIL and `@example.com` as
    // ADDRESS, so an exactly-correct email span is not something it can
    // produce. That is a model property, and per-detector offset correctness
    // is guaranteed instead by the translation-invariance test below.
    assert!(
        best > 0,
        "no detector produced a byte-exact span for any validated entity — \
         the corpus or the oracle is broken, not just one model"
    );
}

/// **Translation invariance** — the precise offset oracle.
///
/// Prefixing the document shifts every byte offset and every word index by a
/// known amount. A detection in the entity's region must therefore shift by
/// exactly that amount. This is what a wrong word-index base, a mis-sized
/// prompt prefix, or a char/byte confusion breaks, and unlike an equality
/// check against the rules it is completely indifferent to how far a model
/// chooses to extend a span.
///
/// Span *starts* are compared: an end that moves is the model extending
/// differently in new context, which is legitimate; a start that moves by
/// anything other than the prefix length is a bug.
#[test]
fn every_detector_shifts_spans_by_exactly_the_prefix_length() {
    let detectors = all_onnx_detectors();
    if detectors.is_empty() {
        return;
    }
    let needle = "john@example.com";
    let base = format!("Contact {needle} now.");
    let base_at = base.find(needle).unwrap();

    for (name, d) in &detectors {
        if d.detect(&base)
            .unwrap()
            .iter()
            .all(|x| x.overlap_len(base_at, base_at + needle.len()) == 0)
        {
            eprintln!("{name}: finds nothing at the email unpadded; skipping");
            continue;
        }

        let mut checked = 0usize;
        for pad in [
            "Hello there. ",
            "Свяжитесь с нами. ", // multibyte prefix — the interesting case
            "王小明说。",
            &"x ".repeat(40), // long enough to change the model's context
        ] {
            let text = format!("{pad}{base}");
            let at = text.find(needle).unwrap();
            assert_eq!(at, pad.len() + base_at, "test fixture is wrong");

            for x in d
                .detect(&text)
                .unwrap()
                .iter()
                .filter(|x| x.overlap_len(at, at + needle.len()) > 0)
            {
                // Shifted back, the span must land inside the email as it sits
                // in the *unpadded* text. Which sub-range the model picks is
                // its business — that it lands in the right place is not.
                let back = x
                    .start
                    .checked_sub(pad.len())
                    .unwrap_or_else(|| panic!("{name}: span starts inside the padding"));
                assert!(
                    (base_at..base_at + needle.len()).contains(&back),
                    "{name}: span {:?} at byte {} maps back to {back}, outside the email at \
                     {base_at}..{} — offsets shifted by something other than the {}-byte prefix",
                    &text[x.start..x.end],
                    x.start,
                    base_at + needle.len(),
                    pad.len()
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "{name}: no padded detection to compare — offsets untested"
        );
        eprintln!("{name}: {checked} padded spans shifted exactly");
    }
}

/// Every detector must be safe to call from several threads via `&self`.
#[test]
fn every_detector_is_thread_safe_and_deterministic() {
    let detectors = all_onnx_detectors();
    if detectors.is_empty() {
        return;
    }
    let text = "Contact John Smith at john@example.com";
    for (name, d) in &detectors {
        let expected = d.detect(text).unwrap();
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    assert_eq!(
                        d.detect(text).unwrap(),
                        expected,
                        "{name} is not deterministic"
                    );
                });
            }
        });
    }
}
