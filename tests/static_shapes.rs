//! Does right-padding change what the GLiNER2 graph computes?
//!
//! This is a gate, not a feature test. Fixing the encoder's input shape is only
//! worth measuring if the padded run produces the same answer as the unpadded
//! one, and that cannot be established by reading the code: GLiNER2 **never
//! pads** in normal operation, so this export was traced with an all-ones
//! attention mask. A tracer is free to constant-fold a mask that was constant
//! throughout tracing. If it did, `attention_mask` is decorative, padded tokens
//! leak into attention, and static shapes are dead — along with batching, which
//! is the larger casualty.
//!
//! # What the measurements actually showed
//!
//! The two padded axes behave differently, and conflating them under one
//! tolerance hides the interesting half:
//!
//! * **`seq_len`** — padding the token sequence is numerically exact, drift
//!   ≤ 4e-6 and flat in the amount of padding. The attention mask is real and
//!   honoured. This is the question that decides whether static shapes (and
//!   batching) are possible at all, and the answer is yes.
//! * **`num_words`** — padding the word-index vector moves `span_probs` by up
//!   to 5e-3. Also flat in the amount: 24 extra words perturbs as much as 512
//!   does. The cause is not mask leakage but the extra rows themselves — pad
//!   words point at the last real token, and the span head's reduction over the
//!   word axis sees them.
//!
//! So `num_words` padding is not bit-exact and this file does not pretend it
//! is. It is held to the standard the export itself was validated against —
//! *span-set parity at thresholds 0.3/0.5/0.7* — plus the export's own
//! documented drift versus PyTorch (0.0057) as a ceiling. A padded run that
//! changes no decision at any operating threshold, while sitting inside the
//! noise the artifact already carries, is equivalent for every purpose the
//! benchmark has.
//!
//! Run-to-run nondeterminism was ruled out first: the same input twice is
//! bit-identical at intra_op 1 and 4, so every difference below is caused by
//! padding and nothing else.
//!
//! `count_logits` is checked by argmax as well as by value, because the count
//! head is a hard global gate: `decode` returns *nothing at all* when its
//! argmax is 0. A gate that collapsed under padding would show up as "no
//! detections", which reads like a quiet model rather than a broken one.

#![cfg(feature = "onnx")]

use std::path::Path;

use dlp_bench::Detector;
use dlp_bench::detectors::gliner2::prompt::FixedShape;
use dlp_bench::detectors::gliner2::{Gliner2Config, Gliner2Detector};
use dlp_bench::detectors::onnx::{TensorOut, ThreadConfig};

const MODEL_DIR: &str = "models/gliner2-pii-onnx";

/// Tolerance for padding the token sequence.
///
/// Deliberately tight. This axis is the attention-mask question, and the
/// measured answer is 4e-6 — a leaking mask would not land near here, so a
/// regression that starts honouring the mask less would fail loudly.
const SEQ_TOL: f32 = 1e-4;

/// Ceiling for padding the word-index vector.
///
/// The export's own card reports a max probability drift of 0.005669 against
/// the PyTorch engine it was converted from. Padding must not perturb the
/// output by more than the artifact's existing error; beyond that it would be
/// introducing a new one.
const WORD_DRIFT_CEILING: f32 = 0.005669;

/// The thresholds the export was validated at, and the ones the CLI exposes.
const OPERATING_THRESHOLDS: [f32; 3] = [0.3, 0.5, 0.7];

/// Tolerance for decoded detection scores, which is what a caller sees.
const SCORE_TOL: f32 = 1e-3;

fn model_present() -> bool {
    Path::new(MODEL_DIR).join("model.quantized.onnx").is_file()
}

fn detector() -> Gliner2Detector {
    let mut cfg = Gliner2Config::new(MODEL_DIR);
    cfg.verify_checksum = false;
    cfg.threads = ThreadConfig::with_intra(4);
    Gliner2Detector::load(cfg).expect("load gliner2")
}

/// Documents spanning the benchmark's buckets, plus the awkward cases.
fn corpus() -> Vec<String> {
    let mut v: Vec<String> = vec![
        // Short, one entity.
        "Please contact John Smith at john.smith@example.com.".to_owned(),
        // Non-Latin script.
        "Свяжитесь с Михаил Ястребцов для уточнения деталей заказа.".to_owned(),
        // Structured, no prose.
        "DATABASE_URL=postgres://svcuser:hunter2xyz@db-prod.internal:5432/orders".to_owned(),
        // Almost certainly gate-closed: nothing sensitive at all. Included on
        // purpose — the count gate's behaviour under padding is the highest
        // -consequence thing here.
        "the quick brown fox jumps over the lazy dog again and again".to_owned(),
        // A single word, the degenerate encoder input.
        "hello".to_owned(),
        // Accented Latin plus an address.
        "Jörg Müller updated the shipping address to 12 Hauptstrasse, Munich.".to_owned(),
    ];
    // Longer documents at the benchmark's own bucket sizes.
    for n in [32usize, 128, 512] {
        v.push(dlp_bench::perf::synthetic_document(n));
    }
    v
}

/// Largest absolute difference over the first `rows` rows of a rank-3 tensor.
///
/// Only the real word rows are compared: padded words deliberately produce
/// duplicate rows, which `decode` discards, and comparing them would fail for a
/// reason that does not matter.
fn max_row_diff(a: &TensorOut, b: &TensorOut, rows: usize) -> f32 {
    let per_row: usize = a.shape[1] * a.shape[2];
    assert_eq!(
        per_row,
        b.shape[1] * b.shape[2],
        "span_probs inner shape changed under padding: {:?} vs {:?}",
        a.shape,
        b.shape
    );
    let n = rows * per_row;
    assert!(a.data.len() >= n && b.data.len() >= n, "too few rows");
    a.data[..n]
        .iter()
        .zip(&b.data[..n])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_diff(a: &TensorOut, b: &TensorOut) -> f32 {
    assert_eq!(a.shape, b.shape, "count_logits shape changed under padding");
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Cells whose side of `t` changes between two runs, over `rows` rows.
fn threshold_crossers(a: &TensorOut, b: &TensorOut, rows: usize, t: f32) -> usize {
    let n = rows * a.shape[1] * a.shape[2];
    a.data[..n]
        .iter()
        .zip(&b.data[..n])
        .filter(|(x, y)| (**x >= t) != (**y >= t))
        .count()
}

#[test]
fn sequence_padding_is_numerically_exact() {
    if !model_present() {
        eprintln!("skipping: run scripts/fetch-models.sh");
        return;
    }
    // The load-bearing question: does this export honour `attention_mask`?
    // It was traced with an all-ones mask, so a tracer could have folded it
    // away. If it had, padded tokens would leak into attention and the drift
    // here would be large and would grow with the pad size.
    //
    // Only `seq_len` is padded; `num_words` is held at its natural value so
    // this measures the mask and nothing else.
    let d = detector();
    let mut checked = 0usize;

    for text in corpus() {
        let words = d.word_count(&text);
        let (seq, nw) = d.encoded_size(&text).expect("encode");
        let (base_spans, base_count) = d.raw_outputs(&text, None).expect("unpadded run");

        // Several pad sizes: mask leakage scales with the padded region, so a
        // drift that is flat across these is not leakage.
        let mut drifts = Vec::new();
        for extra in [1usize, 32, 128, 512] {
            let pad = FixedShape {
                seq_len: seq + extra,
                num_words: nw,
            };
            let (spans, count) = d.raw_outputs(&text, Some(pad)).expect("padded run");
            let rows = words.min(base_spans.shape[0]);
            let drift = max_row_diff(&base_spans, &spans, rows);
            assert!(
                drift < SEQ_TOL,
                "padding {seq} -> {} tokens moved span_probs by {drift} on {:?}: \
                 the export is not honouring attention_mask, which kills static \
                 shapes and batching alike",
                pad.seq_len,
                text.chars().take(50).collect::<String>()
            );
            assert_eq!(
                base_count.argmax(),
                count.argmax(),
                "the count gate flipped on pure sequence padding"
            );
            drifts.push(drift);
        }

        // 512 pad tokens must not drift measurably more than 1 does.
        let (first, last) = (drifts[0], drifts[drifts.len() - 1]);
        assert!(
            last <= first + SEQ_TOL,
            "drift grows with the padded region ({first} -> {last}), which is \
             what a partially-honoured mask looks like"
        );
        checked += 1;
    }
    assert!(checked >= 6, "only {checked} documents exercised");
}

#[test]
fn word_padding_changes_no_decision_at_any_operating_threshold() {
    if !model_present() {
        eprintln!("skipping: run scripts/fetch-models.sh");
        return;
    }
    // Padding `text_word_indices` is *not* bit-exact: the pad rows point at
    // the last real token and the span head's reduction over the word axis
    // sees them. Measured at up to 5e-3, and flat in the amount of padding.
    //
    // The bar is therefore the one the export itself was validated to — span
    // -set parity at the operating thresholds — with the artifact's own
    // documented PyTorch drift as a numeric ceiling.
    let d = detector();
    let mut worst = 0.0f32;

    for text in corpus() {
        let words = d.word_count(&text);
        let (seq, nw) = d.encoded_size(&text).expect("encode");
        let (base_spans, base_count) = d.raw_outputs(&text, None).expect("unpadded run");

        for extra_words in [1usize, 24, 96] {
            let pad = FixedShape {
                seq_len: seq + 128,
                num_words: nw + extra_words,
            };
            let (spans, count) = d.raw_outputs(&text, Some(pad)).expect("padded run");
            assert_eq!(
                spans.shape[0], pad.num_words,
                "graph did not honour the padded word count"
            );

            let rows = words.min(base_spans.shape[0]);
            let drift = max_row_diff(&base_spans, &spans, rows);
            worst = worst.max(drift);
            assert!(
                drift <= WORD_DRIFT_CEILING,
                "padding to {} words moved span_probs by {drift}, beyond the \
                 export's own {WORD_DRIFT_CEILING} drift versus PyTorch — padding \
                 would be adding error rather than staying inside it",
                pad.num_words
            );

            for t in OPERATING_THRESHOLDS {
                let n = threshold_crossers(&base_spans, &spans, rows, t);
                assert_eq!(
                    n,
                    0,
                    "{n} span cells changed side of threshold {t} on {:?} — \
                     padding would change the reported span set",
                    text.chars().take(50).collect::<String>()
                );
            }

            assert!(
                max_diff(&base_count, &count) <= WORD_DRIFT_CEILING,
                "count_logits drifted beyond the export's own error"
            );
            assert_eq!(
                base_count.argmax(),
                count.argmax(),
                "the count gate flipped under word padding — decode would return \
                 an entirely different, possibly empty, span set"
            );
        }
    }

    // Not an assertion, a record: this is the number REPORT.md should quote.
    eprintln!("max span_probs drift from word padding: {worst:.6}");
}

#[test]
fn masked_real_tokens_are_suppressed_not_merely_small() {
    if !model_present() {
        eprintln!("skipping: run scripts/fetch-models.sh");
        return;
    }
    let d = detector();

    // The control. `[PAD]` may embed near zero, in which case the positive test
    // above passes whether or not the mask does anything. Padding with tokens
    // that carry strong PII signal — and masking them — separates the two: if
    // the mask is decorative, these leak into attention and move the outputs.
    let contaminant = " password secret token api_key ssn passport credit card";

    for text in [
        "Please contact John Smith at john.smith@example.com.",
        "the quick brown fox jumps over the lazy dog",
        "Свяжитесь с Михаил Ястребцов для уточнения деталей заказа.",
    ] {
        let words = d.word_count(text);
        let (seq, _) = d.encoded_size(text).expect("encode");
        let (base_spans, base_count) = d.raw_outputs(text, None).expect("unpadded run");

        // Encoding `text + contaminant` and then padding back to a shape that
        // covers only `text`'s real length is not expressible through
        // `raw_outputs`, so instead: encode the concatenation unpadded, and
        // compare against `text` alone padded to the same size. If masking
        // works, the padded-`text` run must match plain `text`, and must
        // *differ* from the concatenation.
        let combined = format!("{text}{contaminant}");
        let (comb_seq, comb_nw) = d.encoded_size(&combined).expect("encode");
        let pad = FixedShape {
            seq_len: comb_seq,
            num_words: comb_nw,
        };

        let (padded_spans, padded_count) = d.raw_outputs(text, Some(pad)).expect("padded run");
        let (comb_spans, _comb_count) = d.raw_outputs(&combined, None).expect("combined run");

        let rows = words.min(base_spans.shape[0]);
        let to_base = max_row_diff(&base_spans, &padded_spans, rows);
        assert!(
            to_base < SEQ_TOL,
            "padding {seq} -> {comb_seq} tokens moved span_probs by {to_base}"
        );
        assert_eq!(base_count.argmax(), padded_count.argmax());

        // And the control's other half: appending that text *unmasked* must
        // actually change the answer. If it does not, the comparison above is
        // vacuous — the model would be ignoring the tail either way, and this
        // test would prove nothing about the mask.
        let to_comb = max_row_diff(&base_spans, &comb_spans, rows);
        assert!(
            to_comb > SEQ_TOL,
            "appending {contaminant:?} unmasked changed span_probs by only \
             {to_comb}, so the no-op result above is not evidence that \
             attention_mask works — this test needs a stronger contaminant"
        );
    }
}

#[test]
fn a_padded_detector_finds_exactly_what_an_unpadded_one_finds() {
    if !model_present() {
        eprintln!("skipping: run scripts/fetch-models.sh");
        return;
    }
    // End-to-end rather than tensor-level: this is the property the benchmark
    // actually reports on, and it exercises `decode`'s clamp of `num_words`,
    // which padding turns from a defensive check into a load-bearing one.
    //
    // Sequential, and the first detector is dropped before the second loads:
    // two 426 MB sessions at once would be ~900 MB of weights for no reason.
    let corpus = corpus();
    let baseline: Vec<Vec<dlp_bench::Detection>> = {
        let d = detector();
        corpus
            .iter()
            .map(|t| d.detect(t).expect("detect"))
            .collect()
    };

    // One shape large enough for the whole corpus, which is what a bucketed
    // deployment would do for its largest bucket.
    let d = detector();
    let (max_seq, max_words) = corpus.iter().fold((0, 0), |(s, w), t| {
        let (a, b) = d.encoded_size(t).expect("encode");
        (s.max(a), w.max(b))
    });
    drop(d);

    let mut cfg = Gliner2Config::new(MODEL_DIR);
    cfg.verify_checksum = false;
    cfg.threads = ThreadConfig::with_intra(4);
    cfg.fixed_shape = Some(FixedShape {
        seq_len: max_seq,
        num_words: max_words,
    });
    let padded = Gliner2Detector::load(cfg).expect("load padded gliner2");

    let mut total = 0usize;
    for (text, want) in corpus.iter().zip(&baseline) {
        let got = padded.detect(text).expect("detect");
        assert_eq!(
            got.len(),
            want.len(),
            "detection count differs on {:?}",
            &text[..text.len().min(60)]
        );
        for (g, w) in got.iter().zip(want) {
            assert_eq!(
                (g.entity_type, g.start, g.end),
                (w.entity_type, w.start, w.end)
            );
            assert!(
                (g.score - w.score).abs() < SCORE_TOL,
                "score drift {} on {:?}",
                (g.score - w.score).abs(),
                &text[..text.len().min(60)]
            );
        }
        total += got.len();
    }

    // A gate that collapsed would make every list empty and every comparison
    // above pass. The corpus has real PII in it; zero detections is a failure.
    assert!(
        total > 0,
        "the padded detector found nothing across {} documents — the count gate \
         has collapsed and the equality above is vacuous",
        corpus.len()
    );
}

#[test]
fn padding_and_dimension_overrides_must_agree() {
    if !model_present() {
        eprintln!("skipping: run scripts/fetch-models.sh");
        return;
    }
    use dlp_bench::runtime::{DimOverride, SessionOptions};

    // Pinning the graph without padding the inputs, or padding to a size the
    // graph was not told about, both fail on every call. Catching it at load
    // turns a per-call shape error into one message naming the actual mistake.
    let mut cfg = Gliner2Config::new(MODEL_DIR);
    cfg.verify_checksum = false;
    cfg.session = SessionOptions {
        dimension_overrides: vec![
            DimOverride {
                name: "seq_len".into(),
                size: 512,
            },
            DimOverride {
                name: "num_words".into(),
                size: 160,
            },
        ],
        ..Default::default()
    };
    let err = match Gliner2Detector::load(cfg.clone()) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a pinned graph with unpadded inputs must not load"),
    };
    assert!(err.contains("no padding was configured"), "{err}");

    cfg.fixed_shape = Some(FixedShape {
        seq_len: 256,
        num_words: 160,
    });
    let err = match Gliner2Detector::load(cfg) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("seq_len 512 against padding 256 must not load"),
    };
    assert!(err.contains("padded to 256"), "{err}");
}
