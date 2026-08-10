//! Performance measurement.
//!
//! Separate from [`crate::eval`] on purpose: quality evaluation parses
//! datasets, allocates metric structures and builds error reports, none of
//! which belong inside a latency measurement.
//!
//! What is timed is the user-visible DLP path and nothing else:
//!
//! ```text
//! text -> tokenize -> inference -> decode spans -> Vec<Detection>
//! ```
//!
//! Model loading, dataset parsing, JSON serialization and logging all happen
//! outside the timed region.

pub mod latency;
pub mod memory;

use serde::{Deserialize, Serialize};

use crate::detector::Detector;
use crate::error::DetectorError;
use crate::stages::{Stage, StageTimer};

pub use latency::{Percentiles, TimingSample, percentiles};
pub use memory::MemorySnapshot;

/// Input size buckets, in approximate whitespace-token counts.
///
/// Approximate is fine and is stated as such: the model's own sub-token count
/// differs, and the report records the real byte and word counts alongside.
pub const DEFAULT_BUCKETS: &[usize] = &[32, 128, 512, 1024];

#[derive(Debug, Clone)]
pub struct PerfConfig {
    pub buckets: Vec<usize>,
    /// Iterations per bucket after warm-up.
    pub iterations: usize,
    /// Warm-up iterations, excluded from the statistics.
    pub warmup: usize,
    /// Thread counts to measure throughput at. Each uses its own detector
    /// instance, so this also exposes the per-instance memory cost.
    pub thread_counts: Vec<usize>,
    pub measure_throughput: bool,
    /// Run a second, instrumented pass to attribute time to each [`Stage`].
    ///
    /// Off by default and never folded into the headline pass: the point of the
    /// headline number is that it measures what a caller pays, and a caller
    /// does not pay for our timers.
    pub stage_timing: bool,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            buckets: DEFAULT_BUCKETS.to_vec(),
            iterations: 30,
            warmup: 3,
            thread_counts: vec![1, 2, 4, 8],
            measure_throughput: true,
            stage_timing: false,
        }
    }
}

/// Cold-start timings, each phase measured separately.
///
/// Measured in-process rather than by spawning a subprocess: process start and
/// dynamic linking are identical for every detector and would only add noise.
/// The methodology is stated in the README so the number is not mistaken for a
/// full process-launch measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdStart {
    /// Loading the model and building the ONNX session.
    pub model_load_ms: f64,
    /// First inference on a warm-up document, from a cold session.
    pub first_inference_ms: f64,
    /// Sum of the two — what an endpoint agent pays before it can inspect a
    /// prompt if it loads lazily.
    pub total_ms: f64,
    pub rss_after_load: Option<MemorySnapshot>,
}

/// Latency for one input-size bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketLatency {
    pub target_tokens: usize,
    pub actual_words: usize,
    pub actual_bytes: usize,
    pub iterations: usize,
    #[serde(flatten)]
    pub percentiles: Percentiles,
    /// Documents per second implied by the median, single-threaded.
    pub docs_per_sec_p50: f64,
}

/// Percentiles for one stage within one bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageStat {
    pub stage: Stage,
    #[serde(flatten)]
    pub percentiles: Percentiles,
    /// This stage's p50 as a fraction of the instrumented call's p50.
    ///
    /// Shares do not sum to 1: each is an independent percentile, and whatever
    /// was never instrumented is missing from all of them. See
    /// [`StageBreakdown::unattributed_p50_ms`].
    pub share_of_p50: f64,
}

/// Where the time went, for one input-size bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageBreakdown {
    pub target_tokens: usize,
    pub actual_words: usize,
    pub iterations: usize,
    /// Wall clock of the *instrumented* calls.
    ///
    /// Reported next to the uninstrumented [`BucketLatency`] for the same
    /// bucket so the cost of the instrumentation is a measurement rather than
    /// an assertion. If these differ materially the breakdown is not describing
    /// the run you care about, and that has to be visible.
    #[serde(flatten)]
    pub instrumented: Percentiles,
    pub stages: Vec<StageStat>,
    /// Instrumented p50 minus the sum of the stage p50s.
    ///
    /// Never clamped, and may be slightly negative: percentiles are computed
    /// per stage, so the median call's stages and the median of each stage are
    /// not the same thing. Printed so nobody reads the breakdown as exhaustive
    /// when it is not — for GLiNER2 it also carries the mutex acquisition,
    /// which is deliberately outside every stage.
    pub unattributed_p50_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputPoint {
    pub threads: usize,
    pub docs_per_sec: f64,
    pub words_per_sec: f64,
    /// Speed-up over the 1-thread measurement.
    pub scaling: f64,
    pub rss: Option<MemorySnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfReport {
    pub detector: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_size_bytes: Option<u64>,
    /// Fixed per-call token cost imposed by a zero-shot label schema, if any.
    /// For GLiNER2 this dominates short-prompt latency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_prefix_tokens: Option<usize>,
    /// How the inference runtime was configured.
    ///
    /// Recorded unconditionally for ONNX detectors. Latency taken under CoreML
    /// and latency taken under CPU are different measurements, and before this
    /// field the only record of which was which was the captured command line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<crate::runtime::RuntimeInfo>,
    pub cold_start: ColdStart,
    pub buckets: Vec<BucketLatency>,
    /// Per-stage attribution from the instrumented second pass, when
    /// `--stage-timing` was requested. Empty otherwise, so a default run emits
    /// the same JSON it always has.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stage_breakdown: Vec<StageBreakdown>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub throughput: Vec<ThroughputPoint>,
    pub warm_rss: Option<MemorySnapshot>,
    pub peak_rss: Option<MemorySnapshot>,
    pub iterations: usize,
    pub warmup: usize,
}

/// Synthetic documents of controlled size, built once outside the timed region.
///
/// Content is representative of AI-DLP traffic — prose with embedded PII plus
/// config and log lines — rather than lorem ipsum, because tokenizer behaviour
/// and the number of candidate spans both depend on what the text looks like.
pub fn synthetic_document(target_words: usize) -> String {
    const FRAGMENTS: &[&str] = &[
        "Please contact John Smith at john.smith@example.com regarding the invoice.",
        "The customer called from +1-415-555-0198 last Tuesday afternoon.",
        "DATABASE_URL=postgres://svcuser:hunter2xyz@db-prod.internal:5432/orders",
        "2026-08-08T10:22:31Z user=maria.garcia@example.es ip=203.0.113.44 action=login",
        "Свяжитесь с Михаил Ястребцов для уточнения деталей заказа.",
        "Card on file ends 4111 1111 1111 1111 and expires next March.",
        "let apiKey = process.env.OPENAI_API_KEY; // do not commit this value",
        "Passport X1234567 was issued in Lisbon and expires 2031-04-02.",
        "Jörg Müller updated the shipping address to 12 Hauptstrasse, Munich.",
        "SELECT id, email FROM customers WHERE created_at > '2026-01-01';",
    ];

    let mut out = String::with_capacity(target_words * 8);
    let mut words = 0usize;
    let mut i = 0usize;
    while words < target_words {
        let f = FRAGMENTS[i % FRAGMENTS.len()];
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(f);
        words += f.split_whitespace().count();
        i += 1;
    }
    out
}

/// Measure cold start: session construction, then a first inference.
///
/// `load` is a closure so the timer brackets exactly the load, with nothing
/// else in scope.
pub fn measure_cold_start<D: Detector>(
    load: impl FnOnce() -> Result<D, DetectorError>,
    warmup_text: &str,
) -> Result<(D, ColdStart), DetectorError> {
    let t0 = std::time::Instant::now();
    let detector = load()?;
    let model_load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let rss_after_load = memory::snapshot();

    let t1 = std::time::Instant::now();
    detector.detect(warmup_text)?;
    let first_inference_ms = t1.elapsed().as_secs_f64() * 1000.0;

    Ok((
        detector,
        ColdStart {
            model_load_ms,
            first_inference_ms,
            total_ms: model_load_ms + first_inference_ms,
            rss_after_load,
        },
    ))
}

/// Measure warm latency across the configured buckets.
pub fn measure_latency(
    detector: &dyn Detector,
    cfg: &PerfConfig,
) -> Result<Vec<BucketLatency>, DetectorError> {
    let mut out = Vec::with_capacity(cfg.buckets.len());

    for &target in &cfg.buckets {
        // Built before timing starts; generating text is not inference.
        let text = synthetic_document(target);
        let actual_words = text.split_whitespace().count();
        let actual_bytes = text.len();

        for _ in 0..cfg.warmup {
            detector.detect(&text)?;
        }

        let mut samples = Vec::with_capacity(cfg.iterations);
        for _ in 0..cfg.iterations {
            let t = std::time::Instant::now();
            let found = detector.detect(&text)?;
            samples.push(t.elapsed());
            // Keep the optimiser from eliminating the call.
            std::hint::black_box(found);
        }

        let p = percentiles(&mut samples);
        out.push(BucketLatency {
            target_tokens: target,
            actual_words,
            actual_bytes,
            iterations: cfg.iterations,
            docs_per_sec_p50: if p.p50_ms > 0.0 {
                1000.0 / p.p50_ms
            } else {
                0.0
            },
            percentiles: p,
        });
    }

    Ok(out)
}

/// Attribute latency to stages, across the configured buckets.
///
/// A separate pass from [`measure_latency`], run after it, for two reasons:
/// the headline numbers must stay uninstrumented, and running both lets the
/// instrumentation overhead be read off the report instead of argued about.
///
/// The caller is expected to have checked
/// [`Detector::supports_stage_timing`]; a detector that has not opted in
/// produces a table of zeros, which is worse than an error.
pub fn measure_stages(
    detector: &dyn Detector,
    cfg: &PerfConfig,
) -> Result<Vec<StageBreakdown>, DetectorError> {
    let mut out = Vec::with_capacity(cfg.buckets.len());

    for &target in &cfg.buckets {
        let text = synthetic_document(target);
        let actual_words = text.split_whitespace().count();

        for _ in 0..cfg.warmup {
            detector.detect(&text)?;
        }

        let mut wall = Vec::with_capacity(cfg.iterations);
        let mut per_stage: Vec<Vec<TimingSample>> = (0..Stage::COUNT)
            .map(|_| Vec::with_capacity(cfg.iterations))
            .collect();

        for _ in 0..cfg.iterations {
            let mut timer = StageTimer::enabled();
            let t = std::time::Instant::now();
            let found = detector.detect_staged(&text, &mut timer)?;
            wall.push(t.elapsed());
            std::hint::black_box(found);

            for s in Stage::ALL {
                per_stage[s.index()].push(std::time::Duration::from_nanos(timer.nanos(s)));
            }
        }

        let instrumented = percentiles(&mut wall);
        let stages: Vec<StageStat> = Stage::ALL
            .iter()
            .map(|&s| {
                let p = percentiles(&mut per_stage[s.index()]);
                StageStat {
                    stage: s,
                    share_of_p50: if instrumented.p50_ms > 0.0 {
                        p.p50_ms / instrumented.p50_ms
                    } else {
                        0.0
                    },
                    percentiles: p,
                }
            })
            .collect();

        let attributed: f64 = stages.iter().map(|s| s.percentiles.p50_ms).sum();
        out.push(StageBreakdown {
            target_tokens: target,
            actual_words,
            iterations: cfg.iterations,
            unattributed_p50_ms: instrumented.p50_ms - attributed,
            instrumented,
            stages,
        });
    }

    Ok(out)
}

/// Measure throughput at several thread counts.
///
/// `build` is called once per worker: each thread gets its own detector, and
/// therefore its own ONNX session. That is the honest way to measure CPU
/// scaling for a detector whose session is not internally parallel-safe, and
/// the resulting RSS is reported so the memory cost of that choice is visible.
pub fn measure_throughput<D: Detector + 'static>(
    build: &(dyn Fn() -> Result<D, DetectorError> + Sync),
    cfg: &PerfConfig,
    docs_per_thread: usize,
) -> Result<Vec<ThroughputPoint>, DetectorError> {
    let text = synthetic_document(128);
    let words = text.split_whitespace().count();
    let mut out = Vec::new();
    let mut baseline: Option<f64> = None;

    for &threads in &cfg.thread_counts {
        let detectors: Vec<D> = (0..threads)
            .map(|_| build())
            .collect::<Result<Vec<_>, _>>()?;

        // Warm every session before the clock starts.
        for d in &detectors {
            d.detect(&text)?;
        }

        let t0 = std::time::Instant::now();
        std::thread::scope(|s| {
            for d in &detectors {
                let text = &text;
                s.spawn(move || {
                    for _ in 0..docs_per_thread {
                        let _ = std::hint::black_box(d.detect(text));
                    }
                });
            }
        });
        let elapsed = t0.elapsed().as_secs_f64();

        let total_docs = (threads * docs_per_thread) as f64;
        let docs_per_sec = if elapsed > 0.0 {
            total_docs / elapsed
        } else {
            0.0
        };
        let rss = memory::snapshot();
        baseline.get_or_insert(docs_per_sec);

        out.push(ThroughputPoint {
            threads,
            docs_per_sec,
            words_per_sec: docs_per_sec * words as f64,
            scaling: baseline.map_or(1.0, |b| if b > 0.0 { docs_per_sec / b } else { 0.0 }),
            rss,
        });
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Detection;

    struct Noop;
    impl Detector for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn detect(&self, _: &str) -> Result<Vec<Detection>, DetectorError> {
            Ok(Vec::new())
        }
    }

    /// Spends a known, lopsided amount of time in two stages.
    struct Staged;
    impl Detector for Staged {
        fn name(&self) -> &str {
            "staged"
        }
        fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
            self.detect_staged(text, &mut StageTimer::disabled())
        }
        fn supports_stage_timing(&self) -> bool {
            true
        }
        fn detect_staged(
            &self,
            _: &str,
            timer: &mut StageTimer,
        ) -> Result<Vec<Detection>, DetectorError> {
            timer.time(Stage::Prepare, || {
                std::thread::sleep(std::time::Duration::from_millis(1))
            });
            timer.time(Stage::SessionRun, || {
                std::thread::sleep(std::time::Duration::from_millis(9))
            });
            Ok(Vec::new())
        }
    }

    #[test]
    fn stage_breakdown_attributes_time_to_the_dominant_stage() {
        let cfg = PerfConfig {
            buckets: vec![32],
            iterations: 5,
            warmup: 1,
            thread_counts: vec![],
            measure_throughput: false,
            stage_timing: true,
        };
        let out = measure_stages(&Staged, &cfg).unwrap();
        assert_eq!(out.len(), 1);
        let b = &out[0];
        assert_eq!(b.stages.len(), Stage::COUNT, "every stage gets a column");

        let by = |s: Stage| b.stages.iter().find(|x| x.stage == s).unwrap();
        assert!(
            by(Stage::SessionRun).share_of_p50 > 0.7,
            "9ms of 10 should dominate, got {}",
            by(Stage::SessionRun).share_of_p50
        );
        assert!(by(Stage::Prepare).percentiles.p50_ms >= 0.5);
        assert_eq!(
            by(Stage::Decode).percentiles.p50_ms,
            0.0,
            "an untouched stage must read zero, not be omitted"
        );

        // Sleeps overshoot, so the attributed sum can exceed the wall clock by
        // a hair; the field is never clamped, only bounded.
        assert!(
            b.unattributed_p50_ms.abs() < b.instrumented.p50_ms * 0.5,
            "unattributed {} vs wall {}",
            b.unattributed_p50_ms,
            b.instrumented.p50_ms
        );
    }

    #[test]
    fn a_detector_without_stage_support_yields_an_all_zero_breakdown() {
        // Documents *why* the CLI must gate on `supports_stage_timing`: this is
        // what the report would otherwise show, and it reads like a finding.
        let cfg = PerfConfig {
            buckets: vec![32],
            iterations: 3,
            warmup: 0,
            thread_counts: vec![],
            measure_throughput: false,
            stage_timing: true,
        };
        let out = measure_stages(&Noop, &cfg).unwrap();
        assert!(out[0].stages.iter().all(|s| s.percentiles.p50_ms == 0.0));
    }

    #[test]
    fn synthetic_documents_reach_the_requested_size_and_are_deterministic() {
        for &target in DEFAULT_BUCKETS {
            let a = synthetic_document(target);
            let b = synthetic_document(target);
            assert_eq!(a, b, "synthetic text must be deterministic");
            let n = a.split_whitespace().count();
            assert!(n >= target, "asked for {target} words, got {n}");
            // Never wildly overshoot: at most one fragment beyond the target.
            assert!(n < target + 20, "overshot {target} by too much: {n}");
        }
    }

    #[test]
    fn synthetic_documents_contain_real_pii_shapes() {
        let d = synthetic_document(128);
        assert!(d.contains('@'), "no email");
        assert!(d.contains("postgres://"), "no connection string");
        assert!(d.contains("Ястребцов"), "no non-ASCII content");
        assert!(d.is_char_boundary(d.len()));
    }

    #[test]
    fn latency_measurement_produces_ordered_percentiles() {
        let cfg = PerfConfig {
            buckets: vec![32, 128],
            iterations: 5,
            warmup: 1,
            thread_counts: vec![],
            measure_throughput: false,
            stage_timing: false,
        };
        let out = measure_latency(&Noop, &cfg).unwrap();
        assert_eq!(out.len(), 2);
        for b in &out {
            assert_eq!(b.iterations, 5);
            assert!(b.actual_words >= b.target_tokens);
            assert!(b.actual_bytes > 0);
            let p = &b.percentiles;
            assert!(p.min_ms <= p.p50_ms && p.p50_ms <= p.p90_ms);
            assert!(p.p90_ms <= p.p95_ms && p.p95_ms <= p.p99_ms && p.p99_ms <= p.max_ms);
        }
    }

    #[test]
    fn cold_start_separates_load_from_first_inference() {
        let (_d, cs) = measure_cold_start(
            || {
                std::thread::sleep(std::time::Duration::from_millis(12));
                Ok(Noop)
            },
            "warm up",
        )
        .unwrap();
        assert!(
            cs.model_load_ms >= 10.0,
            "load not measured: {}",
            cs.model_load_ms
        );
        assert!(cs.first_inference_ms >= 0.0);
        assert!((cs.total_ms - (cs.model_load_ms + cs.first_inference_ms)).abs() < 1e-9);
    }

    #[test]
    fn throughput_reports_scaling_relative_to_one_thread() {
        let cfg = PerfConfig {
            thread_counts: vec![1, 2],
            ..Default::default()
        };
        let out = measure_throughput(&(|| Ok(Noop)), &cfg, 20).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].threads, 1);
        assert!((out[0].scaling - 1.0).abs() < 1e-9, "baseline must be 1.0");
        assert!(out[1].docs_per_sec > 0.0);
        assert!(out[0].words_per_sec > 0.0);
    }
}
