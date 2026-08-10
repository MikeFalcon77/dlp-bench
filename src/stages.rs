//! Per-stage timing inside a single `detect()` call.
//!
//! The benchmark's headline number is one wall-clock measurement around
//! [`Detector::detect`](crate::detector::Detector::detect), which is the honest
//! thing to report but tells you nothing about *where* the time went. For a
//! model missing its latency target by 7x, "is it the transformer or is it our
//! own memory copies" is the first question, and answering it by reasoning about
//! the code is how people spend a week optimising something that was 0.3% of the
//! run.
//!
//! Two constraints shape the design:
//!
//! - **The instrumented path must not become the measured path.** Adding timers
//!   to `detect` would silently change every number already in `out/`. So a
//!   [`StageTimer`] has two states, and the disabled one is a null object: no
//!   allocation, no map lookup, and crucially no `Instant::now()`.
//! - **A stage split must be honest about what it does not cover.** Stages are
//!   recorded, summed and compared against the enclosing wall clock; the
//!   difference is reported as unattributed rather than hidden by normalising
//!   the shares to 100%.
//!
//! This module deliberately depends on nothing — not `ort`, not a detector — so
//! it compiles under `--no-default-features` and can be used by any detector
//! that wants to explain itself.

use std::time::Instant;

use serde::{Deserialize, Serialize};

/// One attributable phase of a detector call.
///
/// The three ORT stages are separate because they answer different questions:
/// [`InputCopy`](Stage::InputCopy) and [`OutputCopy`](Stage::OutputCopy) are
/// *our* cost (`data.to_vec()` on the way in, `to_vec()` on the way out), while
/// [`SessionRun`](Stage::SessionRun) is the runtime's. Collapsing them into one
/// "inference" bar would make the copies unfalsifiable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Word segmentation and text normalisation, before any tokenizer call.
    Prepare,
    /// Building the encoder input: sub-token lookup, prefix concatenation.
    Tokenize,
    /// Copying host buffers into ONNX Runtime tensors.
    InputCopy,
    /// `Session::run` — the forward pass.
    SessionRun,
    /// Extracting output tensors back into owned `Vec<f32>`.
    OutputCopy,
    /// Turning raw scores into validated, deduplicated spans.
    Decode,
}

impl Stage {
    pub const COUNT: usize = 6;

    /// Every stage, in pipeline order. Reported in this order so a printed row
    /// reads left-to-right as the call actually proceeds.
    pub const ALL: [Stage; Self::COUNT] = [
        Stage::Prepare,
        Stage::Tokenize,
        Stage::InputCopy,
        Stage::SessionRun,
        Stage::OutputCopy,
        Stage::Decode,
    ];

    pub const fn index(self) -> usize {
        match self {
            Stage::Prepare => 0,
            Stage::Tokenize => 1,
            Stage::InputCopy => 2,
            Stage::SessionRun => 3,
            Stage::OutputCopy => 4,
            Stage::Decode => 5,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Stage::Prepare => "prepare",
            Stage::Tokenize => "tokenize",
            Stage::InputCopy => "input_copy",
            Stage::SessionRun => "session_run",
            Stage::OutputCopy => "output_copy",
            Stage::Decode => "decode",
        }
    }

    /// Short label for the printed table, padded to a common width by the caller.
    pub const fn short(self) -> &'static str {
        match self {
            Stage::Prepare => "prepare",
            Stage::Tokenize => "tokenize",
            Stage::InputCopy => "in-copy",
            Stage::SessionRun => "session.run",
            Stage::OutputCopy => "out-copy",
            Stage::Decode => "decode",
        }
    }
}

/// Accumulates per-stage durations for one `detect()` call.
///
/// A fixed array indexed by a `Copy` enum: no allocation, no hashing, no string
/// comparison. When disabled every method is a predictable branch that does
/// nothing, which is what lets the same code path serve both the headline
/// measurement and the breakdown.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimer {
    enabled: bool,
    nanos: [u64; Stage::COUNT],
}

impl StageTimer {
    /// A timer that records nothing. This is what the normal path uses.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            nanos: [0; Stage::COUNT],
        }
    }

    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            nanos: [0; Stage::COUNT],
        }
    }

    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start point for a stage, or `None` when disabled.
    ///
    /// Returning `Option<Instant>` rather than calling `Instant::now()`
    /// unconditionally is the whole point: a clock read is tens of nanoseconds
    /// and there are twelve of them per call.
    #[inline]
    pub fn start(&self) -> Option<Instant> {
        if self.enabled {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Attribute the time since `since` to `stage`. A `None` start is ignored.
    ///
    /// Repeated calls for the same stage accumulate, so a stage entered more
    /// than once per document (a chunked forward pass, say) sums correctly.
    #[inline]
    pub fn record(&mut self, stage: Stage, since: Option<Instant>) {
        if let Some(t) = since {
            self.nanos[stage.index()] += t.elapsed().as_nanos() as u64;
        }
    }

    /// Run `f`, attributing its duration to `stage`.
    #[inline]
    pub fn time<T>(&mut self, stage: Stage, f: impl FnOnce() -> T) -> T {
        let t = self.start();
        let out = f();
        self.record(stage, t);
        out
    }

    pub fn nanos(&self, stage: Stage) -> u64 {
        self.nanos[stage.index()]
    }

    /// Sum of all stages. Less than the enclosing wall clock by whatever was
    /// not instrumented — mutex acquisition, the closure boundary, error paths.
    pub fn total_nanos(&self) -> u64 {
        self.nanos.iter().sum()
    }

    pub fn reset(&mut self) {
        self.nanos = [0; Stage::COUNT];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn a_disabled_timer_never_reads_the_clock() {
        let mut t = StageTimer::disabled();
        assert!(!t.is_enabled());
        assert!(t.start().is_none(), "disabled must not produce an Instant");
        let v = t.time(Stage::SessionRun, || {
            sleep(Duration::from_millis(2));
            7
        });
        assert_eq!(v, 7, "the closure still runs");
        assert_eq!(t.total_nanos(), 0, "but nothing is recorded");
    }

    #[test]
    fn an_enabled_timer_accumulates_per_stage_and_across_calls() {
        let mut t = StageTimer::enabled();
        t.time(Stage::Prepare, || sleep(Duration::from_millis(2)));
        t.time(Stage::SessionRun, || sleep(Duration::from_millis(5)));
        // Entered twice: a chunked detector does this, and it must sum.
        t.time(Stage::SessionRun, || sleep(Duration::from_millis(5)));

        let ms = |s: Stage| t.nanos(s) as f64 / 1e6;
        assert!(ms(Stage::Prepare) >= 1.5, "prepare: {}", ms(Stage::Prepare));
        assert!(
            ms(Stage::SessionRun) >= 9.0,
            "two 5ms runs must sum: {}",
            ms(Stage::SessionRun)
        );
        assert_eq!(t.nanos(Stage::Decode), 0, "untouched stages stay zero");
        assert!(t.total_nanos() >= t.nanos(Stage::SessionRun));

        t.reset();
        assert_eq!(t.total_nanos(), 0);
        assert!(t.is_enabled(), "reset must not disable");
    }

    #[test]
    fn stage_all_covers_every_variant_with_a_unique_index() {
        // The exhaustive match makes adding a variant without updating ALL a
        // compile error rather than a silently missing column in the report.
        for s in Stage::ALL {
            match s {
                Stage::Prepare
                | Stage::Tokenize
                | Stage::InputCopy
                | Stage::SessionRun
                | Stage::OutputCopy
                | Stage::Decode => {}
            }
        }
        let mut seen = [false; Stage::COUNT];
        for s in Stage::ALL {
            assert!(!seen[s.index()], "duplicate index for {}", s.as_str());
            seen[s.index()] = true;
        }
        assert!(seen.iter().all(|b| *b), "ALL does not cover every index");
    }

    #[test]
    fn stage_names_are_stable_json_keys() {
        // These strings land in out/*.json; renaming one silently breaks every
        // collator that reads them.
        assert_eq!(Stage::SessionRun.as_str(), "session_run");
        assert_eq!(
            serde_json::to_string(&Stage::InputCopy).unwrap(),
            "\"input_copy\""
        );
    }
}
