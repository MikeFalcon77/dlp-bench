//! The one abstraction the evaluator is allowed to know about.
//!
//! Everything model-specific — label vocabularies, tokenisation, thresholds,
//! tensor layouts — lives inside an implementation of [`Detector`]. If the
//! evaluator ever needs to branch on which detector it is running, that is a
//! design failure.

use crate::entity::Detection;
use crate::error::DetectorError;
use crate::stages::StageTimer;

/// A PII detector.
///
/// Synchronous by design. Inference here is CPU-bound and the benchmark wants
/// to measure exactly one thing — `text -> Vec<Detection>` — without an async
/// runtime's scheduling noise in the timing. Detectors that hold a mutable ORT
/// session use interior mutability rather than forcing `&mut self` on callers.
pub trait Detector: Send + Sync {
    /// Stable identifier, matching the CLI `--detector` value.
    fn name(&self) -> &str;

    /// Detect PII in `text`.
    ///
    /// Returned spans are UTF-8 byte offsets into `text`, validated, sorted by
    /// `(start, end, type)` and free of exact duplicates. Implementations get
    /// this for free via [`finish`].
    fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError>;

    /// Detect, attributing time to each [`Stage`](crate::stages::Stage).
    ///
    /// The default forwards to [`detect`](Detector::detect) and records nothing,
    /// so a detector opts in by overriding this *and*
    /// [`supports_stage_timing`](Detector::supports_stage_timing) together. The
    /// contract is that `detect_staged(text, &mut StageTimer::disabled())` must
    /// return exactly what `detect(text)` returns — the instrumented path is the
    /// same path, not a second implementation that can drift.
    fn detect_staged(
        &self,
        text: &str,
        timer: &mut StageTimer,
    ) -> Result<Vec<Detection>, DetectorError> {
        let _ = timer;
        self.detect(text)
    }

    /// Whether [`detect_staged`](Detector::detect_staged) produces a real
    /// breakdown rather than an all-zero one.
    ///
    /// The perf command refuses `--stage-timing` against a detector that
    /// returns `false`, because a table of zeros reads as "these stages are
    /// free" rather than "this was never measured".
    fn supports_stage_timing(&self) -> bool {
        false
    }

    /// Confidence threshold, if the detector has one.
    ///
    /// `None` means thresholding is not meaningful (the deterministic detector
    /// is either sure or silent), and the sweep command will say so rather than
    /// producing a flat, misleading curve.
    fn threshold(&self) -> Option<f32> {
        None
    }

    /// Return a copy with a different threshold, for sweeps.
    ///
    /// `None` when unsupported. Returning a new instance rather than mutating
    /// avoids reloading a 446 MB session per sweep point.
    fn with_threshold(&self, _threshold: f32) -> Option<Box<dyn Detector>> {
        None
    }

    /// Metadata recorded in the run report so a result can be traced back to an
    /// exact artifact. Empty for detectors with no model.
    fn artifact_info(&self) -> Vec<ArtifactInfo> {
        Vec::new()
    }

    /// Approximate resident size of loaded model artifacts, in bytes.
    fn model_size_bytes(&self) -> Option<u64> {
        None
    }

    /// How the inference runtime was configured, for the run report.
    ///
    /// `None` for detectors with no runtime to describe. Recorded because a
    /// latency number taken under CoreML and one taken under CPU are different
    /// measurements, and until this existed the only record of which was which
    /// was the captured command line.
    fn runtime_info(&self) -> Option<crate::runtime::RuntimeInfo> {
        None
    }

    /// Tokens of fixed prompt overhead paid on every call, if any.
    ///
    /// Zero-shot span models condition on a label schema that is re-encoded
    /// with each document, so their latency has a floor set by the number of
    /// labels rather than by document length. GLiNER2 pays this today and
    /// NVIDIA GLiNER will too, so the perf report needs it as a first-class
    /// field rather than as a detector-specific footnote.
    fn fixed_prompt_tokens(&self) -> Option<usize> {
        None
    }
}

/// Boxed detectors are detectors.
///
/// The CLI resolves `--detector` to a `Box<dyn Detector>`, and the perf harness
/// is generic over `D: Detector`; without this the two cannot meet.
///
/// **Every defaulted method must be forwarded here.** A missing forward is not
/// a compile error — the blanket default takes over and calls `(**self)`'s
/// *other* methods, so it works and returns a plausible wrong answer. For
/// `detect_staged` that would be an all-zero breakdown for every detector the
/// CLI can construct, since the CLI hands out nothing but boxes.
impl<T: Detector + ?Sized> Detector for Box<T> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
        (**self).detect(text)
    }
    fn detect_staged(
        &self,
        text: &str,
        timer: &mut StageTimer,
    ) -> Result<Vec<Detection>, DetectorError> {
        (**self).detect_staged(text, timer)
    }
    fn supports_stage_timing(&self) -> bool {
        (**self).supports_stage_timing()
    }
    fn threshold(&self) -> Option<f32> {
        (**self).threshold()
    }
    fn with_threshold(&self, threshold: f32) -> Option<Box<dyn Detector>> {
        (**self).with_threshold(threshold)
    }
    fn artifact_info(&self) -> Vec<ArtifactInfo> {
        (**self).artifact_info()
    }
    fn model_size_bytes(&self) -> Option<u64> {
        (**self).model_size_bytes()
    }
    fn runtime_info(&self) -> Option<crate::runtime::RuntimeInfo> {
        (**self).runtime_info()
    }
    fn fixed_prompt_tokens(&self) -> Option<usize> {
        (**self).fixed_prompt_tokens()
    }
}

/// Identifies one model artifact precisely enough to reproduce a run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArtifactInfo {
    pub name: String,
    pub path: String,
    /// Upstream repo, e.g. `yethdev/gliner2-pii-onnx`.
    pub repo: Option<String>,
    /// Immutable revision (commit SHA), never a branch name.
    pub revision: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

/// Normalise and validate a detector's output before it leaves the detector.
///
/// Every implementation funnels through this so the evaluator can rely on the
/// invariants without re-checking, and so a detector bug surfaces as a named
/// error at the source rather than as a strange metric later.
pub fn finish(text: &str, mut detections: Vec<Detection>) -> Result<Vec<Detection>, DetectorError> {
    detections.retain(|d| !d.is_empty());
    crate::span::dedup_detections(&mut detections);
    crate::span::validate_all(text, &detections)
        .map_err(|(index, source)| DetectorError::InvalidSpan { index, source })?;
    crate::span::sort_detections(&mut detections);
    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::{DetectionSource, EntityType};

    fn det(start: usize, end: usize, score: f32) -> Detection {
        Detection::new(
            EntityType::Email,
            start,
            end,
            score,
            DetectionSource::recognizer("t"),
        )
    }

    #[test]
    fn finish_sorts_dedups_and_drops_empty() {
        let text = "abcdefghij";
        let out = finish(
            text,
            vec![
                det(5, 8, 0.5),
                det(0, 3, 0.9),
                det(5, 8, 0.7),
                det(2, 2, 1.0),
            ],
        )
        .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].start, out[0].end), (0, 3));
        assert_eq!((out[1].start, out[1].end), (5, 8));
        assert_eq!(out[1].score, 0.7, "duplicate collapsed to the higher score");
    }

    /// Records a sentinel into a stage so a lost forward is visible.
    struct Probe;
    impl Detector for Probe {
        fn name(&self) -> &str {
            "probe"
        }
        fn detect(&self, _: &str) -> Result<Vec<Detection>, DetectorError> {
            Ok(Vec::new())
        }
        fn detect_staged(
            &self,
            _: &str,
            timer: &mut StageTimer,
        ) -> Result<Vec<Detection>, DetectorError> {
            timer.time(crate::stages::Stage::SessionRun, || {
                std::thread::sleep(std::time::Duration::from_millis(2));
            });
            Ok(Vec::new())
        }
        fn supports_stage_timing(&self) -> bool {
            true
        }
    }

    #[test]
    fn stage_timing_survives_the_box_the_cli_hands_out() {
        // Without the forwards in `impl Detector for Box<T>` this test fails
        // with an all-zero timer and `supports_stage_timing() == false`, which
        // is exactly the silent failure the forwards exist to prevent.
        let boxed: Box<dyn Detector> = Box::new(Probe);
        assert!(boxed.supports_stage_timing(), "capability lost through Box");

        let mut t = StageTimer::enabled();
        boxed.detect_staged("anything", &mut t).unwrap();
        assert!(
            t.nanos(crate::stages::Stage::SessionRun) > 0,
            "detect_staged fell through to the default via Box"
        );
    }

    #[test]
    fn a_detector_that_opts_out_reports_no_stage_timing() {
        let boxed: Box<dyn Detector> = Box::new(Noop);
        assert!(!boxed.supports_stage_timing());
        let mut t = StageTimer::enabled();
        boxed.detect_staged("anything", &mut t).unwrap();
        assert_eq!(t.total_nanos(), 0, "the default must record nothing");
    }

    struct Noop;
    impl Detector for Noop {
        fn name(&self) -> &str {
            "noop"
        }
        fn detect(&self, _: &str) -> Result<Vec<Detection>, DetectorError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn finish_rejects_a_detector_bug_rather_than_passing_it_on() {
        // 'ö' occupies bytes 1..3; ending at 2 splits it.
        let err = finish("Jörg", vec![det(0, 2, 1.0)]).unwrap_err();
        assert!(
            matches!(err, DetectorError::InvalidSpan { index: 0, .. }),
            "got {err:?}"
        );
    }
}
