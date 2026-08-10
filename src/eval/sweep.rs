//! Threshold sweep.
//!
//! Emits the full metric set at each threshold so an operating point can be
//! chosen against an actual precision/recall trade-off rather than a single F1.
//!
//! **Methodology guard:** picking a threshold on the same data used for the
//! final comparison inflates the result. The sweep therefore records the
//! dataset role and warns when asked to calibrate on a `Test` set. It does not
//! refuse — sometimes you legitimately want the test-set curve for a report —
//! but the warning and the recorded role make an accidental leak visible.

use serde::{Deserialize, Serialize};

use crate::dataset::{Dataset, DatasetRole};
use crate::detector::Detector;
use crate::error::DetectorError;
use crate::eval::{EvalConfig, evaluate};

#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub start: f32,
    pub end: f32,
    pub step: f32,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            start: 0.10,
            end: 0.95,
            step: 0.05,
        }
    }
}

impl SweepConfig {
    /// Threshold points, inclusive of `end` when it lands on the grid.
    ///
    /// Values are rounded to 4 decimals so accumulated float error does not
    /// produce `0.30000000000000004` in the report or in a CSV join key.
    pub fn points(&self) -> Result<Vec<f32>, String> {
        if self.step <= 0.0 {
            return Err(format!(
                "threshold step must be positive, got {}",
                self.step
            ));
        }
        if self.end < self.start {
            return Err(format!(
                "threshold end {} is below start {}",
                self.end, self.start
            ));
        }
        // Nudge before flooring: `(0.9 - 0.1) / 0.1` is 7.9999995 in f32, which
        // would silently drop the last threshold from the sweep.
        let n = (((self.end - self.start) / self.step) + 1e-4).floor() as usize;
        let limit = round4(self.end);
        let mut v: Vec<f32> = (0..=n)
            .map(|i| round4(self.start + self.step * i as f32))
            .filter(|t| *t <= limit)
            .collect();
        v.dedup();
        Ok(v)
    }
}

fn round4(v: f32) -> f32 {
    (v * 10_000.0).round() / 10_000.0
}

/// One row of the sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepPoint {
    pub threshold: f32,
    pub exact_precision: f64,
    pub exact_recall: f64,
    pub exact_f1: f64,
    pub overlap_precision: f64,
    pub overlap_recall: f64,
    pub overlap_f1: f64,
    pub macro_overlap_f1: f64,
    pub critical_recall: f64,
    pub sensitive_char_recall: f64,
    pub false_positives_per_1k_docs: f64,
    pub predicted_entities: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    pub detector: String,
    pub dataset: String,
    pub dataset_role: DatasetRole,
    pub dataset_sha256: String,
    pub points: Vec<SweepPoint>,
    /// Present when the sweep ran against a held-out set. Kept in the JSON so a
    /// downstream reader cannot miss it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_warning: Option<String>,
}

impl SweepResult {
    /// CSV, for pasting into a spreadsheet or plotting.
    pub fn to_csv(&self) -> String {
        let mut s = String::from(
            "threshold,exact_precision,exact_recall,exact_f1,\
             overlap_precision,overlap_recall,overlap_f1,macro_overlap_f1,\
             critical_recall,sensitive_char_recall,fp_per_1k_docs,predicted_entities\n",
        );
        for p in &self.points {
            s.push_str(&format!(
                "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{}\n",
                p.threshold,
                p.exact_precision,
                p.exact_recall,
                p.exact_f1,
                p.overlap_precision,
                p.overlap_recall,
                p.overlap_f1,
                p.macro_overlap_f1,
                p.critical_recall,
                p.sensitive_char_recall,
                p.false_positives_per_1k_docs,
                p.predicted_entities,
            ));
        }
        s
    }

    /// Highest-F1 point. Advisory only — the caller decides what to do with it,
    /// and the calibration warning still applies.
    pub fn best_by_overlap_f1(&self) -> Option<&SweepPoint> {
        self.points
            .iter()
            .max_by(|a, b| a.overlap_f1.total_cmp(&b.overlap_f1))
    }

    /// Lowest threshold meeting a critical-recall floor, which is usually the
    /// operating point a DLP policy actually wants: satisfy the recall
    /// requirement, then take the best precision available.
    pub fn best_meeting_critical_recall(&self, floor: f64) -> Option<&SweepPoint> {
        self.points
            .iter()
            .filter(|p| p.critical_recall >= floor)
            .max_by(|a, b| a.overlap_precision.total_cmp(&b.overlap_precision))
    }
}

/// Run the sweep. Returns an error if the detector has no meaningful threshold
/// rather than emitting a flat, misleading curve.
pub fn sweep(
    detector: &dyn Detector,
    dataset: &Dataset,
    cfg: &EvalConfig,
    sweep_cfg: &SweepConfig,
) -> Result<SweepResult, DetectorError> {
    if detector.threshold().is_none() {
        return Err(DetectorError::config(format!(
            "detector {:?} has no confidence threshold; a sweep would produce \
             an identical row at every point",
            detector.name()
        )));
    }

    let points_at = sweep_cfg.points().map_err(DetectorError::config)?;
    let mut points = Vec::with_capacity(points_at.len());

    for t in points_at {
        let tuned = detector.with_threshold(t).ok_or_else(|| {
            DetectorError::config(format!(
                "detector {:?} reports a threshold but cannot be retuned",
                detector.name()
            ))
        })?;
        let e = evaluate(tuned.as_ref(), dataset, cfg)?;
        points.push(SweepPoint {
            threshold: t,
            exact_precision: e.quality.exact.micro.precision,
            exact_recall: e.quality.exact.micro.recall,
            exact_f1: e.quality.exact.micro.f1,
            overlap_precision: e.quality.overlap.micro.precision,
            overlap_recall: e.quality.overlap.micro.recall,
            overlap_f1: e.quality.overlap.micro.f1,
            macro_overlap_f1: e.quality.overlap.r#macro.f1,
            critical_recall: e.quality.critical_recall.recall,
            sensitive_char_recall: e.quality.character_coverage.sensitive_char_recall,
            false_positives_per_1k_docs: e.quality.false_positives_per_1k_docs,
            predicted_entities: e.quality.predicted_entities,
        });
    }

    let calibration_warning = (dataset.role == DatasetRole::Test).then(|| {
        format!(
            "swept on dataset {:?} whose role is `test`; a threshold chosen here \
             must not be reported as tuned independently of the final comparison",
            dataset.name
        )
    });
    if let Some(w) = &calibration_warning {
        tracing::warn!(dataset = %dataset.name, "{w}");
    }

    Ok(SweepResult {
        detector: detector.name().to_owned(),
        dataset: dataset.name.clone(),
        dataset_role: dataset.role,
        dataset_sha256: dataset.sha256.clone(),
        points,
        calibration_warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Example, GoldEntity};
    use crate::detector::finish;
    use crate::entity::{Detection, DetectionSource, EntityType};

    #[test]
    fn points_cover_the_range_inclusively_without_float_noise() {
        let p = SweepConfig {
            start: 0.10,
            end: 0.95,
            step: 0.05,
        }
        .points()
        .unwrap();
        assert_eq!(p.first(), Some(&0.10));
        assert_eq!(p.last(), Some(&0.95));
        assert_eq!(p.len(), 18);
        for v in &p {
            assert_eq!(
                *v,
                round4(*v),
                "unrounded threshold {v} would poison a CSV join"
            );
        }
        assert!(p.contains(&0.30), "0.30 must be exact, not 0.30000001");
    }

    #[test]
    fn points_rejects_degenerate_configs() {
        assert!(
            SweepConfig {
                start: 0.1,
                end: 0.9,
                step: 0.0
            }
            .points()
            .is_err()
        );
        assert!(
            SweepConfig {
                start: 0.1,
                end: 0.9,
                step: -0.1
            }
            .points()
            .is_err()
        );
        assert!(
            SweepConfig {
                start: 0.9,
                end: 0.1,
                step: 0.1
            }
            .points()
            .is_err()
        );
        // A single point is legitimate.
        assert_eq!(
            SweepConfig {
                start: 0.5,
                end: 0.5,
                step: 0.1
            }
            .points()
            .unwrap(),
            vec![0.5]
        );
    }

    /// Emits one detection whose score is fixed, filtered by threshold — the
    /// minimum needed to exercise the sweep machinery.
    #[derive(Clone)]
    struct Thresholded {
        threshold: f32,
        score: f32,
    }

    impl Detector for Thresholded {
        fn name(&self) -> &str {
            "thresholded"
        }
        fn detect(&self, text: &str) -> Result<Vec<Detection>, DetectorError> {
            let out = if self.score >= self.threshold {
                vec![Detection::new(
                    EntityType::CreditCard,
                    0,
                    text.len(),
                    self.score,
                    DetectionSource::model("t"),
                )]
            } else {
                vec![]
            };
            finish(text, out)
        }
        fn threshold(&self) -> Option<f32> {
            Some(self.threshold)
        }
        fn with_threshold(&self, t: f32) -> Option<Box<dyn Detector>> {
            Some(Box::new(Thresholded {
                threshold: t,
                score: self.score,
            }))
        }
    }

    fn dataset(role: DatasetRole) -> Dataset {
        Dataset {
            name: "t".into(),
            role,
            examples: vec![Example {
                id: "a".into(),
                source: None,
                language: None,
                text: "4111111111111111".into(),
                entities: vec![GoldEntity {
                    entity_type: EntityType::CreditCard,
                    start: 0,
                    end: 16,
                }],
                tags: vec![],
            }],
            sha256: "0".into(),
        }
    }

    #[test]
    fn recall_falls_and_precision_holds_as_the_threshold_rises() {
        let d = Thresholded {
            threshold: 0.5,
            score: 0.6,
        };
        let r = sweep(
            &d,
            &dataset(DatasetRole::Dev),
            &EvalConfig::default(),
            &SweepConfig {
                start: 0.1,
                end: 0.9,
                step: 0.1,
            },
        )
        .unwrap();

        assert_eq!(r.points.len(), 9);
        let low = r.points.iter().find(|p| p.threshold == 0.5).unwrap();
        let high = r.points.iter().find(|p| p.threshold == 0.7).unwrap();
        assert_eq!(low.overlap_recall, 1.0);
        assert_eq!(low.critical_recall, 1.0);
        assert_eq!(low.sensitive_char_recall, 1.0);
        assert_eq!(
            high.overlap_recall, 0.0,
            "detection suppressed above its score"
        );
        assert_eq!(high.sensitive_char_recall, 0.0);
        assert!(r.calibration_warning.is_none(), "dev set needs no warning");
    }

    #[test]
    fn sweeping_a_test_set_records_a_calibration_warning() {
        let d = Thresholded {
            threshold: 0.5,
            score: 0.6,
        };
        let r = sweep(
            &d,
            &dataset(DatasetRole::Test),
            &EvalConfig::default(),
            &SweepConfig {
                start: 0.4,
                end: 0.6,
                step: 0.1,
            },
        )
        .unwrap();
        let w = r
            .calibration_warning
            .expect("test-set sweep must be flagged");
        assert!(w.contains("test"), "{w}");
        assert_eq!(r.dataset_role, DatasetRole::Test);
    }

    #[test]
    fn sweeping_a_thresholdless_detector_is_an_error_not_a_flat_curve() {
        struct NoThreshold;
        impl Detector for NoThreshold {
            fn name(&self) -> &str {
                "deterministic"
            }
            fn detect(&self, _: &str) -> Result<Vec<Detection>, DetectorError> {
                Ok(vec![])
            }
        }
        let err = sweep(
            &NoThreshold,
            &dataset(DatasetRole::Dev),
            &EvalConfig::default(),
            &SweepConfig::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("no confidence threshold"), "{err}");
    }

    #[test]
    fn operating_point_selectors_behave_sensibly() {
        let d = Thresholded {
            threshold: 0.5,
            score: 0.6,
        };
        let r = sweep(
            &d,
            &dataset(DatasetRole::Dev),
            &EvalConfig::default(),
            &SweepConfig {
                start: 0.1,
                end: 0.9,
                step: 0.1,
            },
        )
        .unwrap();

        assert!(r.best_by_overlap_f1().unwrap().overlap_f1 > 0.9);
        let policy = r.best_meeting_critical_recall(0.95).unwrap();
        assert!(policy.critical_recall >= 0.95);
        assert!(
            r.best_meeting_critical_recall(1.5).is_none(),
            "unreachable floor"
        );
    }

    #[test]
    fn csv_has_one_header_and_one_row_per_point() {
        let d = Thresholded {
            threshold: 0.5,
            score: 0.6,
        };
        let r = sweep(
            &d,
            &dataset(DatasetRole::Dev),
            &EvalConfig::default(),
            &SweepConfig {
                start: 0.2,
                end: 0.4,
                step: 0.1,
            },
        )
        .unwrap();
        let csv = r.to_csv();
        let lines: Vec<_> = csv.lines().collect();
        assert_eq!(lines.len(), 1 + r.points.len());
        assert!(lines[0].starts_with("threshold,"));
        let cols = lines[0].split(',').count();
        for l in &lines[1..] {
            assert_eq!(l.split(',').count(), cols, "ragged CSV row: {l}");
        }
    }
}
