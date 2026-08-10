//! Latency statistics.
//!
//! Percentiles rather than a mean. Inference latency is right-skewed — an
//! occasional allocator or scheduler stall dominates the tail — and for a
//! synchronous DLP hook it is the tail that determines whether a user notices
//! the interception at all.

use std::time::Duration;

use serde::{Deserialize, Serialize};

pub type TimingSample = Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Percentiles {
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    /// Reported alongside, never instead of, the percentiles.
    pub mean_ms: f64,
    /// Standard deviation, as a stability signal: a large value next to a
    /// tight p50 means the measurement was disturbed.
    pub stddev_ms: f64,
    pub samples: usize,
}

/// Compute percentiles. Sorts `samples` in place.
///
/// Uses nearest-rank on the sorted samples, which is exact and needs no
/// interpolation convention that a reader would have to look up.
pub fn percentiles(samples: &mut [TimingSample]) -> Percentiles {
    if samples.is_empty() {
        return Percentiles::default();
    }
    samples.sort_unstable();

    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let n = samples.len();
    let at = |q: f64| -> f64 {
        // Nearest-rank: the smallest value at or above the q-th percentile.
        let rank = (q * n as f64).ceil().max(1.0) as usize;
        ms(samples[rank.min(n) - 1])
    };

    let mean = samples.iter().map(|d| ms(*d)).sum::<f64>() / n as f64;
    let var = samples.iter().map(|d| (ms(*d) - mean).powi(2)).sum::<f64>() / n as f64;

    Percentiles {
        min_ms: ms(samples[0]),
        p50_ms: at(0.50),
        p90_ms: at(0.90),
        p95_ms: at(0.95),
        p99_ms: at(0.99),
        max_ms: ms(samples[n - 1]),
        mean_ms: mean,
        stddev_ms: var.sqrt(),
        samples: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    #[test]
    fn percentiles_are_monotonic_and_exact_on_a_known_distribution() {
        // 1..=100 ms
        let mut s: Vec<_> = (1..=100).map(ms).collect();
        let p = percentiles(&mut s);
        assert_eq!(p.samples, 100);
        assert_eq!(p.min_ms, 1.0);
        assert_eq!(p.max_ms, 100.0);
        assert_eq!(p.p50_ms, 50.0);
        assert_eq!(p.p90_ms, 90.0);
        assert_eq!(p.p95_ms, 95.0);
        assert_eq!(p.p99_ms, 99.0);
        assert!((p.mean_ms - 50.5).abs() < 1e-9);
    }

    #[test]
    fn a_heavy_tail_moves_p99_but_barely_moves_p50() {
        // The exact scenario that makes a mean-only report misleading.
        let mut s: Vec<_> = (0..99)
            .map(|_| ms(10))
            .chain(std::iter::once(ms(1000)))
            .collect();
        let p = percentiles(&mut s);
        assert_eq!(p.p50_ms, 10.0, "median unmoved");
        assert_eq!(p.p99_ms, 10.0);
        assert_eq!(p.max_ms, 1000.0);
        assert!(
            p.mean_ms > 19.0,
            "the mean alone would hide the shape: {}",
            p.mean_ms
        );
        assert!(p.stddev_ms > 90.0, "stddev flags the disturbance");
    }

    #[test]
    fn input_order_does_not_matter() {
        let mut a: Vec<_> = [5u64, 1, 9, 3, 7].iter().map(|v| ms(*v)).collect();
        let mut b: Vec<_> = [9u64, 7, 5, 3, 1].iter().map(|v| ms(*v)).collect();
        assert_eq!(percentiles(&mut a), percentiles(&mut b));
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert_eq!(percentiles(&mut []), Percentiles::default());

        let mut one = vec![ms(42)];
        let p = percentiles(&mut one);
        assert_eq!(
            (p.min_ms, p.p50_ms, p.p99_ms, p.max_ms),
            (42.0, 42.0, 42.0, 42.0)
        );
        assert_eq!(p.stddev_ms, 0.0);
        assert_eq!(p.samples, 1);
    }

    #[test]
    fn percentiles_never_exceed_the_maximum_sample() {
        for n in 1..40usize {
            let mut s: Vec<_> = (0..n).map(|i| ms(i as u64 + 1)).collect();
            let p = percentiles(&mut s);
            assert!(p.p99_ms <= p.max_ms, "n={n}");
            assert!(
                p.p50_ms <= p.p90_ms && p.p90_ms <= p.p95_ms && p.p95_ms <= p.p99_ms,
                "n={n}"
            );
        }
    }
}
