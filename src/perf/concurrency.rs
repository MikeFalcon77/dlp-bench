//! N clients sending requests to one service at the same time.
//!
//! Different question from [`super::measure_throughput`]. That one asks how
//! many documents per second N independent single-threaded detectors manage;
//! this one asks what a *caller* sees when N of them share one service with a
//! fixed CPU budget — the per-request latency including any wait for a free
//! model, and the aggregate rate.
//!
//! How the service is built follows from the detector:
//!
//! * **Shared** — one instance for every client, when
//!   [`Detector::concurrent_detect`] says concurrent calls really run in
//!   parallel (Candle: an immutable model behind `&self`).
//! * **Pool** — otherwise (ONNX Runtime: `ort` forbids concurrent runs on one
//!   session), `min(clients, pool_max)` instances with the thread budget split
//!   evenly between them, and a client waits for whichever is free first.
//!   At most `pool_max` instances are ever loaded at once: the previous pool is
//!   dropped before a larger one is built, so RSS reflects what the service
//!   would hold.
//!
//! Every client runs a closed loop: send, wait for the answer, send again.
//! All clients are released together by a barrier and the wall clock covers
//! the first send to the last answer.

use std::collections::VecDeque;
use std::sync::{Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::latency::{Percentiles, percentiles};
use super::memory::{self, MemorySnapshot};
use super::{ColdStart, measure_cold_start, synthetic_document};
use crate::detector::Detector;
use crate::error::DetectorError;

#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Client counts to measure, e.g. 1..=10.
    pub clients: Vec<usize>,
    /// Document sizes in approximate whitespace tokens; one series each.
    pub docs: Vec<usize>,
    /// Timed requests per client at each level, after one untimed warm-up.
    pub requests_per_client: usize,
    /// Upper bound on loaded instances in pool mode.
    pub pool_max: usize,
    /// Intra-op threads shared by all instances.
    pub thread_budget: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyPoint {
    pub target_tokens: usize,
    pub actual_words: usize,
    pub actual_bytes: usize,
    pub clients: usize,
    /// `true` when every client called the same instance concurrently.
    pub shared: bool,
    /// Model instances loaded while this level ran.
    pub instances: usize,
    pub threads_per_instance: usize,
    pub requests: usize,
    pub wall_ms: f64,
    pub requests_per_sec: f64,
    /// Per request as the client saw it: waiting for an instance included.
    pub latency: Percentiles,
    pub rss: Option<MemorySnapshot>,
}

pub struct ConcurrencyRun<D> {
    /// The first instance's load and first call, for the report header.
    pub cold_start: ColdStart,
    pub points: Vec<ConcurrencyPoint>,
    /// The last instance loaded, so the caller can record its runtime.
    pub last: D,
}

/// `build(threads)` loads one instance with that many intra-op threads.
pub fn measure<D: Detector>(
    build: &dyn Fn(usize) -> Result<D, DetectorError>,
    cfg: &ConcurrencyConfig,
) -> Result<ConcurrencyRun<D>, DetectorError> {
    let budget = cfg.thread_budget.max(1);
    let warm = synthetic_document(32);
    let (first, cold_start) = measure_cold_start(|| build(budget), &warm)?;
    let shared = first.concurrent_detect();

    let docs: Vec<(usize, String)> = cfg
        .docs
        .iter()
        .map(|&t| (t, synthetic_document(t)))
        .collect();
    let mut clients = cfg.clients.clone();
    clients.sort_unstable();
    clients.dedup();

    let mut pool = vec![first];
    let mut threads = budget;
    let mut points = Vec::new();

    for &n in clients.iter().filter(|&&n| n > 0) {
        let want = if shared {
            1
        } else {
            n.min(cfg.pool_max.max(1))
        };
        if pool.len() != want {
            // Drop before building, so no more than `pool_max` are ever held.
            pool.clear();
            threads = (budget / want).max(1);
            for _ in 0..want {
                let d = build(threads)?;
                d.detect(&warm)?;
                pool.push(d);
            }
        }
        for (target, text) in &docs {
            // Untimed round: every client once, so each instance has seen this
            // document size before the clock starts.
            run_level(&pool, shared, n, 1, text)?;
            let (wall, mut samples) = run_level(&pool, shared, n, cfg.requests_per_client, text)?;
            let requests = samples.len();
            let wall_ms = wall.as_secs_f64() * 1000.0;
            points.push(ConcurrencyPoint {
                target_tokens: *target,
                actual_words: text.split_whitespace().count(),
                actual_bytes: text.len(),
                clients: n,
                shared,
                instances: pool.len(),
                threads_per_instance: threads,
                requests,
                wall_ms,
                requests_per_sec: if wall_ms > 0.0 {
                    requests as f64 * 1000.0 / wall_ms
                } else {
                    0.0
                },
                latency: percentiles(&mut samples),
                rss: memory::snapshot(),
            });
        }
    }

    let last = pool.pop().expect("at least one instance is always loaded");
    Ok(ConcurrencyRun {
        cold_start,
        points,
        last,
    })
}

struct Queue {
    free: Vec<usize>,
    waiting: VecDeque<u64>,
    next_ticket: u64,
}

/// `clients` closed-loop clients, `per_client` requests each. Returns the wall
/// time from the shared start to the last answer, and every request's latency.
fn run_level<D: Detector>(
    pool: &[D],
    shared: bool,
    clients: usize,
    per_client: usize,
    text: &str,
) -> Result<(Duration, Vec<Duration>), DetectorError> {
    // First come, first served. A plain mutex + condvar lets a client lose
    // the race for a freed instance several times in a row, which shows up as
    // a p95 several times the p50 that no real request queue would produce.
    let queue = Mutex::new(Queue {
        free: (0..pool.len()).collect(),
        waiting: VecDeque::new(),
        next_ticket: 0,
    });
    let changed = Condvar::new();
    let start = Barrier::new(clients + 1);

    let checkout = || -> usize {
        let mut q = queue.lock().expect("pool queue");
        let ticket = q.next_ticket;
        q.next_ticket += 1;
        q.waiting.push_back(ticket);
        loop {
            if q.waiting.front() == Some(&ticket)
                && let Some(i) = q.free.pop()
            {
                q.waiting.pop_front();
                // The next in line may be able to go too.
                changed.notify_all();
                return i;
            }
            q = changed.wait(q).expect("pool queue");
        }
    };
    let checkin = |i: usize| {
        queue.lock().expect("pool queue").free.push(i);
        changed.notify_all();
    };

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..clients)
            .map(|_| {
                s.spawn(|| -> Result<Vec<Duration>, DetectorError> {
                    start.wait();
                    let mut out = Vec::with_capacity(per_client);
                    for _ in 0..per_client {
                        let t = Instant::now();
                        let found = if shared {
                            pool[0].detect(text)
                        } else {
                            let i = checkout();
                            let r = pool[i].detect(text);
                            checkin(i);
                            r
                        };
                        out.push(t.elapsed());
                        std::hint::black_box(found?);
                    }
                    Ok(out)
                })
            })
            .collect();
        start.wait();
        let t0 = Instant::now();
        let mut samples = Vec::with_capacity(clients * per_client);
        for h in handles {
            samples.extend(h.join().expect("client thread panicked")?);
        }
        Ok((t0.elapsed(), samples))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Detection;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Sleeps, and counts how many calls overlap.
    struct Slow {
        concurrent: bool,
        inflight: AtomicUsize,
        peak: AtomicUsize,
    }
    impl Detector for Slow {
        fn name(&self) -> &str {
            "slow"
        }
        fn detect(&self, _: &str) -> Result<Vec<Detection>, DetectorError> {
            let now = self.inflight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(5));
            self.inflight.fetch_sub(1, Ordering::SeqCst);
            Ok(Vec::new())
        }
        fn concurrent_detect(&self) -> bool {
            self.concurrent
        }
    }

    fn slow(concurrent: bool) -> Slow {
        Slow {
            concurrent,
            inflight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn cfg(clients: Vec<usize>) -> ConcurrencyConfig {
        ConcurrencyConfig {
            clients,
            docs: vec![32],
            requests_per_client: 4,
            pool_max: 2,
            thread_budget: 8,
        }
    }

    #[test]
    fn a_pool_never_exceeds_its_cap_and_splits_the_thread_budget() {
        let built = Mutex::new(Vec::new());
        let run = measure(
            &|threads| {
                built.lock().unwrap().push(threads);
                Ok(slow(false))
            },
            &cfg(vec![1, 2, 5]),
        )
        .unwrap();
        let by_clients: Vec<_> = run
            .points
            .iter()
            .map(|p| (p.clients, p.instances, p.threads_per_instance))
            .collect();
        assert_eq!(by_clients, vec![(1, 1, 8), (2, 2, 4), (5, 2, 4)]);
        // 1 x 8 threads, then the pair of 4s — built once, reused for 5 clients.
        assert_eq!(*built.lock().unwrap(), vec![8, 4, 4]);
        for p in &run.points {
            assert_eq!(p.requests, p.clients * 4);
            assert!(!p.shared);
        }
        // Five clients on two instances must queue: the client-side latency
        // includes the wait, so it exceeds one call.
        let five = &run.points[2];
        assert!(five.latency.p50_ms > 5.0, "{:?}", five.latency);
    }

    #[test]
    fn a_concurrent_detector_is_shared_and_really_called_in_parallel() {
        let run = measure(&|_| Ok(slow(true)), &cfg(vec![4])).unwrap();
        let p = &run.points[0];
        assert!(p.shared);
        assert_eq!(p.instances, 1);
        assert_eq!(
            run.last.peak.load(Ordering::SeqCst),
            4,
            "calls must overlap"
        );
    }
}
