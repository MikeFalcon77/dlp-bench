//! Resident-memory measurement.
//!
//! Isolated behind this module so that platform specifics never leak into the
//! detectors or the evaluator. Where a figure is not available the report says
//! `null` rather than guessing — an invented RSS number is worse than none when
//! the question is "does this fit on an endpoint".
//!
//! Peak RSS is the high-water mark the OS reports for the whole process, so it
//! includes the benchmark harness itself. That is stated in the report and in
//! the README rather than silently attributed to the model.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub rss_bytes: u64,
    pub rss_mb: u64,
    /// Process-wide high-water mark, when the platform exposes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_bytes: Option<u64>,
}

/// Current resident set size, or `None` if unavailable on this platform.
pub fn snapshot() -> Option<MemorySnapshot> {
    let stats = memory_stats::memory_stats()?;
    let rss_bytes = stats.physical_mem as u64;
    Some(MemorySnapshot {
        rss_bytes,
        rss_mb: rss_bytes / (1024 * 1024),
        peak_rss_bytes: peak_rss(),
    })
}

/// Process high-water-mark RSS.
#[cfg(unix)]
fn peak_rss() -> Option<u64> {
    // `getrusage` is POSIX and needs no /proc, so it covers macOS and Linux
    // with one path. `ru_maxrss` is bytes on macOS and kibibytes on Linux —
    // a genuine platform difference, not something to paper over.
    unsafe {
        let mut usage: libc_rusage = std::mem::zeroed();
        if getrusage(0, &mut usage) != 0 {
            return None;
        }
        let raw = usage.ru_maxrss as u64;
        Some(if cfg!(target_os = "macos") {
            raw
        } else {
            raw * 1024
        })
    }
}

#[cfg(not(unix))]
fn peak_rss() -> Option<u64> {
    // Windows would need GetProcessMemoryInfo via the win32 API. Deliberately
    // not implemented in Phase 1 rather than approximated.
    None
}

// Minimal `getrusage` binding. Pulling in the whole `libc` crate for two
// symbols is not worth it, and this keeps the platform surface visible in one
// place.
#[cfg(unix)]
#[repr(C)]
#[derive(Clone, Copy)]
struct libc_rusage {
    ru_utime: [i64; 2],
    ru_stime: [i64; 2],
    ru_maxrss: i64,
    ru_ixrss: i64,
    ru_idrss: i64,
    ru_isrss: i64,
    ru_minflt: i64,
    ru_majflt: i64,
    ru_nswap: i64,
    ru_inblock: i64,
    ru_oublock: i64,
    ru_msgsnd: i64,
    ru_msgrcv: i64,
    ru_nsignals: i64,
    ru_nvcsw: i64,
    ru_nivcsw: i64,
}

#[cfg(unix)]
unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut libc_rusage) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_a_plausible_rss() {
        let Some(s) = snapshot() else {
            eprintln!("RSS unavailable on this platform; reported as null");
            return;
        };
        // A running test process is at least a few MB and well under a terabyte.
        assert!(
            s.rss_bytes > 1024 * 1024,
            "implausibly small RSS: {}",
            s.rss_bytes
        );
        assert!(
            s.rss_bytes < 1 << 40,
            "implausibly large RSS: {}",
            s.rss_bytes
        );
        assert_eq!(s.rss_mb, s.rss_bytes / (1024 * 1024));
    }

    #[test]
    fn peak_is_at_least_current_when_available() {
        let Some(s) = snapshot() else { return };
        if let Some(peak) = s.peak_rss_bytes {
            assert!(peak > 0);
            // Allow slack: the peak may pre-date this sample and the two come
            // from different accounting paths.
            assert!(
                peak >= s.rss_bytes / 2,
                "peak {peak} implausibly below current {}",
                s.rss_bytes
            );
        }
    }

    /// Verifies the reading is live rather than a constant.
    ///
    /// Deliberately does *not* assert that RSS increased across the
    /// allocation: `cargo test` runs tests in parallel, so another thread can
    /// free more memory than this allocates between the two samples, and that
    /// assertion fails intermittently for reasons unrelated to the code under
    /// test. The peak high-water mark is monotonic within the process, so
    /// bounding against it is stable.
    #[test]
    fn reading_accounts_for_a_large_live_allocation() {
        let Some(before) = snapshot() else { return };
        // 64 MB, touched page by page so it is actually resident.
        let mut v = vec![0u8; 64 * 1024 * 1024];
        for i in (0..v.len()).step_by(4096) {
            v[i] = 1;
        }
        std::hint::black_box(&v);

        let after = snapshot().expect("still available");
        assert!(
            after.rss_bytes >= 64 * 1024 * 1024,
            "RSS {} is below the live allocation",
            after.rss_bytes
        );
        if let (Some(pa), Some(pb)) = (after.peak_rss_bytes, before.peak_rss_bytes) {
            assert!(pa >= pb, "peak RSS went backwards: {pb} -> {pa}");
            assert!(
                pa >= after.rss_bytes,
                "peak {pa} below current {}",
                after.rss_bytes
            );
        }
        drop(v);
    }
}
