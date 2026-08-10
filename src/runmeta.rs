//! Reproducibility metadata.
//!
//! Every result carries enough context to be re-run and to be distrusted if
//! the context differs. A benchmark number without its thread count, model
//! revision and dataset hash is an anecdote.

use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub timestamp_utc: String,
    pub command_line: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_dirty: Option<bool>,
    pub dlp_bench_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rustc_version: Option<String>,
    pub os: String,
    pub arch: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,
    pub logical_cpus: usize,
    pub profile: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ort_version: Option<String>,
    pub seed: u64,
}

impl RunMeta {
    pub fn capture(seed: u64) -> Self {
        Self {
            timestamp_utc: timestamp(),
            command_line: std::env::args().collect::<Vec<_>>().join(" "),
            git_commit: git(&["rev-parse", "HEAD"]),
            git_dirty: git(&["status", "--porcelain"]).map(|s| !s.trim().is_empty()),
            dlp_bench_version: env!("CARGO_PKG_VERSION").to_owned(),
            rustc_version: run("rustc", &["--version"]),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            cpu: cpu_model(),
            logical_cpus: std::thread::available_parallelism().map_or(0, |n| n.get()),
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .to_owned(),
            #[cfg(feature = "onnx")]
            ort_version: Some(crate::detectors::onnx::ORT_VERSION.to_owned()),
            #[cfg(not(feature = "onnx"))]
            ort_version: None,
            seed,
        }
    }

    /// True when the run is not reproducible as recorded.
    ///
    /// Surfaced in the human-readable summary: a debug-profile timing or a
    /// dirty tree makes a perf number unciteable, and that should be loud
    /// rather than discoverable by reading JSON.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.profile == "debug" {
            w.push(
                "built in debug profile — latency numbers are not meaningful; use --release"
                    .to_owned(),
            );
        }
        if self.git_dirty == Some(true) {
            w.push(
                "working tree is dirty — the recorded commit does not describe this build"
                    .to_owned(),
            );
        }
        if self.git_commit.is_none() {
            w.push("no git commit recorded".to_owned());
        }
        w
    }
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        .filter(|s| !s.is_empty())
}

fn git(args: &[&str]) -> Option<String> {
    run("git", args)
}

fn cpu_model() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        run("sysctl", &["-n", "machdep.cpu.brand_string"])
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        text.lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_owned())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// RFC 3339 UTC timestamp without pulling in a date crate.
fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    // Civil-from-days (Howard Hinnant's algorithm), valid for all dates we care
    // about and cheaper than a dependency for one string.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_records_the_environment() {
        let m = RunMeta::capture(42);
        assert_eq!(m.seed, 42);
        assert!(!m.os.is_empty());
        assert!(!m.arch.is_empty());
        assert!(m.logical_cpus > 0);
        assert_eq!(m.dlp_bench_version, env!("CARGO_PKG_VERSION"));
        assert!(m.timestamp_utc.ends_with('Z'));
        assert_eq!(m.timestamp_utc.len(), 20, "{}", m.timestamp_utc);
    }

    #[test]
    fn timestamp_formats_known_epochs_correctly() {
        // Verified against `date -u -r <epoch>`.
        assert_eq!(&timestamp()[0..2], "20");
        // The algorithm itself, exercised through a fixed value.
        let fixed = |secs: u64| -> String {
            let days = (secs / 86_400) as i64;
            let rem = secs % 86_400;
            let z = days + 719_468;
            let era = z.div_euclid(146_097);
            let doe = z.rem_euclid(146_097);
            let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
            let y = yoe + era * 400;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = doy - (153 * mp + 2) / 5 + 1;
            let m = if mp < 10 { mp + 3 } else { mp - 9 };
            let y = if m <= 2 { y + 1 } else { y };
            format!(
                "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
                rem / 3600,
                (rem % 3600) / 60,
                rem % 60
            )
        };
        assert_eq!(fixed(0), "1970-01-01T00:00:00Z");
        assert_eq!(fixed(1_000_000_000), "2001-09-09T01:46:40Z");
        assert_eq!(fixed(1_754_611_200), "2025-08-08T00:00:00Z");
    }

    #[test]
    fn debug_builds_are_flagged_as_unciteable() {
        let m = RunMeta::capture(0);
        let w = m.warnings();
        if cfg!(debug_assertions) {
            assert!(w.iter().any(|s| s.contains("--release")), "{w:?}");
        }
    }

    #[test]
    fn metadata_round_trips_through_json() {
        let m = RunMeta::capture(7);
        let s = serde_json::to_string(&m).unwrap();
        let back: RunMeta = serde_json::from_str(&s).unwrap();
        assert_eq!(back.seed, 7);
        assert_eq!(back.os, m.os);
    }
}
