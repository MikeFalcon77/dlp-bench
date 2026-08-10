//! Result serialization and human-readable summaries.
//!
//! JSON carries everything; stdout carries the numbers a person needs to make
//! a decision. The two are generated from the same structures so they cannot
//! disagree.

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::detector::ArtifactInfo;
use crate::eval::Evaluation;
use crate::eval::sweep::SweepResult;
use crate::perf::PerfReport;
use crate::runmeta::RunMeta;

/// Provisional endpoint-DLP acceptance targets.
///
/// **Reference annotations only.** They never affect a metric's computation and
/// never change a detector's behaviour. They exist so a reader can see at a
/// glance where a candidate stands, and the report labels them as provisional.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AcceptanceTargets {
    pub critical_recall: f64,
    pub overall_recall: f64,
    pub precision: f64,
    pub p95_ms_at_128: f64,
    pub p95_ms_at_512: f64,
    pub warm_rss_mb: u64,
    pub cold_start_ms: f64,
}

impl Default for AcceptanceTargets {
    fn default() -> Self {
        Self {
            critical_recall: 0.95,
            overall_recall: 0.90,
            precision: 0.70,
            p95_ms_at_128: 50.0,
            p95_ms_at_512: 150.0,
            warm_rss_mb: 750,
            cold_start_ms: 2000.0,
        }
    }
}

/// One target's status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCheck {
    pub name: String,
    pub target: String,
    pub actual: String,
    pub met: bool,
}

/// The complete result document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub run: RunMeta,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<ArtifactInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<Evaluation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub performance: Option<PerfReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep: Option<SweepResult>,
    /// Provisional targets and whether this run met them. Advisory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acceptance: Vec<TargetCheck>,
}

impl Report {
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, json)
    }

    /// Evaluate the provisional targets against whatever this run measured.
    pub fn check_acceptance(&mut self, t: AcceptanceTargets) {
        let mut out = Vec::new();

        if let Some(q) = &self.quality {
            let m = &q.quality;
            out.push(check(
                "critical_recall",
                t.critical_recall,
                m.critical_recall.recall,
                true,
            ));
            out.push(check(
                "overall_recall",
                t.overall_recall,
                m.overlap.micro.recall,
                true,
            ));
            out.push(check(
                "precision",
                t.precision,
                m.overlap.micro.precision,
                true,
            ));
        }

        if let Some(p) = &self.performance {
            for (bucket, target) in [(128usize, t.p95_ms_at_128), (512, t.p95_ms_at_512)] {
                if let Some(b) = p.buckets.iter().find(|b| b.target_tokens == bucket) {
                    out.push(TargetCheck {
                        name: format!("p95_ms_at_{bucket}"),
                        target: format!("<= {target:.0}"),
                        actual: format!("{:.1}", b.percentiles.p95_ms),
                        met: b.percentiles.p95_ms <= target,
                    });
                }
            }
            out.push(TargetCheck {
                name: "cold_start_ms".into(),
                target: format!("<= {:.0}", t.cold_start_ms),
                actual: format!("{:.0}", p.cold_start.total_ms),
                met: p.cold_start.total_ms <= t.cold_start_ms,
            });
            if let Some(rss) = p.warm_rss {
                out.push(TargetCheck {
                    name: "warm_rss_mb".into(),
                    target: format!("<= {}", t.warm_rss_mb),
                    actual: rss.rss_mb.to_string(),
                    met: rss.rss_mb <= t.warm_rss_mb,
                });
            }
        }

        self.acceptance = out;
    }
}

fn check(name: &str, target: f64, actual: f64, higher_is_better: bool) -> TargetCheck {
    TargetCheck {
        name: name.to_owned(),
        target: format!(
            "{} {:.2}",
            if higher_is_better { ">=" } else { "<=" },
            target
        ),
        actual: format!("{actual:.4}"),
        met: if higher_is_better {
            actual >= target
        } else {
            actual <= target
        },
    }
}

/// Human-readable summary.
pub fn print_summary(w: &mut impl Write, report: &Report) -> std::io::Result<()> {
    let run = &report.run;
    writeln!(w, "\n{:=<78}", "")?;
    writeln!(
        w,
        "dlp-bench {} — {}",
        run.dlp_bench_version, run.timestamp_utc
    )?;
    writeln!(
        w,
        "{} {} · {} · {} logical CPUs · {} profile",
        run.os,
        run.arch,
        run.cpu.as_deref().unwrap_or("unknown CPU"),
        run.logical_cpus,
        run.profile
    )?;
    if let Some(c) = &run.git_commit {
        writeln!(
            w,
            "commit {}{}",
            &c[..c.len().min(12)],
            if run.git_dirty == Some(true) {
                " (dirty)"
            } else {
                ""
            }
        )?;
    }
    for warning in run.warnings() {
        writeln!(w, "!! {warning}")?;
    }
    writeln!(w, "{:=<78}", "")?;

    if let Some(q) = &report.quality {
        print_quality(w, q)?;
    }
    if let Some(p) = &report.performance {
        print_perf(w, p)?;
    }
    if let Some(s) = &report.sweep {
        print_sweep(w, s)?;
    }
    if !report.acceptance.is_empty() {
        writeln!(
            w,
            "\nProvisional endpoint-DLP targets (reference only, not pass/fail):"
        )?;
        for c in &report.acceptance {
            writeln!(
                w,
                "  {} {:<20} target {:<10} actual {}",
                if c.met { "ok  " } else { "MISS" },
                c.name,
                c.target,
                c.actual
            )?;
        }
    }
    writeln!(w)?;
    Ok(())
}

fn print_quality(w: &mut impl Write, e: &Evaluation) -> std::io::Result<()> {
    let m = &e.quality;
    writeln!(
        w,
        "\nQUALITY  detector={}  dataset={} ({:?})",
        e.detector, e.dataset, e.dataset_role
    )?;
    if let Some(t) = e.threshold {
        writeln!(w, "         threshold={t:.2}")?;
    }
    writeln!(
        w,
        "         {} documents · {} gold entities · {} predicted",
        m.documents, m.gold_entities, m.predicted_entities
    )?;

    writeln!(
        w,
        "\n  {:<10} {:>9} {:>9} {:>9}   {:>6} {:>6} {:>6}",
        "", "precision", "recall", "f1", "tp", "fp", "fn"
    )?;
    for (label, a) in [("exact", &m.exact), ("overlap", &m.overlap)] {
        writeln!(
            w,
            "  {:<10} {:>9.4} {:>9.4} {:>9.4}   {:>6} {:>6} {:>6}",
            format!("{label}/micro"),
            a.micro.precision,
            a.micro.recall,
            a.micro.f1,
            a.micro.tp,
            a.micro.fp,
            a.micro.fn_
        )?;
        writeln!(
            w,
            "  {:<10} {:>9.4} {:>9.4} {:>9.4}",
            format!("{label}/macro"),
            a.r#macro.precision,
            a.r#macro.recall,
            a.r#macro.f1
        )?;
    }

    let cc = &m.character_coverage;
    writeln!(
        w,
        "\n  critical recall          {:.4}  ({} tp / {} missed)",
        m.critical_recall.recall, m.critical_recall.tp, m.critical_recall.fn_
    )?;
    writeln!(
        w,
        "  sensitive char recall    {:.4}  ({:.2}% of sensitive bytes left exposed)",
        cc.sensitive_char_recall,
        cc.exposed_fraction * 100.0
    )?;
    writeln!(w, "  over-masked bytes        {}", cc.over_masked_bytes)?;
    writeln!(
        w,
        "  false positives / 1k doc {:.1}",
        m.false_positives_per_1k_docs
    )?;
    writeln!(
        w,
        "  clean negative documents {} / {}",
        m.clean_negative_docs, m.negative_docs
    )?;

    if !m.overlap.per_entity.is_empty() {
        writeln!(w, "\n  per entity (overlap):")?;
        writeln!(
            w,
            "    {:<20} {:>9} {:>9} {:>9} {:>8}",
            "type", "precision", "recall", "f1", "support"
        )?;
        for (t, p) in &m.overlap.per_entity {
            writeln!(
                w,
                "    {:<20} {:>9.4} {:>9.4} {:>9.4} {:>8}",
                t.to_string(),
                p.precision,
                p.recall,
                p.f1,
                p.support()
            )?;
        }
    }

    if !e.by_tag.is_empty() {
        writeln!(w, "\n  by content type (overlap micro):")?;
        writeln!(
            w,
            "    {:<14} {:>6} {:>9} {:>9} {:>9} {:>8}",
            "tag", "docs", "precision", "recall", "f1", "char rec"
        )?;
        for (tag, s) in &e.by_tag {
            writeln!(
                w,
                "    {:<14} {:>6} {:>9.4} {:>9.4} {:>9.4} {:>8.4}",
                tag,
                s.documents,
                s.overlap_micro.precision,
                s.overlap_micro.recall,
                s.overlap_micro.f1,
                s.sensitive_char_recall
            )?;
        }
    }

    if !e.by_language.is_empty() {
        writeln!(w, "\n  by language (overlap micro):")?;
        writeln!(
            w,
            "    {:<10} {:>6} {:>9} {:>9} {:>9} {:>8}",
            "lang", "docs", "precision", "recall", "f1", "char rec"
        )?;
        for (lang, s) in &e.by_language {
            writeln!(
                w,
                "    {:<10} {:>6} {:>9.4} {:>9.4} {:>9.4} {:>8.4}",
                lang,
                s.documents,
                s.overlap_micro.precision,
                s.overlap_micro.recall,
                s.overlap_micro.f1,
                s.sensitive_char_recall
            )?;
        }
    }

    if !e.excluded_types.is_empty() {
        writeln!(
            w,
            "\n  ! scoring restricted; excluded gold types: {}",
            e.excluded_types
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )?;
    }

    if let Some(err) = &e.errors {
        writeln!(
            w,
            "\n  ERRORS  {} false negatives, {} false positives (showing {} / {})",
            err.false_negatives_total,
            err.false_positives_total,
            err.false_negatives_shown + err.false_positives_shown,
            err.false_negatives_total + err.false_positives_total
        )?;
        for (title, list) in [
            ("false negatives (missed)", &err.false_negatives),
            ("false positives (over-detection)", &err.false_positives),
        ] {
            if list.is_empty() {
                continue;
            }
            writeln!(w, "\n    {title}:")?;
            for s in list {
                let score = s
                    .score
                    .map_or_else(|| "  -  ".to_owned(), |v| format!("{v:.3}"));
                writeln!(
                    w,
                    "      [{}] {:<18} {} {}",
                    s.example_id,
                    s.entity_type.to_string(),
                    score,
                    s.context
                )?;
                if !s.competing.is_empty() {
                    let c = s
                        .competing
                        .iter()
                        .map(|c| format!("{}@{:.2}", c.entity_type, c.score))
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(w, "        also predicted here: {c}")?;
                }
            }
        }
    }
    Ok(())
}

/// Print the per-stage attribution, if `--stage-timing` produced one.
///
/// Two rows per bucket, p50 and p95, because "which stage owns the median" and
/// "which stage owns the tail" are different questions and for a synchronous
/// DLP hook the second is the one that decides whether a user notices.
///
/// The `wall` column is the instrumented pass, not the headline pass above.
/// Printing both is what makes the instrumentation cost visible rather than
/// assumed — if they diverge, the breakdown describes a run nobody cares about.
fn print_stage_breakdown(w: &mut impl Write, p: &PerfReport) -> std::io::Result<()> {
    use crate::stages::Stage;

    if p.stage_breakdown.is_empty() {
        return Ok(());
    }

    writeln!(
        w,
        "\n  stage breakdown (instrumented pass; the latency above is uninstrumented):"
    )?;
    write!(w, "    {:>7} {:>4}", "tokens", "")?;
    for s in Stage::ALL {
        write!(w, " {:>12}", s.short())?;
    }
    writeln!(w, " {:>9} {:>9}", "unattr", "wall")?;

    for b in &p.stage_breakdown {
        let get = |s: Stage| {
            b.stages
                .iter()
                .find(|x| x.stage == s)
                .map_or(0.0, |x| x.percentiles.p50_ms)
        };
        write!(w, "    {:>7} {:>4}", b.target_tokens, "p50")?;
        for s in Stage::ALL {
            write!(w, " {:>12.2}", get(s))?;
        }
        writeln!(
            w,
            " {:>9.2} {:>9.2}",
            b.unattributed_p50_ms, b.instrumented.p50_ms
        )?;

        let get95 = |s: Stage| {
            b.stages
                .iter()
                .find(|x| x.stage == s)
                .map_or(0.0, |x| x.percentiles.p95_ms)
        };
        write!(w, "    {:>7} {:>4}", "", "p95")?;
        for s in Stage::ALL {
            write!(w, " {:>12.2}", get95(s))?;
        }
        writeln!(w, " {:>9} {:>9.2}", "-", b.instrumented.p95_ms)?;
    }

    // The headline claim this table exists to settle, stated rather than left
    // for the reader to compute from six columns.
    if let Some(b) = p
        .stage_breakdown
        .iter()
        .find(|b| b.target_tokens == 128)
        .or_else(|| p.stage_breakdown.first())
        && let Some(run) = b.stages.iter().find(|s| s.stage == Stage::SessionRun)
    {
        let copies: f64 = b
            .stages
            .iter()
            .filter(|s| matches!(s.stage, Stage::InputCopy | Stage::OutputCopy))
            .map(|s| s.share_of_p50)
            .sum();
        writeln!(
            w,
            "    at {} tokens: session.run {:.1}% of wall, tensor copies {:.1}%",
            b.target_tokens,
            run.share_of_p50 * 100.0,
            copies * 100.0
        )?;
    }

    Ok(())
}

fn print_perf(w: &mut impl Write, p: &PerfReport) -> std::io::Result<()> {
    writeln!(w, "\nPERFORMANCE  detector={}", p.detector)?;
    if let Some(sz) = p.model_size_bytes {
        writeln!(
            w,
            "  model size            {:.1} MB",
            sz as f64 / 1_048_576.0
        )?;
    }
    if let Some(pref) = p.schema_prefix_tokens {
        writeln!(
            w,
            "  schema prefix         {pref} tokens re-encoded on every call"
        )?;
    }
    if let Some(rt) = &p.runtime {
        write!(
            w,
            "  runtime               {} · intra_op {} · inter_op {} · {}",
            rt.provider, rt.intra_op_threads, rt.inter_op_threads, rt.graph_optimization
        )?;
        if let Some(c) = rt.session.resolved_coreml() {
            write!(
                w,
                " · coreml {}/{}{}{}",
                c.model_format.as_str(),
                c.compute_units.as_str(),
                if c.subgraphs { " +subgraphs" } else { "" },
                if c.static_input_shapes {
                    " +static-shapes"
                } else {
                    ""
                }
            )?;
        }
        if !rt.session.dimension_overrides.is_empty() {
            let dims: Vec<String> = rt
                .session
                .dimension_overrides
                .iter()
                .map(|d| format!("{}={}", d.name, d.size))
                .collect();
            write!(w, " · fixed {}", dims.join(","))?;
        }
        writeln!(w)?;

        // The number that decides whether a CoreML measurement is a
        // measurement of CoreML at all.
        if let Some(share) = &rt.ep_node_share {
            writeln!(
                w,
                "  accelerated           {:.1}% of kernel time off the CPU EP",
                share.accelerated_fraction() * 100.0
            )?;
            for (provider, us) in &share.by_provider_us {
                let pct = if share.total_us > 0 {
                    *us as f64 / share.total_us as f64 * 100.0
                } else {
                    0.0
                };
                writeln!(
                    w,
                    "      {provider:<28} {:>8.1} ms  {:>5.1}%  {} nodes",
                    *us as f64 / 1000.0,
                    pct,
                    share.nodes.get(provider).copied().unwrap_or(0)
                )?;
            }
        }
        if !rt.coreml_compute_plan.is_empty() {
            let mut by_device: std::collections::BTreeMap<&str, usize> = Default::default();
            for e in &rt.coreml_compute_plan {
                *by_device.entry(e.device.as_str()).or_default() += 1;
            }
            let summary: Vec<String> = by_device.iter().map(|(d, n)| format!("{d} {n}")).collect();
            writeln!(
                w,
                "  coreml compute plan   {} ops — {}",
                rt.coreml_compute_plan.len(),
                summary.join(", ")
            )?;
        }
    }

    let cs = &p.cold_start;
    writeln!(w, "  model load            {:.0} ms", cs.model_load_ms)?;
    writeln!(w, "  first inference       {:.1} ms", cs.first_inference_ms)?;
    writeln!(w, "  cold start total      {:.0} ms", cs.total_ms)?;
    if let Some(r) = cs.rss_after_load {
        writeln!(w, "  RSS after load        {} MB", r.rss_mb)?;
    }
    if let Some(r) = p.warm_rss {
        writeln!(w, "  warm RSS              {} MB", r.rss_mb)?;
    }
    if let Some(r) = p.peak_rss.and_then(|r| r.peak_rss_bytes) {
        writeln!(w, "  peak RSS (process)    {} MB", r / 1_048_576)?;
    }

    writeln!(
        w,
        "\n  warm latency ({} iterations after {} warm-up):",
        p.iterations, p.warmup
    )?;
    writeln!(
        w,
        "    {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "tokens", "words", "p50 ms", "p90 ms", "p95 ms", "p99 ms", "max ms", "docs/s"
    )?;
    for b in &p.buckets {
        let q = &b.percentiles;
        writeln!(
            w,
            "    {:>7} {:>7} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1}",
            b.target_tokens,
            b.actual_words,
            q.p50_ms,
            q.p90_ms,
            q.p95_ms,
            q.p99_ms,
            q.max_ms,
            b.docs_per_sec_p50
        )?;
    }

    print_stage_breakdown(w, p)?;

    if !p.throughput.is_empty() {
        writeln!(w, "\n  throughput (independent detector per thread):")?;
        writeln!(
            w,
            "    {:>7} {:>10} {:>12} {:>9} {:>8}",
            "threads", "docs/s", "words/s", "scaling", "RSS MB"
        )?;
        for t in &p.throughput {
            writeln!(
                w,
                "    {:>7} {:>10.1} {:>12.0} {:>8.2}x {:>8}",
                t.threads,
                t.docs_per_sec,
                t.words_per_sec,
                t.scaling,
                t.rss
                    .map_or_else(|| "-".to_owned(), |r| r.rss_mb.to_string())
            )?;
        }
    }
    Ok(())
}

fn print_sweep(w: &mut impl Write, s: &SweepResult) -> std::io::Result<()> {
    writeln!(
        w,
        "\nTHRESHOLD SWEEP  detector={}  dataset={} ({:?})",
        s.detector, s.dataset, s.dataset_role
    )?;
    if let Some(warn) = &s.calibration_warning {
        writeln!(w, "  !! {warn}")?;
    }
    writeln!(
        w,
        "\n  {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "thresh", "ex-prec", "ex-rec", "ex-f1", "ov-prec", "ov-rec", "ov-f1", "crit-rec"
    )?;
    for p in &s.points {
        writeln!(
            w,
            "  {:>6.2} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
            p.threshold,
            p.exact_precision,
            p.exact_recall,
            p.exact_f1,
            p.overlap_precision,
            p.overlap_recall,
            p.overlap_f1,
            p.critical_recall
        )?;
    }
    writeln!(
        w,
        "\n  {:>6} {:>12} {:>16} {:>10}",
        "thresh", "fp/1k docs", "sens char recall", "predicted"
    )?;
    for p in &s.points {
        writeln!(
            w,
            "  {:>6.2} {:>12.1} {:>16.4} {:>10}",
            p.threshold,
            p.false_positives_per_1k_docs,
            p.sensitive_char_recall,
            p.predicted_entities
        )?;
    }
    if let Some(best) = s.best_by_overlap_f1() {
        writeln!(
            w,
            "\n  best overlap F1 at threshold {:.2} (F1 {:.4})",
            best.threshold, best.overlap_f1
        )?;
    }
    if let Some(policy) = s.best_meeting_critical_recall(0.95) {
        writeln!(
            w,
            "  best precision meeting 95% critical recall: threshold {:.2} (precision {:.4})",
            policy.threshold, policy.overlap_precision
        )?;
    } else {
        writeln!(
            w,
            "  no threshold reaches 95% critical recall on this dataset"
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf::{BucketLatency, ColdStart, Percentiles};

    fn perf(p95_128: f64, cold_ms: f64) -> PerfReport {
        PerfReport {
            detector: "t".into(),
            model_size_bytes: Some(1024),
            schema_prefix_tokens: Some(77),
            runtime: None,
            cold_start: ColdStart {
                model_load_ms: cold_ms - 10.0,
                first_inference_ms: 10.0,
                total_ms: cold_ms,
                rss_after_load: None,
            },
            buckets: vec![BucketLatency {
                target_tokens: 128,
                actual_words: 130,
                actual_bytes: 900,
                iterations: 10,
                percentiles: Percentiles {
                    p95_ms: p95_128,
                    ..Default::default()
                },
                docs_per_sec_p50: 0.0,
            }],
            stage_breakdown: vec![],
            throughput: vec![],
            warm_rss: None,
            peak_rss: None,
            iterations: 10,
            warmup: 1,
        }
    }

    fn report(p: PerfReport) -> Report {
        Report {
            run: RunMeta::capture(0),
            artifacts: vec![],
            quality: None,
            performance: Some(p),
            sweep: None,
            acceptance: vec![],
        }
    }

    #[test]
    fn acceptance_targets_are_advisory_and_evaluated_in_both_directions() {
        let mut ok = report(perf(30.0, 1500.0));
        ok.check_acceptance(AcceptanceTargets::default());
        let by = |r: &Report, n: &str| r.acceptance.iter().find(|c| c.name == n).unwrap().met;
        assert!(by(&ok, "p95_ms_at_128"));
        assert!(by(&ok, "cold_start_ms"));

        let mut bad = report(perf(120.0, 4000.0));
        bad.check_acceptance(AcceptanceTargets::default());
        assert!(!by(&bad, "p95_ms_at_128"));
        assert!(!by(&bad, "cold_start_ms"));

        // Crucially: missing a target changes nothing about the measurements.
        assert_eq!(
            bad.performance.as_ref().unwrap().buckets[0]
                .percentiles
                .p95_ms,
            120.0
        );
    }

    #[test]
    fn summary_renders_without_panicking_and_names_the_detector() {
        let mut r = report(perf(30.0, 1500.0));
        r.check_acceptance(AcceptanceTargets::default());
        let mut buf = Vec::new();
        print_summary(&mut buf, &r).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("PERFORMANCE"), "{s}");
        assert!(
            s.contains("schema prefix"),
            "prefix cost must be visible: {s}"
        );
        assert!(
            s.contains("provisional") || s.contains("Provisional"),
            "{s}"
        );
    }

    #[test]
    fn report_round_trips_through_json() {
        let r = report(perf(30.0, 1500.0));
        let s = serde_json::to_string(&r).unwrap();
        let back: Report = serde_json::from_str(&s).unwrap();
        assert_eq!(back.performance.unwrap().detector, "t");
    }

    #[test]
    fn write_json_creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deeper/result.json");
        report(perf(1.0, 1.0)).write_json(&path).unwrap();
        assert!(path.is_file());
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"performance\""));
    }
}
