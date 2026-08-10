//! `dlp-bench` CLI.

mod cli;

use std::collections::BTreeSet;
use std::io::Write;

use clap::Parser;

use dlp_bench::dataset::{Dataset, DatasetRole, transform};
use dlp_bench::detector::Detector;
use dlp_bench::detectors::deterministic::{
    DeterministicConfig, DeterministicDetector, PrivateIpPolicy,
};
use dlp_bench::error::{Error, Result};
use dlp_bench::eval::{EvalConfig, evaluate, sweep::SweepConfig};
use dlp_bench::perf::{self, PerfConfig};
use dlp_bench::report::{AcceptanceTargets, Report, print_summary};
use dlp_bench::runmeta::RunMeta;

use cli::{Cli, Command, DetectorArgs, DetectorKind, EpArg, IpPolicy, PrecisionArg, RoleArg};

fn main() {
    let cli = Cli::parse();
    init_tracing(&cli.log);

    if let Err(e) = run(&cli) {
        eprintln!("\nerror: {e}");
        let mut source = std::error::Error::source(&e);
        while let Some(s) = source {
            eprintln!("  caused by: {s}");
            source = s.source();
        }
        std::process::exit(1);
    }
}

fn init_tracing(level: &str) {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("dlp_bench={level},warn")));
    // stderr so stdout stays a clean summary that can be piped.
    let _ = fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn run(cli: &Cli) -> Result<()> {
    match &cli.command {
        Command::Evaluate(a) => cmd_evaluate(cli, a),
        Command::Sweep(a) => cmd_sweep(cli, a),
        Command::Perf(a) => cmd_perf(cli, a),
        Command::Inspect(a) => cmd_inspect(cli, a),
        Command::Import(a) => cmd_import(cli, a),
    }
}

// ---------------------------------------------------------------------------
// import
// ---------------------------------------------------------------------------

fn cmd_import(cli: &Cli, a: &cli::ImportArgs) -> Result<()> {
    let cli::ImportCorpus::Presidio(p) = &a.corpus;

    let raw = std::fs::read_to_string(&p.input).map_err(|e| {
        Error::io(
            &p.input,
            std::io::Error::new(
                e.kind(),
                format!("{e} — fetch it with `bash scripts/fetch-datasets.sh`"),
            ),
        )
    })?;

    let (mut examples, report) =
        dlp_bench::dataset::import::presidio::import_with(&raw, !p.exclude_gpe)
            .map_err(|e| Error::config(format!("presidio import: {e}")))?;
    eprintln!("{}", report.summary());

    if let Some(limit) = p.limit {
        let before = examples.len();
        examples = dlp_bench::dataset::import::presidio::subset(&examples, limit, cli.seed);
        eprintln!(
            "subset: {} of {before} examples (limit {limit}, seed {})",
            examples.len(),
            cli.seed
        );
    }

    if let Some(parent) = p.output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let mut out = String::new();
    out.push_str(
        "# presidio-research synth_dataset_v2, converted by `dlp-bench import presidio`.\n\
         # MIT licensed. Character offsets converted to UTF-8 bytes and re-verified.\n\
         # Independent third-party corpus: report separately from the local fixtures.\n",
    );
    for ex in &examples {
        out.push_str(&serde_json::to_string(ex)?);
        out.push('\n');
    }
    std::fs::write(&p.output, out).map_err(|e| Error::io(&p.output, e))?;

    let entities: usize = examples.iter().map(|e| e.entities.len()).sum();
    println!(
        "wrote {} examples, {entities} entities to {}",
        examples.len(),
        p.output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Detector construction
// ---------------------------------------------------------------------------

fn build_detector(a: &DetectorArgs) -> Result<Box<dyn Detector>> {
    match a.detector {
        DetectorKind::Deterministic => {
            Ok(Box::new(DeterministicDetector::new(DeterministicConfig {
                private_ip_policy: match a.ip_policy {
                    IpPolicy::Ignore => PrivateIpPolicy::Ignore,
                    IpPolicy::Report => PrivateIpPolicy::Report,
                },
            })))
        }
        DetectorKind::Gliner2Pii => {
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(build_gliner2(a)?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(onnx_disabled("gliner2-pii"))
            }
        }
        DetectorKind::Ai4privacyMdeberta
        | DetectorKind::MinilmL6Pii
        | DetectorKind::DebertaXsmallPii => {
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(build_token_class(a)?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(onnx_disabled(a.detector.as_str()))
            }
        }
        DetectorKind::Gliner1PiiEdge | DetectorKind::Gliner1PiiSmall => {
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(build_gliner1(a)?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(onnx_disabled(a.detector.as_str()))
            }
        }
    }
}

#[cfg(feature = "onnx")]
fn gliner1_model(
    kind: DetectorKind,
) -> Option<&'static dlp_bench::detectors::gliner1::Gliner1Model> {
    use dlp_bench::detectors::gliner1 as g1;
    Some(match kind {
        DetectorKind::Gliner1PiiEdge => &g1::GLINER1_PII_EDGE,
        DetectorKind::Gliner1PiiSmall => &g1::GLINER1_PII_SMALL,
        _ => return None,
    })
}

#[cfg(feature = "onnx")]
fn build_gliner1(a: &DetectorArgs) -> Result<dlp_bench::detectors::gliner1::Gliner1Detector> {
    use dlp_bench::detectors::gliner1::{Gliner1Config, Gliner1Detector};
    use dlp_bench::detectors::onnx::ThreadConfig;

    let model = gliner1_model(a.detector)
        .ok_or_else(|| Error::config(format!("{} is not a GLiNER1 model", a.detector.as_str())))?;
    let mut cfg = Gliner1Config::new(model);
    if a.model_dir.is_some() {
        cfg.model_dir = Some(a.resolved_model_dir());
    }
    cfg.precision = precision(a);
    if let Some(t) = a.threshold {
        cfg.threshold = t;
    }
    cfg.verify_checksum = !a.skip_checksum;
    cfg.provider = execution_provider(a);
    if let Some(t) = a.threads {
        cfg.threads = ThreadConfig::with_intra(t);
    }
    Gliner1Detector::load(cfg).map_err(|source| Error::detector(a.detector.as_str(), source))
}

/// Which descriptor a `--detector` value selects.
#[cfg(feature = "onnx")]
fn token_class_model(
    kind: DetectorKind,
) -> Option<&'static dlp_bench::detectors::token_class::TokenClassModel> {
    use dlp_bench::detectors::token_class as tc;
    Some(match kind {
        DetectorKind::Ai4privacyMdeberta => &tc::AI4PRIVACY_MDEBERTA,
        DetectorKind::MinilmL6Pii => &tc::MINILM_L6_PII,
        DetectorKind::DebertaXsmallPii => &tc::DEBERTA_XSMALL_PII,
        _ => return None,
    })
}

#[cfg(feature = "onnx")]
fn precision(a: &DetectorArgs) -> Option<dlp_bench::detectors::token_class::Precision> {
    use dlp_bench::detectors::token_class::Precision as P;
    a.variant.map(|v| match v {
        PrecisionArg::Fp32 => P::Fp32,
        PrecisionArg::Fp16 => P::Fp16,
        PrecisionArg::Int8 => P::Int8,
    })
}

#[cfg(not(feature = "onnx"))]
fn onnx_disabled(name: &str) -> Error {
    Error::config(format!(
        "{name} requires the `onnx` feature; this binary was built with --no-default-features"
    ))
}

#[cfg(feature = "onnx")]
fn execution_provider(a: &DetectorArgs) -> dlp_bench::detectors::onnx::ExecutionProviderChoice {
    use dlp_bench::detectors::onnx::ExecutionProviderChoice as E;
    match a.ep {
        EpArg::Cpu => E::Cpu,
        EpArg::Coreml => E::CoreMl,
        EpArg::CoremlAne => E::CoreMlAne,
    }
}

#[cfg(feature = "onnx")]
fn build_token_class(
    a: &DetectorArgs,
) -> Result<dlp_bench::detectors::token_class::TokenClassDetector> {
    use dlp_bench::detectors::onnx::ThreadConfig;
    use dlp_bench::detectors::token_class::{TokenClassConfig, TokenClassDetector};

    let model = token_class_model(a.detector).ok_or_else(|| {
        Error::config(format!("{} is not a token classifier", a.detector.as_str()))
    })?;

    let mut cfg = TokenClassConfig::new(model);
    // Only override the directory when the user asked for one, so the
    // descriptor's default stays authoritative.
    if a.model_dir.is_some() {
        cfg.model_dir = Some(a.resolved_model_dir());
    }
    cfg.precision = precision(a);
    if let Some(t) = a.threshold {
        cfg.threshold = t;
    }
    cfg.verify_checksum = !a.skip_checksum;
    cfg.provider = execution_provider(a);
    cfg.merge_adjacent = a.merge_adjacent;
    if let Some(t) = a.threads {
        cfg.threads = ThreadConfig::with_intra(t);
    }
    TokenClassDetector::load(cfg).map_err(|source| Error::detector(a.detector.as_str(), source))
}

#[cfg(feature = "onnx")]
fn build_gliner2(a: &DetectorArgs) -> Result<dlp_bench::detectors::gliner2::Gliner2Detector> {
    use dlp_bench::detectors::gliner2::{Gliner2Config, Gliner2Detector};
    use dlp_bench::detectors::onnx::ThreadConfig;

    let schema = dlp_bench::labels::schema_by_name(&a.schema).ok_or_else(|| {
        Error::config(format!(
            "unknown schema {:?}; choose one of: {}",
            a.schema,
            dlp_bench::labels::SCHEMA_NAMES.join(", ")
        ))
    })?;

    let mut cfg = Gliner2Config::new(a.resolved_model_dir());
    if let Some(t) = a.threshold {
        cfg.threshold = t;
    }
    cfg.schema = schema;
    cfg.verify_checksum = !a.skip_checksum;
    cfg.provider = execution_provider(a);
    if let Some(t) = a.threads {
        cfg.threads = ThreadConfig::with_intra(t);
    }

    Gliner2Detector::load(cfg).map_err(|source| Error::detector("gliner2-pii", source))
}

/// Canonical types a detector's schema can produce, for `--restrict-to-supported`.
fn supported_types(a: &DetectorArgs) -> Option<BTreeSet<dlp_bench::EntityType>> {
    match a.detector {
        // Rules cover only what has structure; restricting to them would hide
        // the very gap the benchmark exists to measure.
        DetectorKind::Deterministic => None,
        DetectorKind::Gliner2Pii => dlp_bench::labels::schema_by_name(&a.schema)
            .map(|s| s.covered_types().into_iter().collect()),
        // The ai4privacy taxonomy has no developer-secret types at all, so
        // restricting to it is the difference between "how good is it at what
        // it was trained on" and "is it enough for our policy".
        DetectorKind::Ai4privacyMdeberta
        | DetectorKind::MinilmL6Pii
        | DetectorKind::DebertaXsmallPii => {
            #[cfg(feature = "onnx")]
            {
                token_class_model(a.detector)
                    .map(|m| m.labels.covered_types().into_iter().collect())
            }
            #[cfg(not(feature = "onnx"))]
            {
                None
            }
        }
        DetectorKind::Gliner1PiiEdge | DetectorKind::Gliner1PiiSmall => Some(
            dlp_bench::labels::GLINER1_PII
                .covered_types()
                .into_iter()
                .collect(),
        ),
    }
}

fn load_dataset(path: &std::path::Path, role: Option<RoleArg>) -> Result<Dataset> {
    let role = role.map(|r| match r {
        RoleArg::Dev => DatasetRole::Dev,
        RoleArg::Test => DatasetRole::Test,
    });
    Dataset::load(path, role).map_err(|source| Error::dataset(path, source))
}

fn eval_config(a: &DetectorArgs, show_errors: usize, restrict: bool) -> EvalConfig {
    EvalConfig {
        error_limit: show_errors,
        restrict_to_supported: restrict.then(|| supported_types(a)).flatten(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

fn cmd_evaluate(cli: &Cli, a: &cli::EvaluateArgs) -> Result<()> {
    let mut dataset = load_dataset(&a.dataset, a.role)?;

    if a.adversarial {
        let extra = transform::expand(&dataset.examples, transform::Transform::ALL, cli.seed);
        eprintln!(
            "adversarial: +{} variants from {} examples",
            extra.len(),
            dataset.len()
        );
        dataset.examples.extend(extra);
        dataset.examples.sort_by(|x, y| x.id.cmp(&y.id));
        dataset.name = format!("{}+adversarial", dataset.name);
    }

    let detector = build_detector(&a.detector)?;
    let cfg = eval_config(&a.detector, a.show_errors, a.restrict_to_supported);

    let evaluation = evaluate(detector.as_ref(), &dataset, &cfg)
        .map_err(|source| Error::detector(detector.name().to_owned(), source))?;

    let mut report = Report {
        run: RunMeta::capture(cli.seed),
        artifacts: detector.artifact_info(),
        quality: Some(evaluation),
        performance: None,
        sweep: None,
        acceptance: vec![],
    };
    report.check_acceptance(AcceptanceTargets::default());
    emit(&report, a.output.as_deref())
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(cli: &Cli, a: &cli::SweepArgs) -> Result<()> {
    let dataset = load_dataset(&a.dataset, a.role)?;
    let detector = build_detector(&a.detector)?;

    let result = dlp_bench::eval::sweep::sweep(
        detector.as_ref(),
        &dataset,
        &eval_config(&a.detector, 0, false),
        &SweepConfig {
            start: a.threshold_start,
            end: a.threshold_end,
            step: a.threshold_step,
        },
    )
    .map_err(|source| Error::detector(detector.name().to_owned(), source))?;

    if let Some(path) = &a.csv {
        std::fs::write(path, result.to_csv()).map_err(|e| Error::io(path, e))?;
        eprintln!("csv written to {}", path.display());
    }

    let report = Report {
        run: RunMeta::capture(cli.seed),
        artifacts: detector.artifact_info(),
        quality: None,
        performance: None,
        sweep: Some(result),
        acceptance: vec![],
    };
    emit(&report, a.output.as_deref())
}

// ---------------------------------------------------------------------------
// perf
// ---------------------------------------------------------------------------

fn cmd_perf(cli: &Cli, a: &cli::PerfArgs) -> Result<()> {
    let cfg = PerfConfig {
        buckets: a.buckets.clone(),
        iterations: a.iterations,
        warmup: a.warmup,
        thread_counts: a.thread_counts.clone(),
        measure_throughput: !a.thread_counts.is_empty(),
    };

    let warmup_text = perf::synthetic_document(32);

    // Cold start brackets construction exactly, with nothing else in scope.
    let (detector, cold_start) = perf::measure_cold_start(
        || match build_detector(&a.detector) {
            Ok(d) => Ok(d),
            Err(Error::Detector { source, .. }) => Err(*source),
            Err(other) => Err(dlp_bench::DetectorError::config(other.to_string())),
        },
        &warmup_text,
    )
    .map_err(|source| Error::detector(a.detector.detector.as_str(), source))?;

    let name = detector.name().to_owned();
    let to_err = |source| Error::detector(name.clone(), source);

    let buckets = perf::measure_latency(detector.as_ref(), &cfg).map_err(to_err)?;
    let warm_rss = perf::memory::snapshot();

    let throughput = if cfg.measure_throughput {
        perf::measure_throughput(&|| build_one(&a.detector), &cfg, a.throughput_docs)
            .map_err(|source| Error::detector(name.clone(), source))?
    } else {
        Vec::new()
    };

    let schema_prefix_tokens = detector.fixed_prompt_tokens();

    let mut report = Report {
        run: RunMeta::capture(cli.seed),
        artifacts: detector.artifact_info(),
        quality: None,
        performance: Some(dlp_bench::perf::PerfReport {
            detector: name,
            model_size_bytes: detector.model_size_bytes(),
            schema_prefix_tokens,
            cold_start,
            buckets,
            throughput,
            warm_rss,
            peak_rss: perf::memory::snapshot(),
            iterations: cfg.iterations,
            warmup: cfg.warmup,
        }),
        sweep: None,
        acceptance: vec![],
    };
    report.check_acceptance(AcceptanceTargets::default());
    emit(&report, a.output.as_deref())
}

/// Build one detector instance for a throughput worker.
///
/// Forces a single intra-op thread per worker. Otherwise N workers each open a
/// session with `min(cores, 8)` intra-op threads and the measurement
/// oversubscribes the machine several times over — throughput flattens and the
/// numbers describe scheduler thrash rather than the detector. With one thread
/// each, `threads = N` means "N documents in flight", which is the question an
/// endpoint agent actually has.
fn build_one(a: &DetectorArgs) -> std::result::Result<Box<dyn Detector>, dlp_bench::DetectorError> {
    let mut single = DetectorArgs {
        threads: Some(1),
        ..a.clone()
    };
    // Checksumming 446 MB once per worker would dominate setup.
    single.skip_checksum = true;
    build_detector(&single).map_err(|e| match e {
        Error::Detector { source, .. } => *source,
        other => dlp_bench::DetectorError::config(other.to_string()),
    })
}

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

fn cmd_inspect(_cli: &Cli, a: &cli::InspectArgs) -> Result<()> {
    let out = std::io::stdout();
    let mut w = out.lock();

    if let Some(path) = &a.dataset {
        let d = load_dataset(path, None)?;
        writeln!(w, "dataset {} ({:?})", d.name, d.role).map_err(|e| Error::io(path, e))?;
        writeln!(w, "  sha256      {}", d.sha256).ok();
        writeln!(w, "  examples    {}", d.len()).ok();
        writeln!(w, "  entities    {}", d.total_entities()).ok();
        writeln!(w, "  negatives   {}", d.negative_count()).ok();
        writeln!(w, "  by entity type:").ok();
        for (t, n) in d.entity_counts() {
            writeln!(w, "    {t:<20} {n}").ok();
        }
        writeln!(w, "  by tag:").ok();
        for (t, n) in d.tag_counts() {
            writeln!(w, "    {t:<20} {n}").ok();
        }
        writeln!(w).ok();
    }

    let detector = build_detector(&a.detector)?;
    writeln!(w, "detector {}", detector.name()).ok();
    if let Some(sz) = detector.model_size_bytes() {
        writeln!(w, "  model size  {:.1} MB", sz as f64 / 1_048_576.0).ok();
    }
    writeln!(w, "  threshold   {:?}", detector.threshold()).ok();
    for art in detector.artifact_info() {
        writeln!(w, "  artifact {}: {}", art.name, art.path).ok();
        if let Some(r) = &art.revision {
            writeln!(w, "    repo {} @ {}", art.repo.as_deref().unwrap_or("-"), r).ok();
        }
    }

    #[cfg(feature = "onnx")]
    if token_class_model(a.detector.detector).is_some() {
        let t = build_token_class(&a.detector)?;
        writeln!(
            w,
            "  labels      {} BIO classes over {} mapped types",
            t.num_labels(),
            t.model().labels.len()
        )
        .ok();
        writeln!(w, "  precision   {}", t.variant().precision.as_str()).ok();
        writeln!(w, "  note        {}", t.model().note).ok();
        let unreachable = dlp_bench::labels::unreachable_types(t.model().labels);
        writeln!(
            w,
            "  cannot detect ({}): {}",
            unreachable.len(),
            unreachable
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
        .ok();
        writeln!(
            w,
            "  chunking    {} tokens, {} overlap",
            t.config().chunk_tokens,
            t.config().chunk_overlap
        )
        .ok();
        writeln!(w, "  graph:\n{}", t.describe_graph()).ok();
    }

    #[cfg(feature = "onnx")]
    if gliner1_model(a.detector.detector).is_some() {
        let g = build_gliner1(&a.detector)?;
        writeln!(w, "  backbone    {}", g.model().backbone).ok();
        writeln!(w, "  precision   {}", g.variant().precision.as_str()).ok();
        writeln!(w, "  schema      {} labels", g.config().schema.len()).ok();
        writeln!(w, "  max width   {} words", g.config().max_width).ok();
        writeln!(w, "  overlap     {:?}", g.config().overlap).ok();
        let broken: Vec<_> = g
            .model()
            .variants
            .iter()
            .filter(|v| v.broken.is_some())
            .map(|v| v.precision.as_str())
            .collect();
        if !broken.is_empty() {
            writeln!(
                w,
                "  unusable    {} (see --variant error for why)",
                broken.join(", ")
            )
            .ok();
        }
        writeln!(w, "  graph:\n{}", g.describe_graph()).ok();
    }

    #[cfg(feature = "onnx")]
    if a.detector.detector == DetectorKind::Gliner2Pii {
        let g = build_gliner2(&a.detector)?;
        writeln!(
            w,
            "  schema      {} labels ({})",
            g.config().schema.len(),
            a.detector.schema
        )
        .ok();
        writeln!(w, "  prefix      {} tokens per call", g.prefix_tokens()).ok();
        writeln!(w, "  graph:\n{}", g.describe_graph()).ok();
    }

    for text in &a.text {
        writeln!(w, "\n--- {text:?}").ok();
        let found = detector
            .detect(text)
            .map_err(|source| Error::detector(detector.name(), source))?;
        if found.is_empty() {
            writeln!(w, "    (nothing detected)").ok();
        }
        for d in &found {
            writeln!(
                w,
                "    {:<20} {:>5.3}  [{}..{}]  {:?}",
                d.entity_type.to_string(),
                d.score,
                d.start,
                d.end,
                &text[d.start..d.end]
            )
            .ok();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------

fn emit(report: &Report, output: Option<&std::path::Path>) -> Result<()> {
    let out = std::io::stdout();
    let mut w = out.lock();
    print_summary(&mut w, report).map_err(|e| Error::io("<stdout>", e))?;

    if let Some(path) = output {
        report.write_json(path).map_err(|e| Error::io(path, e))?;
        writeln!(w, "json written to {}", path.display()).ok();
    }
    Ok(())
}
