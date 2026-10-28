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

use cli::{
    BackendArg, Cli, Command, DetectorArgs, DetectorKind, DeviceArg, EpArg, IpPolicy, PrecisionArg,
    RoleArg,
};

fn main() {
    let cli = Cli::parse();
    init_tracing(&cli.log);
    size_candle_gemm(&cli);

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

/// Make Candle's matmuls split across as many threads as the detector's pool.
///
/// Candle runs every CPU op on the rayon pool it is called from, but sizes the
/// `gemm` split from `RAYON_NUM_THREADS` (else the performance-core count),
/// not from that pool. With `--threads 8` on a 6-performance-core M5 Pro that
/// left two pool threads idle in every matmul, which is 80% of the work:
/// matching the two was worth ~17% of p50. An explicit `RAYON_NUM_THREADS`
/// in the environment wins.
fn size_candle_gemm(cli: &Cli) {
    let a = match &cli.command {
        Command::Evaluate(a) => &a.detector,
        Command::Sweep(a) => &a.detector,
        Command::Perf(a) => &a.detector,
        Command::Concurrency(a) => &a.detector,
        Command::Inspect(a) => &a.detector,
        Command::Import(_) => return,
    };
    let cpu =
        a.backend == BackendArg::Candle && a.device.unwrap_or(DeviceArg::Cpu) == DeviceArg::Cpu;
    if !cpu || std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let threads = a
        .threads
        .unwrap_or_else(|| dlp_bench::runtime::ThreadConfig::default().intra_op);
    // SAFETY: called from `main` before any thread is spawned, so nothing can
    // be reading the environment concurrently.
    unsafe { std::env::set_var("RAYON_NUM_THREADS", threads.to_string()) };
}

fn run(cli: &Cli) -> Result<()> {
    match &cli.command {
        Command::Evaluate(a) => cmd_evaluate(cli, a),
        Command::Sweep(a) => cmd_sweep(cli, a),
        Command::Perf(a) => cmd_perf(cli, a),
        Command::Concurrency(a) => cmd_concurrency(cli, a),
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
    check_backend(a)?;
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

/// Reject backend/device combinations that would silently measure something
/// other than what was asked for.
///
/// Every ONNX Runtime knob is meaningless under Candle, and a result labelled
/// `candle` that quietly ran with `--ep coreml` ignored is exactly the kind of
/// row that later gets compared against the wrong baseline.
fn check_backend(a: &DetectorArgs) -> Result<()> {
    match a.backend {
        BackendArg::Ort => {
            if a.device.is_some() {
                return Err(Error::config(
                    "--device selects a Candle device; under --backend ort use --ep".to_owned(),
                ));
            }
        }
        BackendArg::Candle => {
            if a.detector != DetectorKind::Gliner2Pii {
                return Err(Error::config(format!(
                    "--backend candle is implemented for gliner2-pii only, not {}",
                    a.detector.as_str()
                )));
            }
            let r = &a.runtime;
            let ort_flag = a.ep != EpArg::Cpu
                || r.coreml_format.is_some()
                || r.coreml_compute_units.is_some()
                || r.coreml_subgraphs.is_some()
                || r.coreml_static_shapes.is_some()
                || r.coreml_specialization.is_some()
                || r.coreml_profile
                || r.coreml_cache_dir.is_some()
                || r.coreml_fp16_accum.is_some()
                || r.disable_cpu_fallback
                || r.no_intra_op_spinning
                || r.ort_profile.is_some()
                || r.ort_log.is_some()
                || !r.dim_override.is_empty();
            if ort_flag {
                return Err(Error::config(
                    "--ep and the ONNX Runtime flags have no effect under --backend candle; \
                     use --device cpu|metal"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn backend(a: &DetectorArgs) -> dlp_bench::runtime::Backend {
    use dlp_bench::runtime::{Backend, CandleDevice};
    match a.backend {
        BackendArg::Ort => Backend::Ort,
        BackendArg::Candle => Backend::Candle(match a.device.unwrap_or(DeviceArg::Cpu) {
            DeviceArg::Cpu => CandleDevice::Cpu,
            DeviceArg::Metal => CandleDevice::Metal,
        }),
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
    cfg.session = session_options(a)?;
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

/// Translate the CLI into a [`SessionOptions`], rejecting combinations that
/// would silently measure something other than what was asked for.
///
/// Every check here exists because the alternative is a plausible-looking
/// number: a `--coreml-*` flag under `--ep cpu` does nothing, a compute-plan
/// request under NeuralNetwork produces an empty plan, and a profile without
/// `ORT_LOG` set produces an empty file. All three read as findings.
#[cfg(feature = "onnx")]
fn session_options(a: &DetectorArgs) -> Result<dlp_bench::runtime::SessionOptions> {
    use dlp_bench::runtime::{
        CoreMlComputeUnits as U, CoreMlModelFormat as F, CoreMlOptions, CoreMlSpecialization as S,
        DimOverride, OrtLogLevel, SessionOptions,
    };

    let r = &a.runtime;
    let provider = execution_provider(a);

    let touched_coreml = r.coreml_format.is_some()
        || r.coreml_compute_units.is_some()
        || r.coreml_subgraphs.is_some()
        || r.coreml_static_shapes.is_some()
        || r.coreml_specialization.is_some()
        || r.coreml_profile
        || r.coreml_cache_dir.is_some()
        || r.coreml_fp16_accum.is_some();

    if touched_coreml && !provider.is_coreml() {
        return Err(Error::config(
            "--coreml-* flags need --ep coreml or --ep coreml-ane; under --ep cpu \
             they would be silently ignored and the result would be labelled cpu"
                .to_owned(),
        ));
    }
    if r.disable_cpu_fallback && !provider.is_coreml() {
        return Err(Error::config(
            "--disable-cpu-fallback under --ep cpu would refuse to place the graph \
             on the only provider available"
                .to_owned(),
        ));
    }

    let coreml = provider.is_coreml().then(|| {
        // Start from what the provider implies, so `--ep coreml-ane` plus an
        // unrelated flag does not silently revert the compute units to All.
        let base = SessionOptions::with_provider(provider)
            .resolved_coreml()
            .unwrap_or_default();
        CoreMlOptions {
            model_format: match r.coreml_format {
                Some(cli::CoreMlFormatArg::NeuralNetwork) => F::NeuralNetwork,
                Some(cli::CoreMlFormatArg::MlProgram) => F::MlProgram,
                None => base.model_format,
            },
            compute_units: match r.coreml_compute_units {
                Some(cli::CoreMlUnitsArg::All) => U::All,
                Some(cli::CoreMlUnitsArg::CpuAne) => U::CpuAndNeuralEngine,
                Some(cli::CoreMlUnitsArg::CpuGpu) => U::CpuAndGpu,
                Some(cli::CoreMlUnitsArg::CpuOnly) => U::CpuOnly,
                None => base.compute_units,
            },
            subgraphs: r.coreml_subgraphs.unwrap_or(base.subgraphs),
            static_input_shapes: r.coreml_static_shapes.unwrap_or(base.static_input_shapes),
            specialization: match r.coreml_specialization {
                Some(cli::CoreMlSpecializationArg::Default) => S::Default,
                Some(cli::CoreMlSpecializationArg::FastPrediction) => S::FastPrediction,
                None => base.specialization,
            },
            profile_compute_plan: r.coreml_profile,
            low_precision_accumulation_on_gpu: r
                .coreml_fp16_accum
                .unwrap_or(base.low_precision_accumulation_on_gpu),
            model_cache_dir: r.coreml_cache_dir.clone(),
        }
    });

    if r.coreml_profile && coreml.as_ref().map(|c| c.model_format) != Some(F::MlProgram) {
        return Err(Error::config(
            "--coreml-profile needs --coreml-format ml-program: Apple's MLComputePlan \
             enumerates MLProgram operations and reports nothing for NeuralNetwork"
                .to_owned(),
        ));
    }

    // `ort` reads ORT_LOG once, when the environment is created on the first
    // session build, and defaults to ERROR. Setting only the session level
    // produces a session that would log and an environment that discards it.
    if r.coreml_profile || r.ort_log.is_some() {
        let env = std::env::var("ORT_LOG").unwrap_or_default();
        if !matches!(env.as_str(), "info" | "verbose") {
            return Err(Error::config(format!(
                "ONNX Runtime discards log records below its environment severity, \
                 which is {:?} here. Re-run with ORT_LOG=verbose in the environment.",
                if env.is_empty() {
                    "error (default)"
                } else {
                    &env
                }
            )));
        }
    }

    let dimension_overrides: Vec<DimOverride> = r
        .dim_override
        .iter()
        .map(|(name, size)| DimOverride {
            name: name.clone(),
            size: *size,
        })
        .collect();

    // Only GLiNER2 knows how to pad its inputs to a fixed shape; for anything
    // else an override turns every call into a shape-mismatch error at run
    // time, which is a confusing way to learn the flag was unsupported.
    if !dimension_overrides.is_empty() && a.detector != DetectorKind::Gliner2Pii {
        return Err(Error::config(format!(
            "--dim-override is only implemented for gliner2-pii; {} has no padding \
             path and every call would fail on a shape mismatch",
            a.detector.as_str()
        )));
    }

    Ok(SessionOptions {
        provider,
        coreml,
        disable_cpu_fallback: r.disable_cpu_fallback,
        disable_intra_op_spinning: r.no_intra_op_spinning,
        dimension_overrides,
        log_level: r.ort_log.map(|l| match l {
            cli::OrtLogArg::Error => OrtLogLevel::Error,
            cli::OrtLogArg::Warning => OrtLogLevel::Warning,
            cli::OrtLogArg::Info => OrtLogLevel::Info,
            cli::OrtLogArg::Verbose => OrtLogLevel::Verbose,
        }),
        profile_json: r.ort_profile.clone(),
        capture_ort_log: r.coreml_profile || r.ort_log.is_some(),
    })
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
    cfg.session = session_options(a)?;
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
    cfg.session = session_options(a)?;
    cfg.backend = backend(a);
    // Derived from the same overrides the session gets, so the two cannot
    // disagree; `Gliner2Detector::load` re-checks and refuses if they do.
    cfg.fixed_shape = fixed_shape(&cfg.session)?;
    if let Some(t) = a.threads {
        cfg.threads = ThreadConfig::with_intra(t);
    }

    Gliner2Detector::load(cfg).map_err(|source| Error::detector("gliner2-pii", source))
}

/// Turn `--dim-override seq_len=N --dim-override num_words=M` into the padding
/// GLiNER2 must apply for a session built with those overrides.
///
/// Both or neither: a session pinned on one axis and variable on the other
/// fails on every call, and the error would point at the graph rather than at
/// the missing flag.
#[cfg(feature = "onnx")]
fn fixed_shape(
    session: &dlp_bench::runtime::SessionOptions,
) -> Result<Option<dlp_bench::detectors::gliner2::prompt::FixedShape>> {
    use dlp_bench::detectors::gliner2::prompt::FixedShape;
    match (
        session.dimension_override("seq_len"),
        session.dimension_override("num_words"),
    ) {
        (None, None) => Ok(None),
        (Some(seq_len), Some(num_words)) => Ok(Some(FixedShape {
            seq_len: seq_len as usize,
            num_words: num_words as usize,
        })),
        (seq, _) => Err(Error::config(format!(
            "static shapes need both axes pinned: --dim-override {} is set but {} is not",
            if seq.is_some() {
                "seq_len"
            } else {
                "num_words"
            },
            if seq.is_some() {
                "num_words"
            } else {
                "seq_len"
            },
        ))),
    }
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
        stage_timing: a.stage_timing,
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

    // Second pass, after the headline numbers are already banked, so the
    // timers cannot influence them. Refuse rather than print zeros: a
    // breakdown of all-zero stages reads as "these are free".
    let stage_breakdown = if cfg.stage_timing {
        if !detector.supports_stage_timing() {
            return Err(Error::config(format!(
                "{} does not implement stage timing; --stage-timing would report \
                 zeros for every stage. Supported today: gliner2-pii.",
                name
            )));
        }
        perf::measure_stages(detector.as_ref(), &cfg).map_err(to_err)?
    } else {
        Vec::new()
    };

    let throughput = if cfg.measure_throughput {
        perf::measure_throughput(&|| build_one(&a.detector), &cfg, a.throughput_docs)
            .map_err(|source| Error::detector(name.clone(), source))?
    } else {
        Vec::new()
    };

    let schema_prefix_tokens = detector.fixed_prompt_tokens();
    // Last, because it flushes ONNX Runtime's node profile: anything measured
    // after this point would not appear in the EP breakdown.
    let runtime = detector.runtime_info();

    let mut report = Report {
        run: RunMeta::capture(cli.seed),
        artifacts: detector.artifact_info(),
        quality: None,
        performance: Some(dlp_bench::perf::PerfReport {
            detector: name,
            model_size_bytes: detector.model_size_bytes(),
            schema_prefix_tokens,
            runtime,
            cold_start,
            buckets,
            stage_breakdown,
            throughput,
            concurrency: Vec::new(),
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

// ---------------------------------------------------------------------------
// concurrency
// ---------------------------------------------------------------------------

fn cmd_concurrency(cli: &Cli, a: &cli::ConcurrencyArgs) -> Result<()> {
    let budget = a
        .detector
        .threads
        .unwrap_or_else(|| dlp_bench::runtime::ThreadConfig::default().intra_op);
    let cfg = perf::concurrency::ConcurrencyConfig {
        clients: a.clients.clone(),
        docs: a.docs.clone(),
        requests_per_client: a.requests_per_client,
        pool_max: a.pool_max,
        thread_budget: budget,
    };
    let build = |threads: usize| {
        let args = DetectorArgs {
            threads: Some(threads),
            ..a.detector.clone()
        };
        build_detector(&args).map_err(|e| match e {
            Error::Detector { source, .. } => *source,
            other => dlp_bench::DetectorError::config(other.to_string()),
        })
    };
    let run = perf::concurrency::measure(&build, &cfg)
        .map_err(|source| Error::detector(a.detector.detector.as_str(), source))?;
    let d = &run.last;

    let mut report = Report {
        run: RunMeta::capture(cli.seed),
        artifacts: d.artifact_info(),
        quality: None,
        performance: Some(dlp_bench::perf::PerfReport {
            detector: d.name().to_owned(),
            model_size_bytes: d.model_size_bytes(),
            schema_prefix_tokens: d.fixed_prompt_tokens(),
            runtime: d.runtime_info(),
            cold_start: run.cold_start,
            buckets: Vec::new(),
            stage_breakdown: Vec::new(),
            throughput: Vec::new(),
            concurrency: run.points,
            warm_rss: perf::memory::snapshot(),
            peak_rss: perf::memory::snapshot(),
            iterations: a.requests_per_client,
            warmup: 1,
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

        // Encoder sizes at each perf bucket, so a static-shape run can be sized
        // from measured sub-token counts instead of a guessed words-to-pieces
        // ratio. Rounded up to the next multiple of 32 for headroom, and
        // printed in the exact form `--dim-override` takes.
        writeln!(w, "  bucket shapes (for --dim-override):").ok();
        for &n in dlp_bench::perf::DEFAULT_BUCKETS {
            let doc = dlp_bench::perf::synthetic_document(n);
            match g.encoded_size(&doc) {
                Ok((seq, words)) => {
                    let round = |v: usize| v.div_ceil(32) * 32;
                    writeln!(
                        w,
                        "    {n:>5} words -> {seq:>5} tokens, {words:>4} words   \
                         --dim-override seq_len={} --dim-override num_words={}",
                        round(seq),
                        round(words)
                    )
                    .ok();
                }
                Err(e) => {
                    writeln!(w, "    {n:>5} words -> error: {e}").ok();
                }
            }
        }
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
