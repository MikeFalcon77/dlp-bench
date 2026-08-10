//! Command-line interface.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "dlp-bench",
    version,
    about = "Benchmark local PII detectors for endpoint DLP / AI gateways",
    long_about = "Compares local PII detection approaches under one taxonomy, one dataset \
                  format, one span convention and one evaluator.\n\n\
                  Detection quality and endpoint suitability are measured separately: use \
                  `evaluate` for quality and `perf` for latency and memory. Always build with \
                  --release before quoting a latency number."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Log level (error, warn, info, debug, trace).
    ///
    /// Logging is metadata-oriented: benchmark text is PII by construction and
    /// is never logged. Use `--show-errors` or `inspect` to see content.
    #[arg(long, global = true, default_value = "warn")]
    pub log: String,

    /// Seed for any randomised step. Recorded in the result.
    #[arg(long, global = true, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Measure detection quality on a dataset.
    Evaluate(EvaluateArgs),
    /// Sweep a detector's confidence threshold.
    Sweep(SweepArgs),
    /// Measure cold start, warm latency, throughput and memory.
    Perf(PerfArgs),
    /// Show what is loaded: detector configuration, graph I/O, dataset stats.
    Inspect(InspectArgs),
    /// Convert a third-party corpus into the benchmark's JSONL format.
    Import(ImportArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct ImportArgs {
    #[command(subcommand)]
    pub corpus: ImportCorpus,
}

#[derive(Subcommand, Debug, Clone)]
pub enum ImportCorpus {
    /// `presidio-research` synth_dataset_v2 (MIT, 1500 docs, 2863 spans).
    ///
    /// Character offsets are converted to UTF-8 bytes and every span is
    /// re-verified against the surface text the corpus supplies.
    Presidio(PresidioArgs),
}

#[derive(clap::Args, Debug, Clone)]
pub struct PresidioArgs {
    /// Raw upstream JSON; fetch with `bash scripts/fetch-datasets.sh`.
    #[arg(long, default_value = "datasets/raw/presidio-synth-v2.json")]
    pub input: PathBuf,

    #[arg(long, default_value = "datasets/presidio-synth.jsonl")]
    pub output: PathBuf,

    /// Keep at most N examples, stratified so rare entity types survive.
    #[arg(long)]
    pub limit: Option<usize>,

    /// Do not score bare place names (`GPE`) as ADDRESS.
    ///
    /// 411 of the corpus's spans are standalone cities and countries. Scoring
    /// them favours models with granular location labels over models with one
    /// broad `address` label, so both settings are worth measuring.
    #[arg(long)]
    pub exclude_gpe: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorKind {
    /// Rule-based recognizers with structural validators.
    Deterministic,
    /// Fastino GLiNER2 PII via a local ONNX export.
    #[value(name = "gliner2-pii")]
    Gliner2Pii,
    /// ai4privacy/OpenPII BIO token classifier (mdeberta-v3-base) via ONNX.
    #[value(name = "ai4privacy-mdeberta")]
    Ai4privacyMdeberta,
    /// MiniLM-L6 BIO token classifier: 6 layers x 384, the small end.
    #[value(name = "minilm-l6-pii")]
    MinilmL6Pii,
    /// DeBERTa-v3-xsmall BIO token classifier, 9 entity types.
    #[value(name = "deberta-xsmall-pii")]
    DebertaXsmallPii,
    /// GLiNER1 zero-shot span model, ettin-encoder-32m.
    #[value(name = "gliner1-pii-edge")]
    Gliner1PiiEdge,
    /// GLiNER1 zero-shot span model, ettin-encoder-68m.
    #[value(name = "gliner1-pii-small")]
    Gliner1PiiSmall,
}

impl DetectorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DetectorKind::Deterministic => "deterministic",
            DetectorKind::Gliner2Pii => "gliner2-pii",
            DetectorKind::Ai4privacyMdeberta => "ai4privacy-mdeberta",
            DetectorKind::MinilmL6Pii => "minilm-l6-pii",
            DetectorKind::DebertaXsmallPii => "deberta-xsmall-pii",
            DetectorKind::Gliner1PiiEdge => "gliner1-pii-edge",
            DetectorKind::Gliner1PiiSmall => "gliner1-pii-small",
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum IpPolicy {
    /// Private, loopback and link-local addresses are not PII (default).
    Ignore,
    /// All valid IP addresses are PII.
    Report,
}

/// ONNX Runtime execution provider.
///
/// There is no Metal EP in ONNX Runtime; on Apple silicon CoreML is the route
/// to the GPU and Neural Engine, and it partitions the graph itself.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpArg {
    /// CPU only (default, and the Phase-1 baseline).
    Cpu,
    /// CoreML with all compute units, CPU fallback for unsupported subgraphs.
    Coreml,
    /// CoreML restricted to CPU + Neural Engine, no GPU.
    CoremlAne,
}

/// CoreML model representation.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlFormatArg {
    /// CoreML 3. The default, and the only format measured so far.
    NeuralNetwork,
    /// CoreML 5. More operators, and the only format Apple's compute-plan
    /// profiler can introspect.
    MlProgram,
}

/// Hardware CoreML may schedule onto.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlUnitsArg {
    All,
    CpuAne,
    CpuGpu,
    CpuOnly,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlSpecializationArg {
    Default,
    /// Trade session-load time and memory for prediction latency.
    FastPrediction,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrtLogArg {
    Error,
    Warning,
    Info,
    Verbose,
}

/// Numeric precision of the weight file to load.
///
/// Explicit because it is a first-class benchmark axis: the GLiNER models ship
/// the same weights at three precisions, and a latency number is meaningless
/// without knowing which one produced it.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionArg {
    Fp32,
    Fp16,
    Int8,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoleArg {
    Dev,
    Test,
}

#[derive(clap::Args, Debug, Clone)]
pub struct DetectorArgs {
    #[arg(long, value_enum)]
    pub detector: DetectorKind,

    /// Directory holding the model artifacts. Defaults per detector.
    #[arg(long)]
    pub model_dir: Option<PathBuf>,

    /// Confidence threshold for ML detectors.
    ///
    /// Unset means "use the model's own default", which differs per model —
    /// GLiNER1's card recommends 0.3, the BIO classifiers use 0.5. Imposing a
    /// single number on all of them would silently mis-tune whichever was not
    /// calibrated for it.
    #[arg(long)]
    pub threshold: Option<f32>,

    /// Zero-shot label schema: minimal (5), default (17) or full (42).
    ///
    /// Affects both quality and latency — the label prompt is re-encoded on
    /// every call, so schema size is a real cost.
    #[arg(long, default_value = "default")]
    pub schema: String,

    /// ONNX intra-op thread count. Defaults to min(cores, 8).
    #[arg(long)]
    pub threads: Option<usize>,

    /// Execution provider. CPU is the baseline; compare before switching.
    #[arg(long, value_enum, default_value = "cpu")]
    pub ep: EpArg,

    /// Weight precision. Defaults per model; an error if unavailable.
    #[arg(long, value_enum)]
    pub variant: Option<PrecisionArg>,

    /// Whether private/loopback IPs count as PII (deterministic detector).
    #[arg(long, value_enum, default_value = "ignore")]
    pub ip_policy: IpPolicy,

    /// Skip the model sha256 check. Faster startup; makes the run unciteable.
    #[arg(long)]
    pub skip_checksum: bool,

    /// Merge adjacent same-type spans (token classifiers only).
    ///
    /// On by default: the ai4privacy taxonomy splits a person into FIRSTNAME +
    /// LASTNAME and an address into BUILDINGNUMBER + STREET + CITY, so without
    /// merging one gold entity is matched once and counted against N-1 times.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub merge_adjacent: bool,

    #[command(flatten)]
    pub runtime: RuntimeArgs,
}

/// ONNX Runtime session knobs.
///
/// Flattened into [`DetectorArgs`] rather than into `perf` alone, because the
/// static-shape correctness gate runs `evaluate` under the same session
/// configuration as the latency measurement — if these lived on `perf` only,
/// the two could not be compared.
///
/// Every one of these defaults to "leave it alone", so a command that sets none
/// of them builds exactly the session the benchmark has always built.
#[derive(clap::Args, Debug, Clone)]
#[command(next_help_heading = "ONNX Runtime")]
pub struct RuntimeArgs {
    /// CoreML model format. MLProgram is untested on the GLiNER2 graph.
    #[arg(long, value_enum)]
    pub coreml_format: Option<CoreMlFormatArg>,

    /// Hardware CoreML may use. Overrides what `--ep` implies.
    #[arg(long, value_enum)]
    pub coreml_compute_units: Option<CoreMlUnitsArg>,

    /// Compile CoreML subgraphs inside Loop/Scan/If bodies.
    ///
    /// Not main-graph partitioning, which is unconditional. This was turned on
    /// at the same time as MLProgram when the graph last broke, so neither has
    /// been tested alone.
    #[arg(long)]
    pub coreml_subgraphs: Option<bool>,

    /// Only let CoreML claim nodes whose inputs have static shapes.
    ///
    /// Restrictive on its own — pair it with `--dim-override`.
    #[arg(long)]
    pub coreml_static_shapes: Option<bool>,

    #[arg(long, value_enum)]
    pub coreml_specialization: Option<CoreMlSpecializationArg>,

    /// Log which hardware each CoreML operator ran on.
    ///
    /// Requires `--coreml-format ml-program`: Apple's compute-plan API
    /// enumerates MLProgram operations and reports nothing for NeuralNetwork.
    #[arg(long)]
    pub coreml_profile: bool,

    /// Reuse the compiled CoreML model across runs, instead of recompiling the
    /// 426 MB graph on every process start.
    #[arg(long)]
    pub coreml_cache_dir: Option<PathBuf>,

    /// Allow fp16 accumulation on the GPU.
    #[arg(long)]
    pub coreml_fp16_accum: Option<bool>,

    /// Fail to build unless the graph places entirely on the accelerator.
    ///
    /// A diagnostic, not an optimisation: the error names the nodes CoreML
    /// could not take, which is the question `--ep coreml` cannot answer.
    #[arg(long)]
    pub disable_cpu_fallback: bool,

    /// Turn off ONNX Runtime's intra-op spin-wait.
    #[arg(long)]
    pub no_intra_op_spinning: bool,

    /// Write ONNX Runtime's per-node profile here and summarise kernel time by
    /// execution provider in the report.
    #[arg(long)]
    pub ort_profile: Option<PathBuf>,

    /// ONNX Runtime session log level. Also needs `ORT_LOG` in the environment.
    #[arg(long, value_enum)]
    pub ort_log: Option<OrtLogArg>,

    /// Pin a free graph dimension, e.g. `--dim-override seq_len=512`.
    ///
    /// Repeatable. Applies to graph inputs only; computed output dimensions are
    /// not free dimensions and cannot be set. GLiNER2 pads to match.
    #[arg(long, value_parser = parse_dim_override)]
    pub dim_override: Vec<(String, i64)>,
}

fn parse_dim_override(s: &str) -> std::result::Result<(String, i64), String> {
    let (name, size) = s
        .split_once('=')
        .ok_or_else(|| format!("expected name=size, got {s:?}"))?;
    let size: i64 = size
        .trim()
        .parse()
        .map_err(|_| format!("{size:?} is not an integer"))?;
    if size <= 0 {
        return Err(format!("dimension {name} must be positive, got {size}"));
    }
    Ok((name.trim().to_owned(), size))
}

impl DetectorArgs {
    /// Model directory, defaulting to the conventional location per detector.
    pub fn resolved_model_dir(&self) -> PathBuf {
        self.model_dir.clone().unwrap_or_else(|| {
            PathBuf::from(match self.detector {
                DetectorKind::Deterministic | DetectorKind::Gliner2Pii => "models/gliner2-pii-onnx",
                DetectorKind::Ai4privacyMdeberta => "models/ai4privacy-mdeberta",
                DetectorKind::MinilmL6Pii => "models/minilm-l6-pii",
                DetectorKind::DebertaXsmallPii => "models/deberta-xsmall-pii",
                DetectorKind::Gliner1PiiEdge => "models/gliner-pii-edge",
                DetectorKind::Gliner1PiiSmall => "models/gliner-pii-small",
            })
        })
    }
}

#[derive(clap::Args, Debug, Clone)]
pub struct EvaluateArgs {
    #[command(flatten)]
    pub detector: DetectorArgs,

    #[arg(long)]
    pub dataset: PathBuf,

    /// Override the dataset role inferred from the filename.
    #[arg(long, value_enum)]
    pub role: Option<RoleArg>,

    /// Show up to N false negatives and false positives.
    #[arg(long, default_value_t = 0)]
    pub show_errors: usize,

    /// Score only the entity types the detector's schema can produce.
    ///
    /// Answers "how good is it at what we asked for" rather than "is it enough
    /// for our policy". The report records which was used.
    #[arg(long)]
    pub restrict_to_supported: bool,

    /// Also evaluate seeded adversarial variants of every example.
    #[arg(long)]
    pub adversarial: bool,

    /// Write the full result as JSON.
    #[arg(long, short)]
    pub output: Option<PathBuf>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct SweepArgs {
    #[command(flatten)]
    pub detector: DetectorArgs,

    #[arg(long)]
    pub dataset: PathBuf,

    #[arg(long, value_enum)]
    pub role: Option<RoleArg>,

    #[arg(long, default_value_t = 0.10)]
    pub threshold_start: f32,
    #[arg(long, default_value_t = 0.95)]
    pub threshold_end: f32,
    #[arg(long, default_value_t = 0.05)]
    pub threshold_step: f32,

    #[arg(long, short)]
    pub output: Option<PathBuf>,

    /// Also write the sweep as CSV.
    pub csv: Option<PathBuf>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct PerfArgs {
    #[command(flatten)]
    pub detector: DetectorArgs,

    /// Input size buckets in approximate whitespace tokens.
    #[arg(long, value_delimiter = ',', default_values_t = [32usize, 128, 512, 1024])]
    pub buckets: Vec<usize>,

    #[arg(long, default_value_t = 30)]
    pub iterations: usize,

    #[arg(long, default_value_t = 3)]
    pub warmup: usize,

    /// Thread counts for the throughput measurement.
    ///
    /// Pass the flag with no values (`--thread-counts`) to skip throughput
    /// entirely. A latency-only run matters for the CoreML profiling cells:
    /// each throughput worker opens its own session, and each session with
    /// `--ort-profile` writes its own profile file.
    #[arg(long, value_delimiter = ',', num_args = 0.., default_values_t = [1usize, 2, 4, 8])]
    pub thread_counts: Vec<usize>,

    /// Documents per thread in the throughput measurement.
    #[arg(long, default_value_t = 20)]
    pub throughput_docs: usize,

    /// Also attribute latency to tokenize / inference / decode stages.
    ///
    /// Runs a second, instrumented pass after the headline measurement, so the
    /// reported percentiles above stay uninstrumented. Errors out on a detector
    /// that does not implement it rather than printing a table of zeros.
    #[arg(long)]
    pub stage_timing: bool,

    #[arg(long, short)]
    pub output: Option<PathBuf>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct InspectArgs {
    #[command(flatten)]
    pub detector: DetectorArgs,

    /// Describe a dataset instead of just the detector.
    #[arg(long)]
    pub dataset: Option<PathBuf>,

    /// Run the detector over these texts and print the spans it finds.
    #[arg(long)]
    pub text: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn evaluate_parses_the_documented_invocation() {
        let c = Cli::try_parse_from([
            "dlp-bench",
            "evaluate",
            "--detector",
            "gliner2-pii",
            "--model-dir",
            "models/gliner2-pii-onnx",
            "--dataset",
            "datasets/dlp-v1.jsonl",
            "--threshold",
            "0.35",
            "--show-errors",
            "100",
            "--output",
            "out/result.json",
        ])
        .unwrap();
        let Command::Evaluate(a) = c.command else {
            panic!("wrong subcommand")
        };
        assert_eq!(a.detector.detector, DetectorKind::Gliner2Pii);
        assert_eq!(a.detector.threshold, Some(0.35));
        assert_eq!(a.show_errors, 100);
        assert_eq!(a.output.unwrap().to_str().unwrap(), "out/result.json");
        assert_eq!(
            a.detector.resolved_model_dir().to_str().unwrap(),
            "models/gliner2-pii-onnx"
        );
        assert_eq!(c.seed, 42, "seed defaults to the documented value");
    }

    #[test]
    fn deterministic_evaluate_needs_no_model_dir() {
        let c = Cli::try_parse_from([
            "dlp-bench",
            "evaluate",
            "--detector",
            "deterministic",
            "--dataset",
            "datasets/dlp-v1.jsonl",
        ])
        .unwrap();
        let Command::Evaluate(a) = c.command else {
            panic!()
        };
        assert_eq!(a.detector.detector, DetectorKind::Deterministic);
        assert_eq!(
            a.detector.ip_policy,
            IpPolicy::Ignore,
            "policy default is explicit"
        );
    }

    #[test]
    fn sweep_defaults_match_the_documented_range() {
        let c = Cli::try_parse_from([
            "dlp-bench",
            "sweep",
            "--detector",
            "gliner2-pii",
            "--dataset",
            "datasets/dlp-dev.jsonl",
        ])
        .unwrap();
        let Command::Sweep(a) = c.command else {
            panic!()
        };
        assert_eq!(
            (a.threshold_start, a.threshold_end, a.threshold_step),
            (0.10, 0.95, 0.05)
        );
    }

    #[test]
    fn perf_buckets_and_thread_counts_are_comma_separated() {
        let c = Cli::try_parse_from([
            "dlp-bench",
            "perf",
            "--detector",
            "gliner2-pii",
            "--buckets",
            "32,128",
            "--thread-counts",
            "1,4",
        ])
        .unwrap();
        let Command::Perf(a) = c.command else {
            panic!()
        };
        assert_eq!(a.buckets, vec![32, 128]);
        assert_eq!(a.thread_counts, vec![1, 4]);
    }

    #[test]
    fn threshold_is_unset_by_default_so_each_model_keeps_its_own() {
        let c =
            Cli::try_parse_from(["dlp-bench", "perf", "--detector", "gliner1-pii-edge"]).unwrap();
        let Command::Perf(a) = c.command else {
            panic!()
        };
        assert_eq!(a.detector.threshold, None);
    }

    #[test]
    fn execution_provider_defaults_to_cpu_and_accepts_coreml() {
        let c = Cli::try_parse_from(["dlp-bench", "perf", "--detector", "gliner2-pii"]).unwrap();
        let Command::Perf(a) = c.command else {
            panic!()
        };
        assert_eq!(
            a.detector.ep,
            EpArg::Cpu,
            "CPU must stay the default baseline"
        );

        let c = Cli::try_parse_from([
            "dlp-bench",
            "perf",
            "--detector",
            "gliner2-pii",
            "--ep",
            "coreml",
        ])
        .unwrap();
        let Command::Perf(a) = c.command else {
            panic!()
        };
        assert_eq!(a.detector.ep, EpArg::Coreml);
    }

    #[test]
    fn each_detector_has_a_default_model_directory() {
        for (name, want) in [
            ("gliner2-pii", "models/gliner2-pii-onnx"),
            ("ai4privacy-mdeberta", "models/ai4privacy-mdeberta"),
            ("minilm-l6-pii", "models/minilm-l6-pii"),
            ("deberta-xsmall-pii", "models/deberta-xsmall-pii"),
            ("gliner1-pii-edge", "models/gliner-pii-edge"),
            ("gliner1-pii-small", "models/gliner-pii-small"),
        ] {
            let c = Cli::try_parse_from(["dlp-bench", "perf", "--detector", name]).unwrap();
            let Command::Perf(a) = c.command else {
                panic!()
            };
            assert_eq!(a.detector.resolved_model_dir().to_str().unwrap(), want);
        }
    }

    #[test]
    fn an_unknown_detector_is_rejected_with_the_valid_choices() {
        let err = Cli::try_parse_from([
            "dlp-bench",
            "evaluate",
            "--detector",
            "nope",
            "--dataset",
            "d.jsonl",
        ])
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("deterministic") && msg.contains("gliner2-pii"),
            "{msg}"
        );
    }
}
