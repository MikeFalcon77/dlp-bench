//! How the inference runtime was configured, as plain data.
//!
//! Two jobs, and they pull in the same direction.
//!
//! **Reproducibility.** Until now no `out/*.json` recorded the execution
//! provider, the thread count or any CoreML setting — the only trace was the
//! command line captured in [`RunMeta`](crate::runmeta::RunMeta). That is fine
//! until someone compares a CoreML row against a CPU row taken on a different
//! day with a different flag, which is exactly what the provider comparison in
//! the README is. A benchmark that pins its model by sha256 and its runtime by
//! exact version should not leave *how the session was built* to prose.
//!
//! **Experiment surface.** The CoreML EP has eight knobs and the code used to
//! hardcode one combination of them. That combination was arrived at by
//! changing `ModelFormat` and `EnableOnSubgraphs` together when the graph broke,
//! which means neither was ever tested alone. Making every knob independently
//! settable is what turns "CoreML is slower" into a statement about a specific
//! configuration rather than about CoreML.
//!
//! Nothing here imports `ort`. The types are consumed by `perf`, `report` and
//! `detector`, none of which should learn about ONNX Runtime, and they must
//! compile under `--no-default-features` and off macOS. Translation to `ort`
//! calls happens in exactly one place,
//! [`detectors::onnx`](crate::detectors::onnx).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// CPU thread configuration.
///
/// Recorded in every benchmark result. Comparing two detectors under different
/// thread settings is meaningless, so this is explicit rather than inherited
/// from an environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadConfig {
    pub intra_op: usize,
    pub inter_op: usize,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        // Intra-op across physical cores; inter-op 1 because this is a single
        // sequential graph and a second op-level pool only adds contention.
        //
        // On the M1 Pro this box measures on, `available_parallelism()` is 10
        // (8 performance + 2 efficiency), so the cap lands on exactly the
        // performance-core count. That is a coincidence rather than a design,
        // and `scripts/perf-matrix.sh` sweeps it rather than trusting it.
        Self {
            intra_op: std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
            inter_op: 1,
        }
    }
}

impl ThreadConfig {
    pub fn single() -> Self {
        Self {
            intra_op: 1,
            inter_op: 1,
        }
    }

    pub fn with_intra(intra_op: usize) -> Self {
        Self {
            intra_op: intra_op.max(1),
            inter_op: 1,
        }
    }
}

/// Which inference library runs a token classifier.
///
/// Orthogonal to [`ExecutionProviderChoice`], which only means anything under
/// ONNX Runtime. Candle reads the same ONNX weight file, so switching backend
/// changes the runtime and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    #[default]
    Ort,
    Candle(CandleDevice),
}

/// Where Candle runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandleDevice {
    #[default]
    Cpu,
    /// Apple GPU. The accelerator ONNX Runtime cannot reach directly.
    Metal,
}

impl CandleDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            CandleDevice::Cpu => "cpu",
            CandleDevice::Metal => "metal",
        }
    }

    /// Whether this build can actually honour the request.
    pub fn is_available(self) -> bool {
        match self {
            CandleDevice::Cpu => cfg!(feature = "candle"),
            CandleDevice::Metal => cfg!(all(feature = "metal", target_os = "macos")),
        }
    }
}

/// Which ONNX Runtime execution provider to use.
///
/// ONNX Runtime has **no Metal execution provider**. On Apple silicon the route
/// to the GPU and Neural Engine is CoreML, which partitions the graph itself
/// and decides per subgraph whether to run on ANE, GPU or CPU — anything it
/// cannot handle silently falls back to the CPU EP. That used to make "did it
/// use the GPU?" unanswerable from the outside; [`EpNodeShare`] now answers it
/// from ONNX Runtime's own node profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProviderChoice {
    /// CPU only. The Phase-1 baseline and the default.
    #[default]
    Cpu,
    /// CoreML, falling back to CPU for unsupported subgraphs.
    CoreMl,
    /// CoreML restricted to CPU + Neural Engine (no GPU).
    CoreMlAne,
}

impl ExecutionProviderChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            ExecutionProviderChoice::Cpu => "cpu",
            ExecutionProviderChoice::CoreMl => "coreml",
            ExecutionProviderChoice::CoreMlAne => "coreml-ane",
        }
    }

    /// Whether this build can actually honour the request.
    pub fn is_available(self) -> bool {
        match self {
            ExecutionProviderChoice::Cpu => true,
            ExecutionProviderChoice::CoreMl | ExecutionProviderChoice::CoreMlAne => {
                cfg!(all(feature = "coreml", target_os = "macos"))
            }
        }
    }

    pub fn is_coreml(self) -> bool {
        !matches!(self, ExecutionProviderChoice::Cpu)
    }

    /// The compute units this choice implies when the caller sets none.
    fn implied_compute_units(self) -> CoreMlComputeUnits {
        match self {
            ExecutionProviderChoice::CoreMlAne => CoreMlComputeUnits::CpuAndNeuralEngine,
            _ => CoreMlComputeUnits::All,
        }
    }
}

/// The CoreML model representation the EP compiles to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlModelFormat {
    /// CoreML 3 (macOS 10.15+). The default, and what currently runs.
    #[default]
    NeuralNetwork,
    /// CoreML 5 (macOS 12+). Supports more operators and is the only format
    /// Apple's `MLComputePlan` can introspect, so
    /// [`CoreMlOptions::profile_compute_plan`] is useless without it.
    MlProgram,
}

impl CoreMlModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            CoreMlModelFormat::NeuralNetwork => "neural-network",
            CoreMlModelFormat::MlProgram => "ml-program",
        }
    }
}

/// Hardware CoreML is allowed to schedule onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlComputeUnits {
    #[default]
    All,
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

impl CoreMlComputeUnits {
    pub fn as_str(self) -> &'static str {
        match self {
            CoreMlComputeUnits::All => "all",
            CoreMlComputeUnits::CpuAndNeuralEngine => "cpu-ane",
            CoreMlComputeUnits::CpuAndGpu => "cpu-gpu",
            CoreMlComputeUnits::CpuOnly => "cpu-only",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlSpecialization {
    #[default]
    Default,
    /// Trade session-load time and memory for prediction latency.
    FastPrediction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrtLogLevel {
    #[default]
    Error,
    Warning,
    Info,
    Verbose,
}

/// Every CoreML EP knob, independently settable.
///
/// [`Default`] is exactly the configuration that shipped before this struct
/// existed: legacy `NeuralNetwork`, all compute units, subgraph compilation off.
/// That is load-bearing — see `coreml_defaults_reproduce_the_shipped_config` in
/// [`detectors::onnx`](crate::detectors::onnx) — because every CoreML number
/// already in the README was taken under it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreMlOptions {
    pub model_format: CoreMlModelFormat,
    pub compute_units: CoreMlComputeUnits,
    /// Compile CoreML subgraphs inside the body of a `Loop`/`Scan`/`If`.
    ///
    /// Commonly misread as "let CoreML partition the main graph", which it is
    /// not — main-graph partitioning happens through `GetCapability` and is not
    /// affected by this flag at all.
    pub subgraphs: bool,
    /// Only let CoreML claim nodes whose inputs already have static shapes.
    ///
    /// Restrictive, not enabling: on its own it makes CoreML take *fewer*
    /// nodes. It is only meaningful together with
    /// [`SessionOptions::dimension_overrides`].
    pub static_input_shapes: bool,
    pub specialization: CoreMlSpecialization,
    /// Log the hardware each operator was dispatched to.
    ///
    /// Drives Apple's `MLComputePlan` (macOS 14.4+), which enumerates
    /// **MLProgram** operations — under `NeuralNetwork` it produces nothing.
    pub profile_compute_plan: bool,
    pub low_precision_accumulation_on_gpu: bool,
    /// Reuse the compiled CoreML model across session loads.
    ///
    /// Without this the 426 MB graph is recompiled on every process start,
    /// which is most of the measured 3.9 s CoreML load penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_cache_dir: Option<PathBuf>,
}

/// Pin one of the graph's free dimensions to a fixed size.
///
/// Applies to graph *inputs* only (ONNX Runtime's
/// `AddFreeDimensionOverrideByName`). Computed output dims such as GLiNER2's
/// `Mul_span_probs_dim_1` are not free dimensions and cannot be set this way.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimOverride {
    pub name: String,
    pub size: i64,
}

/// Everything about the session that is not the model file itself.
///
/// `Default` is the CPU baseline, so an existing caller that constructs one and
/// sets nothing gets the behaviour it always had.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionOptions {
    pub provider: ExecutionProviderChoice,
    /// CoreML settings. `None` means "the caller expressed no opinion", which
    /// resolves to the defaults implied by `provider` — the distinction is kept
    /// so the recorded config can say which it was.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coreml: Option<CoreMlOptions>,
    /// Refuse to place unsupported nodes on the CPU EP.
    ///
    /// A diagnostic, not an optimisation: the session fails to commit unless
    /// the graph lands entirely on the accelerator, and the failure message
    /// names the nodes that could not. A build failure here is a *result*.
    pub disable_cpu_fallback: bool,
    /// Turn off ONNX Runtime's intra-op spin-wait.
    ///
    /// Spinning burns a core to shave wake-up latency, which is the wrong trade
    /// on the two efficiency cores of an Apple silicon part.
    pub disable_intra_op_spinning: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dimension_overrides: Vec<DimOverride>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_level: Option<OrtLogLevel>,
    /// Write ONNX Runtime's per-node profile here. Feeds [`EpNodeShare`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_json: Option<PathBuf>,
    /// Capture ONNX Runtime's log lines into the report instead of stderr.
    pub capture_ort_log: bool,
}

impl SessionOptions {
    pub fn cpu() -> Self {
        Self::default()
    }

    pub fn with_provider(provider: ExecutionProviderChoice) -> Self {
        Self {
            provider,
            ..Self::default()
        }
    }

    /// CoreML settings to apply, filling in whatever the caller left unsaid.
    ///
    /// Returns `None` for a CPU-only session. This is the single place the
    /// `provider`-implies-compute-units rule lives, so `--ep coreml-ane` and
    /// `--ep coreml --coreml-compute-units cpu-ane` cannot drift apart.
    pub fn resolved_coreml(&self) -> Option<CoreMlOptions> {
        if !self.provider.is_coreml() {
            return None;
        }
        Some(self.coreml.clone().unwrap_or(CoreMlOptions {
            compute_units: self.provider.implied_compute_units(),
            ..CoreMlOptions::default()
        }))
    }

    pub fn dimension_override(&self, name: &str) -> Option<i64> {
        self.dimension_overrides
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.size)
    }
}

/// One operator's hardware assignment, as CoreML reported it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputePlanEntry {
    pub operator: String,
    /// `"ANE"`, `"GPU"`, `"CPU"` or `"Unknown"`, verbatim from CoreML.
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
}

/// How ONNX Runtime's kernel time divided across execution providers.
///
/// The number that decides the CoreML question. A CoreML session that hands
/// most of the graph back to the CPU EP is not a measurement of CoreML; it is a
/// measurement of CPU inference plus partition overhead, and no amount of
/// tuning on the CoreML side will move it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EpNodeShare {
    /// Microseconds of kernel time per provider name, e.g.
    /// `{"CoreMLExecutionProvider": 12000, "CPUExecutionProvider": 340000}`.
    pub by_provider_us: BTreeMap<String, u64>,
    /// Node executions counted per provider. A provider can own few nodes and
    /// most of the time, or the reverse; both are worth seeing.
    pub nodes: BTreeMap<String, usize>,
    pub total_us: u64,
}

impl EpNodeShare {
    /// Fraction of kernel time on providers other than the CPU EP.
    pub fn accelerated_fraction(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        let cpu = self
            .by_provider_us
            .get("CPUExecutionProvider")
            .copied()
            .unwrap_or(0);
        (self.total_us - cpu.min(self.total_us)) as f64 / self.total_us as f64
    }
}

/// The full runtime configuration a result was produced under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub provider: String,
    pub intra_op_threads: usize,
    pub inter_op_threads: usize,
    pub graph_optimization: String,
    pub session: SessionOptions,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coreml_compute_plan: Vec<ComputePlanEntry>,
    /// Raw ONNX Runtime log lines, when capture was on. Kept alongside the
    /// parsed plan so a parser miss is recoverable rather than silent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ort_log: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ep_node_share: Option<EpNodeShare>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_config_is_cpu_sane() {
        let t = ThreadConfig::default();
        assert!(t.intra_op >= 1 && t.intra_op <= 8);
        assert_eq!(t.inter_op, 1, "a second op pool only adds contention here");
        assert_eq!(ThreadConfig::single(), ThreadConfig::with_intra(1));
        assert_eq!(
            ThreadConfig::with_intra(0).intra_op,
            1,
            "must never request 0 threads"
        );
    }

    #[test]
    fn provider_availability_is_reported_honestly() {
        assert!(ExecutionProviderChoice::Cpu.is_available());
        assert!(!ExecutionProviderChoice::Cpu.is_coreml());
        assert_eq!(
            ExecutionProviderChoice::default(),
            ExecutionProviderChoice::Cpu
        );
        let expected = cfg!(all(feature = "coreml", target_os = "macos"));
        assert_eq!(ExecutionProviderChoice::CoreMl.is_available(), expected);
        assert_eq!(ExecutionProviderChoice::CoreMl.as_str(), "coreml");
    }

    #[test]
    fn an_unspecified_coreml_config_resolves_from_the_provider() {
        // The two spellings of "CoreML on CPU+ANE" must agree, or `--ep
        // coreml-ane` and `--coreml-compute-units cpu-ane` silently measure
        // different things.
        let implied = SessionOptions::with_provider(ExecutionProviderChoice::CoreMlAne)
            .resolved_coreml()
            .unwrap();
        assert_eq!(
            implied.compute_units,
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(implied.model_format, CoreMlModelFormat::NeuralNetwork);
        assert!(!implied.subgraphs, "subgraphs off is the shipped config");

        let explicit = SessionOptions {
            provider: ExecutionProviderChoice::CoreMl,
            coreml: Some(CoreMlOptions {
                compute_units: CoreMlComputeUnits::CpuAndNeuralEngine,
                ..Default::default()
            }),
            ..Default::default()
        }
        .resolved_coreml()
        .unwrap();
        assert_eq!(implied, explicit);

        assert!(
            SessionOptions::cpu().resolved_coreml().is_none(),
            "a CPU session must not resolve CoreML options"
        );
    }

    #[test]
    fn a_default_session_serializes_to_almost_nothing() {
        // Every added field is `skip_serializing_if`, so a default CPU run does
        // not grow the JSON that existing collators read.
        let json = serde_json::to_string(&SessionOptions::default()).unwrap();
        assert_eq!(
            json,
            r#"{"provider":"cpu","disable_cpu_fallback":false,"disable_intra_op_spinning":false,"capture_ort_log":false}"#
        );
        let back: SessionOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SessionOptions::default());
        // And an empty object still deserializes, so older files load.
        let empty: SessionOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, SessionOptions::default());
    }

    #[test]
    fn ep_node_share_reports_the_accelerated_fraction() {
        let mut s = EpNodeShare {
            total_us: 1000,
            ..Default::default()
        };
        s.by_provider_us.insert("CPUExecutionProvider".into(), 800);
        s.by_provider_us
            .insert("CoreMLExecutionProvider".into(), 200);
        assert!((s.accelerated_fraction() - 0.2).abs() < 1e-9);

        // The degenerate cases must not panic or produce a fraction > 1.
        assert_eq!(EpNodeShare::default().accelerated_fraction(), 0.0);
        let mut all_cpu = EpNodeShare {
            total_us: 500,
            ..Default::default()
        };
        all_cpu
            .by_provider_us
            .insert("CPUExecutionProvider".into(), 900);
        assert_eq!(all_cpu.accelerated_fraction(), 0.0, "must clamp, not wrap");
    }
}
