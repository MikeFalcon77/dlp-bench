//! The ONNX Runtime layer.
//!
//! Every `ort` type in this crate is confined to this file. That is deliberate:
//! `ort` is a release-candidate whose API has churned across patch versions
//! (`Session::run` moved to `&mut self` in rc.10; `try_extract_*` was reshaped),
//! so an upgrade should touch one file. It also means a second ONNX detector —
//! GLiNER1, ModernBERT — reuses this layer rather than growing a parallel one,
//! which is what keeps the latency comparison fair: one runtime version, one
//! thread-pool configuration, one set of graph-optimisation settings.
//!
//! This layer knows nothing about GLiNER2. It loads a graph, asserts a declared
//! I/O contract, and runs `i64` inputs to `f32` outputs.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::ep::CPU;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::error::DetectorError;
use crate::stages::{Stage, StageTimer};

// The runtime configuration types live in `crate::runtime` so that `perf`,
// `report` and `detector` can name them without depending on `ort`. Re-exported
// here because this module is where callers have always found them.
pub use crate::runtime::{
    ComputePlanEntry, CoreMlComputeUnits, CoreMlModelFormat, CoreMlOptions, CoreMlSpecialization,
    DimOverride, EpNodeShare, ExecutionProviderChoice, OrtLogLevel, RuntimeInfo, SessionOptions,
    ThreadConfig,
};

/// What a caller expects the graph to look like.
///
/// Checked at load. Failing loudly here is the difference between "this model
/// is not the one you think" and silently producing plausible garbage from a
/// transposed tensor.
#[derive(Debug, Clone)]
pub struct GraphContract {
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
}

/// A loaded ONNX graph with a verified contract.
pub struct OrtGraph {
    session: Session,
    path: PathBuf,
    threads: ThreadConfig,
    options: SessionOptions,
    /// Lines ONNX Runtime logged while this session was built, when
    /// `capture_ort_log` was set. Behind a lock because the C callback fires
    /// from whichever thread the runtime happens to be on.
    log: Arc<Mutex<Vec<String>>>,
}

impl OrtGraph {
    /// Load `path` and verify it matches `contract`.
    pub fn load(
        path: &Path,
        contract: &GraphContract,
        threads: ThreadConfig,
    ) -> Result<Self, DetectorError> {
        Self::load_with_options(path, contract, threads, &SessionOptions::default())
    }

    /// Load with an explicit execution provider and otherwise default options.
    pub fn load_with(
        path: &Path,
        contract: &GraphContract,
        threads: ThreadConfig,
        provider: ExecutionProviderChoice,
    ) -> Result<Self, DetectorError> {
        Self::load_with_options(
            path,
            contract,
            threads,
            &SessionOptions::with_provider(provider),
        )
    }

    /// Load with the full session configuration.
    pub fn load_with_options(
        path: &Path,
        contract: &GraphContract,
        threads: ThreadConfig,
        options: &SessionOptions,
    ) -> Result<Self, DetectorError> {
        if !options.provider.is_available() {
            return Err(DetectorError::config(format!(
                "execution provider {:?} is not available in this build \
                 (needs the `coreml` feature on macOS)",
                options.provider.as_str()
            )));
        }
        if !path.is_file() {
            return Err(DetectorError::MissingArtifact {
                path: path.to_path_buf(),
                model_id: path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_owned(),
            });
        }

        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        // Each `with_*` returns `Result<SessionBuilder, Error<SessionBuilder>>`
        // — the error carries the builder back, so the error types differ per
        // step and the chain cannot be `and_then`ed. Collapse to `String` at
        // each step instead.
        //
        let session = (|| -> Result<Session, String> {
            let b = Session::builder().map_err(|e| e.to_string())?;
            // CPU is always appended last as the fallback: CoreML partitions
            // the graph and hands back whatever it cannot compile.
            let b = b
                .with_execution_providers(providers(options))
                .map_err(|e| e.to_string())?;
            let b = b
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?;
            let b = b
                .with_intra_threads(threads.intra_op)
                .map_err(|e| e.to_string())?;
            let b = b
                .with_inter_threads(threads.inter_op)
                .map_err(|e| e.to_string())?;

            // Fix free dimensions before commit; shape inference runs during
            // it, which is the whole point of setting them.
            let mut b = b;
            for d in &options.dimension_overrides {
                b = b
                    .with_dimension_override(&d.name, d.size)
                    .map_err(|e| format!("dimension override {}={}: {e}", d.name, d.size))?;
            }
            if options.disable_cpu_fallback {
                b = b.with_disable_cpu_fallback().map_err(|e| e.to_string())?;
            }
            if options.disable_intra_op_spinning {
                b = b.with_intra_op_spinning(false).map_err(|e| e.to_string())?;
            }
            if let Some(p) = &options.profile_json {
                b = b.with_profiling(p).map_err(|e| e.to_string())?;
            }
            if let Some(level) = options.log_level {
                b = b
                    .with_log_level(log_level(level))
                    .map_err(|e| e.to_string())?;
                b = b.with_log_verbosity(1).map_err(|e| e.to_string())?;
            }
            if options.capture_ort_log {
                // A Rust callback rather than stderr: ONNX Runtime's own logger
                // is gated by the `ORT_LOG` env var *and* by this crate's
                // `tracing` filter, so scraping the terminal means getting both
                // right at once. This gets the lines into the report instead.
                let sink = Arc::clone(&log);
                b = b
                    .with_logger(Arc::new(
                        move |_level: ort::logging::LogLevel,
                              category: &str,
                              _id: &str,
                              _loc: &str,
                              message: &str| {
                            if let Ok(mut v) = sink.lock() {
                                v.push(format!("{category}: {message}"));
                            }
                        },
                    ))
                    .map_err(|e| e.to_string())?;
            }

            b.commit_from_file(path).map_err(|e| e.to_string())
        })()
        .map_err(|e| {
            DetectorError::graph(format!(
                "failed to build session from {}: {e}",
                path.display()
            ))
        })?;

        let got_in: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        let got_out: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();

        if got_in != contract.inputs {
            return Err(DetectorError::graph(format!(
                "input names differ: expected {:?}, graph has {:?}",
                contract.inputs, got_in
            )));
        }
        if got_out != contract.outputs {
            return Err(DetectorError::graph(format!(
                "output names differ: expected {:?}, graph has {:?}",
                contract.outputs, got_out
            )));
        }

        tracing::info!(
            model = %path.display(),
            provider = options.provider.as_str(),
            intra_op = threads.intra_op,
            inter_op = threads.inter_op,
            inputs = ?got_in,
            outputs = ?got_out,
            "onnx session ready"
        );

        Ok(Self {
            session,
            path: path.to_path_buf(),
            threads,
            options: options.clone(),
            log,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn threads(&self) -> ThreadConfig {
        self.threads
    }

    pub fn provider(&self) -> ExecutionProviderChoice {
        self.options.provider
    }

    pub fn session_options(&self) -> &SessionOptions {
        &self.options
    }

    /// ONNX Runtime log lines captured during construction.
    pub fn captured_log(&self) -> Vec<String> {
        self.log.lock().map(|v| v.clone()).unwrap_or_default()
    }

    /// Flush the node profile and return the file ONNX Runtime wrote.
    ///
    /// `None` when profiling was not requested. The file is not readable until
    /// this is called — the runtime buffers records and only serialises them
    /// here — so a caller that forgets gets an empty or truncated profile.
    pub fn end_profiling(&mut self) -> Option<PathBuf> {
        self.options.profile_json.as_ref()?;
        match self.session.end_profiling() {
            Ok(p) => Some(PathBuf::from(p)),
            Err(e) => {
                tracing::warn!(error = %e, "end_profiling failed");
                None
            }
        }
    }

    /// Kernel time per execution provider, from ONNX Runtime's node profile.
    ///
    /// Call after [`end_profiling`](Self::end_profiling). This is the only
    /// direct answer to "how much of the graph did CoreML actually take": the
    /// EP list is a *request*, and the runtime hands back whatever it cannot
    /// place without saying so.
    pub fn ep_node_share(profile: &Path) -> Result<EpNodeShare, DetectorError> {
        let raw = std::fs::read_to_string(profile).map_err(|e| DetectorError::io(profile, e))?;
        let events: Vec<serde_json::Value> = serde_json::from_str(&raw).map_err(|e| {
            DetectorError::inference(format!("{}: not an ORT profile: {e}", profile.display()))
        })?;

        let mut share = EpNodeShare::default();
        for ev in &events {
            // Node-level records are the ones carrying a provider; session and
            // model-load records are not, and counting them would inflate the
            // denominator with graph compilation.
            if ev.get("cat").and_then(|c| c.as_str()) != Some("Node") {
                continue;
            }
            let Some(provider) = ev
                .get("args")
                .and_then(|a| a.get("provider"))
                .and_then(|p| p.as_str())
            else {
                continue;
            };
            let dur = ev.get("dur").and_then(|d| d.as_u64()).unwrap_or(0);
            *share.by_provider_us.entry(provider.to_owned()).or_default() += dur;
            *share.nodes.entry(provider.to_owned()).or_default() += 1;
            share.total_us += dur;
        }
        Ok(share)
    }

    /// The runtime configuration this graph ran under, for the report.
    ///
    /// `&mut` because it flushes the node profile: ONNX Runtime buffers those
    /// records and only writes them on `end_profiling`, so a caller who took
    /// `&self` would read an empty file and report a 0% CoreML share.
    ///
    /// Call once, after the measurement. A failure to read back the profile
    /// downgrades `ep_node_share` to `None` rather than failing the run — the
    /// latency numbers are still valid without it.
    pub fn runtime_info(&mut self) -> RuntimeInfo {
        let ep_node_share = self.end_profiling().and_then(|p| match Self::ep_node_share(&p) {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(profile = %p.display(), error = %e, "could not summarise ORT profile");
                None
            }
        });
        let log = self.captured_log();
        RuntimeInfo {
            provider: self.options.provider.as_str().to_owned(),
            intra_op_threads: self.threads.intra_op,
            inter_op_threads: self.threads.inter_op,
            graph_optimization: "level3".to_owned(),
            session: self.options.clone(),
            coreml_compute_plan: parse_compute_plan(&log),
            ort_log: log,
            ep_node_share,
        }
    }

    /// Human-readable I/O description, for `dlp-bench inspect`.
    pub fn describe(&self) -> String {
        let fmt = |v: &[ort::value::Outlet]| {
            v.iter()
                .map(|o| format!("{}: {:?}", o.name(), o.dtype()))
                .collect::<Vec<_>>()
                .join("\n    ")
        };
        format!(
            "  inputs:\n    {}\n  outputs:\n    {}",
            fmt(self.session.inputs()),
            fmt(self.session.outputs())
        )
    }

    /// Run the graph with `i64` inputs, returning `f32` outputs.
    ///
    /// Shapes are explicit per input because this graph mixes ranks:
    /// `input_ids`/`attention_mask` are `[1, seq_len]` (batch dim present)
    /// while `text_word_indices`/`schema_special_indices` are `[n]`. Passing
    /// rank-1 for the first two is silently accepted by some runtimes and
    /// produces nonsense, so the caller states the shape and we check it.
    ///
    /// `&mut self` because `Session::run` requires it from rc.10 onward. The
    /// detector wraps this in a `Mutex` to keep the `Detector` trait's `&self`
    /// signature; see `gliner2::mod` for why that trade is the right one.
    pub fn run_i64(
        &mut self,
        inputs: &[GraphInput<'_>],
        want_outputs: &[&'static str],
    ) -> Result<Vec<TensorOut>, DetectorError> {
        self.run_i64_timed(inputs, want_outputs, &mut StageTimer::disabled())
    }

    /// [`run_i64`](Self::run_i64), splitting the cost three ways.
    ///
    /// The split exists to make one specific claim falsifiable: that the
    /// `data.to_vec()` on the way in and the `to_vec()` on the way out are
    /// negligible next to the forward pass. They plausibly are — but
    /// `span_probs` is `[num_words, 8, num_labels]`, which at 1000 words is
    /// half a megabyte copied per call, and "plausibly negligible" is how
    /// people end up optimising the wrong thing.
    pub fn run_i64_timed(
        &mut self,
        inputs: &[GraphInput<'_>],
        want_outputs: &[&'static str],
        timer: &mut StageTimer,
    ) -> Result<Vec<TensorOut>, DetectorError> {
        let t_in = timer.start();
        let mut values: Vec<(&'static str, ort::value::DynValue)> =
            Vec::with_capacity(inputs.len());
        for GraphInput { name, shape, data } in inputs {
            let expect: usize = shape.iter().product();
            if expect != data.len() {
                return Err(DetectorError::inference(format!(
                    "input {name}: shape {shape:?} implies {expect} elements, got {}",
                    data.len()
                )));
            }
            let t = Tensor::from_array((shape.clone(), data.to_vec()))
                .map_err(|e| DetectorError::inference(format!("tensor {name}: {e}")))?;
            values.push((*name, t.into_dyn()));
        }
        timer.record(Stage::InputCopy, t_in);

        let t_run = timer.start();
        let outputs = self
            .session
            .run(values)
            .map_err(|e| DetectorError::inference(format!("session.run: {e}")))?;
        timer.record(Stage::SessionRun, t_run);

        let t_out = timer.start();
        let extracted = want_outputs
            .iter()
            .map(|name| {
                let v = outputs
                    .get(*name)
                    .ok_or_else(|| DetectorError::inference(format!("missing output {name}")))?;
                let (shape, data) = v.try_extract_tensor::<f32>().map_err(|e| {
                    DetectorError::inference(format!("output {name} is not f32: {e}"))
                })?;
                Ok(TensorOut {
                    shape: shape.iter().map(|d| *d as usize).collect(),
                    data: data.to_vec(),
                })
            })
            .collect();
        timer.record(Stage::OutputCopy, t_out);
        extracted
    }
}

/// Build the execution-provider list, CPU always last.
///
/// The CoreML arm applies every option unconditionally rather than only the
/// ones that differ from ONNX Runtime's own defaults: a recorded configuration
/// that omits a field because it "happened to match" is a configuration nobody
/// can reproduce after the runtime's defaults change.
///
/// The two were once suspected together and are now separated. Measured on the
/// GLiNER2 graph, M1 Pro, macOS 14.6, ort 2.0.0-rc.13:
///
/// * `MLProgram` **alone** fails, at `/Mul_10`, with "Attempting to broadcast an
///   axis by a dimension other than 1. 6 by 768". Note the error comes from
///   `providers/cpu/math/element_wise_ops.h` — the *CPU* kernel, on a tensor
///   CoreML handed back with the wrong shape. It is a partitioning bug, not an
///   unsupported operator.
/// * `with_subgraphs(true)` **alone** runs fine (232 ms p50 at 42 words, versus
///   139 ms on CPU). It was never the problem; it only ever governed CoreML
///   inside `Loop`/`Scan`/`If` bodies, which this graph does not have.
/// * Both together fail identically to `MLProgram` alone.
///
/// So `NeuralNetwork` is required, and the default stays there.
fn providers(options: &SessionOptions) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        use ort::ep::CoreML;
        use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};

        let Some(c) = options.resolved_coreml() else {
            return vec![CPU::default().build()];
        };

        let mut ep = CoreML::default()
            .with_model_format(match c.model_format {
                CoreMlModelFormat::NeuralNetwork => ModelFormat::NeuralNetwork,
                CoreMlModelFormat::MlProgram => ModelFormat::MLProgram,
            })
            .with_compute_units(match c.compute_units {
                CoreMlComputeUnits::All => ComputeUnits::All,
                CoreMlComputeUnits::CpuAndNeuralEngine => ComputeUnits::CPUAndNeuralEngine,
                CoreMlComputeUnits::CpuAndGpu => ComputeUnits::CPUAndGPU,
                CoreMlComputeUnits::CpuOnly => ComputeUnits::CPUOnly,
            })
            .with_subgraphs(c.subgraphs)
            .with_static_input_shapes(c.static_input_shapes)
            .with_specialization_strategy(match c.specialization {
                CoreMlSpecialization::Default => SpecializationStrategy::Default,
                CoreMlSpecialization::FastPrediction => SpecializationStrategy::FastPrediction,
            })
            .with_profile_compute_plan(c.profile_compute_plan)
            .with_low_precision_accumulation_on_gpu(c.low_precision_accumulation_on_gpu);
        if let Some(dir) = &c.model_cache_dir {
            ep = ep.with_model_cache_dir(dir.display().to_string());
        }

        // CPU last even when `disable_cpu_fallback` is set: that option acts on
        // the *session*, refusing to place leftovers, and the list is what the
        // runtime tries in order.
        vec![ep.build(), CPU::default().build()]
    }
    #[cfg(not(all(feature = "coreml", target_os = "macos")))]
    {
        let _ = options;
        vec![CPU::default().build()]
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn log_level(level: OrtLogLevel) -> ort::logging::LogLevel {
    use ort::logging::LogLevel as L;
    match level {
        OrtLogLevel::Error => L::Error,
        OrtLogLevel::Warning => L::Warning,
        OrtLogLevel::Info => L::Info,
        OrtLogLevel::Verbose => L::Verbose,
    }
}

#[cfg(not(all(feature = "coreml", target_os = "macos")))]
fn log_level(level: OrtLogLevel) -> ort::logging::LogLevel {
    use ort::logging::LogLevel as L;
    match level {
        OrtLogLevel::Error => L::Error,
        OrtLogLevel::Warning => L::Warning,
        OrtLogLevel::Info => L::Info,
        OrtLogLevel::Verbose => L::Verbose,
    }
}

/// Recover per-operator hardware assignments from CoreML's compute-plan log.
///
/// Lenient by construction: the format is an implementation detail of ONNX
/// Runtime's CoreML EP and carries no stability promise, so a parse miss must
/// degrade to "no entries" rather than to an error. The raw lines are kept in
/// [`RuntimeInfo::ort_log`] so a miss is recoverable by hand.
fn parse_compute_plan(lines: &[String]) -> Vec<ComputePlanEntry> {
    let mut out = Vec::new();
    for line in lines {
        // Shape: "... Operator Type: [op] ... Device Usage: [dev] ..."
        let Some(op) = field(line, "Operator Type:") else {
            continue;
        };
        let device = field(line, "Device Usage:")
            .or_else(|| field(line, "Compute Device:"))
            .unwrap_or_else(|| "Unknown".to_owned());
        out.push(ComputePlanEntry {
            operator: op,
            device,
            estimated_cost: field(line, "Estimated Cost:").and_then(|v| v.parse().ok()),
        });
    }
    out
}

fn field(line: &str, key: &str) -> Option<String> {
    let rest = line.split_once(key)?.1.trim_start();
    let value = rest
        .split([',', ';', '\n'])
        .next()?
        .trim()
        .trim_matches(|c| c == '[' || c == ']' || c == '"');
    (!value.is_empty()).then(|| value.to_owned())
}

/// One named input tensor with its explicit shape.
#[derive(Debug)]
pub struct GraphInput<'a> {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub data: &'a [i64],
}

impl<'a> GraphInput<'a> {
    /// `[1, n]` — a batched sequence input.
    pub fn batched(name: &'static str, data: &'a [i64]) -> Self {
        Self {
            name,
            shape: vec![1, data.len()],
            data,
        }
    }

    /// `[n]` — a flat index vector.
    pub fn flat(name: &'static str, data: &'a [i64]) -> Self {
        Self {
            name,
            shape: vec![data.len()],
            data,
        }
    }
}

/// An extracted `f32` output tensor.
#[derive(Debug, Clone)]
pub struct TensorOut {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorOut {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Index a rank-3 tensor. Panics on rank mismatch, which would be a bug in
    /// this crate rather than bad input — the shape is asserted at load.
    pub fn at3(&self, i: usize, j: usize, k: usize) -> f32 {
        debug_assert_eq!(self.shape.len(), 3);
        let (_, d1, d2) = (self.shape[0], self.shape[1], self.shape[2]);
        self.data[(i * d1 + j) * d2 + k]
    }

    /// Index of the maximum value, first-max wins (matching `torch.argmax`).
    pub fn argmax(&self) -> usize {
        let mut best = 0usize;
        for (i, v) in self.data.iter().enumerate() {
            if *v > self.data[best] {
                best = i;
            }
        }
        best
    }
}

/// The `ort` binding version, for run metadata.
///
/// Exact-pinned in `Cargo.toml`, and it in turn pins the ONNX Runtime build
/// fetched by `download-binaries` — so this one string identifies the whole
/// inference stack for reproducibility purposes.
pub const ORT_VERSION: &str = "2.0.0-rc.13";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provider_list_ends_with_cpu_as_the_fallback() {
        // CoreML hands back any subgraph it cannot compile; without a CPU
        // fallback in the list the session would fail to build.
        for c in [
            ExecutionProviderChoice::Cpu,
            ExecutionProviderChoice::CoreMl,
            ExecutionProviderChoice::CoreMlAne,
        ] {
            let list = providers(&SessionOptions::with_provider(c));
            assert!(!list.is_empty(), "{c:?} produced no providers");
            let want = if c.is_coreml() && cfg!(all(feature = "coreml", target_os = "macos")) {
                2
            } else {
                1
            };
            assert_eq!(list.len(), want, "{c:?} produced {} entries", list.len());
        }
    }

    #[test]
    fn coreml_defaults_reproduce_the_shipped_config() {
        // Every CoreML number in the README was measured under exactly this
        // configuration. If adding knobs changed what "no knobs" means, those
        // numbers would silently stop describing the current code.
        let shipped = SessionOptions::with_provider(ExecutionProviderChoice::CoreMl)
            .resolved_coreml()
            .expect("coreml provider must resolve options");
        assert_eq!(shipped.model_format, CoreMlModelFormat::NeuralNetwork);
        assert_eq!(shipped.compute_units, CoreMlComputeUnits::All);
        assert!(!shipped.subgraphs);
        assert!(!shipped.static_input_shapes);
        assert_eq!(shipped.specialization, CoreMlSpecialization::Default);
        assert!(!shipped.profile_compute_plan);
        assert!(shipped.model_cache_dir.is_none());

        let ane = SessionOptions::with_provider(ExecutionProviderChoice::CoreMlAne)
            .resolved_coreml()
            .unwrap();
        assert_eq!(ane.compute_units, CoreMlComputeUnits::CpuAndNeuralEngine);
        assert_eq!(
            CoreMlOptions {
                compute_units: CoreMlComputeUnits::CpuAndNeuralEngine,
                ..shipped
            },
            ane,
            "the ANE arm must differ from the All arm in exactly one field"
        );
    }

    #[test]
    fn ep_node_share_sums_kernel_time_per_provider() {
        // Shape taken from a real ONNX Runtime profile: an array of trace
        // events, only `cat == "Node"` ones carrying a provider. Session and
        // model-load records must not inflate the denominator.
        let profile = r#"[
          {"cat":"Session","name":"model_loading_uri","dur":1700000,"args":{}},
          {"cat":"Node","name":"a_kernel","dur":100,"args":{"provider":"CoreMLExecutionProvider"}},
          {"cat":"Node","name":"b_kernel","dur":300,"args":{"provider":"CPUExecutionProvider"}},
          {"cat":"Node","name":"c_fence","dur":0,"args":{}},
          {"cat":"Node","name":"d_kernel","dur":600,"args":{"provider":"CPUExecutionProvider"}}
        ]"#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prof.json");
        std::fs::write(&path, profile).unwrap();

        let s = OrtGraph::ep_node_share(&path).unwrap();
        assert_eq!(s.total_us, 1000, "session records must be excluded");
        assert_eq!(s.by_provider_us["CPUExecutionProvider"], 900);
        assert_eq!(s.by_provider_us["CoreMLExecutionProvider"], 100);
        assert_eq!(s.nodes["CPUExecutionProvider"], 2);
        assert!((s.accelerated_fraction() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn a_compute_plan_line_yields_an_operator_and_a_device() {
        let lines = vec![
            "CoreMLExecutionProvider: Operator Type: [conv], Device Usage: [ANE], Estimated Cost: 0.25".to_owned(),
            "CoreMLExecutionProvider: Operator Type: [matmul], Device Usage: [CPU]".to_owned(),
            "something else entirely".to_owned(),
        ];
        let plan = parse_compute_plan(&lines);
        assert_eq!(plan.len(), 2, "unrecognised lines are skipped, not fatal");
        assert_eq!(plan[0].operator, "conv");
        assert_eq!(plan[0].device, "ANE");
        assert_eq!(plan[0].estimated_cost, Some(0.25));
        assert_eq!(plan[1].device, "CPU");
        assert_eq!(plan[1].estimated_cost, None);

        // The format carries no stability promise, so a total miss must be
        // empty rather than an error — the raw lines are kept regardless.
        assert!(parse_compute_plan(&["nothing parseable".to_owned()]).is_empty());
    }

    #[test]
    fn missing_artifact_error_names_the_fetch_script() {
        let contract = GraphContract {
            inputs: &["a"],
            outputs: &["b"],
        };
        let msg = match OrtGraph::load(
            Path::new("models/does-not-exist/model.onnx"),
            &contract,
            ThreadConfig::single(),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a nonexistent model must not load"),
        };
        assert!(msg.contains("fetch-models.sh"), "unhelpful error: {msg}");
        assert!(msg.contains("does-not-exist"), "{msg}");
    }

    #[test]
    fn tensor_out_indexing_and_argmax() {
        // shape [2, 2, 3]
        let t = TensorOut {
            shape: vec![2, 2, 3],
            data: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        };
        assert_eq!(t.numel(), 12);
        assert_eq!(t.at3(0, 0, 0), 0.0);
        assert_eq!(t.at3(0, 1, 2), 5.0);
        assert_eq!(t.at3(1, 0, 0), 6.0);
        assert_eq!(t.at3(1, 1, 2), 11.0);
        assert_eq!(t.argmax(), 11);

        // First-max wins, matching torch.argmax.
        let ties = TensorOut {
            shape: vec![4],
            data: vec![1.0, 5.0, 5.0, 2.0],
        };
        assert_eq!(ties.argmax(), 1);
    }
}
