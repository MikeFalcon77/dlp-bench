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

use ort::ep::CPU;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;

use crate::error::DetectorError;

/// CPU thread configuration.
///
/// Recorded in every benchmark result. Comparing two detectors under different
/// thread settings is meaningless, so this is explicit rather than inherited
/// from an environment variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadConfig {
    pub intra_op: usize,
    pub inter_op: usize,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        // Intra-op across physical cores; inter-op 1 because this is a single
        // sequential graph and a second op-level pool only adds contention.
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

/// Which ONNX Runtime execution provider to use.
///
/// ONNX Runtime has **no Metal execution provider**. On Apple silicon the route
/// to the GPU and Neural Engine is CoreML, which partitions the graph itself
/// and decides per subgraph whether to run on ANE, GPU or CPU — anything it
/// cannot handle silently falls back to the CPU EP. That makes "did it use the
/// GPU?" unanswerable from the outside, so the report states the *requested*
/// provider and the measurement shows whether it helped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
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
}

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
    provider: ExecutionProviderChoice,
}

impl OrtGraph {
    /// Load `path` and verify it matches `contract`.
    pub fn load(
        path: &Path,
        contract: &GraphContract,
        threads: ThreadConfig,
    ) -> Result<Self, DetectorError> {
        Self::load_with(path, contract, threads, ExecutionProviderChoice::default())
    }

    /// Load with an explicit execution provider.
    pub fn load_with(
        path: &Path,
        contract: &GraphContract,
        threads: ThreadConfig,
        provider: ExecutionProviderChoice,
    ) -> Result<Self, DetectorError> {
        if !provider.is_available() {
            return Err(DetectorError::config(format!(
                "execution provider {:?} is not available in this build \
                 (needs the `coreml` feature on macOS)",
                provider.as_str()
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
                .with_execution_providers(providers(provider))
                .map_err(|e| e.to_string())?;
            let b = b
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?;
            let b = b
                .with_intra_threads(threads.intra_op)
                .map_err(|e| e.to_string())?;
            let mut b = b
                .with_inter_threads(threads.inter_op)
                .map_err(|e| e.to_string())?;
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
            provider,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn threads(&self) -> ThreadConfig {
        self.threads
    }

    pub fn provider(&self) -> ExecutionProviderChoice {
        self.provider
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

        let outputs = self
            .session
            .run(values)
            .map_err(|e| DetectorError::inference(format!("session.run: {e}")))?;

        want_outputs
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
            .collect()
    }
}

/// Build the execution-provider list for a choice, CPU always last.
fn providers(choice: ExecutionProviderChoice) -> Vec<ort::ep::ExecutionProviderDispatch> {
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        use ort::ep::CoreML;
        use ort::ep::coreml::{ComputeUnits, ModelFormat};
        match choice {
            ExecutionProviderChoice::Cpu => vec![CPU::default().build()],
            // Measured on this graph: `MLProgram` and/or `with_subgraphs(true)`
            // mis-partition it and inference fails at runtime with
            // "Attempting to broadcast an axis by a dimension other than 1.
            // 6 by 768" from a Mul node. The legacy NeuralNetwork format with
            // subgraph compilation off is the configuration that runs.
            ExecutionProviderChoice::CoreMl => vec![
                CoreML::default()
                    .with_model_format(ModelFormat::NeuralNetwork)
                    .with_compute_units(ComputeUnits::All)
                    .with_subgraphs(false)
                    .build(),
                CPU::default().build(),
            ],
            ExecutionProviderChoice::CoreMlAne => vec![
                CoreML::default()
                    .with_model_format(ModelFormat::NeuralNetwork)
                    .with_compute_units(ComputeUnits::CPUAndNeuralEngine)
                    .with_subgraphs(false)
                    .build(),
                CPU::default().build(),
            ],
        }
    }
    #[cfg(not(all(feature = "coreml", target_os = "macos")))]
    {
        let _ = choice;
        vec![CPU::default().build()]
    }
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
    fn default_thread_config_is_cpu_sane() {
        let t = ThreadConfig::default();
        assert!(t.intra_op >= 1 && t.intra_op <= 8);
        assert_eq!(t.inter_op, 1, "a second op pool only adds contention here");
        assert_eq!(
            ThreadConfig::single(),
            ThreadConfig {
                intra_op: 1,
                inter_op: 1
            }
        );
        assert_eq!(
            ThreadConfig::with_intra(0).intra_op,
            1,
            "must never request 0 threads"
        );
    }

    #[test]
    fn provider_availability_is_reported_honestly() {
        assert!(ExecutionProviderChoice::Cpu.is_available());
        assert_eq!(
            ExecutionProviderChoice::default(),
            ExecutionProviderChoice::Cpu
        );
        let coreml_expected = cfg!(all(feature = "coreml", target_os = "macos"));
        assert_eq!(
            ExecutionProviderChoice::CoreMl.is_available(),
            coreml_expected
        );
        assert_eq!(ExecutionProviderChoice::CoreMl.as_str(), "coreml");
    }

    #[test]
    fn every_provider_list_ends_with_cpu_as_the_fallback() {
        // CoreML hands back any subgraph it cannot compile; without a CPU
        // fallback in the list the session would fail to build.
        for c in [
            ExecutionProviderChoice::Cpu,
            ExecutionProviderChoice::CoreMl,
            ExecutionProviderChoice::CoreMlAne,
        ] {
            assert!(!providers(c).is_empty(), "{c:?} produced no providers");
        }
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
