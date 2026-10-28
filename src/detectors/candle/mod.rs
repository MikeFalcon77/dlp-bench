//! The Candle layer.
//!
//! Every `candle_*` type in this crate is confined to this module, for the same
//! reason [`super::onnx`] confines `ort`: a version bump should touch one place.
//!
//! What Candle adds to the benchmark is a **second runtime over identical
//! weights** for `gliner2-pii`. The network is rebuilt from the very ONNX file
//! the ONNX Runtime path uses ([`onnx_weights`]), so tokenisation, the prompt,
//! decoding and the weights are all shared, and the only thing that varies is
//! the implementation of the forward pass — eager Candle kernels on CPU or
//! Metal against ONNX Runtime's optimised graph. Metal matters because ONNX
//! Runtime has no Metal execution provider; CoreML is its only route to the
//! Apple GPU, and the README records why that route loses.

pub mod deberta;
pub mod gliner2;
pub mod onnx_weights;

use candle_core::Device;

use crate::error::DetectorError;
use crate::runtime::{CandleDevice, RuntimeInfo, SessionOptions, ThreadConfig};

/// Pinned in `Cargo.toml`; recorded so a result names the runtime version the
/// same way an ONNX result names `ort`'s.
pub(crate) const CANDLE_VERSION: &str = "0.11.0";

/// Device, thread pool and the report entry that describes them.
///
/// Shared by every Candle model so they are configured — and recorded —
/// identically, the way [`super::onnx`] does it for ONNX Runtime sessions.
pub(crate) struct Runtime {
    pub(crate) device: Device,
    kind: CandleDevice,
    /// Private pool so `threads` bounds this model alone, as ONNX Runtime's
    /// intra-op setting does. Candle's CPU kernels parallelise on whichever
    /// rayon pool is current; without this every model would share one
    /// process-wide pool and throughput workers would oversubscribe it.
    pool: Option<rayon::ThreadPool>,
    threads: ThreadConfig,
}

impl Runtime {
    pub(crate) fn new(kind: CandleDevice, threads: ThreadConfig) -> Result<Self, DetectorError> {
        if !kind.is_available() {
            return Err(DetectorError::config(format!(
                "candle device {} is not compiled into this binary; rebuild with \
                 --features {}",
                kind.as_str(),
                match kind {
                    CandleDevice::Cpu => "candle",
                    CandleDevice::Metal => "metal",
                }
            )));
        }
        let (device, pool) = match kind {
            CandleDevice::Cpu => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads.intra_op)
                    .build()
                    .map_err(|e| DetectorError::config(format!("candle thread pool: {e}")))?;
                (Device::Cpu, Some(pool))
            }
            CandleDevice::Metal => {
                let device = Device::new_metal(0)
                    .map_err(|e| DetectorError::config(format!("no Metal device: {e}")))?;
                (device, None)
            }
        };
        Ok(Self {
            device,
            kind,
            pool,
            threads,
        })
    }

    /// Run `f` on this model's pool, mapping Candle errors to inference errors.
    pub(crate) fn install<T: Send>(
        &self,
        f: impl FnOnce() -> candle_core::Result<T> + Send,
    ) -> Result<T, DetectorError> {
        let r = match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        };
        r.map_err(|e| DetectorError::inference(format!("candle: {e}")))
    }

    pub(crate) fn kind(&self) -> CandleDevice {
        self.kind
    }

    pub(crate) fn info(&self) -> RuntimeInfo {
        let provider = match self.kind {
            CandleDevice::Cpu if cfg!(feature = "accelerate") => "candle-cpu-accelerate",
            CandleDevice::Cpu => "candle-cpu",
            CandleDevice::Metal => "candle-metal",
        };
        RuntimeInfo {
            provider: provider.to_owned(),
            intra_op_threads: if self.pool.is_some() {
                self.threads.intra_op
            } else {
                0
            },
            inter_op_threads: 1,
            graph_optimization: format!(
                "eager (candle {CANDLE_VERSION}, gemm split {})",
                candle_core::utils::get_num_threads()
            ),
            session: SessionOptions::default(),
            coreml_compute_plan: Vec::new(),
            ort_log: Vec::new(),
            ep_node_share: None,
        }
    }
}
