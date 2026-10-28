//! mdeberta-v3-base encoder, written for this workload.
//!
//! Replaces `candle_transformers::models::debertav2`, whose CPU profile on a
//! 512-token GLiNER2 prompt was 54% `where_cond` — `XSoftmax` masking twice
//! with an all-ones mask — 13% building and applying that mask, 10% gathers,
//! 7% single-threaded `erf`, and 2.6% matmul. The arithmetic is the same
//! (HF `modeling_deberta_v2`); what changes is where it happens:
//!
//! * **No masking.** Inputs are never padded on this path, so the mask is all
//!   ones and `XSoftmax` is plain softmax.
//! * **Relative-position projections at load.** `query_proj` and `key_proj`
//!   applied to the normalised `rel_embeddings` depend on no input
//!   (`share_att_key`), so each layer's 512 x 768 position keys and queries
//!   are computed once, with the `1/sqrt(3 * head_dim)` scale folded in.
//! * **Q, K and V in one matmul** per layer.
//! * **On CPU, one parallel pass** builds the disentangled scores
//!   (`QKᵀ + c2p[r(i,j)] + p2c[r(i,j)]`, both gathers and the p2c transpose)
//!   and applies softmax, and GELU runs in parallel too. Candle's own
//!   element-wise CPU kernels are single-threaded, and these two steps are
//!   where the time went.
//! * **On other devices** the same math runs on stock Candle ops (gather +
//!   fused `softmax_last_dim`), so Metal keeps working without custom kernels.
//!
//! `candle_reproduces_the_onnx_graph_outputs` in `detectors::gliner2` is the
//! correctness gate: every `span_probs` cell within 1e-3 of ONNX Runtime.

use candle_core::{CpuStorage, CustomOp1, CustomOp3, D, Device, Layout, Module, Shape, Tensor};
use candle_nn::{LayerNorm, Linear, VarBuilder};
use rayon::prelude::*;

/// The shape constants of mdeberta-v3-base as GLiNER2 ships it
/// (`encoder_config/config.json` of the upstream checkpoint).
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub layer_norm_eps: f64,
    /// `position_buckets`; relative positions index `2 * buckets` embeddings.
    pub position_buckets: usize,
    /// `max_relative_positions`, or `max_position_embeddings` when unset.
    pub max_relative_positions: usize,
}

pub const MDEBERTA_V3_BASE: Config = Config {
    hidden: 768,
    layers: 12,
    heads: 12,
    intermediate: 3072,
    layer_norm_eps: 1e-7,
    position_buckets: 256,
    max_relative_positions: 512,
};

struct Layer {
    /// `[query; key; value]` stacked, one matmul.
    qkv: Linear,
    /// `key_proj(rel)`, `[heads, 2 * buckets, head_dim]`.
    pos_key: Tensor,
    /// `query_proj(rel) / sqrt(3 * head_dim)`, same shape.
    pos_query: Tensor,
    attn_out: Linear,
    attn_norm: LayerNorm,
    intermediate: Linear,
    out: Linear,
    out_norm: LayerNorm,
}

pub struct Encoder {
    cfg: Config,
    layers: Vec<Layer>,
    device: Device,
}

impl Encoder {
    /// `vb` is rooted at the encoder (`…encoder.encoder`), holding `layer.N.*`,
    /// `rel_embeddings.weight` and `LayerNorm.*`.
    pub fn load(vb: VarBuilder, cfg: Config) -> candle_core::Result<Self> {
        let h = cfg.hidden;
        let head_dim = h / cfg.heads;
        let span = 2 * cfg.position_buckets;
        let scale = 1.0 / ((3 * head_dim) as f64).sqrt();

        let rel = vb.get((span, h), "rel_embeddings.weight")?;
        let rel =
            candle_nn::layer_norm(h, cfg.layer_norm_eps, vb.pp("LayerNorm"))?.forward(&rel)?;
        let per_head = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((span, cfg.heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };

        let layers = (0..cfg.layers)
            .map(|i| {
                let vb = vb.pp(format!("layer.{i}"));
                let sa = vb.pp("attention.self");
                let q = candle_nn::linear(h, h, sa.pp("query_proj"))?;
                let k = candle_nn::linear(h, h, sa.pp("key_proj"))?;
                let v = candle_nn::linear(h, h, sa.pp("value_proj"))?;
                let pos_key = per_head(k.forward(&rel)?)?;
                let pos_query = per_head((q.forward(&rel)? * scale)?)?;
                let bias = |l: &Linear| l.bias().cloned().expect("linear() always has a bias");
                let qkv = Linear::new(
                    Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?,
                    Some(Tensor::cat(&[bias(&q), bias(&k), bias(&v)], 0)?),
                );
                Ok(Layer {
                    qkv,
                    pos_key,
                    pos_query,
                    attn_out: candle_nn::linear(h, h, vb.pp("attention.output.dense"))?,
                    attn_norm: candle_nn::layer_norm(
                        h,
                        cfg.layer_norm_eps,
                        vb.pp("attention.output.LayerNorm"),
                    )?,
                    intermediate: candle_nn::linear(
                        h,
                        cfg.intermediate,
                        vb.pp("intermediate.dense"),
                    )?,
                    out: candle_nn::linear(cfg.intermediate, h, vb.pp("output.dense"))?,
                    out_norm: candle_nn::layer_norm(
                        h,
                        cfg.layer_norm_eps,
                        vb.pp("output.LayerNorm"),
                    )?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self {
            cfg,
            layers,
            device: vb.device().clone(),
        })
    }

    /// `x` is `[seq, hidden]`, already embedded and normalised; returns the
    /// same shape. Unpadded input only — there is no attention mask.
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (n, h) = x.dims2()?;
        let heads = self.cfg.heads;
        let head_dim = h / heads;
        let scale = 1.0 / ((3 * head_dim) as f64).sqrt();
        let rel = relative_index(
            n,
            self.cfg.position_buckets,
            self.cfg.max_relative_positions,
        );
        let attend = Attend::new(&rel, n, 2 * self.cfg.position_buckets, &self.device)?;

        let split = |t: Tensor| -> candle_core::Result<Tensor> {
            t.reshape((n, heads, head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };

        let mut x = x.clone();
        for l in &self.layers {
            let qkv = l.qkv.forward(&x)?;
            let q = (split(qkv.narrow(1, 0, h)?)? * scale)?;
            let k = split(qkv.narrow(1, h, h)?)?;
            let v = split(qkv.narrow(1, 2 * h, h)?)?;

            let content = q.matmul(&k.t()?)?; // [heads, n, n]
            let c2p = q.matmul(&l.pos_key.t()?)?; // [heads, n, span]
            let p2c = k.matmul(&l.pos_query.t()?)?; // [heads, n, span]
            let probs = attend.apply(&content, &c2p, &p2c)?;

            let ctx = probs.matmul(&v)?.transpose(0, 1)?.reshape((n, h))?;
            let a = l.attn_norm.forward(&(l.attn_out.forward(&ctx)? + &x)?)?;
            let m = gelu(&l.intermediate.forward(&a)?)?;
            x = l.out_norm.forward(&(l.out.forward(&m)? + a)?)?;
        }
        Ok(x)
    }
}

/// `r(i, j)` for every query `i` and key `j`, row-major: the index into the
/// `2 * buckets` relative embeddings, `clamp(bucket(i - j) + buckets)`.
///
/// Both the c2p and the p2c term use this one index — HF computes p2c from
/// `-bucket(j - i)`, and the bucketing is odd-symmetric.
fn relative_index(n: usize, buckets: usize, max_position: usize) -> Vec<u32> {
    let span = 2 * buckets as i64;
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n as i64 {
        for j in 0..n as i64 {
            let r = log_bucket(i - j, buckets as i64, max_position as i64) + buckets as i64;
            out.push(r.clamp(0, span - 1) as u32);
        }
    }
    out
}

/// HF `make_log_bucket_position`, in the same f32 arithmetic, so a bucket
/// boundary falls where the reference puts it.
fn log_bucket(rel: i64, buckets: i64, max_position: i64) -> i64 {
    let mid = buckets / 2;
    let abs = if rel < mid && rel > -mid {
        mid - 1
    } else {
        rel.abs()
    };
    if abs <= mid {
        return rel;
    }
    let log_pos = ((abs as f32 / mid as f32).ln()
        / ((max_position as f32 - 1.0) / mid as f32).ln()
        * (mid - 1) as f32)
        .ceil()
        + mid as f32;
    log_pos as i64 * rel.signum()
}

/// Disentangled attention scores plus softmax.
enum Attend {
    Cpu(DisentangledSoftmax),
    /// Index tensors for `gather`: `r(i,j)` and its transpose, `[1, n, n]`.
    Ops {
        c2p: Tensor,
        p2c: Tensor,
    },
}

impl Attend {
    fn new(rel: &[u32], n: usize, span: usize, device: &Device) -> candle_core::Result<Self> {
        if device.is_cpu() {
            return Ok(Attend::Cpu(DisentangledSoftmax {
                index: rel.to_vec(),
                n,
                span,
            }));
        }
        let mut transposed = vec![0u32; n * n];
        for i in 0..n {
            for j in 0..n {
                transposed[j * n + i] = rel[i * n + j];
            }
        }
        Ok(Attend::Ops {
            c2p: Tensor::from_slice(rel, (1, n, n), device)?,
            p2c: Tensor::from_vec(transposed, (1, n, n), device)?,
        })
    }

    fn apply(&self, content: &Tensor, c2p: &Tensor, p2c: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Attend::Cpu(op) => content.apply_op3_no_bwd(c2p, p2c, op),
            Attend::Ops { c2p: ic, p2c: ip } => {
                let shape = content.shape();
                let c2p = c2p.gather(&ic.broadcast_as(shape)?.contiguous()?, D::Minus1)?;
                // [h, j, i] = p2c[h, j, r(i, j)], then to [h, i, j].
                let p2c = p2c
                    .gather(&ip.broadcast_as(shape)?.contiguous()?, D::Minus1)?
                    .transpose(1, 2)?;
                candle_nn::ops::softmax_last_dim(&((content + c2p)? + p2c)?.contiguous()?)
            }
        }
    }
}

/// `softmax_j(content[h,i,j] + c2p[h,i,r(i,j)] + p2c[h,j,r(i,j)])`, one row per
/// rayon task.
struct DisentangledSoftmax {
    index: Vec<u32>,
    n: usize,
    span: usize,
}

fn contiguous<'a>(s: &'a CpuStorage, l: &Layout) -> candle_core::Result<&'a [f32]> {
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&s.as_slice::<f32>()?[a..b]),
        None => candle_core::bail!("disentangled attention expects contiguous input"),
    }
}

impl CustomOp3 for DisentangledSoftmax {
    fn name(&self) -> &'static str {
        "deberta-disentangled-softmax"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (n, span) = (self.n, self.span);
        let heads = l1.shape().dims()[0];
        if l1.shape().dims() != [heads, n, n]
            || l2.shape().dims() != [heads, n, span]
            || l3.shape().dims() != [heads, n, span]
        {
            candle_core::bail!(
                "disentangled attention shapes {:?} {:?} {:?} for n={n} span={span}",
                l1.shape(),
                l2.shape(),
                l3.shape()
            );
        }
        let (content, c2p, p2c) = (
            contiguous(s1, l1)?,
            contiguous(s2, l2)?,
            contiguous(s3, l3)?,
        );

        let mut out = vec![0f32; heads * n * n];
        out.par_chunks_mut(n).enumerate().for_each(|(row, dst)| {
            let (head, i) = (row / n, row % n);
            let content = &content[row * n..(row + 1) * n];
            let c2p = &c2p[row * span..(row + 1) * span];
            let p2c = &p2c[head * n * span..(head + 1) * n * span];
            let index = &self.index[i * n..(i + 1) * n];
            let mut max = f32::NEG_INFINITY;
            for j in 0..n {
                let r = index[j] as usize;
                let v = content[j] + c2p[r] + p2c[j * span + r];
                dst[j] = v;
                max = max.max(v);
            }
            let mut sum = 0f32;
            for v in dst.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            let inv = 1.0 / sum;
            for v in dst.iter_mut() {
                *v *= inv;
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((heads, n, n))))
    }
}

/// Exact (erf) GELU. Candle's `gelu_erf` runs single-threaded on CPU; this is
/// the same formula — `candle_core::cpu::erf::erf_f32`, as `gelu_erf` uses —
/// split across the current pool.
fn gelu(x: &Tensor) -> candle_core::Result<Tensor> {
    if x.device().is_cpu() {
        x.apply_op1_no_bwd(&ParallelGelu)
    } else {
        x.gelu_erf()
    }
}

struct ParallelGelu;

impl CustomOp1 for ParallelGelu {
    fn name(&self) -> &'static str {
        "parallel-gelu-erf"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let src = contiguous(s, l)?;
        let mut out = vec![0f32; src.len()];
        out.par_chunks_mut(4096)
            .zip(src.par_chunks(4096))
            .for_each(|(d, s)| {
                for (d, &v) in d.iter_mut().zip(s) {
                    *d = (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2)
                        + 1.0)
                        * 0.5
                        * v;
                }
            });
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HF's bucketing, checked at the points that matter: identity inside
    /// the window, symmetric, monotone outside, saturating at the span.
    #[test]
    fn log_buckets_match_the_reference_shape() {
        let (b, m) = (256, 512);
        for d in -128..=128 {
            assert_eq!(
                log_bucket(d, b, m),
                d,
                "inside the window positions are exact"
            );
        }
        for d in 129..600 {
            assert_eq!(
                log_bucket(-d, b, m),
                -log_bucket(d, b, m),
                "odd symmetry at {d}"
            );
            assert!(
                log_bucket(d, b, m) >= log_bucket(d - 1, b, m),
                "monotone at {d}"
            );
        }
        // max_position - 1 lands on the last bucket, mid + (mid - 1).
        assert_eq!(log_bucket(511, b, m), 255);
        let idx = relative_index(600, 256, 512);
        assert!(idx.iter().all(|&r| r < 512));
        assert_eq!(idx[0], 256, "r(0, 0) is the centre");
    }

    /// The fused CPU op against the stock-op path it replaces on GPUs.
    #[test]
    fn fused_cpu_attention_matches_the_gather_path() {
        let (heads, n, span) = (3, 7, 16);
        let dev = Device::Cpu;
        let content = Tensor::randn(0f32, 1.0, (heads, n, n), &dev).unwrap();
        let c2p = Tensor::randn(0f32, 1.0, (heads, n, span), &dev).unwrap();
        let p2c = Tensor::randn(0f32, 1.0, (heads, n, span), &dev).unwrap();
        let rel: Vec<u32> = (0..n * n)
            .map(|k| ((k / n) as i64 - (k % n) as i64 + 8).clamp(0, 15) as u32)
            .collect();

        let fused = Attend::new(&rel, n, span, &dev).unwrap();
        let mut transposed = vec![0u32; n * n];
        for i in 0..n {
            for j in 0..n {
                transposed[j * n + i] = rel[i * n + j];
            }
        }
        let ops = Attend::Ops {
            c2p: Tensor::from_slice(&rel, (1, n, n), &dev).unwrap(),
            p2c: Tensor::from_vec(transposed, (1, n, n), &dev).unwrap(),
        };
        let a = fused
            .apply(&content, &c2p, &p2c)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = ops
            .apply(&content, &c2p, &p2c)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(diff < 1e-6, "fused and gather paths differ by {diff}");
    }

    #[test]
    fn parallel_gelu_is_candles_gelu_erf() {
        let x = Tensor::randn(0f32, 3.0, (5, 9001), &Device::Cpu).unwrap();
        let a = gelu(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = x
            .gelu_erf()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(a, b, "same formula, same erf: bit-identical");
    }
}
