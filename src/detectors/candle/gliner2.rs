//! GLiNER2 PII on Candle, rebuilt from the fused ONNX export.
//!
//! The ONNX graph is not just an encoder: it carries GLiNER2's whole extraction
//! head. What it computes, read off the pinned export node by node:
//!
//! 1. `H = mdeberta-v3-base(input_ids)`, batch axis squeezed.
//! 2. `schema = H[schema_special_indices]`: row 0 is the `[P]` prompt token,
//!    rows `1..` are one `[E]` token per label. `words = H[text_word_indices]`.
//! 3. `count_logits = count_pred([P])` (Linear-ReLU-Linear, 20 classes), and a
//!    hard gate `argmax(count_logits) > 0` on everything below.
//! 4. Span representations, `SpanMarkerV0`: for word `i` and width `w < 8`,
//!    `out_project(relu(cat(project_start(words[i]), project_end(words[i+w]))))`.
//!    Spans running past the last word are rewritten to `(0, 0)` exactly as
//!    the export does, so their (ignored) scores match too.
//! 5. One step of the count GRU — the export was traced at count 1, so the
//!    input is `pos_embedding[0]` and the hidden state is the label embedding —
//!    then `projector(cat(h, labels))`.
//! 6. `span_probs = sigmoid(spans · structᵀ) * gate`, shaped `[words, 8, labels]`.
//!
//! The word-embedding table is stored int8 with per-row scales and the graph
//! dequantizes after `Gather`; this does the same, per call, for the rows it
//! needs. Dequantizing the whole 250k x 768 table at load would be numerically
//! identical and cost ~730 MB of RSS that ONNX Runtime does not pay.

use std::path::{Path, PathBuf};

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::deberta::{self, MDEBERTA_V3_BASE};
use super::onnx_weights::{self, Int8Table, OnnxWeights};
use super::{CANDLE_VERSION, Runtime};
use crate::detectors::onnx::TensorOut;
use crate::error::DetectorError;
use crate::runtime::{CandleDevice, RuntimeInfo, ThreadConfig};

/// The span-width axis the export was built with.
pub const MAX_WIDTH: usize = 8;
const HIDDEN: usize = 768;

const WORD_TABLE: &str = "encoder.embeddings.word_embeddings.weight_q8";
const WORD_SCALES: &str = "encoder.embeddings.word_embeddings.weight_row_scales";

/// `Linear -> ReLU -> Linear`, the shape of every head MLP in this model.
struct Mlp {
    a: Linear,
    b: Linear,
}

impl Mlp {
    fn load(
        vb: VarBuilder,
        input: usize,
        inner: usize,
        output: usize,
        second: &str,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            a: candle_nn::linear(input, inner, vb.pp("0"))?,
            b: candle_nn::linear(inner, output, vb.pp(second))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.b.forward(&self.a.forward(x)?.relu()?)
    }
}

pub struct Gliner2Candle {
    rt: Runtime,
    word_q8: Int8Table,
    word_scales: Vec<f32>,
    /// The whole of mdeberta's embedding stage once the word rows are given:
    /// no position embeddings (`position_biased_input: false`), no segment
    /// embeddings (`type_vocab_size: 0`), and an all-ones mask multiply.
    embed_norm: candle_nn::LayerNorm,
    encoder: deberta::Encoder,
    count_pred: Mlp,
    project_start: Mlp,
    project_end: Mlp,
    out_project: Mlp,
    gru_ih: Linear,
    gru_hh: Linear,
    /// `pos_embedding[0]`, `[1, 768]`. The only count step the export kept.
    count_pos0: Tensor,
    projector: Mlp,
    num_params: usize,
    path: PathBuf,
}

impl Gliner2Candle {
    pub fn load(
        path: &Path,
        device: CandleDevice,
        threads: ThreadConfig,
    ) -> Result<Self, DetectorError> {
        let rt = Runtime::new(device, threads)?;
        let mut w = onnx_weights::load(path, &rt.device)?;
        // Not needed on this path: the rows come from the int8 table instead.
        w.tensors
            .remove("encoder.embeddings.word_embeddings.weight");
        let mismatch = |detail: String| DetectorError::ArtifactMismatch {
            path: path.to_path_buf(),
            detail,
        };

        let word_q8 = w
            .int8
            .remove(WORD_TABLE)
            .ok_or_else(|| mismatch(format!("no int8 table {WORD_TABLE}")))?;
        let scales = w
            .tensors
            .remove(WORD_SCALES)
            .ok_or_else(|| mismatch(format!("no row scales {WORD_SCALES}")))?;
        let word_scales: Vec<f32> = scales
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .map_err(|e| mismatch(e.to_string()))?;
        if word_q8.dims != [word_scales.len(), HIDDEN] {
            return Err(mismatch(format!(
                "word table {:?} does not match {} row scales",
                word_q8.dims,
                word_scales.len()
            )));
        }
        let count_pos0 = count_pos0(&w).ok_or_else(|| {
            mismatch(
                "cannot find the count GRU's positional input (Expand -> Gemm gru.weight_ih_l0)"
                    .into(),
            )
        })?;

        let num_params =
            w.tensors.values().map(|t| t.elem_count()).sum::<usize>() + word_q8.data.len();
        let cfg = MDEBERTA_V3_BASE;
        let vb = VarBuilder::from_tensors(w.tensors, DType::F32, &rt.device);
        let load_err = |e: candle_core::Error| mismatch(format!("weights do not fit GLiNER2: {e}"));

        let h = HIDDEN;
        Ok(Self {
            embed_norm: candle_nn::layer_norm(
                HIDDEN,
                cfg.layer_norm_eps,
                vb.pp("encoder.embeddings.LayerNorm"),
            )
            .map_err(load_err)?,
            encoder: deberta::Encoder::load(vb.pp("encoder.encoder"), cfg).map_err(load_err)?,
            count_pred: Mlp::load(vb.pp("count_pred"), h, 2 * h, 20, "2").map_err(load_err)?,
            project_start: Mlp::load(vb.pp("span_marker.project_start"), h, 4 * h, h, "3")
                .map_err(load_err)?,
            project_end: Mlp::load(vb.pp("span_marker.project_end"), h, 4 * h, h, "3")
                .map_err(load_err)?,
            out_project: Mlp::load(vb.pp("span_marker.out_project"), 2 * h, 4 * h, h, "3")
                .map_err(load_err)?,
            gru_ih: linear_named(&vb, "gru", "weight_ih_l0", "bias_ih_l0", h, 3 * h)
                .map_err(load_err)?,
            gru_hh: linear_named(&vb, "gru", "weight_hh_l0", "bias_hh_l0", h, 3 * h)
                .map_err(load_err)?,
            count_pos0,
            projector: Mlp::load(vb.pp("projector"), 2 * h, 4 * h, h, "2").map_err(load_err)?,
            word_q8,
            word_scales,
            rt,
            num_params,
            path: path.to_path_buf(),
        })
    }

    /// `(span_probs [words, 8, labels], count_logits [1, 20])`, the same two
    /// tensors the ONNX graph returns.
    ///
    /// `input_copy` is the host-side part — dequantizing the embedding rows and
    /// uploading them — so the caller can attribute it the way it attributes
    /// ONNX Runtime's input copy.
    pub fn run(
        &self,
        input_ids: &[i64],
        text_word_indices: &[i64],
        schema_special_indices: &[i64],
        input_copy: &mut dyn FnMut(),
    ) -> Result<(TensorOut, TensorOut), DetectorError> {
        let embeds = self.embed(input_ids)?;
        let embeds = Tensor::from_vec(embeds, (input_ids.len(), HIDDEN), &self.rt.device)
            .map_err(|e| DetectorError::inference(format!("candle: {e}")))?;
        input_copy();

        let words = index_tensor(text_word_indices, input_ids.len(), &self.rt.device)?;
        let schema = index_tensor(schema_special_indices, input_ids.len(), &self.rt.device)?;
        let num_words = text_word_indices.len();
        let num_labels = schema_special_indices.len().saturating_sub(1);
        if num_words == 0 || num_labels == 0 {
            return Err(DetectorError::inference(
                "gliner2 needs at least one word and one label",
            ));
        }
        let (starts, ends) = span_endpoints(num_words);
        let starts = Tensor::new(starts, &self.rt.device)
            .map_err(|e| DetectorError::inference(e.to_string()))?;
        let ends = Tensor::new(ends, &self.rt.device)
            .map_err(|e| DetectorError::inference(e.to_string()))?;

        let (probs, count) = self.rt.install(|| {
            let x = self.embed_norm.forward(&embeds)?;
            let h = self.encoder.forward(&x)?;

            let schema = h.index_select(&schema, 0)?;
            let prompt = schema.narrow(0, 0, 1)?;
            let labels = schema.narrow(0, 1, num_labels)?;
            let words = h.index_select(&words, 0)?;

            let count_logits = self.count_pred.forward(&prompt)?;

            let start = self
                .project_start
                .forward(&words)?
                .index_select(&starts, 0)?;
            let end = self.project_end.forward(&words)?.index_select(&ends, 0)?;
            let spans = self
                .out_project
                .forward(&Tensor::cat(&[start, end], D::Minus1)?.relu()?)?;

            // GRU, one step: x = pos_embedding[0], h0 = the label embedding.
            let gi = self.gru_ih.forward(&self.count_pos0)?; // [1, 3H], same for every label
            let gh = self.gru_hh.forward(&labels)?; // [M, 3H]
            let gate = |t: &Tensor, i: usize| t.narrow(1, i * HIDDEN, HIDDEN);
            let r = candle_nn::ops::sigmoid(&gate(&gi, 0)?.broadcast_add(&gate(&gh, 0)?)?)?;
            let z = candle_nn::ops::sigmoid(&gate(&gi, 1)?.broadcast_add(&gate(&gh, 1)?)?)?;
            let n = gate(&gi, 2)?
                .broadcast_add(&(&r * gate(&gh, 2)?)?)?
                .tanh()?;
            let hidden = ((z.ones_like()? - &z)? * n)?.add(&(&z * &labels)?)?;
            let structure = self
                .projector
                .forward(&Tensor::cat(&[hidden, labels], D::Minus1)?)?;

            let probs = candle_nn::ops::sigmoid(&spans.matmul(&structure.t()?)?)?;
            // Reading back is also what synchronises a Metal command buffer, so
            // the timed region ends when the GPU has actually finished.
            Ok((
                probs.flatten_all()?.to_vec1::<f32>()?,
                count_logits.flatten_all()?.to_vec1::<f32>()?,
            ))
        })?;

        // `argmax(count_logits) > 0`, first index on ties as ONNX ArgMax does.
        let mut best = 0;
        for (i, v) in count.iter().enumerate() {
            if *v > count[best] {
                best = i;
            }
        }
        let gate = if best > 0 { 1.0 } else { 0.0 };
        let data = probs.into_iter().map(|p| p * gate).collect();

        Ok((
            TensorOut {
                shape: vec![num_words, MAX_WIDTH, num_labels],
                data,
            },
            TensorOut {
                shape: vec![1, count.len()],
                data: count,
            },
        ))
    }

    /// Dequantized embedding rows for `ids`: `int8 * row_scale`, in fp32, which
    /// is exactly the graph's `Cast` then `Mul`.
    fn embed(&self, ids: &[i64]) -> Result<Vec<f32>, DetectorError> {
        let mut out = Vec::with_capacity(ids.len() * HIDDEN);
        for &id in ids {
            let row = usize::try_from(id)
                .ok()
                .filter(|&r| r < self.word_scales.len())
                .ok_or_else(|| {
                    DetectorError::inference(format!("token id {id} outside the vocabulary"))
                })?;
            let scale = self.word_scales[row];
            let q = &self.word_q8.data[row * HIDDEN..(row + 1) * HIDDEN];
            out.extend(q.iter().map(|&v| f32::from(v) * scale));
        }
        Ok(out)
    }

    pub fn runtime_info(&self) -> RuntimeInfo {
        self.rt.info()
    }

    pub fn describe(&self) -> String {
        format!(
            "  candle {CANDLE_VERSION} on {}\n  weights: {:.1} M parameters (word table int8), read from {}",
            self.rt.kind().as_str(),
            self.num_params as f64 / 1e6,
            self.path.display()
        )
    }
}

/// `(start, end)` word index per `(word, width)` cell, row-major.
///
/// A span that would run past the last word becomes `(0, 0)`, matching the
/// export's `Where(end >= num_words, 0, ...)` on both endpoints.
fn span_endpoints(num_words: usize) -> (Vec<u32>, Vec<u32>) {
    let mut starts = Vec::with_capacity(num_words * MAX_WIDTH);
    let mut ends = Vec::with_capacity(num_words * MAX_WIDTH);
    for i in 0..num_words {
        for w in 0..MAX_WIDTH {
            let e = i + w;
            let (s, e) = if e >= num_words { (0, 0) } else { (i, e) };
            starts.push(s as u32);
            ends.push(e as u32);
        }
    }
    (starts, ends)
}

fn index_tensor(idx: &[i64], len: usize, device: &Device) -> Result<Tensor, DetectorError> {
    let idx: Vec<u32> = idx
        .iter()
        .map(|&i| {
            usize::try_from(i)
                .ok()
                .filter(|&i| i < len)
                .map(|i| i as u32)
                .ok_or_else(|| {
                    DetectorError::inference(format!("index {i} outside a {len}-token sequence"))
                })
        })
        .collect::<Result<_, _>>()?;
    Tensor::new(idx, device).map_err(|e| DetectorError::inference(format!("candle: {e}")))
}

/// A `Linear` whose parameters do not follow the `weight`/`bias` convention,
/// like `nn.GRU`'s `weight_ih_l0` / `bias_ih_l0`.
fn linear_named(
    vb: &VarBuilder,
    prefix: &str,
    weight: &str,
    bias: &str,
    input: usize,
    output: usize,
) -> candle_core::Result<Linear> {
    let vb = vb.pp(prefix);
    Ok(Linear::new(
        vb.get((output, input), weight)?,
        Some(vb.get(output, bias)?),
    ))
}

/// Find `pos_embedding[0]` by where it goes rather than by its export-assigned
/// name: it is the initializer `Expand`ed into the `Gemm` with
/// `gru.weight_ih_l0`.
fn count_pos0(w: &OnnxWeights) -> Option<Tensor> {
    let gemm = w.nodes.iter().find(|n| {
        n.op_type == "Gemm" && n.inputs.get(1).map(String::as_str) == Some("gru.weight_ih_l0")
    })?;
    let x = gemm.inputs.first()?;
    let expand = w
        .nodes
        .iter()
        .find(|n| n.op_type == "Expand" && n.outputs.first() == Some(x))?;
    let t = w.tensors.get(expand.inputs.first()?)?;
    (t.dims() == [1, HIDDEN]).then(|| t.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_past_the_last_word_collapse_to_the_first_word() {
        let (s, e) = span_endpoints(3);
        assert_eq!(s.len(), 3 * MAX_WIDTH);
        // word 0: widths 0..2 valid, the rest overflow.
        assert_eq!(&s[..4], &[0, 0, 0, 0]);
        assert_eq!(&e[..4], &[0, 1, 2, 0]);
        // word 2, width 0 is valid; width 1 overflows.
        let w2 = 2 * MAX_WIDTH;
        assert_eq!((s[w2], e[w2]), (2, 2));
        assert_eq!((s[w2 + 1], e[w2 + 1]), (0, 0));
    }
}
