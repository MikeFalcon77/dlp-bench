//! Read a transformer's weights out of its ONNX export.
//!
//! The Candle backend runs the **same weight file** the ONNX Runtime detector
//! runs — same path, same pinned sha256 — so a quality or latency difference
//! between the two is a property of the runtime, not of a second artifact that
//! might have been exported from a different checkpoint. MiniLM-L6 publishes no
//! safetensors at all, so for it this is also the only option.
//!
//! An `optimum`/`torch.onnx` export keeps most parameter names verbatim
//! (`bert.embeddings.word_embeddings.weight`, every bias, every LayerNorm), but
//! renames each `nn.Linear` weight to an anonymous `onnx::MatMul_NNN` stored
//! **transposed** (`[in, out]`). The module path survives in the name of the
//! `MatMul` node that consumes it — `/bert/encoder/layer.0/attention/self/query/MatMul`
//! — so that node name is used to recover `…query.weight` and the tensor is
//! transposed back to PyTorch's `[out, in]`.
//!
//! Weight-compressed exports are read the way the graph itself reads them:
//! fp16 weights behind a `Cast` to fp32 are widened at load (exact), and an
//! int8 table that the graph dequantizes *after* `Gather` is handed over
//! still quantized, so the caller can do the same per row instead of holding
//! a 4x larger fp32 copy.
//!
//! Only the handful of protobuf fields that path needs are decoded. There is no
//! `onnx`/`prost` dependency: the wire format is a dozen lines, and pulling in a
//! code generator for five fields would be the larger risk.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::error::DetectorError;

// ONNX protobuf field numbers (onnx.proto3).
const MODEL_GRAPH: u64 = 7;
const GRAPH_NODE: u64 = 1;
const GRAPH_INITIALIZER: u64 = 5;
const NODE_INPUT: u64 = 1;
const NODE_OUTPUT: u64 = 2;
const NODE_NAME: u64 = 3;
const NODE_OP_TYPE: u64 = 4;
const NODE_ATTRIBUTE: u64 = 5;
const ATTR_NAME: u64 = 1;
const ATTR_INT: u64 = 3;
const TENSOR_DIMS: u64 = 1;
const TENSOR_DATA_TYPE: u64 = 2;
const TENSOR_FLOAT_DATA: u64 = 4;
const TENSOR_NAME: u64 = 8;
const TENSOR_RAW_DATA: u64 = 9;
const TENSOR_DATA_LOCATION: u64 = 14;

const ONNX_FLOAT: u64 = 1;
const ONNX_INT8: u64 = 3;
const ONNX_FLOAT16: u64 = 10;

/// Weights recovered from an ONNX file.
pub struct OnnxWeights {
    /// Float weights, as fp32 on the target device, under PyTorch names.
    pub tensors: HashMap<String, Tensor>,
    /// int8 initializers, left quantized on the host.
    pub int8: HashMap<String, Int8Table>,
    /// The graph's nodes, for weights that can only be found structurally.
    pub nodes: Vec<Node>,
}

pub struct Int8Table {
    pub dims: Vec<usize>,
    pub data: Vec<i8>,
}

/// Load every float initializer as a Candle tensor under its PyTorch name.
pub fn load(path: &Path, device: &Device) -> Result<OnnxWeights, DetectorError> {
    let bytes = std::fs::read(path).map_err(|e| DetectorError::io(path, e))?;
    let mut graph = parse_graph(&bytes).map_err(|e| bad(path, e))?;
    let (tensors, int8) = to_tensors(&graph, device).map_err(|e| bad(path, e))?;
    Ok(OnnxWeights {
        tensors,
        int8,
        nodes: std::mem::take(&mut graph.nodes),
    })
}

fn bad(path: &Path, detail: String) -> DetectorError {
    DetectorError::ArtifactMismatch {
        path: path.to_path_buf(),
        detail,
    }
}

struct Initializer<'a> {
    name: String,
    dims: Vec<usize>,
    data_type: u64,
    raw: Option<&'a [u8]>,
    /// `float_data`, packed little-endian on the wire like `raw_data`.
    floats: Option<&'a [u8]>,
    external: bool,
}

pub struct Node {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// The `to` attribute of a `Cast`.
    cast_to: Option<u64>,
}

struct Graph<'a> {
    initializers: Vec<Initializer<'a>>,
    nodes: Vec<Node>,
}

type Loaded = (HashMap<String, Tensor>, HashMap<String, Int8Table>);

fn to_tensors(graph: &Graph<'_>, device: &Device) -> Result<Loaded, String> {
    // `Identity` nodes alias one initializer under another value name. DeBERTa
    // with `share_att_key` routes its query/key projections through them,
    // because the relative-position path reuses the same weights. A `Cast` to
    // fp32 is the same thing for an fp16-compressed weight: widening is exact,
    // so the loader does it once and the alias stands.
    let mut alias: HashMap<&str, &str> = HashMap::new();
    for n in &graph.nodes {
        let aliasing =
            n.op_type == "Identity" || (n.op_type == "Cast" && n.cast_to == Some(ONNX_FLOAT));
        if aliasing && let (Some(i), Some(o)) = (n.inputs.first(), n.outputs.first()) {
            alias.insert(o.as_str(), i.as_str());
        }
    }
    let by_name: HashMap<&str, &Initializer<'_>> = graph
        .initializers
        .iter()
        .map(|i| (i.name.as_str(), i))
        .collect();

    // Anonymous MatMul weights, named after their consumer.
    let mut renamed: HashMap<&str, String> = HashMap::new();
    for n in graph.nodes.iter().filter(|n| n.op_type == "MatMul") {
        let Some(init) = n.inputs.get(1).and_then(|v| resolve(&alias, &by_name, v)) else {
            continue; // activation x activation, e.g. Q·Kᵀ
        };
        if !init.name.starts_with("onnx::") {
            continue;
        }
        let Some(module) = linear_module(&n.name) else {
            return Err(format!(
                "cannot recover a module path from MatMul node {:?}",
                n.name
            ));
        };
        let target = format!("{module}.weight");
        match renamed.get(init.name.as_str()) {
            Some(prev) if *prev != target => {
                return Err(format!(
                    "initializer {} feeds both {prev} and {target}",
                    init.name
                ));
            }
            _ => {
                renamed.insert(init.name.as_str(), target);
            }
        }
    }

    // Two anonymous tensors claiming one name would silently drop one of them.
    let mut claimed: HashMap<&str, &str> = HashMap::new();
    for (src, dst) in &renamed {
        if let Some(other) = claimed.insert(dst.as_str(), src)
            && other != *src
        {
            return Err(format!("{dst} is claimed by both {other} and {src}"));
        }
    }

    let mut out = HashMap::new();
    let mut int8 = HashMap::new();
    for init in &graph.initializers {
        let (name, transpose) = match renamed.get(init.name.as_str()) {
            Some(n) => (n.clone(), true),
            None => (init.name.clone(), false),
        };
        if init.external {
            return Err(format!(
                "{}: external tensor data is not supported",
                init.name
            ));
        }
        let bytes = init.raw.or(init.floats).unwrap_or(&[]);
        let numel: usize = init.dims.iter().product();
        let width = match init.data_type {
            ONNX_FLOAT => 4,
            ONNX_FLOAT16 => 2,
            ONNX_INT8 => 1,
            // Shape and index constants the graph computes with, not weights.
            _ if init.name.starts_with("onnx::") && !transpose => continue,
            other => {
                return Err(format!(
                    "{} has ONNX data type {other}; the Candle backend reads fp32, \
                     fp16 and int8 weights only",
                    init.name
                ));
            }
        };
        if bytes.len() != numel * width {
            return Err(format!(
                "{}: {} bytes of data for shape {:?}",
                init.name,
                bytes.len(),
                init.dims
            ));
        }
        if init.data_type == ONNX_INT8 {
            if transpose {
                return Err(format!("{} is an int8 MatMul weight", init.name));
            }
            let data = bytes.iter().map(|&b| b as i8).collect();
            int8.insert(
                name,
                Int8Table {
                    dims: init.dims.clone(),
                    data,
                },
            );
            continue;
        }
        let err = |e: candle_core::Error| format!("{}: {e}", init.name);
        let dtype = if width == 4 { DType::F32 } else { DType::F16 };
        let mut t = Tensor::from_raw_buffer(bytes, dtype, &init.dims, &Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .map_err(err)?;
        if transpose {
            t = t.t().and_then(|t| t.contiguous()).map_err(err)?;
        }
        out.insert(name, t.to_device(device).map_err(err)?);
    }
    Ok((out, int8))
}

/// The initializer a value name refers to, following `Identity` aliases.
fn resolve<'i>(
    alias: &HashMap<&str, &str>,
    by_name: &HashMap<&str, &'i Initializer<'i>>,
    v: &str,
) -> Option<&'i Initializer<'i>> {
    let mut cur = v;
    // Bounded: a cycle, or a chain no exporter would produce, resolves to None.
    for _ in 0..8 {
        if let Some(i) = by_name.get(cur) {
            return Some(i);
        }
        cur = alias.get(cur)?;
    }
    None
}

/// `/bert/encoder/layer.0/attention/self/query/MatMul` → `bert.encoder.layer.0.attention.self.query`.
///
/// A module called twice in one forward pass gets a `_N` suffix on its second
/// call site (`query_proj_1`); that suffix is dropped, and the caller checks
/// that both call sites really do share one initializer.
///
/// Children of an `nn.Sequential` repeat their parent's name in the scope
/// (`/span_marker/project_start/project_start.0/MatMul`); the repeat is
/// folded so the result is `span_marker.project_start.0`.
fn linear_module(node: &str) -> Option<String> {
    let path = node.strip_prefix('/')?.strip_suffix("/MatMul")?;
    let mut parts: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        if let Some(prev) = parts.last()
            && seg
                .strip_prefix(*prev)
                .is_some_and(|rest| rest.starts_with('.'))
        {
            parts.pop();
        }
        parts.push(seg);
    }
    let last = parts.last_mut()?;
    if let Some((base, n)) = last.rsplit_once('_')
        && !base.is_empty()
        && !n.is_empty()
        && n.bytes().all(|b| b.is_ascii_digit())
    {
        *last = base;
    }
    Some(parts.join("."))
}

// ---------------------------------------------------------------------------
// Protobuf wire format
// ---------------------------------------------------------------------------

enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

struct Fields<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Fields<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b < 0x80 {
                return Ok(v);
            }
        }
        Err("varint longer than 10 bytes".into())
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or("field runs past the end of its message")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn next_field(&mut self) -> Option<Result<(u64, Value<'a>), String>> {
        if self.pos >= self.buf.len() {
            return None;
        }
        Some((|| {
            let key = self.varint()?;
            let value = match key & 7 {
                0 => Value::Varint(self.varint()?),
                1 => {
                    self.take(8)?;
                    Value::Fixed
                }
                2 => {
                    let n = usize::try_from(self.varint()?).map_err(|e| e.to_string())?;
                    Value::Bytes(self.take(n)?)
                }
                5 => {
                    self.take(4)?;
                    Value::Fixed
                }
                w => return Err(format!("unsupported protobuf wire type {w}")),
            };
            Ok((key >> 3, value))
        })())
    }
}

fn utf8(b: &[u8]) -> Result<String, String> {
    std::str::from_utf8(b)
        .map(str::to_owned)
        .map_err(|e| e.to_string())
}

fn parse_graph(model: &[u8]) -> Result<Graph<'_>, String> {
    let mut graph_bytes = None;
    let mut f = Fields::new(model);
    while let Some(field) = f.next_field() {
        if let (MODEL_GRAPH, Value::Bytes(b)) = field? {
            graph_bytes = Some(b);
        }
    }
    let graph_bytes = graph_bytes.ok_or("no graph in ModelProto — is this an ONNX file?")?;

    let mut graph = Graph {
        initializers: Vec::new(),
        nodes: Vec::new(),
    };
    let mut f = Fields::new(graph_bytes);
    while let Some(field) = f.next_field() {
        match field? {
            (GRAPH_NODE, Value::Bytes(b)) => graph.nodes.push(parse_node(b)?),
            (GRAPH_INITIALIZER, Value::Bytes(b)) => graph.initializers.push(parse_tensor(b)?),
            _ => {}
        }
    }
    Ok(graph)
}

fn parse_node(b: &[u8]) -> Result<Node, String> {
    let mut n = Node {
        name: String::new(),
        op_type: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        cast_to: None,
    };
    let mut f = Fields::new(b);
    while let Some(field) = f.next_field() {
        match field? {
            (NODE_INPUT, Value::Bytes(s)) => n.inputs.push(utf8(s)?),
            (NODE_OUTPUT, Value::Bytes(s)) => n.outputs.push(utf8(s)?),
            (NODE_NAME, Value::Bytes(s)) => n.name = utf8(s)?,
            (NODE_OP_TYPE, Value::Bytes(s)) => n.op_type = utf8(s)?,
            (NODE_ATTRIBUTE, Value::Bytes(a)) => {
                let (mut name, mut int) = (None, None);
                let mut f = Fields::new(a);
                while let Some(field) = f.next_field() {
                    match field? {
                        (ATTR_NAME, Value::Bytes(s)) => name = Some(s),
                        (ATTR_INT, Value::Varint(v)) => int = Some(v),
                        _ => {}
                    }
                }
                if name == Some(b"to".as_slice()) {
                    n.cast_to = int;
                }
            }
            _ => {}
        }
    }
    Ok(n)
}

fn parse_tensor(b: &[u8]) -> Result<Initializer<'_>, String> {
    let mut t = Initializer {
        name: String::new(),
        dims: Vec::new(),
        data_type: 0,
        raw: None,
        floats: None,
        external: false,
    };
    let mut f = Fields::new(b);
    while let Some(field) = f.next_field() {
        match field? {
            (TENSOR_DIMS, Value::Varint(d)) => t.dims.push(dim(d)?),
            // Packed repeated int64.
            (TENSOR_DIMS, Value::Bytes(p)) => {
                let mut inner = Fields::new(p);
                while inner.pos < p.len() {
                    t.dims.push(dim(inner.varint()?)?);
                }
            }
            (TENSOR_DATA_TYPE, Value::Varint(v)) => t.data_type = v,
            (TENSOR_FLOAT_DATA, Value::Bytes(p)) => t.floats = Some(p),
            (TENSOR_NAME, Value::Bytes(s)) => t.name = utf8(s)?,
            (TENSOR_RAW_DATA, Value::Bytes(p)) => t.raw = Some(p),
            (TENSOR_DATA_LOCATION, Value::Varint(v)) => t.external = v == 1,
            _ => {}
        }
    }
    Ok(t)
}

fn dim(d: u64) -> Result<usize, String> {
    usize::try_from(d).map_err(|_| format!("dimension {d} does not fit in usize"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_paths_are_recovered_from_matmul_node_names() {
        assert_eq!(
            linear_module("/bert/encoder/layer.0/attention/self/query/MatMul").as_deref(),
            Some("bert.encoder.layer.0.attention.self.query")
        );
        assert_eq!(
            linear_module("/classifier/MatMul").as_deref(),
            Some("classifier")
        );
        // The second call site of a shared projection.
        assert_eq!(
            linear_module("/deberta/encoder/layer.3/attention/self/key_proj_1/MatMul").as_deref(),
            Some("deberta.encoder.layer.3.attention.self.key_proj")
        );
        // `layer.0` is a module index, not a call-site suffix.
        assert_eq!(
            linear_module("/m/layer.0/MatMul").as_deref(),
            Some("m.layer.0")
        );
        assert_eq!(
            linear_module("/span_marker/project_start/project_start.0/MatMul").as_deref(),
            Some("span_marker.project_start.0")
        );
        assert_eq!(linear_module("MatMul_17"), None);
        assert_eq!(linear_module("/a/b/Add"), None);
    }

    /// Hand-assemble a two-initializer model and read it back.
    #[test]
    fn a_minimal_graph_round_trips_with_the_weight_transposed() {
        fn key(field: u64, wire: u64, out: &mut Vec<u8>) {
            varint((field << 3) | wire, out);
        }
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            while v >= 0x80 {
                out.push((v as u8) | 0x80);
                v >>= 7;
            }
            out.push(v as u8);
        }
        fn bytes(field: u64, b: &[u8], out: &mut Vec<u8>) {
            key(field, 2, out);
            varint(b.len() as u64, out);
            out.extend_from_slice(b);
        }
        fn tensor(name: &str, dims: &[u64], data: &[f32]) -> Vec<u8> {
            let mut t = Vec::new();
            for &d in dims {
                key(TENSOR_DIMS, 0, &mut t);
                varint(d, &mut t);
            }
            key(TENSOR_DATA_TYPE, 0, &mut t);
            varint(ONNX_FLOAT, &mut t);
            bytes(TENSOR_NAME, name.as_bytes(), &mut t);
            let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            bytes(TENSOR_RAW_DATA, &raw, &mut t);
            t
        }
        fn node(name: &str, op: &str, inputs: &[&str], output: &str) -> Vec<u8> {
            let mut n = Vec::new();
            for i in inputs {
                bytes(NODE_INPUT, i.as_bytes(), &mut n);
            }
            bytes(NODE_OUTPUT, output.as_bytes(), &mut n);
            bytes(NODE_NAME, name.as_bytes(), &mut n);
            bytes(NODE_OP_TYPE, op.as_bytes(), &mut n);
            n
        }

        let mut g = Vec::new();
        bytes(
            GRAPH_INITIALIZER,
            &tensor("onnx::MatMul_9", &[2, 3], &[1., 2., 3., 4., 5., 6.]),
            &mut g,
        );
        bytes(
            GRAPH_INITIALIZER,
            &tensor("head.bias", &[3], &[0.5, 0.5, 0.5]),
            &mut g,
        );
        bytes(
            GRAPH_NODE,
            &node("Identity_0", "Identity", &["onnx::MatMul_9"], "w_alias"),
            &mut g,
        );
        bytes(
            GRAPH_NODE,
            &node("/head/MatMul", "MatMul", &["x", "onnx::MatMul_9"], "y"),
            &mut g,
        );
        bytes(
            GRAPH_NODE,
            &node("/head_1/MatMul", "MatMul", &["x2", "w_alias"], "y2"),
            &mut g,
        );
        let mut model = Vec::new();
        bytes(MODEL_GRAPH, &g, &mut model);

        let graph = parse_graph(&model).unwrap();
        let (w, int8) = to_tensors(&graph, &Device::Cpu).unwrap();
        assert!(int8.is_empty());
        assert_eq!(w.len(), 2);
        let weight = &w["head.weight"];
        assert_eq!(weight.dims(), &[3, 2], "stored [in, out], loaded [out, in]");
        assert_eq!(
            weight.to_vec2::<f32>().unwrap(),
            vec![vec![1., 4.], vec![2., 5.], vec![3., 6.]]
        );
        assert_eq!(w["head.bias"].to_vec1::<f32>().unwrap(), vec![0.5; 3]);
    }

    #[test]
    fn a_non_onnx_file_is_rejected_rather_than_read_as_empty() {
        assert!(parse_graph(b"").is_err());
        assert!(parse_graph(&[0x0a, 0x05, b'h']).is_err(), "truncated field");
    }
}
