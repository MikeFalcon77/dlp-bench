#!/usr/bin/env bash
# gliner2-pii latency on three runtimes: ONNX Runtime, Candle, PyTorch.
#
# Six configurations, run strictly one at a time — an overlapping run once
# inflated a GLiNER2 measurement ~2.4x (README). Same documents, same 17-label
# schema, same thread count, same timed region everywhere; throughput is
# skipped so this is a pure latency comparison.
#
# Prerequisites:
#   bash scripts/fetch-models.sh gliner2-pii-onnx
#   bash scripts/fetch-models.sh gliner2-pii-pytorch
#   python3.14 -m venv .venv-pytorch && .venv-pytorch/bin/pip install -r scripts/requirements-pytorch.txt
#
# Usage: bash scripts/bench-runtimes.sh [out-dir]     then: python3 scripts/collate-runtimes.py [out-dir]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-out/runtimes}"
mkdir -p "$OUT"

THREADS="${THREADS:-8}"
ITER="${ITER:-30}"
WARMUP="${WARMUP:-3}"
BUCKETS="${BUCKETS:-32,128,512,1024}"
PY=.venv-pytorch/bin/python

cargo build --release
cargo build --release --features metal,accelerate --target-dir target/apple
GEMM=./target/release/dlp-bench
APPLE=./target/apple/release/dlp-bench

"$PY" -I -m pip freeze >"$OUT/pip-freeze.txt"

# --skip-checksum: the PyTorch loader does not hash its weights either, so
# hashing here would add ~0.4 s to the Rust model-load figures only.
RUST=(perf --detector gliner2-pii --skip-checksum --threads "$THREADS"
      --iterations "$ITER" --warmup "$WARMUP" --buckets "$BUCKETS" --thread-counts)

run() {
  local name=$1; shift
  echo "=== $(date +%T) $name"
  "$@" >"$OUT/$name.log" 2>&1
}

run ort-cpu            "$GEMM"  "${RUST[@]}"                                  -o "$OUT/ort-cpu.json"
run candle-cpu-gemm    "$GEMM"  "${RUST[@]}" --backend candle --device cpu    -o "$OUT/candle-cpu-gemm.json"
run candle-cpu-accel   "$APPLE" "${RUST[@]}" --backend candle --device cpu    -o "$OUT/candle-cpu-accel.json"
run candle-metal       "$APPLE" "${RUST[@]}" --backend candle --device metal  -o "$OUT/candle-metal.json"
run pytorch-cpu        "$PY" -I scripts/bench_pytorch_gliner2.py --device cpu --threads "$THREADS" \
                         --iterations "$ITER" --warmup "$WARMUP" --buckets "$BUCKETS" -o "$OUT/pytorch-cpu.json"
run pytorch-mps        "$PY" -I scripts/bench_pytorch_gliner2.py --device mps --threads "$THREADS" \
                         --iterations "$ITER" --warmup "$WARMUP" --buckets "$BUCKETS" -o "$OUT/pytorch-mps.json"

echo "=== $(date +%T) done -> $OUT"
