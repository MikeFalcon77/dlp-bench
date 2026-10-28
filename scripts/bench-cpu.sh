#!/usr/bin/env bash
# gliner2-pii on CPU only — ONNX Runtime, Candle, PyTorch — latency and N
# concurrent clients. Portable: the same script runs on macOS or a Linux VM.
#
# CPU-only on purpose. The product runs on Linux VMs, so macOS-only paths are
# excluded even when run on a Mac: Candle is built without `accelerate`/`metal`
# (its pure-Rust gemm is what Linux gets). PyTorch on macOS still links Apple
# Accelerate for BLAS (the CPU's AMX/SME matrix unit) and cannot be told not
# to, so its Mac numbers flatter it relative to a Linux VM.
#
# Runs are strictly sequential; an overlapping run once inflated a GLiNER2
# measurement ~2.4x (README).
#
# Usage: bash scripts/bench-cpu.sh [out-dir]     then: python3 scripts/collate-runtimes.py [out-dir]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${1:-out/cpu-$(uname -s | tr '[:upper:]' '[:lower:]')}"
mkdir -p "$OUT"

THREADS="${THREADS:-8}"            # the whole service's CPU budget
ITER="${ITER:-30}"
WARMUP="${WARMUP:-3}"
BUCKETS="${BUCKETS:-32,128,512}"
CLIENTS="${CLIENTS:-1,2,3,4,5,6,7,8,9,10}"
DOCS="${DOCS:-32,128}"
REQ="${REQ:-20}"                   # timed requests per client per level
POOL_MAX="${POOL_MAX:-2}"          # ONNX Runtime sessions held at most
PY="${PYTORCH_PY:-.venv-pytorch/bin/python}"

cargo build --release --no-default-features --features onnx,candle
BIN="${CARGO_TARGET_DIR:-target}/release/dlp-bench"

{
  uname -a
  echo "nproc $(getconf _NPROCESSORS_ONLN)"
  if command -v lscpu >/dev/null; then lscpu; else sysctl -n machdep.cpu.brand_string; fi
  "$PY" -I -m pip freeze
} >"$OUT/environment.txt" 2>&1

# --skip-checksum: the PyTorch loader does not hash its weights either.
COMMON=(--detector gliner2-pii --skip-checksum --threads "$THREADS")
LAT=(perf "${COMMON[@]}" --iterations "$ITER" --warmup "$WARMUP" --buckets "$BUCKETS" --thread-counts)
CONC=(concurrency "${COMMON[@]}" --clients "$CLIENTS" --docs "$DOCS"
      --requests-per-client "$REQ" --pool-max "$POOL_MAX")
PYB=(-I scripts/bench_pytorch_gliner2.py --device cpu --threads "$THREADS")

run() {
  local name=$1; shift
  echo "=== $(date +%T) $name"
  "$@" >"$OUT/$name.log" 2>&1
}

run lat-ort-cpu       "$BIN" "${LAT[@]}"                                -o "$OUT/lat-ort-cpu.json"
run lat-candle-cpu    "$BIN" "${LAT[@]}" --backend candle --device cpu  -o "$OUT/lat-candle-cpu.json"
run lat-pytorch-cpu   "$PY" "${PYB[@]}" --iterations "$ITER" --warmup "$WARMUP" --buckets "$BUCKETS" \
                        -o "$OUT/lat-pytorch-cpu.json"

run conc-ort-cpu      "$BIN" "${CONC[@]}"                               -o "$OUT/conc-ort-cpu.json"
run conc-candle-cpu   "$BIN" "${CONC[@]}" --backend candle --device cpu -o "$OUT/conc-candle-cpu.json"
run conc-pytorch-cpu  "$PY" "${PYB[@]}" --clients "$CLIENTS" --docs "$DOCS" --requests-per-client "$REQ" \
                        -o "$OUT/conc-pytorch-cpu.json"

echo "=== $(date +%T) done -> $OUT"
