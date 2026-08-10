#!/usr/bin/env bash
#
# Sweep GLiNER2 across thread counts and execution-provider configurations.
#
# One process per cell, deliberately. An in-process sweep would be marginally
# faster (each cell needs a fresh session regardless — intra_op is baked at
# build time — so process launch is ~10 ms against a ~1.4 s model load) and
# wrong in three ways:
#
#   * ONNX Runtime's arena allocators, CoreML's compiled-model cache and the
#     mmap'd 426 MB weight file all carry state across sessions within a
#     process, so cell N would be measured with cell N-1's arena warmed.
#   * `perf/memory.rs` reports peak RSS process-wide. Six sequential sessions
#     in one process invalidates the RSS column in every cell, not just one.
#   * `RunMeta.command_line` reproduces each cell verbatim. That property is
#     the reason this repo's results are citable, and an in-process sweep
#     collapses six results onto one command line.
#
# Two passes, forward then reverse. A monotonic 1 -> 10 thread sweep on a
# laptop chassis otherwise attributes thermal drift to thread count; the
# collator takes the min per cell and flags disagreement.
#
# ONNX Runtime's node profiler is NOT enabled on timing cells. Measured on this
# graph it costs ~6% at p50 and ~12% at p95 — small, but it is overhead on the
# exact quantity these cells exist to measure, and it would land unevenly
# (10368 CPU-EP nodes to instrument versus 1128 on CoreML, so the CPU cells
# would be penalised hardest, which is precisely the comparison at stake).
# Provider share is collected separately in the `profile` pass, at low
# iteration counts, where only the ratio matters and the wall clock is ignored.
#
# Usage:
#   scripts/perf-matrix.sh                 # everything
#   PASS=threads scripts/perf-matrix.sh    # just the thread sweep
#   PASS=ep      scripts/perf-matrix.sh    # just the provider matrix
#   PASS=static  scripts/perf-matrix.sh    # just the static-shape cells
#   ALLOW_DIRTY=1 ITERATIONS=5 scripts/perf-matrix.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="target/release/dlp-bench"
PASS="${PASS:-all}"
ITERATIONS="${ITERATIONS:-20}"
WARMUP="${WARMUP:-3}"
BUCKETS="${BUCKETS:-32,128,512}"
RUN="${RUN:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$ROOT/out/matrix/$RUN"

# --- guards ----------------------------------------------------------------
# Each of these is a mistake that would silently waste a 40-minute sweep.

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found — run: cargo build --release" >&2
    exit 1
fi
if [[ -n "$(find src -name '*.rs' -newer "$BIN" -print -quit 2>/dev/null)" ]]; then
    echo "error: sources are newer than $BIN; the sweep would measure stale code." >&2
    echo "       run: cargo build --release" >&2
    exit 1
fi
if [[ -z "${ALLOW_DIRTY:-}" && -n "$(git status --porcelain 2>/dev/null)" ]]; then
    echo "error: working tree is dirty; results would record an unreproducible commit." >&2
    echo "       commit, or re-run with ALLOW_DIRTY=1 (RunMeta will flag it)." >&2
    exit 1
fi

mkdir -p "$OUT/a" "$OUT/b"

# --- cells -----------------------------------------------------------------
# "tag|args". A cell that fails to build is recorded, not skipped: for
# --disable-cpu-fallback the failure *is* the result, and its message names the
# nodes CoreML could not place.

CELLS=()

if [[ "$PASS" == "all" || "$PASS" == "threads" ]]; then
    # E1. This box is 8 performance + 2 efficiency cores, and the current
    # default of min(cores, 8) already lands on the P-core count, so the
    # expected outcome is confirmation rather than a win. 10 should be worse:
    # the two E-cores become stragglers at every intra-op barrier.
    for t in 1 2 4 6 8 10; do
        CELLS+=("cpu-t$t|--ep cpu --threads $t")
    done
    # ORT's spin-wait burns a core to shave wake-up latency, which is exactly
    # the wrong trade on an efficiency core.
    CELLS+=("cpu-t8-nospin|--ep cpu --threads 8 --no-intra-op-spinning")
    # Control, and the only lever here with order-of-magnitude potential: the
    # 17-label schema is 77 tokens that all 12 layers reprocess every call.
    CELLS+=("cpu-t8-minimal|--ep cpu --threads 8 --schema minimal")
    CELLS+=("cpu-t8-full|--ep cpu --threads 8 --schema full")
fi

if [[ "$PASS" == "all" || "$PASS" == "ep" ]]; then
    # E5/E2. MLProgram is known to fail on this graph (see the note in
    # detectors/onnx.rs); it is kept in the matrix so the failure is recorded
    # alongside everything else rather than remembered.
    CELLS+=("cpu-base|--ep cpu --threads 8")
    CELLS+=("coreml|--ep coreml --threads 8")
    CELLS+=("coreml-mlprogram|--ep coreml --threads 8 --coreml-format ml-program")
    CELLS+=("coreml-subgraphs|--ep coreml --threads 8 --coreml-subgraphs true")
    CELLS+=("coreml-ane|--ep coreml-ane --threads 8")
    CELLS+=("coreml-gpu|--ep coreml --threads 8 --coreml-compute-units cpu-gpu")
    CELLS+=("coreml-cpuonly|--ep coreml --threads 8 --coreml-compute-units cpu-only")
    CELLS+=("coreml-fastpred|--ep coreml --threads 8 --coreml-specialization fast-prediction")
    CELLS+=("coreml-cached|--ep coreml --threads 8 --coreml-cache-dir .cache/coreml")
    CELLS+=("coreml-nofallback|--ep coreml --threads 8 --disable-cpu-fallback")
fi

if [[ "$PASS" == "all" || "$PASS" == "static" ]]; then
    # E3. One fixed shape per invocation: four 426 MB sessions in one process
    # would be ~1.7 GB against a 750 MB target and would corrupt the RSS column
    # in every cell. Shapes come from `inspect`, rounded up to a multiple of 32.
    #
    # Bucket 128 only — enough to answer "does pinning the shape let CoreML
    # take more of the graph", which is what E3 asks.
    S=544
    W=224
    DIMS="--dim-override seq_len=$S --dim-override num_words=$W"
    CELLS+=("static-cpu|--ep cpu --threads 8 $DIMS")
    CELLS+=("static-coreml|--ep coreml --threads 8 $DIMS")
    CELLS+=("static-coreml-req|--ep coreml --threads 8 $DIMS --coreml-static-shapes true")
    CELLS+=("static-coreml-nofallback|--ep coreml --threads 8 $DIMS --disable-cpu-fallback")
fi

if [[ "$PASS" == "all" || "$PASS" == "profile" ]]; then
    # E2a. Timings from these cells are meaningless and the collator ignores
    # them; the only output that matters is the kernel-time split across
    # execution providers, which is what decides whether a CoreML measurement
    # is a measurement of CoreML at all.
    S=544
    W=224
    DIMS="--dim-override seq_len=$S --dim-override num_words=$W"
    CELLS+=("prof-cpu|--ep cpu --threads 8")
    CELLS+=("prof-coreml|--ep coreml --threads 8")
    CELLS+=("prof-coreml-ane|--ep coreml-ane --threads 8")
    CELLS+=("prof-coreml-gpu|--ep coreml --threads 8 --coreml-compute-units cpu-gpu")
    CELLS+=("prof-static-coreml|--ep coreml --threads 8 $DIMS")
    CELLS+=("prof-static-coreml-req|--ep coreml --threads 8 $DIMS --coreml-static-shapes true")
fi

# --- run -------------------------------------------------------------------

run_cell() {
    local dir="$1" tag="$2" args="$3"
    local json="$dir/$tag.json" err="$dir/$tag.err"
    local buckets="$BUCKETS"
    local iterations="$ITERATIONS"
    local profile=()

    # A static-shape session accepts exactly one input size.
    [[ "$tag" == static-* ]] && buckets=128
    # Profile cells exist only to read the provider split off `ep_node_share`;
    # their timings are discarded, so they run short. Every other cell runs
    # unprofiled so the instrumentation cannot bias the number being reported.
    if [[ "$tag" == prof-* ]]; then
        profile=(--ort-profile "$dir/$tag-trace")
        buckets=128
        iterations=3
    fi

    printf '  %-28s ' "$tag"
    # shellcheck disable=SC2086
    if caffeinate -i "$BIN" perf \
        --detector gliner2-pii --skip-checksum \
        $args \
        --buckets "$buckets" --iterations "$iterations" --warmup "$WARMUP" \
        --thread-counts \
        "${profile[@]}" \
        -o "$json" >"$dir/$tag.log" 2>"$err"
    then
        python3 - "$json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))["performance"]
b = p["buckets"][0]
rt = p.get("runtime") or {}
share = rt.get("ep_node_share")
acc = ""
if share and share.get("total_us"):
    cpu = share["by_provider_us"].get("CPUExecutionProvider", 0)
    acc = f"  accel {100*(share['total_us']-cpu)/share['total_us']:.0f}%"
print(f"p50 {b['p50_ms']:8.1f} ms   load {p['cold_start']['model_load_ms']:6.0f} ms{acc}")
PY
    else
        # Record it. `--disable-cpu-fallback` failing to commit is the answer to
        # "can CoreML take this graph", not a skipped cell.
        echo "FAILED — see $err"
        head -c 400 "$err" | tr '\n' ' ' | sed 's/^/      /' >>"$dir/$tag.failed"
        echo >>"$dir/$tag.failed"
    fi
}

{
    echo "{"
    echo "  \"run\": \"$RUN\","
    echo "  \"pass\": \"$PASS\","
    echo "  \"iterations\": $ITERATIONS,"
    echo "  \"buckets\": \"$BUCKETS\","
    printf '  "cells": ['
    sep=""
    for cell in "${CELLS[@]}"; do
        printf '%s"%s"' "$sep" "${cell%%|*}"
        sep=", "
    done
    echo "]"
    echo "}"
} >"$OUT/manifest.json"

echo "matrix $RUN — ${#CELLS[@]} cells x 2 passes -> $OUT"

echo
echo "pass A (forward):"
for cell in "${CELLS[@]}"; do
    run_cell "$OUT/a" "${cell%%|*}" "${cell#*|}"
done

echo
echo "pass B (reverse — separates thermal drift from the variable under test):"
for ((i = ${#CELLS[@]} - 1; i >= 0; i--)); do
    cell="${CELLS[$i]}"
    run_cell "$OUT/b" "${cell%%|*}" "${cell#*|}"
done

echo
echo "done. collate with:"
echo "  python3 scripts/collate-matrix.py $OUT"
