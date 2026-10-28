"""Collate runtime-comparison results into markdown tables.

Reads every `*.json` report in a directory — `dlp-bench perf`,
`dlp-bench concurrency` and `scripts/bench_pytorch_gliner2.py` all write the
same `performance` shape — and prints a latency table and, per request size,
a concurrency table.

Refuses to compare a bucket unless every run measured the same document (byte
and word counts agree): the PyTorch harness builds its documents from a port of
the Rust generator, and a drift there would otherwise read as a runtime
difference.

Usage: python3 scripts/collate-runtimes.py [out-dir]
"""

import glob
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/runtimes"

LABELS = {
    "cpu": "ONNX Runtime, CPU",
    "coreml": "ONNX Runtime, CoreML",
    "candle-cpu": "Candle, CPU (gemm)",
    "candle-cpu-accelerate": "Candle, CPU (Accelerate)",
    "candle-metal": "Candle, Metal",
    "pytorch-cpu": "PyTorch, CPU",
    "pytorch-mps": "PyTorch, MPS",
}
ORDER = list(LABELS)
BASELINE = "cpu"


def provider(p):
    return (p.get("runtime") or {}).get("provider", p.get("detector", "?"))


runs = []
for path in sorted(glob.glob(os.path.join(OUT, "*.json"))):
    with open(path) as f:
        p = json.load(f).get("performance")
    if p:
        runs.append(p)
if not runs:
    sys.exit(f"no results in {OUT}")
runs.sort(key=lambda p: ORDER.index(provider(p)) if provider(p) in ORDER else len(ORDER))


def check_same_docs(rows, key):
    shapes = {}
    for r in rows:
        shapes.setdefault(r[key], set()).add((r["actual_words"], r["actual_bytes"]))
    for k, s in shapes.items():
        if len(s) != 1:
            sys.exit(f"{key}={k}: runs measured different documents {s}; not comparable")


def label(p):
    return LABELS.get(provider(p), provider(p))


# ---- latency ---------------------------------------------------------------
lat = [p for p in runs if p.get("buckets")]
if lat:
    check_same_docs([b for p in lat for b in p["buckets"]], "target_tokens")
    buckets = sorted({b["target_tokens"] for p in lat for b in p["buckets"]})
    words = {b["target_tokens"]: b["actual_words"] for p in lat for b in p["buckets"]}
    base = next(({b["target_tokens"]: b for b in p["buckets"]} for p in lat if provider(p) == BASELINE), {})

    print("### Single-request latency, p50 / p95 ms (speed-up of p50 vs ONNX Runtime)\n")
    head = ["runtime", "load ms", "1st call ms"] + [f"{t} tok ({words[t]} w)" for t in buckets]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for p in lat:
        cs = p["cold_start"]
        by = {b["target_tokens"]: b for b in p["buckets"]}
        row = [label(p), f"{cs['model_load_ms']:.0f}", f"{cs['first_inference_ms']:.0f}"]
        for t in buckets:
            b = by.get(t)
            if not b:
                row.append("-")
                continue
            cell = f"{b['p50_ms']:.1f} / {b['p95_ms']:.1f}"
            if provider(p) != BASELINE and t in base:
                cell += f" ({base[t]['p50_ms'] / b['p50_ms']:.2f}x)"
            row.append(cell)
        print("| " + " | ".join(row) + " |")
    print()

# ---- concurrency -----------------------------------------------------------
conc = [p for p in runs if p.get("concurrency")]
if conc:
    check_same_docs([c for p in conc for c in p["concurrency"]], "target_tokens")
    for t in sorted({c["target_tokens"] for p in conc for c in p["concurrency"]}):
        clients = sorted({c["clients"] for p in conc for c in p["concurrency"] if c["target_tokens"] == t})
        w = next(c["actual_words"] for p in conc for c in p["concurrency"] if c["target_tokens"] == t)
        print(f"### {t} tok ({w} words): throughput req/s | client latency p50 / p95 ms\n")
        head = ["runtime", "serving"] + [f"N={n}" for n in clients]
        print("| " + " | ".join(head) + " |")
        print("|" + "|".join(["---", "---"] + ["---:"] * len(clients)) + "|")
        for p in conc:
            by = {c["clients"]: c for c in p["concurrency"] if c["target_tokens"] == t}
            first = next(iter(by.values()))
            if first.get("serialised"):
                serving = "1 shared, serialised"
            elif first["shared"]:
                serving = "1 shared"
            else:
                serving = f"pool <= {max(c['instances'] for c in by.values())}"
            row = [label(p), serving]
            for n in clients:
                c = by.get(n)
                row.append(
                    f"{c['requests_per_sec']:.1f} \\| {c['latency']['p50_ms']:.0f} / {c['latency']['p95_ms']:.0f}"
                    if c
                    else "-"
                )
            print("| " + " | ".join(row) + " |")
        print()
    for p in conc:
        rss = [c["rss"]["rss_mb"] for c in p["concurrency"] if c.get("rss")]
        if rss:
            print(f"{label(p)}: RSS {min(rss)}-{max(rss)} MB across levels")
