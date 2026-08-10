#!/usr/bin/env python3
"""Collate a `perf-matrix.sh` run into markdown for REPORT.md.

Reads the two passes the sweep writes and reports the *minimum* p50 per cell,
because the failure mode this guards against is thermal: on a laptop chassis a
monotonic sweep makes later cells look worse than earlier ones regardless of
what is being varied. Where the two passes disagree by more than DRIFT_WARN the
cell is flagged rather than averaged — an average of two measurements taken
under different thermal conditions is not a measurement of anything.

Cells that failed to build are printed with their error. That is deliberate:
`--disable-cpu-fallback` failing is the answer to "can CoreML take this graph",
and dropping it would turn a result into a gap.

Usage: python3 scripts/collate-matrix.py out/matrix/<run>
"""

import json
import pathlib
import sys

DRIFT_WARN = 0.10


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def bucket(perf, tokens):
    for b in perf["buckets"]:
        if b["target_tokens"] == tokens:
            return b
    return None


def cell_rows(root, tag):
    """(pass_a, pass_b, failure_text) for one cell."""
    out = []
    for p in ("a", "b"):
        d = root / p
        failed = d / f"{tag}.failed"
        rep = load(d / f"{tag}.json")
        if rep is None:
            out.append(("failed", failed.read_text().strip() if failed.exists() else "no output"))
        else:
            out.append(("ok", rep["performance"]))
    return out


def fmt(v, width=8, prec=1):
    return f"{v:>{width}.{prec}f}" if v is not None else " " * (width - 1) + "-"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = pathlib.Path(sys.argv[1])
    manifest = load(root / "manifest.json")
    if manifest is None:
        sys.exit(f"no manifest.json in {root}")

    tokens = [int(t) for t in manifest["buckets"].split(",")]
    all_tags = manifest["cells"]
    # `prof-*` cells run ONNX Runtime's node profiler, which costs ~6% at p50
    # and ~12% at p95 on this graph. Their timings are excluded from the
    # latency table entirely rather than footnoted — a slower row sitting next
    # to comparable ones invites exactly the comparison it cannot support.
    tags = [t for t in all_tags if not t.startswith("prof-")]
    prof_tags = [t for t in all_tags if t.startswith("prof-")]

    print(f"### perf matrix `{manifest['run']}` "
          f"({manifest['iterations']} iterations, min of two passes)\n")

    header = ["cell", "ep", "thr"]
    for t in tokens:
        header += [f"p50@{t}", f"p95@{t}"]
    header += ["load ms", "RSS MB", "accel", "drift"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")

    failures, notes = [], []

    for tag in tags:
        (sa, va), (sb, vb) = cell_rows(root, tag)
        if sa == "failed" and sb == "failed":
            failures.append((tag, va))
            continue
        perfs = [v for s, v in ((sa, va), (sb, vb)) if s == "ok"]
        # Pass A is authoritative for anything that is not a timing.
        ref = perfs[0]
        rt = ref.get("runtime") or {}
        session = rt.get("session", {})
        coreml = session.get("coreml") or {}

        ep = rt.get("provider", "?")
        extra = []
        if coreml.get("model_format") == "ml_program":
            extra.append("mlprog")
        if coreml.get("subgraphs"):
            extra.append("subgraphs")
        if coreml.get("static_input_shapes"):
            extra.append("reqstatic")
        if coreml.get("compute_units") not in (None, "all"):
            extra.append(coreml["compute_units"].replace("_", "-"))
        if coreml.get("specialization") == "fast_prediction":
            extra.append("fastpred")
        if coreml.get("model_cache_dir"):
            extra.append("cached")
        if session.get("disable_cpu_fallback"):
            extra.append("nofallback")
        if session.get("disable_intra_op_spinning"):
            extra.append("nospin")
        if session.get("dimension_overrides"):
            extra.append("static")
        if extra:
            ep += " +" + "+".join(extra)

        row = [f"`{tag}`", ep, str(rt.get("intra_op_threads", "?"))]
        worst_drift = 0.0
        for t in tokens:
            bs = [bucket(p, t) for p in perfs]
            bs = [b for b in bs if b]
            if not bs:
                row += ["-", "-"]
                continue
            p50 = min(b["p50_ms"] for b in bs)
            p95 = min(b["p95_ms"] for b in bs)
            if len(bs) == 2:
                lo, hi = sorted(b["p50_ms"] for b in bs)
                if lo > 0:
                    worst_drift = max(worst_drift, (hi - lo) / lo)
            row += [f"{p50:.1f}", f"{p95:.1f}"]

        row.append(f"{min(p['cold_start']['model_load_ms'] for p in perfs):.0f}")
        rss = [p["warm_rss"]["rss_mb"] for p in perfs if p.get("warm_rss")]
        row.append(str(min(rss)) if rss else "-")

        row.append("-")  # provider share comes from the profile pass, not here

        flag = f"{100 * worst_drift:.0f}%" + (" !!" if worst_drift > DRIFT_WARN else "")
        row.append(flag if len(perfs) == 2 else "1 pass")
        print("| " + " | ".join(row) + " |")

    for tag in prof_tags:
        (sa, va), (_sb, _vb) = cell_rows(root, tag)
        if sa == "failed":
            failures.append((tag, va))
            continue
        rt = va.get("runtime") or {}
        if rt.get("ep_node_share"):
            notes.append((tag, rt["ep_node_share"]))

    if failures:
        print("\n### cells that failed to build\n")
        print("These are results, not gaps: a session that refuses to commit is "
              "telling you the graph cannot be placed as requested.\n")
        for tag, err in failures:
            print(f"* **`{tag}`** — {err.strip()[:500]}")

    if notes:
        print("\n### execution-provider node share\n")
        print("Kernel time by execution provider, from ONNX Runtime's own node "
              "profile. Wall-clock numbers from these runs are omitted "
              "deliberately — profiling costs ~6% here, so only the ratio is "
              "trustworthy. A CoreML session that hands most of the graph back "
              "to the CPU EP is not a measurement of CoreML.\n")
        print("| cell | provider | kernel ms | share | node execs |")
        print("|---|---|---|---|---|")
        for tag, share in notes:
            for provider, us in sorted(share["by_provider_us"].items(),
                                       key=lambda kv: -kv[1]):
                pct = 100 * us / share["total_us"] if share["total_us"] else 0
                n = share.get("nodes", {}).get(provider, 0)
                print(f"| `{tag}` | {provider} | {us / 1000:.0f} | {pct:.1f}% | {n} |")

    # Stage breakdown, when any cell was run with --stage-timing.
    staged = []
    for tag in tags:
        rep = load(root / "a" / f"{tag}.json")
        if rep and rep["performance"].get("stage_breakdown"):
            staged.append((tag, rep["performance"]["stage_breakdown"]))
    if staged:
        print("\n### stage breakdown (p50 ms)\n")
        stages = [s["stage"] for s in staged[0][1][0]["stages"]]
        print("| cell | tokens | " + " | ".join(stages) + " | unattr | wall |")
        print("|" + "|".join("---" for _ in range(len(stages) + 4)) + "|")
        for tag, breakdown in staged:
            for b in breakdown:
                by = {s["stage"]: s["percentiles"]["p50_ms"] for s in b["stages"]}
                cells = " | ".join(f"{by.get(s, 0):.2f}" for s in stages)
                print(f"| `{tag}` | {b['target_tokens']} | {cells} | "
                      f"{b['unattributed_p50_ms']:.2f} | {b['instrumented']['p50_ms']:.1f} |")


if __name__ == "__main__":
    main()
