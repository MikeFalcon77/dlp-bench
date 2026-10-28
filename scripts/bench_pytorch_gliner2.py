"""Latency of gliner2-pii on the reference PyTorch engine.

The PyTorch row of the runtime comparison (scripts/bench-runtimes.sh). Mirrors
`dlp-bench perf` so the numbers sit in one table with ONNX Runtime and Candle:

* the same documents — `synthetic_document` is a line-for-line port of
  `perf::synthetic_document` in src/perf/mod.rs, and collate-runtimes.py
  refuses to compare buckets whose byte and word counts differ;
* the same schema — the 17 labels of `GLINER2_PII` in src/labels.rs, in order;
* the same timed region — text in, entities out, tokenisation and decoding
  included, as in `Detector::detect`;
* the same statistics — nearest-rank percentiles and population standard
  deviation, as in src/perf/latency.rs.

Process start and `import torch` are outside cold start, as process launch is
for the Rust binary; the import time is recorded separately.

With `--clients`, measures N concurrent clients instead, mirroring
`dlp-bench concurrency` (src/perf/concurrency.rs): one shared model, N Python
threads in a closed loop released together by a barrier, client-side latency
per request and the aggregate rate. torch releases the GIL inside its kernels,
so the threads overlap there; tokenisation and decoding in `gliner2` are
Python and do not.

Usage:
    .venv-pytorch/bin/python scripts/bench_pytorch_gliner2.py --device cpu -o out.json
    .venv-pytorch/bin/python scripts/bench_pytorch_gliner2.py --device cpu --clients 1,2,4 -o out.json
"""

import argparse
import json
import math
import platform
import sys
import threading
import time

T_IMPORT = time.perf_counter()
import psutil  # noqa: E402
import torch  # noqa: E402
import gliner2  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402

IMPORT_MS = (time.perf_counter() - T_IMPORT) * 1000.0

LABELS = [
    "person",
    "email",
    "phone_number",
    "address",
    "date_of_birth",
    "national_id_number",
    "passport_number",
    "drivers_license_number",
    "card_number",
    "bank_account",
    "iban",
    "ip_address",
    "username",
    "api_key",
    "access_token",
    "password",
    "secret",
]

FRAGMENTS = [
    "Please contact John Smith at john.smith@example.com regarding the invoice.",
    "The customer called from +1-415-555-0198 last Tuesday afternoon.",
    "DATABASE_URL=postgres://svcuser:hunter2xyz@db-prod.internal:5432/orders",
    "2026-08-08T10:22:31Z user=maria.garcia@example.es ip=203.0.113.44 action=login",
    "Свяжитесь с Михаил Ястребцов для уточнения деталей заказа.",
    "Card on file ends 4111 1111 1111 1111 and expires next March.",
    "let apiKey = process.env.OPENAI_API_KEY; // do not commit this value",
    "Passport X1234567 was issued in Lisbon and expires 2031-04-02.",
    "Jörg Müller updated the shipping address to 12 Hauptstrasse, Munich.",
    "SELECT id, email FROM customers WHERE created_at > '2026-01-01';",
]


def synthetic_document(target_words: int) -> str:
    out, words, i = "", 0, 0
    while words < target_words:
        f = FRAGMENTS[i % len(FRAGMENTS)]
        out = f if not out else out + " " + f
        words += len(f.split())
        i += 1
    return out


def percentiles(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)

    def at(q: float) -> float:
        rank = max(1, math.ceil(q * n))
        return s[min(rank, n) - 1]

    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    return {
        "min_ms": s[0],
        "p50_ms": at(0.50),
        "p90_ms": at(0.90),
        "p95_ms": at(0.95),
        "p99_ms": at(0.99),
        "max_ms": s[-1],
        "mean_ms": mean,
        "stddev_ms": math.sqrt(var),
        "samples": n,
    }


def rss() -> dict:
    b = psutil.Process().memory_info().rss
    return {"rss_bytes": b, "rss_mb": b // (1024 * 1024)}


class FifoLock:
    """A lock that admits waiters in arrival order.

    `threading.Lock` makes no fairness promise, so a thread can win it back
    repeatedly and push another client's tail latency far past what a real
    request queue would show. Same policy as the pool queue in
    src/perf/concurrency.rs.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._next = 0
        self._serving = 0

    def acquire(self) -> None:
        with self._cond:
            ticket = self._next
            self._next += 1
            while ticket != self._serving:
                self._cond.wait()

    def release(self) -> None:
        with self._cond:
            self._serving += 1
            self._cond.notify_all()


def run_level(model, detect, clients: int, per_client: int, text: str):
    """`clients` closed-loop threads on one shared model; wall ms and samples."""
    start = threading.Barrier(clients + 1)
    results: list[list[float]] = [[] for _ in range(clients)]
    errors: list[BaseException] = []

    def client(i: int) -> None:
        start.wait()
        try:
            for _ in range(per_client):
                t = time.perf_counter()
                detect(model, text)
                results[i].append((time.perf_counter() - t) * 1000.0)
        except BaseException as e:  # surfaced after join, not swallowed
            errors.append(e)

    with torch.inference_mode():
        threads = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
        for th in threads:
            th.start()
        start.wait()
        t0 = time.perf_counter()
        for th in threads:
            th.join()
        wall_ms = (time.perf_counter() - t0) * 1000.0
    if errors:
        raise errors[0]
    return wall_ms, [x for r in results for x in r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--buckets", default="32,128,512,1024")
    ap.add_argument("--model-dir", default="models/gliner2-pii-pytorch")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--clients", help="concurrency mode: client counts, e.g. 1,2,3")
    ap.add_argument("--docs", default="32,128", help="concurrency mode: request sizes")
    ap.add_argument("--requests-per-client", type=int, default=20)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    # Same intra-op count as the Rust runs; inter-op 1 for the same reason as
    # `ThreadConfig::default` — one sequential graph, a second pool only adds
    # contention. Set on MPS too: the host-side ops still run on the CPU.
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)

    if a.device == "mps":
        sync = torch.mps.synchronize
    else:
        def sync() -> None:
            pass

    # torch's MPS backend is not safe for concurrent use from several threads:
    # two threads encoding into one Metal command buffer abort the process
    # (`failed assertion _status < MTLCommandBufferStatusCommitted`). A real
    # service on MPS has to serialise calls, so the benchmark does too, and
    # says so in its output rather than measuring something that cannot ship.
    serialised = a.device == "mps"
    lock = FifoLock()

    def detect(model, text: str) -> dict:
        if serialised:
            lock.acquire()
        try:
            r = model.extract_entities(text, LABELS, threshold=a.threshold)
            # Results are host Python objects, so the GPU work has been read
            # back already; synchronising keeps the clock honest if that changes.
            sync()
        finally:
            if serialised:
                lock.release()
        return r

    warmup_text = synthetic_document(32)

    with torch.inference_mode():
        t0 = time.perf_counter()
        model = AutoExtractor.from_pretrained(a.model_dir, map_location=a.device)
        model.eval()
        sync()
        model_load_ms = (time.perf_counter() - t0) * 1000.0
        rss_after_load = rss()

        t1 = time.perf_counter()
        detect(model, warmup_text)
        first_inference_ms = (time.perf_counter() - t1) * 1000.0

        concurrency = []
        if a.clients:
            for n in sorted({int(c) for c in a.clients.split(",") if int(c) > 0}):
                for target in [int(d) for d in a.docs.split(",")]:
                    text = synthetic_document(target)
                    run_level(model, detect, n, 1, text)  # untimed round
                    wall_ms, samples = run_level(model, detect, n, a.requests_per_client, text)
                    concurrency.append(
                        {
                            "target_tokens": target,
                            "actual_words": len(text.split()),
                            "actual_bytes": len(text.encode("utf-8")),
                            "clients": n,
                            "shared": True,
                            "serialised": serialised,
                            "instances": 1,
                            "threads_per_instance": a.threads,
                            "requests": len(samples),
                            "wall_ms": wall_ms,
                            "requests_per_sec": len(samples) * 1000.0 / wall_ms,
                            "latency": percentiles(samples),
                            "rss": rss(),
                        }
                    )
                    c = concurrency[-1]
                    print(
                        f"  {target:>5} tok  {n:>2} clients  {c['requests_per_sec']:6.2f} req/s"
                        f"  p50 {c['latency']['p50_ms']:8.1f} ms  p95 {c['latency']['p95_ms']:8.1f} ms",
                        file=sys.stderr,
                    )

        buckets = []
        for target in [int(b) for b in a.buckets.split(",")] if not a.clients else []:
            text = synthetic_document(target)
            for _ in range(a.warmup):
                detect(model, text)
            samples = []
            found = None
            for _ in range(a.iterations):
                t = time.perf_counter()
                found = detect(model, text)
                samples.append((time.perf_counter() - t) * 1000.0)
            p = percentiles(samples)
            buckets.append(
                {
                    "target_tokens": target,
                    "actual_words": len(text.split()),
                    "actual_bytes": len(text.encode("utf-8")),
                    "iterations": a.iterations,
                    **p,
                    "docs_per_sec_p50": 1000.0 / p["p50_ms"] if p["p50_ms"] > 0 else 0.0,
                    # Sanity, not quality: the same document should yield
                    # roughly as many entities here as under the Rust runtimes.
                    "entities_found": sum(len(v) for v in found["entities"].values()),
                }
            )
            print(
                f"  {target:>5} tok  p50 {p['p50_ms']:9.1f} ms  p95 {p['p95_ms']:9.1f} ms",
                file=sys.stderr,
            )

        warm_rss = rss()

    report = {
        "performance": {
            "detector": f"gliner2-pii-pytorch-{a.device}",
            "runtime": {
                "provider": f"pytorch-{a.device}",
                "intra_op_threads": torch.get_num_threads(),
                "inter_op_threads": torch.get_num_interop_threads(),
                "graph_optimization": "eager",
            },
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gliner2": getattr(gliner2, "__version__", None),
            },
            "cold_start": {
                "import_ms": IMPORT_MS,
                "model_load_ms": model_load_ms,
                "first_inference_ms": first_inference_ms,
                "total_ms": model_load_ms + first_inference_ms,
                "rss_after_load": rss_after_load,
            },
            "buckets": buckets,
            "concurrency": concurrency,
            "warm_rss": warm_rss,
            "iterations": a.iterations,
            "warmup": a.warmup,
            "threshold": a.threshold,
            "labels": LABELS,
        }
    }
    with open(a.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"json written to {a.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
