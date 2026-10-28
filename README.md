# dlp-bench

A reproducible, Rust-only benchmark for **local PII detection models** intended for endpoint DLP
agents and AI gateways.

> **[REPORT.md](REPORT.md)** — conclusions and recommended product direction. Start there if you
> want the decision rather than the methodology.

## 1. Purpose

We need to choose a detector that can sit in front of ChatGPT, Claude, Gemini, Cursor or Claude
Code and inspect a prompt *before* it leaves the machine. That decision needs answers to seven
questions, and this benchmark is shaped around them:

1. Will the detector **miss** sensitive information?
2. How often will it **block harmless prompts**?
3. Can the sensitive span be **reliably redacted**?
4. Is it fast enough to run **synchronously** before sending a prompt?
5. Is the **memory footprint** acceptable for an endpoint agent?
6. Can it run **entirely locally**?
7. How does it behave on **source code, logs, JSON, YAML and configs** — not just prose?

Every detector is compared under the same canonical taxonomy, the same input datasets, the same
entity-span representation, the same evaluator, the same runtime and the same performance
methodology. Adding a model must not require touching the evaluator.

## 2. Architecture

```text
entity.rs / span.rs / labels.rs     canonical taxonomy, UTF-8 byte spans, label mappings
        |
detector.rs — trait Detector        the only abstraction the evaluator sees
        |
detectors/
  onnx.rs                           ALL `ort` types; one runtime, one thread pool,
                                    one optimisation level for every model
  words.rs                          Word (byte offsets) + splitter plumbing
  candle/                           ALL `candle_*` types; gliner2-pii rebuilt from its ONNX file
    onnx_weights.rs  initializers -> named tensors (fp16 widened, int8 kept)
    deberta.rs       mdeberta-v3 encoder: unmasked, fused disentangled attention on CPU
    gliner2.rs       span/count/GRU head on top, same outputs as the graph
  deterministic/                    pattern -> validator -> Detection
    recognizer.rs, validators.rs, patterns/{contact,financial,secrets}.rs
  token_class/                      generic BIO token classifiers
    mod.rs      TokenClassModel descriptor table (3 models), chunking
    decode.rs   BIO aggregation, adjacent-span merge, trimming
  gliner1/                          GLiNER1 zero-shot span, token_level mode
    prompt.rs   <<ENT>>/<<SEP>> prompt + words_mask
    decode.rs   start/end/inside decoding, flat suppression
  gliner2/                          Fastino GLiNER2 PII, fused single graph
    words.rs    text fixup + the reference's URL-aware splitter
    prompt.rs   schema prefix + per-document encoder input
    decode.rs   span extraction, count gate, per-label suppression
        |
dataset/  -> eval/  -> report.rs    quality
dataset/  -> perf/  -> report.rs    latency / memory
```

Detectors depend on nothing in `eval/` or `perf/`. They are the implementations we would ship in
an agent; the benchmark is a consumer of them, not the other way round.

`cargo build --no-default-features` drops `ort` and `tokenizers` entirely and still builds the
deterministic detector and the full evaluator — useful for CI without a 446 MB download.

## 3. Supported detectors

| id | what it is | status |
|---|---|---|
| `deterministic` | Regex candidates + structural validators (Luhn, IBAN mod-97, `Ipv4Addr::from_str`, JWT header decode, PEM markers, vendor key prefixes) | Phase 1, complete |
| `gliner2-pii` | `fastino/gliner2-privacy-filter-PII-multi` via a fused single-graph ONNX export, CPU, no Python | Phase 1, complete |
| `minilm-l6-pii` | MiniLM-L6 BIO token classifier (6 layers x 384) | Phase 2, complete |
| `ai4privacy-mdeberta` | ai4privacy/OpenPII BIO classifier on mdeberta-v3-base | Phase 2, complete |
| `deberta-xsmall-pii` | DeBERTa-v3-xsmall BIO classifier, 9 types | Phase 2, complete |
| `gliner1-pii-edge` / `-small` | GLiNER1 zero-shot span, ettin-encoder-32m / -68m, fp32 & int8 | Phase 2, complete |
| `nvidia-gliner-pii` | `nvidia/gliner-PII` | not started |
| `hybrid` | deterministic ∪ best ML, with span merging and conflict resolution | Phase 2 |

## 4. Canonical taxonomy

Twenty types in `src/entity.rs`, serialised as stable `SCREAMING_SNAKE_CASE`:

```
PERSON EMAIL PHONE ADDRESS DATE_OF_BIRTH
NATIONAL_ID PASSPORT DRIVERS_LICENSE
CREDIT_CARD BANK_ACCOUNT IBAN
IP_ADDRESS USERNAME
API_KEY ACCESS_TOKEN PASSWORD SECRET
URL PRIVATE_KEY CONNECTION_STRING
```

Model labels are **never** compared directly. Each model gets an explicit table in `src/labels.rs`
mapping its vocabulary to this taxonomy; an unmapped label is counted and reported, never silently
dropped or coerced into a neighbouring type.

The default GLiNER2 schema deliberately requests **one label per canonical type** (17 labels). The
model advertises 42, but several are overlapping views of the same span — `person` / `full_name` /
`first_name` / `last_name`, `card_number` / `payment_card`, `bank_account` / `account_number`.
Requesting all of them without a normalisation policy makes the model emit N predictions over one
gold span, which the one-to-one matcher then scores as 1 TP and N−1 FP: a precision artefact of the
schema, not a property of the model. The full 42-label schema is available as `--schema full` for
the latency-vs-schema-size measurement, where quality is not being scored.

## 5. Obtaining the model files

```bash
bash scripts/fetch-models.sh
```

Downloads by **pinned commit revision** with sha256 and size verification into `models/`
(gitignored). Nothing is downloaded implicitly during a benchmark run; a missing artifact produces
a startup error naming this script. Every detector re-verifies its weight file's sha256 at load
unless `--skip-checksum` is passed.

Fetch one model instead of all nine weight files (~2.2 GB total):

```bash
bash scripts/fetch-models.sh minilm-l6-pii
```

The GLiNER2 artifact is `yethdev/gliner2-pii-onnx` @ `7d82c702…`, 446,391,629 bytes, sha256
`489edc52…`. See §12 for what we do and do not know about its provenance.

## 6. Preparing datasets

Internal format is JSONL with **UTF-8 byte offsets**:

```json
{"id":"nl-001","source":"custom","language":"en",
 "text":"Please contact John Smith at john@example.com.",
 "entities":[{"type":"PERSON","start":15,"end":25},{"type":"EMAIL","start":29,"end":45}],
 "tags":["prose"]}
```

Loading rejects duplicate ids, invalid or non-char-boundary spans, overlapping gold entities and
unknown labels, each with the line number and example id. Ordering is by id, so two runs iterate
identically.

Shipped fixtures, regenerated by `python3 scripts/gen_fixtures.py` (a one-off authoring tool; it
computes offsets from the text so annotations cannot drift — it is not part of the runtime):

- `datasets/dlp-v1.jsonl` — 47 examples, 50 entities, **18 negatives**. Held-out test split.
- `datasets/dlp-dev.jsonl` — 17 examples, 21 entities, 6 negatives. Calibration split.

Both cover prose, multilingual prose (ru/de/es/fr/zh/pl/ja), source code, config, JSON, YAML, logs,
SQL and AI/coding prompts. Tags slice the metrics by content type.

Negative examples are mandatory and make up ~38% of the test split: GitHub issue numbers, version
strings, UUIDs, SHA-256 hashes, `127.0.0.1:8080`, port ranges, integer arrays, build ids, colour
tokens, documentation placeholders and public company names.

`--adversarial` expands a dataset with seeded, deterministic transformations (email obfuscation,
phone/card separator variants, case shifts, Markdown/JSON/key-value wrapping, line breaks). Every
transformation rebuilds the document and its annotations together, so a transformed example is as
trustworthy as the original.

Third-party corpora live behind importers in `src/dataset/import/`, which are the only place
allowed to know a foreign schema or offset convention:

- `datasets/presidio-synth.jsonl` — `presidio-research` synth_dataset_v2 (MIT), 1342 examples,
  2236 entities. Independent English prose. Fetch with `scripts/fetch-datasets.sh`, convert with
  `dlp-bench import presidio`.
- `datasets/dlp-multilingual.jsonl` — 43 examples, 18 languages, 7 scripts.

Results from each are reported **separately**, never pooled: they ask different questions, and
several candidate models have training-data proximity to the public ones.

## 7. Running a quality evaluation

```bash
cargo build --release

./target/release/dlp-bench evaluate \
  --detector deterministic \
  --dataset datasets/dlp-v1.jsonl \
  --output out/deterministic-v1.json

./target/release/dlp-bench evaluate \
  --detector gliner2-pii \
  --model-dir models/gliner2-pii-onnx \
  --dataset datasets/dlp-v1.jsonl \
  --threshold 0.5 \
  --show-errors 100 \
  --output out/gliner2-v1.json
```

Useful flags: `--restrict-to-supported` (score only the types the schema can produce),
`--adversarial`, `--schema minimal|default|full`, `--ip-policy ignore|report`.

## 8. Running a threshold sweep

```bash
./target/release/dlp-bench sweep \
  --detector gliner2-pii \
  --dataset datasets/dlp-dev.jsonl \
  --threshold-start 0.10 --threshold-end 0.95 --threshold-step 0.05 \
  --output out/sweep-dev.json
```

Each point reports precision, recall, F1 (exact and overlap), macro F1, critical recall, sensitive
character recall and false positives per 1,000 documents. `--csv` also writes a flat table.

**Calibrate on `dlp-dev`, report on `dlp-v1`.** The sweep records the dataset role and emits a
`calibration_warning` into the JSON when run against a `test` set. It does not refuse — sometimes
you want the test-set curve for a report — but an accidental leak is visible rather than silent.

## 9. Running a performance benchmark

```bash
./target/release/dlp-bench perf \
  --detector gliner2-pii \
  --model-dir models/gliner2-pii-onnx \
  --iterations 30 \
  --thread-counts 1,2,4,8 \
  --output out/perf-gliner2.json
```

Release builds only; a debug build is flagged in the summary as unciteable.

**Methodology.** Only `text -> tokenize -> inference -> decode -> Vec<Detection>` is timed. Dataset
parsing, JSON serialization, model loading and logging are outside the timed region. Synthetic
documents are built before the clock starts.

- **Cold start** is measured in-process: model/session construction and the first inference are
  timed separately and reported separately. Process launch and dynamic linking are excluded —
  they are identical for every detector and would only add noise. If you need a true
  process-launch figure, time the binary externally; the in-process number is the reproducible
  approximation and is labelled as such.
- **Warm latency** uses buckets of ~32/128/512/1024 whitespace tokens, with warm-up iterations
  discarded, and reports p50/p90/p95/p99 plus min/max, mean and standard deviation. Never a bare
  average.
- **Throughput** builds one independent detector per worker, each pinned to a **single** intra-op
  thread. Without that pin, N workers each open a session with `min(cores, 8)` intra-op threads,
  oversubscribe the machine several times over, and the resulting numbers describe scheduler
  thrash rather than the detector. The per-worker RSS cost of N sessions is reported alongside.
- **Memory** is behind `perf/memory.rs`. RSS after load and warm RSS are the model's; **peak RSS is
  process-wide** and includes the harness and every concurrently loaded session, so it is an upper
  bound, not the model's footprint. Where a figure is unavailable the report says `null` rather
  than guessing.

### Execution providers, and why the Mac GPU is not used by default

ONNX Runtime has **no Metal execution provider**. On Apple silicon the only route to the GPU and
Neural Engine is the **CoreML EP**, which partitions the graph itself and decides per subgraph
whether to run on ANE, GPU or CPU — silently handing anything it cannot compile back to the CPU EP.
"Did it use the GPU?" is therefore not answerable from the outside; only the measurement is.

It is wired and opt-in:

```bash
./target/release/dlp-bench perf --detector gliner2-pii --ep coreml       # ANE + GPU + CPU
./target/release/dlp-bench perf --detector gliner2-pii --ep coreml-ane   # ANE + CPU, no GPU
```

**Measured on this machine (Apple M1 Pro, 10 cores), CoreML is a regression:**

| ~tokens | CPU p50 | CoreML p50 | CoreML-ANE p50 |
|---:|---:|---:|---:|
| 32 | **140 ms** | 230 ms | 229 ms |
| 128 | **379 ms** | 663 ms | 611 ms |
| model load | **1721 ms** | 3931 ms | 2671 ms |
| first inference | **293 ms** | 845 ms | 467 ms |

So **CPU remains the default**, and that is a measured decision rather than an unexamined one.

Two findings worth recording, because both cost real time to discover:

1. **`ModelFormat::MLProgram` and `with_subgraphs(true)` break this graph.** Inference fails at
   runtime with `Attempting to broadcast an axis by a dimension other than 1. 6 by 768` from a
   `Mul` node — CoreML mis-partitions it. The configuration that actually runs is the legacy
   `ModelFormat::NeuralNetwork` with subgraph compilation **off**. This is counter-intuitive:
   MLProgram is the newer format and supports more operators in general.
2. **Why CoreML loses.** The export is weight-quantised with fp32 compute — fp16 MatMul weights
   cast up at load, an int8 embedding table dequantised after `Gather` — and every shape in the
   graph is dynamic (`seq_len`, `num_words`, `num_labels`). CoreML compiles almost none of that, so
   it splits the graph into many small subgraphs, pays a conversion cost at each boundary, and the
   heavy matrix multiplies still land on the CPU. The overhead is the whole story.

A CoreML win would most likely require a differently exported model: static or bucketed shapes and
plain fp32 or fp16 weights rather than the cast-and-dequantise pattern. That is a model-export
question, not a runtime one, and belongs to whoever owns the export.

Adding CUDA, DirectML or OpenVINO later is one more arm in `providers()` in
`detectors/gliner2/graph.rs` plus one `ort` feature; nothing else in the crate is affected.

### gliner2-pii on three runtimes: ONNX Runtime, Candle, PyTorch

`gliner2-pii` also runs on [Candle](https://github.com/huggingface/candle) (`--backend candle
--device cpu|metal`), rebuilt from the **same ONNX file** ONNX Runtime loads: fp16 weights are
widened exactly, the int8 word table is dequantized per row after the gather as the graph does,
and the GLiNER2 head (span marker, count gate, one GRU step, projector) is reimplemented from the
export's nodes. Tokenisation, prompt and decoding are shared with the ORT path, and
`candle_reproduces_the_onnx_graph_outputs` holds every `span_probs` cell within 1e-3 of ORT.
Candle on Metal needs `--features metal`; `--features accelerate` routes Candle's CPU matmuls
through Apple Accelerate.

The reference is the upstream PyTorch engine (`gliner2` 2.1.0, torch 2.14.1, Python 3.14) on
the original fp32 checkpoint, `scripts/fetch-models.sh gliner2-pii-pytorch`. Its tokenizer is
byte-identical to the export's, and on the same text it returns the same spans with the same
scores as ORT and Candle.

```bash
bash scripts/bench-runtimes.sh          # six configurations, strictly sequential
python3 scripts/collate-runtimes.py
```

Apple M5 Pro, 48 GB. Same synthetic documents (the collator refuses buckets whose bytes
differ), same 17-label schema, threshold 0.5, 8 intra-op threads, 30 iterations after 3
warm-ups. Timed region is text → entities, tokenisation and decoding included. p50 latency, with
speed-up against ONNX Runtime CPU in brackets:

| runtime | load | 1st call | 32 tok (42 w) | 128 tok (132 w) | 512 tok (522 w) | 1024 tok (1031 w) |
|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime, CPU | 535 ms | 59 ms | 60 ms | 167 ms | 853 ms | 2412 ms |
| Candle, CPU (gemm) | 412 ms | 378 ms | 379 ms (0.16×) | 1052 ms (0.16×) | 7682 ms (0.11×) | 25490 ms (0.09×) |
| Candle, CPU (Accelerate) | 437 ms | 270 ms | 235 ms (0.26×) | 791 ms (0.21×) | 6582 ms (0.13×) | 22741 ms (0.11×) |
| Candle, Metal | 420 ms | 148 ms | 41 ms (1.49×) | 121 ms (1.38×) | 1129 ms (0.76×) | 3857 ms (0.63×) |
| PyTorch, CPU | 4157 ms | 121 ms | 79 ms (0.77×) | 153 ms (1.09×) | 739 ms (1.15×) | 2130 ms (1.13×) |
| **PyTorch, MPS** | 4010 ms | 287 ms | **27 ms (2.22×)** | **78 ms (2.12×)** | **713 ms (1.20×)** | 2554 ms (0.94×) |

The Candle rows were measured with `candle-transformers`' DeBERTa-v2, before the encoder
rewrite described in the next section; the current CPU numbers are there.

p95 is within 5% of p50 everywhere. PyTorch's load excludes a further ~2.2–2.4 s of
`import torch`. Rust loads ran with `--skip-checksum`, because the PyTorch loader does not hash
its weights.

How to read it:

- **PyTorch MPS is the fastest configuration for prompt-sized input**, about 2.2× ORT on CPU up
  to 128 tokens. Past 512 tokens its advantage fades, and at 1024 it is slightly behind.
- **Candle on Metal beats ORT only on short input** (1.4–1.5×) and stays ~1.5× behind PyTorch
  MPS throughout. The gap widens with length, which points at Candle's DeBERTa-v2
  disentangled-attention ops rather than its matmuls.
- **Candle on CPU is not competitive**, 4–11× slower than ORT. Accelerate helps only marginally,
  so the matmuls are not the bottleneck there either.
- **PyTorch on CPU is level with ORT** (0.8–1.15×), but pays ~4 s to load plus ~2.3 s to import
  — about 12× ORT's cold start.

Caveats. ORT and Candle run the community fp16/int8-compressed export, and PyTorch runs the
original fp32 weights. Compute is fp32 in all three, so speed is comparable, but the weights are
not bit-identical. Accelerate manages its own threads and ignores `--threads`. On Metal/MPS the
memory is unified, so process RSS does not show GPU buffers, and the "RSS" columns in
`out/runtimes/*.json` are not comparable across CPU and GPU rows.

### gliner2-pii on CPU under concurrent clients

The product runs on Linux VMs, so this pass is **CPU only**. It is built without the macOS-only
paths: Candle is the pure-Rust `gemm` build, with neither `accelerate` nor `metal`.

```bash
bash scripts/bench-cpu.sh out/cpu-macos       # latency, then N = 1..10 clients
python3 scripts/collate-runtimes.py out/cpu-macos
```

`dlp-bench concurrency` runs N closed-loop clients, released together by a barrier. Latency is
measured as the client sees it, including any wait for a free model. The service has a fixed
budget of 8 threads:

- **ONNX Runtime** gets a pool of `min(N, 2)` sessions, with the budget split between them
  (1 × 8 threads, then 2 × 4). `ort` forbids concurrent runs on one session, and holding more
  than two ~1.5 GB copies was ruled out. Waiting clients are served first come, first served.
- **Candle** serves every client from one shared model with no lock: its forward pass is
  `&self`. Its DeBERTa encoder is this crate's own (`detectors/candle/deberta.rs`, see below).
- **PyTorch** uses one shared model and N Python threads. torch releases the GIL inside its
  kernels.

Apple M5 Pro. Single request, p50 (speed-up against ORT):

| | 32 tok | 128 tok | 512 tok |
|---|---:|---:|---:|
| ONNX Runtime | 60 ms | 148 ms | 824 ms |
| Candle (own encoder) | 122 ms (0.49×) | 267 ms (0.55×) | 1167 ms (0.71×) |
| PyTorch | 81 ms (0.74×) | 159 ms (0.93×) | 729 ms (1.13×) |

Under load, 20 timed requests per client per level. Each cell is req/s | p95 ms:

| 32 tok (42 words) | N=1 | N=2 | N=4 | N=6 | N=8 | N=10 |
|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime (pool ≤ 2) | 14.1 \| 74 | 21.4 \| 104 | 22.0 \| 192 | 21.9 \| 285 | 21.7 \| 382 | 22.4 \| 472 |
| Candle (shared) | 7.9 \| 131 | 8.4 \| 359 | 8.7 \| 1442 | 8.5 \| 2204 | 8.8 \| 2652 | 8.8 \| 3422 |
| PyTorch (shared) | 12.4 \| 82 | 16.5 \| 129 | 25.1 \| 199 | 27.5 \| 277 | 26.3 \| 317 | 27.0 \| 397 |

| 128 tok (132 words) | N=1 | N=2 | N=4 | N=6 | N=8 | N=10 |
|---|---:|---:|---:|---:|---:|---:|
| ONNX Runtime (pool ≤ 2) | 6.1 \| 171 | 8.8 \| 241 | 8.7 \| 479 | 8.8 \| 711 | 8.8 \| 948 | 8.4 \| 1378 |
| Candle (shared) | 3.7 \| 279 | 3.7 \| 864 | 3.8 \| 2952 | 3.8 \| 3993 | 3.9 \| 5328 | 3.9 \| 7937 |
| PyTorch (shared) | 6.4 \| 164 | 9.0 \| 229 | 12.2 \| 401 | 10.7 \| 580 | 12.7 \| 655 | 12.9 \| 816 |

RSS across the levels: ORT 1.8 → 3.0 GB (one session, then two), PyTorch 1.9–2.8 GB, Candle
1.3–1.7 GB.

- **ORT stops scaling at N=2**, at ~22 req/s (32 tok) and ~8.8 req/s (128 tok). Its throughput is
  set by the two-session cap, so beyond that every extra client adds a full queue slot to the
  latency.
- **PyTorch on one model keeps scaling to N≈4–6** and plateaus 1.2–1.5× above capped ORT, with a
  lower p95 at high N.
- **Candle gains nothing from concurrency.** One request already occupies the whole 8-thread
  pool, so throughput is flat at 1/latency (~8.7 and ~3.8 req/s), about 0.4× ORT. Concurrent
  forward passes then interleave on that pool in no order, so the tail is far worse than a queue
  would give (p95 7.9 s at N=10, 128 tok). Serialising requests through a FIFO would keep the
  same throughput and bound p95 near N × the single-call latency.

**Candle's encoder.** `candle-transformers`' DeBERTa-v2 spent 54% of a CPU forward pass in
single-threaded `where_cond` inside `XSoftmax`, masking with an all-ones mask, plus 13% building
that mask, 10% in gathers, 7% in single-threaded `erf`, and 2.6% in matmuls.
`detectors/candle/deberta.rs` replaces it with the same arithmetic:

- no mask on unpadded input;
- relative-position projections computed once at load, with the scale folded in;
- Q, K and V in one matmul;
- on CPU, one parallel pass doing both gathers, the add and the softmax, plus a parallel GELU
  that is bit-identical to `gelu_erf`;
- Candle's `gemm` split matched to `--threads`.

That is 3–6× faster than before (362 → 122 ms at 32 tokens, 7.7 → 1.2 s at 512).
`candle_reproduces_the_onnx_graph_outputs` still holds every `span_probs` cell within 1e-3 of
ONNX Runtime, now including a ~1100-token document that exercises log-bucketed and clamped
positions. 80% of the remaining time is in the `gemm` crate's matmul kernels, so the gap to ORT
is now a matmul-kernel gap: ORT uses its own MLAS kernels. On x86 Linux, Candle's `mkl` feature
is the next lever.

**Transfer to a Linux VM.** Relative order is the point of this table, and one cell is
flattered. PyTorch on macOS links Apple Accelerate for BLAS, which runs on the CPU's AMX/SME matrix
unit. A Linux VM has no such unit: torch there uses MKL/oneDNN (x86) or OpenBLAS (ARM). Expect
PyTorch's lead over ORT to shrink or reverse on the VM. ORT (MLAS) and Candle (`gemm`) use their
own kernels on both systems. A process-level check confirms the PyTorch-CPU run never touched the
GPU: every parameter is on `cpu`, the MPS allocator holds 0 bytes, and the process kept ~2 cores
busy.

## 10. Interpreting the metrics

**High recall matters most for DLP.** A missed credit card is an incident. Recall is reported
overall, per entity, and separately for a configurable **critical group** (`NATIONAL_ID`,
`PASSPORT`, `DRIVERS_LICENSE`, `CREDIT_CARD`, `BANK_ACCOUNT`, `IBAN`, `API_KEY`, `ACCESS_TOKEN`,
`PASSWORD`, `SECRET`, `PRIVATE_KEY`, `CONNECTION_STRING`) — because macro F1 still lets a
catastrophic `PASSPORT` recall hide behind an easy `EMAIL`.

**False positives still matter.** A tool that blocks legitimate AI use gets switched off, and then
its recall is zero. `false_positives_per_1k_docs` and `clean_negative_docs` are the usability
numbers; the negative examples exist to produce them.

**Exact F1 and character-level leakage answer different questions.** Gold `john.smith@example.com`,
predicted `smith@example.com` is an exact-entity *failure* and a 77% redaction *success*. Both
facts are reported: `exact` vs `overlap` F1, and `sensitive_char_recall` /`exposed_fraction`
(type-agnostic, because a mislabelled span still gets masked) alongside `over_masked_bytes` (the
benign text a redactor would destroy).

**Macro vs micro.** Micro is dominated by frequent classes. Macro tells you whether the detector is
uniformly competent. Both are reported for both match modes.

**Matching is strictly one-to-one.** A prediction satisfies at most one gold span and vice versa,
resolved greedily by descending IoU with a total tie-break on `(start, end)` so the result is
independent of input order. Without this, a detector that emits sloppier, larger spans scores
*better* — the opposite of what a redaction system wants. Overlap requires at least one shared
byte; abutting spans `[0,5)` and `[5,9)` do not overlap.

**Public and synthetic datasets are not substitutes for real AI-DLP prompts.** The shipped fixtures
are hand-authored to look like traffic to a coding assistant. They are small. Treat them as a
correctness harness and a smoke test, not as evidence about production behaviour.

### Measured results, and how to read them

Phase-1 run on `datasets/dlp-v1.jsonl` (47 documents, 50 entities, 18 negatives), Apple M1 Pro,
release build, GLiNER2 at threshold 0.5.

|  | deterministic | gliner2-pii |
|---|---|---|
| overlap micro P / R / F1 | 0.88 / 0.58 / **0.70** | 0.59 / 0.88 / **0.70** |
| macro overlap F1 | 0.67 | **0.80** |
| sensitive-char recall | 0.72 | **0.88** |
| FP per 1k docs | **85** | 660 |
| clean negatives | **18/18** | 10/18 |
| p95 @128 tok · warm RSS · cold start | **0.1 ms · 12 MB · 6 ms** | 425 ms · 545 MB · 2879 ms |

**The identical 0.70 F1 is a coincidence, and reading it as "the model is no better than regex" is
the single most likely misreading of this table.** The two detectors are good at disjoint things:

| entity | support | rules R | gliner R | who can do it at all |
|---|---:|---:|---:|---|
| PERSON | 13 | 0.00 | 0.92 | **only the model** |
| ADDRESS | 1 | 0.00 | 1.00 | **only the model** |
| DATE_OF_BIRTH | 1 | 0.00 | 1.00 | **only the model** |
| PASSPORT | 1 | 0.00 | 1.00 | **only the model** |
| DRIVERS_LICENSE | 1 | 0.00 | 1.00 | **only the model** |
| CONNECTION_STRING | 2 | 1.00 | 0.00 | **only the rules** |
| PRIVATE_KEY | 1 | 1.00 | 0.00 | **only the rules** |
| CREDIT_CARD / IBAN / API_KEY / EMAIL / PHONE | 24 | 0.80–1.00 | 0.75–1.00 | both |

Rules score **0.00 recall on PERSON**, which is 13 of 50 gold entities. Together with ADDRESS,
DATE_OF_BIRTH, PASSPORT and DRIVERS_LICENSE that is **34% of the dataset that pattern matching
cannot reach at any level of effort** — its 0.58 recall has a hard ceiling not far above that.
Conversely the model scores 0.00 on connection strings and PEM keys, which the rules nail exactly.

The precision gap is also less damning than it looks. Splitting GLiNER2's false positives:

- **15 of 28 are mislabels** — the model found the sensitive text and named it wrong
  (`Anna Kowalska` → `USERNAME`, `s3cretpw99@prod-db.internal` → `EMAIL`, an AWS key → `SECRET`).
  Typed F1 punishes these, but **the bytes still get redacted**, which is why its
  sensitive-character recall is 0.88 against the rules' 0.72.
- **13 of 28 are genuine over-detection**, and these are the real usability cost. Sorted by
  confidence they are almost entirely developer-corpus noise: `#ff00aa` / `#00ffcc` / `#1a1a1a`
  as `ACCESS_TOKEN` **at 0.98**, a git SHA at 0.91, a UUID as `IBAN` at 0.86, `@1.2.3` from an npm
  spec as `IP_ADDRESS` at 0.85. **Raising the threshold will not remove these** — they are more
  confident than many true positives. The sweep (on `dlp-dev`) agrees: precision climbs only from
  0.77 at 0.45 to 0.83 at 0.85, and past 0.85 recall starts dropping faster than precision rises.

One of the 13 is a policy disagreement rather than an error: `127.0.0.1` flagged at 1.00, which the
fixtures label as not-PII under the default private-IP policy. The model has no way to know the
policy; the rules take it as configuration.

**Conclusion: neither is a substitute for the other, and the aggregate F1 is the wrong number to
decide on.** Rules give near-perfect precision, microsecond latency and a 12 MB footprint on
everything with structure; the model reaches the semantic entities that structure cannot express.
That is the concrete, measured case for the Phase-2 hybrid — and the false-positive breakdown says
its first job is cross-label conflict resolution, not better recall.

### Does dropping the label prompt actually buy speed?

GLiNER2 pays 77 tokens of schema prefix on every call, re-encoded by all 12
layers. `ai4privacy-mdeberta` is the controlled test of what that costs: **the
same `mdeberta-v3-base` backbone**, same depth and width, differing only in
having a plain 111-class BIO head and no prompt (plus 256-token chunking, which
GLiNER2 does not do).

| ~tokens | rules p50 | gliner2 p50 | ai4privacy p50 | speed-up |
|---:|---:|---:|---:|---:|
| 32 | 0.03 ms | 142 ms | **66 ms** | 2.1× |
| 128 | 0.1 ms | 374 ms | **279 ms** | 1.3× |
| 512 | 0.4 ms | 2014 ms | **1189 ms** | 1.7× |
| 1024 | 0.9 ms | 7159 ms | **2325 ms** | 3.1× |
| cold start | 6 ms | 2879 ms | **1612 ms** | 1.8× |
| warm RSS | 12 MB | 545 MB | 594 MB | — |

**The prompt is not the bottleneck.** Removing it buys 1.3–3.1×, not the
order of magnitude one might expect from a 77-token prefix. And most of the
win at the long end is chunking, not the missing prompt: GLiNER2 runs 1024
words as one O(n²) sequence while this detector runs five 256-token chunks.
At 128 tokens, where chunking does not engage, the gain is only 1.3×.

**The conclusion for the product is therefore unchanged, and firmer: neither
model is viable on the synchronous path.** 287 ms p95 at 128 tokens is still
5.7× over the 50 ms target, and 1.2 s at 512 tokens is not something a user
will wait for. Swapping architectures does not rescue it; the tiered design
(rules synchronously, ML asynchronously or on short input only) is required
either way.

### Three-way quality comparison

`datasets/dlp-v1.jsonl`, threshold 0.5, overlap micro unless stated.

| | deterministic | gliner2-pii | ai4privacy-mdeberta |
|---|---|---|---|
| precision | **0.88** | 0.59 | 0.55 |
| recall | 0.58 | **0.88** | 0.56 |
| F1 | **0.70** | **0.70** | 0.55 |
| macro F1 | 0.67 | **0.80** | 0.60 |
| critical recall | **0.78** | 0.72 | 0.28 |
| sensitive-char recall | 0.72 | **0.88** | 0.57 |
| FP per 1k docs | **85** | 660 | 489 |

**`ai4privacy-mdeberta` is not a better detector for this use case**, and the
reasons are structural rather than incidental:

1. **Its taxonomy has no developer secrets.** No API key, access token, secret,
   private key or connection string — and no passport or driver's licence
   either. That is 10 of 50 gold entities it cannot reach at any threshold, and
   it is why critical recall collapses to 0.28. For AI-DLP, where pasted configs
   and keys are a primary leak channel, this is close to disqualifying.
   `dlp-bench inspect --detector ai4privacy-mdeberta` prints the unreachable
   list so the zero is never mistaken for a model failure.
2. **It is markedly weaker outside English.** Inspecting per-token output shows
   the model tagging `▁Я` as `O` and then `стреб` as `B-LASTNAME`, so
   `Ястребцов` comes back as `стребцов` — the first letter left exposed. It
   misses `Müller` in `Jörg Müller` entirely. Despite the multilingual
   backbone, the ai4privacy v2 training data does not carry it over.
3. **PERSON recall is 0.54 against GLiNER2's 0.92** on the same documents.

It is genuinely better at `PHONE` (1.00/1.00) and ties on `IBAN`, and it is the
cheapest of the two models — but neither compensates for the taxonomy gap.

Two caveats on fairness. This is a *different model*, not just a different
architecture, so the comparison confounds "token classifier vs zero-shot span"
with "ai4privacy v2 training data vs Fastino's". And no ModernBERT PII model
has an ONNX export at all — five candidates checked, all safetensors-only, and
AI4Privacy publish no first-party ModernBERT — so this is a stand-in for the
task's "AI4Privacy ModernBERT" slot rather than that exact model.


### Phase 2: how small can a model be and still work?

Phase 1 ended with nothing viable on the synchronous path, and the ai4privacy
experiment refuted the theory that GLiNER2's label prompt was the bottleneck.
The remaining lever was **model size**. Six more configurations, same dataset,
same evaluator, same runtime:

| detector | MB | load ms | p50@128 | **p95@128** | p95@512 | P | R | F1 | crit | FP/1k | clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | – | 0 | 0.1 | **0.1** | 0.6 | **0.88** | 0.58 | 0.70 | **0.78** | **85** | **18/18** |
| **minilm-l6-pii** | 86 | **119** | 33.6 | **35.6** | 153.7 | 0.70 | 0.70 | 0.70 | 0.28 | 319 | 16/18 |
| gliner1-pii-edge int8 | **44** | 264 | 84.5 | 88.1 | 385 | 0.36 | 0.54 | 0.43 | 0.22 | 1043 | 6/18 |
| gliner1-pii-edge fp32 | 173 | 339 | 182 | 224 | 1072 | 0.27 | 0.60 | 0.37 | 0.28 | 1745 | 2/18 |
| gliner1-pii-small int8 | 79 | 789 | 179 | 229 | 3355 | 0.31 | 0.56 | 0.40 | 0.50 | 1319 | 3/18 |
| gliner1-pii-small fp32 | 312 | 645 | 251 | 358 | 1084 | 0.29 | 0.62 | 0.40 | 0.50 | 1596 | 1/18 |
| deberta-xsmall-pii | 83 | 617 | 97.4 | 99.8 | 444 | 0.13 | 0.38 | 0.19 | 0.00 | 2702 | 2/18 |
| ai4privacy-mdeberta | 1062 | 1434 | 247 | 268 | 1075 | 0.55 | 0.56 | 0.55 | 0.28 | 489 | 9/18 |
| gliner2-pii | 426 | 1323 | 348 | 358 | 2010 | 0.59 | **0.88** | 0.70 | 0.72 | 660 | 10/18 |

Every perf run was executed alone, one process at a time; an earlier batch that
overlapped two of them inflated GLiNER2's numbers by ~2.4x and was discarded.

**MiniLM-L6 is the only model that meets the latency target.** 35.6 ms p95 at
128 tokens against a 50 ms goal, 154 ms at 512 against 150, 119 ms cold start,
256 MB RSS — and it does that at F1 0.70, equal to both GLiNER2 and the rules,
with 16/18 negative documents clean. It is ~10x faster than GLiNER2 for the
same aggregate F1.

The catch is the same one that sank ai4privacy: **critical recall 0.28**. It
maps `api_key`, `password` and `pin` — better than ai4privacy, which has no
credential types at all — but no access token, private key or connection
string, and it is English-only (wordpiece vocab 30522), so it finds the email
in a Russian sentence and not the name.

**Size alone does not buy quality.** The GLiNER1 family and DeBERTa-xsmall are
all small and all substantially worse than the rules on precision, with
1000–2700 false positives per 1000 documents. `deberta-xsmall-pii` splits
`john@example.com` into an EMAIL `john` plus an ADDRESS `@example.com`, tags
`#4411` as an address, and reaches critical recall 0.00.

### What quantization costs

The GLiNER1 checkpoints publish the same weights at three precisions, which
isolates precision from size. Only two are usable — see below.

| | fp32 | int8 | change |
|---|---:|---:|---|
| edge, size | 173 MB | 44 MB | **-75%** |
| edge, p50@128 | 182 ms | 84.5 ms | **-54%** |
| edge, F1 | 0.37 | 0.43 | +0.06 |
| edge, FP/1k | 1745 | 1043 | -40% |
| small, size | 312 MB | 79 MB | -75% |
| small, p50@128 | 251 ms | 179 ms | -29% |
| small, F1 | 0.40 | 0.40 | 0.00 |

int8 is **free or better** here: a quarter of the size, roughly half the
latency, and no quality loss — F1 actually improves slightly on `edge`. That is
consistent with the model card's claim of quantization-aware training rather
than post-training quantization, and it suggests quantizing GLiNER2 (which we
have only ever run at its published int8) would be worth measuring for any
model we do adopt.

**The published fp16 exports are broken.** Both checkpoints fail to load:

```
Type Error: Type (tensor(float16)) of output arg
(/core/token_rep_layer/bert_layer/model/layers.0/attn/rotary_emb/Cast_1_output_0)
of node (...) does not match expected type (tensor(float))
```

A defect in the upstream export — fp32 and int8 from the same repository load
fine. Recorded as data in the model descriptor (`ModelVariant::broken`) so that
`--variant fp16` reports the reason rather than surfacing a raw ONNX Runtime
type error, and so a reader comparing precisions knows why fp16 is absent from
the results rather than assuming it was skipped.

### DataFog could not be run

`DataFog/pii-small-en` was requested as the "small structured NER + CRF"
experiment. It declares `architectures: ["PiiNerModel"], model_type: "pii_ner"`
— a custom class whose card describes a `CharCNN + CRF` head over
DeBERTa-v3-xsmall. It ships safetensors only, contains **no modeling file** (so
`trust_remote_code` would also fail), a CRF's Viterbi decode is not ordinarily
exportable to ONNX, and PyPI is unreachable here, so no export path exists at
all. `megaloxorafitis/pii-ner-deberta-v3-xsmall` was substituted: a real
DeBERTa-v3-xsmall token classifier with a published quantized ONNX. It answers
the size question and **not** the CRF question.

### Where this leaves the design

The tiered architecture is unchanged, but the ML tier now has a plausible
occupant. MiniLM-L6 is fast enough to run synchronously and roughly as accurate
overall as models 5–12x its size; what it cannot do is cover credentials or
non-English text. That maps cleanly onto the tiering already argued for:

* **Tier 0 — rules, synchronous.** Still the only thing with 0.88 precision,
  0.1 ms and 12 MB, and still the only detector that covers connection strings
  and private keys.
* **Tier 1 — MiniLM-L6, synchronous.** 36 ms p95 buys `PERSON`, `ADDRESS` and
  `DATE_OF_BIRTH`, which rules cannot reach at any effort.
* **Tier 2 — GLiNER2, asynchronous.** Still the recall leader (0.88 overlap,
  0.72 critical) and still far too slow to block on.

The open question is no longer "is anything fast enough" but "can a
credential-aware, multilingual model be found at MiniLM-L6's size" — which is a
fine-tuning question, not a benchmark one.


### Multilingual coverage — and why it reverses the recommendation

An endpoint DLP agent has to work in the languages the organisation actually
uses. That requirement disqualifies the Phase-2 winner, and it does so
**structurally rather than empirically**: four of the six models cannot
represent non-Latin text at all.

| model | tokenizer | vocab | multilingual |
|---|---|---:|---|
| gliner2-pii | Unigram (SentencePiece) | 250 101 | **yes** — mdeberta-v3 |
| ai4privacy-mdeberta | Unigram (SentencePiece) | 250 101 | **yes** — mdeberta-v3 |
| minilm-l6-pii | WordPiece | 30 522 | no — English BERT |
| deberta-xsmall-pii | Unigram | 128 000 | no — English DeBERTa-v3 |
| gliner1-pii-edge / -small | ByteLevel BPE | 50 280 | no — English Ettin |

`datasets/dlp-multilingual.jsonl` measures it: 43 documents, 75 entities, **18
languages across 7 scripts** (Latin, Cyrillic, Greek, CJK, Arabic, Hebrew,
Devanagari), each carrying a PERSON plus a universal structured entity so the
two failure modes can be told apart. Recall, overlap micro:

| script | langs | rules | **gliner2** | ai4privacy | minilm-l6 | gliner1-edge | deberta-xsmall |
|---|---|---:|---:|---:|---:|---:|---:|
| Latin | en de fr es it pt nl pl tr | 0.51 | **0.98** | 0.84 | 0.90 | 0.69 | 0.70 |
| Cyrillic | ru uk | 0.50 | **1.00** | 0.90 | 0.40 | 0.19 | 0.50 |
| Greek | el | 0.50 | **1.00** | 0.50 | 0.50 | 0.50 | 1.00 |
| CJK | zh ja ko | 0.50 | 0.75 | **0.85** | 0.34 | **0.06** | 0.70 |
| Arabic | ar | 0.50 | **1.00** | 1.00 | 0.50 | **0.00** | 1.00 |
| Hebrew | he | 0.50 | **1.00** | 1.00 | 0.50 | **0.00** | 1.00 |
| Devanagari | hi | 0.50 | **1.00** | 0.50 | **0.00** | **0.00** | 1.00 |
| **overall** | | 0.51 / P 0.95 | **0.95 / P 0.89** | 0.83 / P 0.83 | 0.68 / P 0.63 | 0.45 / P 0.37 | 0.69 / P 0.39 |

**GLiNER2 is the only ML detector that holds up.** F1 0.92 overall, recall 1.00
in eleven of eighteen languages including Arabic, Hebrew, Korean, Hindi, Greek,
Russian and Ukrainian.

**MiniLM-L6 collapses outside Latin script** — 0.00 on Hindi, 0.25 on Chinese,
0.38 on Russian. Its non-zero scores there are almost entirely the *email*,
which is ASCII in every language; it essentially never finds the name. The
Phase-2 conclusion that it was the best endpoint candidate holds only for an
English-only deployment.

**The deterministic detector is the quiet winner on robustness**: a flat 0.50
recall at 0.95 precision in *every one* of the eighteen languages. It finds the
structured entity and never the person, identically everywhere, because
checksums and grammars do not care what script surrounds them. For a
multilingual product that predictability is worth a great deal.

Two cautions on this table. Support is 1–5 documents per language, so a single
document moves a cell by 0.25–0.50; treat the script-level rows and the overall
column as the signal and individual cells as indicative. And `deberta-xsmall`'s
high non-Latin recall is an artefact of over-tagging — it reaches 1.00 on
Arabic and Hindi at 0.39 precision overall, having tagged whole clauses.

#### Why "small **and** multilingual" is hard

A 250 000-token vocabulary is not free. At hidden size 768 the embedding table
alone is `251000 x 768 x 4 B = 735 MB` in fp32 — **69% of ai4privacy's 1062 MB
ONNX file is the embedding matrix**, before a single transformer layer. MiniLM
gets to 86 MB partly by having 6 layers, but as much by having a 30 522-token
English vocabulary.

So the lever for a small multilingual model is not depth, it is **quantizing
the embedding table**. That is exactly what the GLiNER2 export does — int8
embeddings with fp32 compute — reaching 426 MB on a backbone whose fp32 form is
1.2 GB. Combined with the GLiNER1 finding that int8 costs nothing in quality,
the promising direction is a *more aggressively quantized multilingual* model
rather than a smaller English one.

#### Revised recommendation

With multilingual support required, the tiering collapses back to two usable
components and the fast ML tier is empty:

* **Tier 0 — rules, synchronous, language-agnostic.** 0.1 ms, 12 MB, 0.95
  precision, and the same 0.50 recall in all eighteen languages.
* **Tier 2 — GLiNER2, asynchronous.** The only detector that is both accurate
  and multilingual, and at 358 ms p95 still far too slow to block on.
* **Tier 1 is vacant.** Every model fast enough to run synchronously is
  English-only. Filling it needs a quantized multilingual model in the
  100–200 MB range — a fine-tuning and export question, not a benchmark one.
  `onnx-community/gliner_multi_pii-v1` (mdeberta backbone, published at int8,
  uint8 and q4) is the obvious next thing to measure.


### Independent corpus: presidio-research

The local fixtures are hand-written and small — several canonical types have a
support of 1, which makes their per-entity numbers meaningless. `presidio-research`
(MIT, `data-privacy-stack/presidio-research`) supplies an independent check:
**1 500 documents, 2 863 annotated spans**, pinned by commit and sha256.

```bash
bash scripts/fetch-datasets.sh
cargo run --release -- import presidio          # -> datasets/presidio-synth.jsonl
```

**Its offsets are Unicode character indices, ours are UTF-8 bytes.** Measured,
not assumed: reading them as bytes mis-slices **581 of 2 863 spans**, because
320 sit after non-ASCII text — the templates are English but the generated
values are international (`Krisztián Szöllösy`, `Koskikatu`). The importer
converts through `span::char_to_byte_table` and then **re-verifies every span
against the `entity_value` the corpus ships**, so a convention error is a hard
failure rather than a quietly shifted annotation.

Import is auditable rather than lossy: 1 342 of 1 500 records and 2 236 of
2 863 spans are kept; the six unmapped types are printed with the reason each
was excluded. The 158 dropped records are those whose entities were *all*
unmapped — they are **not** emitted as negatives, because the text does contain
PII of a kind we chose not to score and a detection there is not a false
positive.

Results (overlap micro, 1 342 docs / 2 236 entities / 113 negatives):

| | rules | gliner2 | minilm-l6 | ai4privacy |
|---|---:|---:|---:|---:|
| precision | **0.94** | 0.84 | 0.77 | 0.68 |
| recall | 0.10 | 0.58 | **0.75** | 0.63 |
| F1 | 0.18 | 0.68 | **0.76** | 0.66 |
| macro F1 | 0.62 | **0.84** | 0.77 | 0.77 |
| sensitive-char recall | 0.12 | 0.58 | **0.82** | 0.53 |
| FP / 1k docs | **10** | 186 | 377 | 497 |
| clean negatives | **113/113** | 83/113 | **113/113** | 104/113 |

Per-entity recall, with support — this is what the corpus is *for*:

| entity | support | rules | gliner2 | minilm-l6 | ai4privacy |
|---|---:|---:|---:|---:|---:|
| PERSON | 857 | 0.00 | **0.97** | 0.89 | 0.62 |
| ADDRESS | 1046 | 0.00 | 0.15 | **0.64** | 0.60 |
| CREDIT_CARD | 136 | 0.77 | **1.00** | 0.69 | 0.85 |
| PHONE | 92 | 0.33 | 0.79 | **0.84** | 0.62 |
| EMAIL | 49 | **1.00** | **1.00** | **1.00** | 0.73 |
| IBAN | 21 | 0.95 | **1.00** | 0.00 | **1.00** |
| NATIONAL_ID | 16 | 0.00 | 0.75 | **1.00** | **1.00** |
| IP_ADDRESS | 14 | **1.00** | 0.93 | **1.00** | **1.00** |
| DRIVERS_LICENSE | 5 | 0.00 | **1.00** | 0.00 | 0.00 |

Reading it honestly:

* **85% of the gold is PERSON or ADDRESS**, so the aggregate is close to "can
  you find people and places". The rules score F1 0.18 here for exactly the
  reason they score 0.70 on the AI-DLP set — different corpora ask different
  questions, which is why they are reported separately rather than pooled.
* **MiniLM-L6 leads on this corpus** (F1 0.76). It is English template prose:
  its home ground, and precisely the conditions under which the multilingual
  result above shows it should not be trusted.
* **`minilm-l6`'s IBAN recall of 0.00 is structural** — its 55-type taxonomy
  has no IBAN label. Likewise `DRIVERS_LICENSE`. Zeroes of this kind are a
  property of the training taxonomy and `dlp-bench inspect` lists them.
* **False positives collapse for everyone** relative to the AI-DLP fixtures
  (10–497 vs 85–2702 per 1k). This corpus contains no code, hex, UUIDs or build
  ids, so it does not exercise the false-positive surface that matters most for
  a coding assistant. It measures recall well and usability poorly.

#### The GPE decision moves the result, so it is a flag

411 of the corpus's spans are `GPE` — a bare city or country. Mapping them to
`ADDRESS` (as the ai4privacy table maps `CITY`/`STATE`/`COUNTRY`) is a
judgement call that is **not** cosmetic: it favours models with granular
location labels (`minilm-l6` has `city`, `state`, `country`, `postcode`) over
models with one broad label (GLiNER2 asks only for `address`), and it is the
main reason GLiNER2's ADDRESS recall reads 0.15 against MiniLM's 0.64.

It is therefore a switch rather than a buried constant:

```bash
cargo run --release -- import presidio --exclude-gpe \
  --output datasets/presidio-synth-nogpe.jsonl     # 1215 docs, 1825 entities
```

**Measured, the switch reverses the ranking:**

| | ADDRESS recall (with GPE) | (no GPE) | overall F1 (with) | (no GPE) |
|---|---:|---:|---:|---:|
| gliner2 | 0.15 | 0.24 | 0.68 | **0.78** |
| minilm-l6 | **0.64** | **0.64** | **0.76** | 0.73 |

With bare place names scored, MiniLM leads 0.76 to 0.68. Without them, GLiNER2
leads 0.78 to 0.73. MiniLM is genuinely better at street addresses either way,
but *which model wins the headline number* is decided by an annotation
convention, not by the models. Any single-number comparison on this corpus
should be read with that in mind — it is the clearest illustration in this
benchmark of why a label mapping is a methodological choice and not plumbing.

An initial hypothesis that GLiNER2's 8-word `max_width` cap explained the gap
was **wrong** and is recorded here as such: only 12% of ADDRESS spans exceed 8
words, and 477 of 1 046 are single words — the single-word GPE spans.

### Provisional acceptance targets

Reported as reference annotations only. They never affect a metric's computation and never change a
detector's behaviour.

```
critical recall >= 95%     overall recall >= 90%     precision >= 70%
p95 <= 50 ms @ ~128 tokens    p95 <= 150 ms @ ~512 tokens
warm RSS <= 750 MB            cold start <= 2 s
```

## 11. Known limitations

**The ONNX export is community-produced and unverified.** `yethdev/gliner2-pii-onnx` is an
unaffiliated export of Fastino's model with 0 downloads at the time of writing. Its card claims
span-set parity with the PyTorch engine at thresholds 0.3/0.5/0.7 and a max probability drift of
0.0057 — but it also reports a *higher* SPY F1 than the original it derives from (.535 vs .477),
which a lossy weight-quantised export should not achieve. We treat those numbers as unverified and
report only our own measurements.

**Reference parity could not be established directly.** The plan was to verify prompt construction,
tensor layout and span decoding against the GLiNER2 Python reference using
`uv run --with tokenizers` / `--with onnxruntime`. **PyPI is unreachable from this machine** through
a TLS-intercepting proxy (`invalid peer certificate: UnknownIssuer`), so no Python package can be
installed and no reference output produced.

What replaces it is `tests/gliner2_agreement.rs`, a self-contained oracle that is arguably stronger
for the property that matters. The deterministic detector locates emails, IBANs and credit cards by
regex plus checksum — no tokenizer, no model, no word splitter. GLiNER2 reaches the same spans
through a completely independent path (prepare text → split words → build prompt → ONNX → decode
word-index spans → byte offsets). The tests assert those spans are **byte-identical**. A transposed
axis, an off-by-one in `text_word_indices`, a forgotten trailing `.` or a char/byte confusion would
all break that agreement immediately. Also covered: spans always slice the caller's text, spans
track the entity under padding, thresholds are monotone, inference is deterministic and
thread-safe, and a tampered artifact is refused.

Still unverified: absolute probability agreement with PyTorch. If reference parity becomes
important, generate goldens on a machine with PyPI access and commit them.

**GLiNER2 behaviours the benchmark exposes rather than hides:**

- *Cross-label duplication.* Suppression is per label, matching the reference, so one span can be
  emitted as `API_KEY`, `ACCESS_TOKEN` and `SECRET` simultaneously, or as `EMAIL` and `USERNAME`.
  This is the single largest source of its false positives and the clearest argument for the hybrid
  detector's conflict resolution.
- *Connection strings read as emails.* `postgres://admin:pw@host/db` is reported as an `EMAIL`
  spanning `pw@host`. The deterministic detector gets this exactly right, which is a direct hybrid
  motivation.
- *Quadratic cost with length.* mdeberta uses relative attention with no absolute position table,
  so there is no 512-token limit — long inputs simply cost O(n²). The 1024-token bucket is several
  times the 512-token bucket. Chunking exists in the reference as a separate opt-in API and is not
  applied automatically here; automatic chunking would change the span set silently.
- *The label prompt dominates short-prompt latency.* The 17-label schema is 77 tokens that all 12
  encoder layers reprocess on every call, and it cannot be cached because the encoder is
  bidirectional. `schema_prefix_tokens` is in the perf report for this reason.

**Other limitations:** CPU only (the architecture does not prevent adding execution providers — one
more entry in the array in `graph.rs`). Peak RSS is process-wide. The fixture datasets are small.
`BANK_ACCOUNT` and `URL` have no coverage in the current fixtures. Windows peak-RSS is not
implemented, and reports `null` rather than an approximation.

## 12. Development

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test                          # includes model-dependent tests if models/ is populated
cargo test --no-default-features    # evaluator + deterministic detector only, no ONNX
```

Model-dependent tests **skip with a printed notice** rather than being `#[ignore]`d, so the GLiNER2
suite cannot rot unnoticed while still passing on a machine without the artifacts.

Logging is metadata-oriented: benchmark text is PII by construction and is never logged. Use
`--show-errors` or `dlp-bench inspect --text ...` to see content; both are explicit opt-ins.
