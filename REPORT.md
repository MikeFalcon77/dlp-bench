# AI-DLP prompt filtering — findings and recommended direction

**Status:** conclusions from the `dlp-bench` benchmark, 2026-08.
**Scope:** which local PII detector to ship in front of ChatGPT / Claude / Gemini /
Cursor / Copilot, and what has to be built around it.
**Basis:** 9 detector configurations, 3 independent corpora, one evaluator, one
runtime. All numbers below are measured on an Apple M1 Pro, release build, CPU,
one process at a time. Raw results are in `out/*.json`; methodology is in the
[README](README.md).

---

## Executive summary

1. **No single model does the job.** GLiNER2 has the best recall and is the only
   multilingual option, but it scores **0.00** on connection strings and PEM
   private keys — which the 20-line regex layer gets exactly right. The rules
   score **0.00** on `PERSON`, which is 26% of our AI-DLP gold and unreachable
   by pattern matching at any effort. The two are complements, not competitors.
2. **Large prompts are the binding constraint.** GLiNER2 costs 7.2 seconds on a
   1 031-word document, and the cost per word is still rising at that size.
   Pasting a source file into Cursor is the normal case, not the tail. Any
   design that blocks on a transformer over the whole prompt is dead on arrival.
3. **Classical filtering is mandatory and must come first.** Validated regex is
   0.1 ms, 12 MB, 0.94–0.95 precision, and behaves *identically in all 18
   languages we tested*. It is the only component that can run synchronously on
   every keystroke-to-send path.
4. **The hard part is delivery, not detection.** A detector that cannot see the
   prompt is worth nothing. We need browser extensions across **Chrome, Edge,
   Firefox and Safari** — plus targeted integration into desktop applications
   (Cursor, Claude Code, the ChatGPT and Claude desktop apps) that never touch a
   browser at all. This is the same surface Nightfall covers with "browser
   plugins and endpoint agents" for ChatGPT, Copilot, Gemini, Claude and
   Perplexity.

---

## 1. What was measured

Three corpora, deliberately kept separate because they answer different
questions and pooling them would hide exactly the failures we care about.

| corpus | size | what it answers |
|---|---|---|
| `dlp-v1` | 47 docs, 50 entities, 18 negatives | AI-DLP reality: code, config, JSON, YAML, logs, SQL, coding prompts, and developer secrets |
| `dlp-multilingual` | 43 docs, 75 entities | 18 languages across 7 scripts |
| `presidio-synth` | 1 342 docs, 2 236 entities | independent third-party check (MIT, `presidio-research`), statistical weight on common types |

Nine configurations: deterministic rules, GLiNER2-PII, ai4privacy-mdeberta,
MiniLM-L6, DeBERTa-v3-xsmall, and GLiNER1 edge/small at fp32 and int8.

---

## 2. Conclusion 1 — GLiNER alone is not enough

GLiNER2 is the strongest single model we measured. It is still not sufficient,
and the gap is structural rather than a matter of tuning.

### It cannot see developer secrets that the rules get perfectly

Per-entity recall on `dlp-v1`:

| entity | rules | gliner2 | who can do it |
|---|---:|---:|---|
| CONNECTION_STRING | **1.00** | 0.00 | only the rules |
| PRIVATE_KEY | **1.00** | 0.00 | only the rules |
| API_KEY | **1.00** | 0.75 | rules better |
| CREDIT_CARD / IBAN | **1.00** | 1.00 | both |
| PERSON | 0.00 | **0.92** | only the model |
| ADDRESS / DOB / PASSPORT / DRIVERS_LICENSE | 0.00 | **1.00** | only the model |

GLiNER2 reports `postgres://admin:pw@host/db` as an **EMAIL** covering
`pw@host` — a plausible mistake for a model that has never been trained on
connection-string grammar, and one that a checksum-and-grammar recognizer
cannot make.

Conversely the rules score 0.00 on PERSON, ADDRESS, DATE_OF_BIRTH, PASSPORT and
DRIVERS_LICENSE — **34% of the AI-DLP gold set is unreachable by pattern
matching**, and no amount of regex work changes that.

### Its precision is not deployable on its own

660 false positives per 1 000 documents, and only 10 of 18 negative documents
came back clean. Splitting the 28 sampled false positives:

- **15 are mislabels** — the model found the sensitive text and named it wrong
  (`Anna Kowalska` → `USERNAME`). The bytes still get redacted; this is why its
  sensitive-character recall is 0.88 against the rules' 0.72.
- **13 are genuine over-detection**, and they are developer noise at high
  confidence: `#ff00aa` as `ACCESS_TOKEN` at **0.98**, a git SHA at 0.91, a UUID
  as `IBAN` at 0.86. **Thresholding will not remove these** — they are more
  confident than many true positives. The sweep confirms it: precision climbs
  only 0.77 → 0.83 across thresholds 0.45–0.85.

### Nothing else we tested is a better single choice

- **MiniLM-L6** is fast (36 ms p95 @128 tokens, the only model to meet the
  target) and equal on aggregate F1 — but it is a 30 522-token English
  WordPiece model. Recall on Hindi 0.00, Chinese 0.25, Russian 0.38. Its
  non-zero scores outside Latin script are almost entirely the *email*, which
  is ASCII everywhere; it essentially never finds the name.
- **GLiNER1 edge/small, DeBERTa-xsmall, ai4privacy** are all worse than the
  rules on precision (1 000–2 700 FP/1k) and none is both fast and multilingual.
- **Only two of six models are structurally multilingual** (250 101-token
  SentencePiece): GLiNER2 and ai4privacy. GLiNER2 reaches recall 1.00 in eleven
  of eighteen languages including Arabic, Hebrew, Korean, Hindi and Greek.

---

## 3. Conclusion 2 — large prompts break the synchronous path

Measured, GLiNER2-PII, isolated runs:

| document | p50 | ms per word |
|---:|---:|---:|
| 42 words | 142 ms | 3.37 |
| 132 words | 374 ms | 2.84 |
| 522 words | 2 014 ms | 3.86 |
| **1 031 words** | **7 159 ms** | **6.94** |

Cost per word roughly **doubles** between 132 and 1 031 words: mdeberta uses
relative attention with no absolute position table, so there is no length limit
— attention is simply O(n²). Beyond 1 031 words we did not measure; the curve
implies well over 15 s for a 2 000-word paste, but that is an extrapolation and
should be treated as one.

**This is the normal case, not the tail.** A pasted 100-line source file is
~800 words. "Here is my config, why does it fail" is 200–600. A stack trace with
surrounding code is 500+. The p95 target of 50 ms at 128 tokens is already
missed by 7×; at realistic prompt sizes the miss is three orders of magnitude.

Two mitigations exist and neither is sufficient alone:

- **Chunking.** Splitting into 256-token windows keeps every pass in the cheap
  part of the curve. Measured indirectly: the chunked token classifiers gain up
  to 3.1× at 1 024 words. It does not get us to 50 ms; it gets us from
  "unusable" to "tolerable in the background".
- **Quantization.** On GLiNER1 int8 was **free or better** — a quarter of the
  size, roughly half the latency, no quality loss (F1 0.37 → 0.43 on `edge`),
  consistent with quantization-aware training. GLiNER2 is already published
  int8-quantized, so this lever is partly spent for it, but it is the right
  first thing to try on any replacement.

**Design consequence:** the ML tier must be asynchronous, or restricted to short
prompts, or both. Blocking a user's Enter key on a transformer pass over their
whole prompt is not a product.

---

## 4. Conclusion 3 — classical filtering first, and it carries more than expected

The deterministic layer is not a baseline to be replaced. It is the only
component that satisfies the synchronous constraint, and it has three properties
no model matched:

| property | rules | best model |
|---|---|---|
| p95 @128 tokens | **0.1 ms** | 36 ms (MiniLM, English only) |
| warm RSS | **12 MB** | 256 MB |
| cold start | **6 ms** | 119 ms |
| precision (`dlp-v1`) | **0.88** | 0.70 |
| clean negatives | **18/18** | 16/18 |
| **language variance** | **none — 0.50 recall in all 18 languages** | collapses outside training languages |

That last row is the underrated one. Checksums and grammars do not care what
script surrounds them: Luhn, IBAN mod-97, `Ipv4Addr::from_str` and a JWT header
decode behave identically in Russian, Arabic and Japanese text. For a product
that must ship in many locales, **predictability is worth as much as accuracy**.

Beyond finding PII, the rule layer should do two more jobs:

1. **Negative recognition as a veto.** The 13 genuine model false positives were
   `#ff00aa`, git SHAs, UUIDs, semver and npm specs — all of which a
   deterministic recognizer identifies with certainty. A model detection
   overlapping a *confirmed-benign* construct should be suppressed. Thresholding
   cannot remove a 0.98-confidence error; a veto can.
2. **Gating the expensive tier.** Most prompts contain no person or address. A
   cheap pre-filter avoids paying hundreds of milliseconds on the majority that
   need nothing.

---

## 5. Conclusion 4 — delivery is the harder half

None of the above matters if the prompt is never seen. Two distinct surfaces,
and we need both.

### 5.1 Browser extensions — and not Chrome only

Most AI usage is a web app: chatgpt.com, claude.ai, gemini.google.com,
perplexity.ai, copilot.microsoft.com, plus whatever appears next quarter. The
interception point is the DOM — the composer element and the network request
that leaves it.

**Chrome-only is not sufficient coverage** for an enterprise product:

| browser | engine | extension API | notes |
|---|---|---|---|
| Chrome | Blink | MV3 | the baseline |
| Edge | Blink | MV3 | near-identical to Chrome; low marginal cost |
| Firefox | Gecko | WebExtensions | MV3 differs; background scripts and blocking APIs are not the same |
| Safari | WebKit | Safari Web Extension | requires an Xcode wrapper, notarization and App Store distribution |

Realistically Chrome and Edge share one build; Firefox needs a variant; Safari
needs its own packaging and release channel. That is a real, ongoing engineering
cost and it should be planned rather than discovered.

Constraints an extension imposes on the detector:

- **The detector cannot be a 426 MB ONNX model inside the extension.** It must
  live in a native host process (the endpoint agent) that the extension talks
  to over local IPC, or the fast rule layer must run in the extension in
  WASM/JS while the ML tier stays native.
- **Interception must be non-blocking** for anything slower than a few
  milliseconds, which is exactly the tiering above.
- Enterprise deployment via policy (Chrome/Edge ADMX, Firefox policies.json,
  Safari MDM) matters as much as the code.

This mirrors the shape Nightfall publicly describes — "browser plugins and
endpoint agents monitor AI interactions" across ChatGPT, Copilot, Gemini, Claude
and Perplexity, with "Endpoints & Browsers" as a first-class product surface.
The market has converged on this answer.

### 5.2 Targeted application integration

A browser extension sees nothing when the user is in **Cursor, VS Code +
Copilot, Claude Code, the ChatGPT or Claude desktop apps, JetBrains AI, or a
terminal agent**. For a developer audience — the people most likely to paste a
config with live credentials — this is the *primary* channel, not an edge case.

Options, roughly in order of robustness:

1. **IDE/editor extensions** (VS Code, Cursor, JetBrains). First-class, supported
   APIs; the best option where they exist. Cursor is VS Code-compatible, so one
   extension covers a large share.
2. **Local proxy / MITM on the AI API endpoints.** Catches desktop apps and CLI
   agents uniformly, independent of UI. Costs a trusted local CA, breaks on
   certificate pinning, and is a heavy thing to ask of endpoint IT.
3. **Endpoint agent with per-application hooks** — accessibility APIs, clipboard
   monitoring, input capture. Broadest reach, most fragile, most invasive, and
   the most likely to break on every application update.

**Note:** see §5.3 for evidence that per-app DOM injection is a losing bet — the
one vendor we can inspect built it and then removed it.

**Recommendation:** treat this as a portfolio. Extensions for browsers and
VS Code-family editors cover the bulk at acceptable cost; a local proxy covers
desktop and CLI clients; a hooks-based agent is a last resort for named
high-value applications only. The detection library must be a **reusable native
component with a stable local API**, because it will be called from at least
four different host processes.

This is why the benchmark's detectors were built behind one `Detector` trait
with no dependency on the evaluation code — they are meant to be lifted into
these hosts unchanged.

---


### 5.3 How Nightfall actually solves the latency problem — they avoid it

Read from the shipped extension itself (`Nightfall DLP for Browsers` v8.15.0,
Chrome Web Store id `jgmgecncmjklkabkejnjfgfkglapfgek`; it ships source maps
with `sourcesContent`, so the original TypeScript is recoverable). Their
documentation does not describe this; the code does.

**The extension is a sensor and an enforcement point. It contains no detector.**
It talks to a *local endpoint agent* over `ws://localhost:<port>` (with a
`wss://` fallback) plus `nativeMessaging`. The whole agent protocol is two
request types: `CLIPBOARD_PERMISSION` and `FILE_UPLOAD_PERMISSION`.

**For pasted text they never inspect the content in the decision path at all.**
The permission request carries only `{domain, url, title, userEmail}` — *where
you are and who you are*, not *what you are pasting*. The paste handler is
fully synchronous and reads a cached boolean:

```ts
doc.addEventListener("paste", event => {
  const pasteContent = event.clipboardData?.getData("text") ?? "";
  if (!allowPaste) {                    // pre-computed, never awaited
    event.preventDefault();
    event.stopPropagation();
    sendClipboardEvent("paste", pasteContent, { blocked: true, ... });
  } else {
    sendClipboardEvent("paste", pasteContent, { blocked: false, ... });
  }
}, true);
```

`allowPaste` is refreshed ahead of time — on init, `visibilitychange`, `focus`,
`pageshow` and SPA URL changes — so by the time the user pastes, the answer is
already in a variable. The content *is* shipped to the agent, but by
`sendClipboardEvent`, fire-and-forget, **after** the decision. That is
detection-and-alerting, not prevention.

So the honest description of their text path is: **coarse policy prevention,
fine-grained detection after the fact.** They dodge our entire latency problem
by not asking a content question on the hot path.

**For file uploads they do ask a content question**, and the pattern there is
worth stealing — *swallow and replay*:

1. capture the `drop` / `paste` / `change` event and `preventDefault()` it;
2. request a verdict from the agent;
3. if allowed, dispatch a **synthetic event with the original files** so the page
   proceeds as if nothing happened (`replayedEvents` is a `WeakSet` so their own
   capture-phase listener ignores the replay).

This converts a blocking decision into an asynchronous one without the user
losing the action. It is also fragile in exactly the way one would expect —
folder drag-and-drop is explicitly unsupported because "a synthetic
DataTransfer can't reconstruct FileSystemDirectoryEntry".

**Everything fails open, on a 3-second floor.** Both paths use
`CLIPBOARD_PERMISSION_TIMEOUT_MS = 3000` / `UPLOAD_VERDICT_TIMEOUT_MS = 3_000`,
and both default to *allow*:

```ts
export const DEFAULT_ALLOWED_PASTE_PERMISSION = true;
```

Their own comment on the upload path is the important part: *"The agent replies
promptly — it never long-holds (scan-pending uploads are blocked immediately
with a 'scanning' notification rather than waiting)."* When the scan is not
finished, the agent **blocks and tells the user it is scanning** rather than
making them wait. That is a third option we had not considered alongside
"block-and-wait" and "allow-and-warn-later", and it is probably the right
default for a slow detector: *deny with an explanation, let the user retry.*

Two further details that show how fragile DOM interception is in practice, and
that we should budget for:

- Copilot surfaces (`copilot.microsoft.com`, `copilot.com`, `copilot.fun`)
  swallow Ctrl+V at `keydown`, so the normal `paste` event never fires and they
  need a separate keydown-capture blocker — Edge-only.
- `claude.ai` needs a **MAIN-world** injected script to capture the user's email
  from a Segment `identify` call, with a DOM-menu-clicking fallback that
  synthesises `pointerdown` + `click` on a Radix dropdown. That is per-site
  reverse engineering that breaks on redesign.

**What this means for us.** Their design is a legitimate answer to the latency
problem, but it is a *weaker security posture* than the one we have been
scoping: a domain/user policy cannot tell a prompt containing an AWS key from a
prompt asking about the weather. Our measured Tier 0 changes the trade: at
0.1 ms and 12 MB, validated regex is fast enough to make a **content-aware**
allow/deny decision inline, which their architecture cannot do. The right
synthesis is:

| | Nightfall (observed) | proposed here |
|---|---|---|
| hot-path decision | policy on (domain, user) | **content-aware** Tier 0 rules, 0.1 ms |
| content inspection | asynchronous, after the fact | Tier 0 inline + Tier 1 asynchronous |
| slow-scan behaviour | block with "scanning" notice | same — adopt this |
| timeout | 3 s, fail open | same floor; fail-open is a policy choice, not a default |
| detector location | local agent over localhost WS | same |

The "block with a scanning notice" pattern and the swallow-and-replay mechanism
should both be adopted. The policy-only text path should not.


#### They built per-app injection, shipped it, and are removing it

This is the most consequential thing in the extension, because it is a direct
verdict on the "inject into ChatGPT / Cursor specifically" idea.

The extension **does** recognise AI sites — but only to answer *who the user
is*, never *whether to block*. `aiAppsAccount.js` is injected on
`*.chatgpt.com`, `claude.ai` and `*.perplexity.ai` purely to scrape the account
email (from ChatGPT's `client-bootstrap` JSON, from a Claude Segment
`identify` call intercepted in the MAIN world, from Perplexity's
`localStorage`). The site list is named for what it is:

```ts
/** The set of sites the extension extracts user identity (account email) from. */
export const ACCOUNT_DETECTION_MATCH_PATTERNS: string[] = [ ... ];
```

The blocking decision has no site awareness at all: `ClipboardMonitor` runs on
`<all_urls>` and asks the agent about `{domain, url, title, userEmail}`.

But an **earlier architecture did exactly what we proposed**. Dead code in
`constant.ts` shows a `TextMonitor` that hooked the composer of nine named
applications:

```ts
// export const InputQuerySelectors: Record<GENAI_APPS, string> = {
//   [GENAI_APPS.CHATGPT]:    "#prompt-textarea",
//   [GENAI_APPS.CLAUDE]:     'div[contenteditable="true"]',
//   [GENAI_APPS.GEMINI]:     '[contenteditable="true"][role="textbox"]:not([data-placeholder^="Example"])',
//   [GENAI_APPS.DEEPSEEK]:   "textarea[id='chat-input']",
//   [GENAI_APPS.PERPLEXITY]: "#ask-input, textarea[placeholder^='Ask'], textarea[placeholder='Research anything…']",
//   [GENAI_APPS.GROK]:       "textarea[dir='auto']",
//   ...
// };
// export const SubmitQuerySelectors: Record<GENAI_APPS, string> = { ... };
// export const GenAIAppConfig = {
//   [GENAI_APPS.CHATGPT]: { submitSelector: "button", textBoxRemoval: true },
//   [GENAI_APPS.CLAUDE]:  { submitSelector: "button", textBoxRemoval: true, donotShowModal: true },
//   ...
// };
```

That is composer interception with a per-app send-button hook and even
`textBoxRemoval` — clearing the user's input on a violation. There was a
matching `MessageType.SCAN` / `CREATE_VIOLATION` protocol and a
`ScanMessageSenderResponse` type: a real scan-the-text path.

**It is gone from the shipped bundle.** `grep` for `prompt-textarea`,
`TextMonitor` or `InputQuerySelectors` in the built JavaScript returns zero
matches; only dead type declarations survive in source, every one tagged
`// TODO: Remove ... once TextMonitor removal is confirmed`.

The reason is legible in the selectors themselves. `textarea[placeholder=
'Research anything…']` keys off a **localised UI string**. `div._7436101[role=
button]` keys off a **hashed CSS class**. These break on any redesign, any copy
change, any non-English locale — for nine applications that ship continuously.
They replaced a per-app surface that rots with an app-agnostic one
(clipboard + upload) that does not.

**Implication for our plan.** The report's §5.2 proposal to "inject into ChatGPT
and Cursor specifically" should be **demoted**. A vendor with a dedicated team
tried it, shipped it, and retreated to clipboard/upload interception. Our
sequencing should be:

1. **App-agnostic interception first** — clipboard and upload on `<all_urls>`.
   One implementation, no per-site maintenance, covers every AI app including
   ones that do not exist yet.
2. **IDE extensions second** (VS Code / Cursor), because those are *supported
   APIs* rather than scraped DOM, and so do not carry the rot.
3. **Per-app DOM injection last, and only where a named customer demands it**,
   with the maintenance cost stated up front.

The gap this leaves is honest and worth stating: clipboard interception does not
see **typed** text. Someone who types a password character by character is not
blocked. For AI-DLP that is a tolerable trade — secrets and configs arrive by
paste, essentially always — but it is a real hole, and it is the hole Nightfall
chose to accept rather than keep paying the per-app maintenance bill.

## 6. Recommended architecture

```text
                     ┌──────────────────────────────────────────┐
  browser ext ──┐    │  Tier 0  deterministic rules             │  synchronous
  IDE ext ──────┼───▶│    validated regex + checksums           │  0.1 ms, 12 MB
  local proxy ──┤    │    language-invariant                    │  BLOCKS
  desktop agent─┘    │    + negative recognizers (veto)         │
                     └────────────────┬─────────────────────────┘
                                      │ gate: does this need semantics?
                     ┌────────────────▼─────────────────────────┐
                     │  Tier 1  ML detector (currently GLiNER2)  │ asynchronous
                     │    chunked 256 tokens, quantized          │ warn / redact
                     │    multilingual                           │ after the fact
                     └────────────────┬─────────────────────────┘
                     ┌────────────────▼─────────────────────────┐
                     │  Arbitration                              │
                     │   validated rule beats ML on same span    │
                     │   one label per span                      │
                     │   negative veto overrides ML              │
                     │   provenance retained                     │
                     └────────────────┬─────────────────────────┘
                     ┌────────────────▼─────────────────────────┐
                     │  Policy (separate from detection)         │
                     │   private IPs, placeholders, per-type     │
                     │   action: block / redact / warn / allow   │
                     └───────────────────────────────────────────┘
```

The arbitration layer is not optional: **15 of GLiNER2's 28 sampled false
positives are duplicate labels on one span** (`API_KEY` + `ACCESS_TOKEN` +
`SECRET` on the same AWS key). One-label-per-span removes roughly half its
precision problem with no model change.

Policy must be a separate layer from detection. `127.0.0.1` is PII under some
policies and noise under others; the detector should report the span and let
policy decide. The benchmark already models this as an explicit flag rather than
a buried default.

---

## 7. What we do not know yet

Stated plainly, because these gaps bound how much weight the conclusions carry.

- **Tier 1 has no ideal occupant.** Every model fast enough to run
  synchronously is English-only; the only good multilingual model is too slow.
  Filling this needs a quantized multilingual model in the 100–200 MB range.
  `onnx-community/gliner_multi_pii-v1` (mdeberta backbone, published at int8,
  uint8 and q4) is the obvious next measurement.
- **Corpus size.** `dlp-v1` is 47 documents and `dlp-multilingual` is 43. They
  are a correctness harness and a directional signal, not production evidence.
  Several per-language cells rest on a single document. `presidio-synth` adds
  1 342 documents but only for English prose.
- **The hybrid detector is designed, not built.** Its expected gain is estimated
  from the false-positive breakdown, not measured.
- **No measurement above 1 031 words.** The quadratic claim is solid within the
  measured range; the "15 s for 2 000 words" figure is extrapolation.
- **The GLiNER2 ONNX export is community-produced and unaudited** (0 downloads,
  single unaffiliated author, and it self-reports a *higher* SPY F1 than the
  original it derives from, which a lossy export should not achieve). Our own
  agreement tests verify the offset pipeline, not the export's fidelity to the
  original weights.
- **Delivery cost is unestimated.** Nothing in this report quantifies the
  engineering effort for four browser targets plus IDE and proxy integration.
  That estimate is the next thing needed for planning, and it is likely larger
  than the remaining detection work.

---

## 8. Immediate next steps

1. Measure `onnx-community/gliner_multi_pii-v1` at int8/uint8/q4 — the only
   plausible candidate for a fast *and* multilingual Tier 1.
2. Build the hybrid detector: arbitration, one-label-per-span, negative veto.
   The benchmark can measure the gain the day it exists.
3. Add chunking to GLiNER2 and measure the real large-prompt curve.
4. Prototype one browser extension end-to-end against a native host, to find
   out what the IPC latency budget actually is. Everything above assumes local
   IPC is cheap; that assumption is untested.
5. Scope the delivery portfolio properly — four browsers, VS Code family,
   proxy — and cost it.
