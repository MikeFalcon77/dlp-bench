#!/usr/bin/env bash
# Download benchmark model artifacts by pinned revision, with sha256 verification.
#
# Adapted from the pinned-revision + per-file-checksum pattern in
# quarry/crates/quarry-models/{assets/hub-models.toml,src/download.rs,src/hash.rs}.
# Nothing here runs implicitly during a benchmark; `dlp-bench` fails with a clear
# error naming this script when artifacts are missing.
#
# Usage: bash scripts/fetch-models.sh [model-id ...]     (default: all)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${DLP_BENCH_MODELS_DIR:-$ROOT/models}"

# ---------------------------------------------------------------------------
# Manifest: <local-dir>|<hf-repo>|<revision>|<remote-path>|<local-name>|<sha256>|<bytes>
# Revisions are immutable commit SHAs. `main` is never accepted.
# ---------------------------------------------------------------------------
MANIFEST=(
  "gliner2-pii-onnx|yethdev/gliner2-pii-onnx|7d82c70220c2b0891d4bda612402602b63be1c61|model.quantized.onnx|model.quantized.onnx|489edc522a34b8c6a0630c4710998abf0227b11acf45600ba2bce95bcd943930|446391629"
  "gliner2-pii-onnx|yethdev/gliner2-pii-onnx|7d82c70220c2b0891d4bda612402602b63be1c61|tokenizer.json|tokenizer.json|f6df10ec83bea993035b2dd7c39345a3d4fcf23421c2adb6cb4ffc1e6d1bc4b5|16020604"
  "gliner2-pii-onnx|yethdev/gliner2-pii-onnx|7d82c70220c2b0891d4bda612402602b63be1c61|tokenizer_config.json|tokenizer_config.json|233beed1f1095cccfc7907cde31a8d90a0c6aa4fdfaf6493f8e55fd162e81ae6|711"
  "gliner2-pii-onnx|yethdev/gliner2-pii-onnx|7d82c70220c2b0891d4bda612402602b63be1c61|export_info.json|export_info.json||"

  # ai4privacy/OpenPII token classifier on the same mdeberta-v3-base backbone as
  # GLiNER2 — the controlled comparison for "does a plain token classifier beat
  # zero-shot span extraction on latency". 111 BIO labels over 56 entity types.
  #
  # Note: no ModernBERT PII model has an ONNX export (checked five candidates),
  # so this is the closest available stand-in for the task's "AI4Privacy
  # ModernBERT" slot. See the README.
  "ai4privacy-mdeberta|Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2|316240516ad48a82ae1b13567670093901e41b5e|onnx/model.onnx|model.onnx|1c529747fec02680c565f8e840c0aed87ab2b67e8c6b993effdc4f2a2b57ed9a|1114083751"
  "ai4privacy-mdeberta|Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2|316240516ad48a82ae1b13567670093901e41b5e|onnx/tokenizer.json|tokenizer.json|2483e98ba6128a9a21cb2cc2acfa9532da53fd97b56d5e8fe34c315c76ca731d|16316179"
  "ai4privacy-mdeberta|Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2|316240516ad48a82ae1b13567670093901e41b5e|onnx/config.json|config.json||"
  "ai4privacy-mdeberta|Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2|316240516ad48a82ae1b13567670093901e41b5e|onnx/tokenizer_config.json|tokenizer_config.json||"

  # ---- Phase 2: the small end of the space -------------------------------
  # Everything measured so far is a ~12x768 encoder and none of it fits the
  # synchronous path. These probe whether anything smaller does.

  # MiniLM-L6: 6 layers x 384, the lower bound for an off-the-shelf transformer
  # NER. English wordpiece; its 111 BIO labels include api_key, which the
  # ai4privacy taxonomy lacks entirely.
  "minilm-l6-pii|Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2|80a05dbac61de613ea01aee8542b465bb51c54b2|onnx/model.onnx|model.onnx|8341930b1c41fb05ffffc737c334cb78ff073c161a7806654a944d7355be893b|90542099"
  "minilm-l6-pii|Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2|80a05dbac61de613ea01aee8542b465bb51c54b2|tokenizer.json|tokenizer.json||"
  "minilm-l6-pii|Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2|80a05dbac61de613ea01aee8542b465bb51c54b2|config.json|config.json||"
  "minilm-l6-pii|Negative-Star-Innovators/MiniLM-L6-finetuned-pii-detection-v2|80a05dbac61de613ea01aee8542b465bb51c54b2|tokenizer_config.json|tokenizer_config.json||"

  # DeBERTa-v3-xsmall: 12 layers x 384. Stands in for DataFog/pii-small-en,
  # which cannot be run here — custom PiiNerModel (CharCNN + CRF), safetensors
  # only, no modeling file, and no export path with PyPI unreachable. This one
  # answers the size question but has no CRF; only 14 labels.
  "deberta-xsmall-pii|megaloxorafitis/pii-ner-deberta-v3-xsmall|54e8f7565f739dfa9febb58ddb89291342256995|onnx/model_quantized.onnx|model.onnx|b024fbfc5d5aeaefd3190a59b7a757a5d821103c75503b5da9c702109faacb2a|87334561"
  "deberta-xsmall-pii|megaloxorafitis/pii-ner-deberta-v3-xsmall|54e8f7565f739dfa9febb58ddb89291342256995|tokenizer.json|tokenizer.json||"
  "deberta-xsmall-pii|megaloxorafitis/pii-ner-deberta-v3-xsmall|54e8f7565f739dfa9febb58ddb89291342256995|config.json|config.json||"
  "deberta-xsmall-pii|megaloxorafitis/pii-ner-deberta-v3-xsmall|54e8f7565f739dfa9febb58ddb89291342256995|tokenizer_config.json|tokenizer_config.json||"

  # GLiNER edge (ettin-encoder-32m) and small (-68m), each at fp32/fp16/int8.
  # Same weights at three precisions, so the size axis and the precision axis
  # are isolated from one another. The quantized variants are
  # quantization-aware trained per the model card, not post-training quantized.
  "gliner-pii-edge|knowledgator/gliner-pii-edge-v1.0|9b7f39b0a2da971a5beea78d35f1539d4009c891|onnx/model.onnx|model.onnx|4ca588722e6d79447ad4c9c230eeba3d9d472c672a9598184a34e9f77fc35836|181078966"
  "gliner-pii-edge|knowledgator/gliner-pii-edge-v1.0|9b7f39b0a2da971a5beea78d35f1539d4009c891|onnx/model_fp16.onnx|model_fp16.onnx|af499f06532909628765ce1cebd65311fa2d9a16f74a34e7b6e1b9369230a3a7|90845497"
  "gliner-pii-edge|knowledgator/gliner-pii-edge-v1.0|9b7f39b0a2da971a5beea78d35f1539d4009c891|onnx/model_quint8.onnx|model_quint8.onnx|988acb03456b26e2d9f2521016d820310c2ed64deb4a846297d3289f0c2eb7e4|45820894"
  "gliner-pii-edge|knowledgator/gliner-pii-edge-v1.0|9b7f39b0a2da971a5beea78d35f1539d4009c891|tokenizer.json|tokenizer.json||"
  "gliner-pii-edge|knowledgator/gliner-pii-edge-v1.0|9b7f39b0a2da971a5beea78d35f1539d4009c891|gliner_config.json|gliner_config.json||"

  "gliner-pii-small|knowledgator/gliner-pii-small-v1.0|d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e|onnx/model.onnx|model.onnx|c60fd671a30798c1bfad5828f8bbf6214eb44cbc5e2f0fb968563ae935a17454|326820148"
  "gliner-pii-small|knowledgator/gliner-pii-small-v1.0|d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e|onnx/model_fp16.onnx|model_fp16.onnx|522d6dad97e2bfb836bcc7be33d792472e1fc24dc45e99f86c19ecd3020143f1|163945472"
  "gliner-pii-small|knowledgator/gliner-pii-small-v1.0|d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e|onnx/model_quint8.onnx|model_quint8.onnx|891589426ee96f2748b16439f44fad8c3f97e198e002a6637e58dee989500216|82680500"
  "gliner-pii-small|knowledgator/gliner-pii-small-v1.0|d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e|tokenizer.json|tokenizer.json||"
  "gliner-pii-small|knowledgator/gliner-pii-small-v1.0|d21aad5b4a7ec82b3d0970fd1ac74a12c087d85e|gliner_config.json|gliner_config.json||"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

want="${1:-}"
downloaded=0
verified=0

for row in "${MANIFEST[@]}"; do
  IFS='|' read -r dir repo rev remote name sha bytes <<<"$row"
  [[ -n "$want" && "$want" != "$dir" ]] && continue

  case "$rev" in
    main|master|HEAD) echo "refusing unpinned revision '$rev' for $repo" >&2; exit 1 ;;
  esac

  dest="$MODELS_DIR/$dir"
  out="$dest/$name"
  mkdir -p "$dest"

  if [[ -f "$out" && -n "$sha" ]]; then
    if [[ "$(sha256_of "$out")" == "$sha" ]]; then
      echo "ok       $dir/$name (cached)"
      verified=$((verified + 1))
      continue
    fi
    echo "stale    $dir/$name — checksum mismatch, re-downloading" >&2
    rm -f "$out"
  elif [[ -f "$out" && -z "$sha" ]]; then
    echo "ok       $dir/$name (cached, unchecked)"
    continue
  fi

  url="https://huggingface.co/$repo/resolve/$rev/$remote"
  echo "fetch    $dir/$name  <-  $repo@${rev:0:8}"
  curl -fSL --retry 3 --retry-delay 2 -o "$out.part" "$url"
  mv "$out.part" "$out"
  downloaded=$((downloaded + 1))

  if [[ -n "$sha" ]]; then
    got="$(sha256_of "$out")"
    if [[ "$got" != "$sha" ]]; then
      echo "FAIL     $dir/$name sha256 mismatch" >&2
      echo "         expected $sha" >&2
      echo "         actual   $got" >&2
      rm -f "$out"
      exit 1
    fi
    actual_bytes="$(wc -c <"$out" | tr -d ' ')"
    if [[ -n "$bytes" && "$actual_bytes" != "$bytes" ]]; then
      echo "FAIL     $dir/$name size mismatch: expected $bytes, got $actual_bytes" >&2
      rm -f "$out"
      exit 1
    fi
    echo "verified $dir/$name"
    verified=$((verified + 1))
  fi
done

# Record what we resolved so run metadata can cite an exact artifact revision.
for row in "${MANIFEST[@]}"; do
  IFS='|' read -r dir repo rev _ _ _ _ <<<"$row"
  [[ -n "$want" && "$want" != "$dir" ]] && continue
  printf '{"repo":"%s","revision":"%s"}\n' "$repo" "$rev" >"$MODELS_DIR/$dir/.revision.json"
done

echo
echo "downloaded=$downloaded verified=$verified  ->  $MODELS_DIR"
