#!/usr/bin/env bash
# Download third-party evaluation corpora by pinned commit, with sha256
# verification. Same discipline as scripts/fetch-models.sh: an unpinned or
# unverified corpus makes every number computed from it unciteable.
#
# Downloads the raw upstream file only. Converting it into the benchmark's
# JSONL format is a separate, deterministic step:
#
#   bash scripts/fetch-datasets.sh
#   cargo run --release -- import presidio \
#     --input datasets/raw/presidio-synth-v2.json \
#     --output datasets/presidio-synth.jsonl

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/datasets/raw"

# <local-name>|<repo>|<commit>|<remote-path>|<sha256>|<bytes>
MANIFEST=(
  "presidio-synth-v2.json|data-privacy-stack/presidio-research|1663688c99fa5c8c6d8cf0412e1d71861de4f621|data/synth_dataset_v2.json|ec08a771ba8135314cafb60752b2295212222ba3a4cd75d73811839c699e0012|1023794"
)

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

mkdir -p "$RAW"
for row in "${MANIFEST[@]}"; do
  IFS='|' read -r name repo commit remote sha bytes <<<"$row"
  case "$commit" in
    main|master|HEAD) echo "refusing unpinned ref '$commit' for $repo" >&2; exit 1 ;;
  esac

  out="$RAW/$name"
  if [[ -f "$out" && "$(sha256_of "$out")" == "$sha" ]]; then
    echo "ok       $name (cached)"
    continue
  fi

  url="https://raw.githubusercontent.com/$repo/$commit/$remote"
  echo "fetch    $name  <-  $repo@${commit:0:8}"
  curl -fSL --retry 3 --retry-delay 2 -o "$out.part" "$url"
  mv "$out.part" "$out"

  got="$(sha256_of "$out")"
  if [[ "$got" != "$sha" ]]; then
    echo "FAIL     $name sha256 mismatch" >&2
    echo "         expected $sha" >&2
    echo "         actual   $got" >&2
    rm -f "$out"
    exit 1
  fi
  actual_bytes="$(wc -c <"$out" | tr -d ' ')"
  if [[ "$actual_bytes" != "$bytes" ]]; then
    echo "FAIL     $name size mismatch: expected $bytes, got $actual_bytes" >&2
    rm -f "$out"
    exit 1
  fi
  printf '{"repo":"%s","commit":"%s","path":"%s"}\n' "$repo" "$commit" "$remote" \
    >"$RAW/${name%.json}.provenance.json"
  echo "verified $name"
done

echo
echo "raw corpora in $RAW — convert with \`dlp-bench import\`"
