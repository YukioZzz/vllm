#!/usr/bin/env bash
set -euo pipefail

IMG="${IMG:-vllm/vllm-openai-rocm:latest}"
HF_CACHE="${HF_CACHE:-/data/hf-hub-cache}"
ROOT="$HF_CACHE/models--amd--Kimi-K2.5-MXFP4"
SNAP="${SNAP:-}"
if [ -z "$SNAP" ]; then
  SNAP="$(find "$ROOT/snapshots" -maxdepth 1 -mindepth 1 -type d | sort | head -1)"
fi

docker run --rm --entrypoint bash -v "$HF_CACHE:$HF_CACHE" "$IMG" -lc "
set -e
ROOT='$ROOT'
SNAP='$SNAP'
for f in \"\$SNAP\"/*.py; do
  [ -e \"\$f\" ] || continue
  cp -f \"\$(readlink -f \"\$f\")\" \"\$ROOT/blobs/\$(basename \"\$f\")\"
done
ls -l \"\$ROOT\"/blobs/*.py | sed -n '1,120p'
python3 - <<'PY'
from pathlib import Path
root = Path('$ROOT')
for f in ['tool_declaration_ts.py', 'tokenization_kimi.py', 'media_utils.py']:
    p = root / 'blobs' / f
    print(f, p.exists(), p.read_text(errors='ignore')[:80].replace('\\n', ' '))
PY
"
