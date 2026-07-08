#!/usr/bin/env bash
set -euo pipefail

HF_CACHE="${HF_CACHE:-/data/hf-hub-cache}"
DRAFT_REPO="${DRAFT_REPO:-lightseekorg/kimi-k2.6-eagle3.1-mla}"
IMG="${IMG:-vllm/vllm-openai-rocm:latest}"
DOCKER_NETWORK_ARGS="${DOCKER_NETWORK_ARGS:-}"

echo "--- download/check draft: $DRAFT_REPO ---"
docker run --rm --entrypoint bash \
  $DOCKER_NETWORK_ARGS \
  -v "$HF_CACHE:$HF_CACHE" \
  -e HF_HOME="$HF_CACHE" \
  -e HF_HUB_CACHE="$HF_CACHE" \
  "$IMG" -lc "
set -e
python3 - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

path = Path(snapshot_download(repo_id='$DRAFT_REPO', cache_dir='$HF_CACHE', local_files_only=False))
print('SNAPSHOT', path)
cfg = json.loads((path / 'config.json').read_text())
for k in [
    'architectures', 'model_type', 'eagle_config', 'num_hidden_layers',
    'num_attention_heads', 'kv_lora_rank', 'qk_rope_head_dim',
    'hidden_size', 'head_dim', 'torch_dtype'
]:
    print(k, cfg.get(k))
PY
"

echo "--- snapshots ---"
find "$HF_CACHE" -maxdepth 5 -type d -path '*/snapshots/*' 2>/dev/null \
  | grep -Ei 'Kimi|kimi|eagle|EAGLE' | sed -n '1,260p'
