#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN="${RUN:-/tmp/kimi_k25_eagle31_mla_dcp_atom_aiter_moe}"
export NAME="${NAME:-kimi_k25_eagle31_mla_dcp_atom_aiter_moe}"
export PORT="${PORT:-18089}"
export WORK="${WORK:-/tmp/kimi_mtp_dcp_verify}"
export DRAFT="${DRAFT:-/data/hf-hub-cache/models--lightseekorg--kimi-k2.6-eagle3.1-mla/snapshots/35194ee8feb2826812f716eb42a924f99a5404f3}"
export GLOBAL_ATOM_TRITON=1
export MOE_BACKEND=aiter
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export PATCH_TORCH_MERGE_ATTN=1

docker rm -f \
  kimi_k25_eagle3_nodcp \
  kimi_k25_eagle3_dcp \
  kimi_k25_eagle31_mla_nodcp \
  kimi_k25_eagle31_mla_dcp \
  kimi_k25_target_dcp_atom_aiter_moe \
  "$NAME" >/dev/null 2>&1 || true

bash "$SCRIPT_DIR/start_server.sh"
