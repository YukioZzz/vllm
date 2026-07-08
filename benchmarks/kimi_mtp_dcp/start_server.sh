#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK="${WORK:-/tmp/kimi_mtp_dcp_verify}"
RUN="${RUN:-/tmp/kimi_k25_eagle3_dcp}"
NAME="${NAME:-kimi_k25_eagle3_dcp}"
IMG="${IMG:-vllm/vllm-openai-rocm:latest}"
HOST_HF_CACHE="${HOST_HF_CACHE:-/data/hf-hub-cache}"
HOST_AITER_SRC="${HOST_AITER_SRC:-/data/zejun/aiter}"
TARGET="${TARGET:-/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/snapshots/6b0ab7ed538724ea46517351234660bdf36e2d73}"
DRAFT="${DRAFT:-/data/hf-hub-cache/models--lightseekorg--kimi-k2.5-eagle3}"
PORT="${PORT:-18085}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
ENABLE_SPEC="${ENABLE_SPEC:-1}"

if [ -d "$DRAFT/snapshots" ]; then
  DRAFT="$(find "$DRAFT/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)"
fi
if [ -d "$TARGET/snapshots" ]; then
  TARGET="$(find "$TARGET/snapshots" -maxdepth 1 -mindepth 1 -type d | head -1)"
fi

rm -rf "$RUN"
mkdir -p "$RUN"

cat > "$RUN/start.sh" <<'SH'
set -euo pipefail

export HF_MODULES_CACHE=/tmp/transformers_modules
export AITER_GLUON_TRITON_PATH=/atom_triton
if [ "${GLOBAL_ATOM_TRITON:-0}" = "1" ]; then
  export PYTHONPATH="/atom_triton:${PYTHONPATH:-}"
  export AITER_GLUON_TRITON_SWITCHED=1
fi
mkdir -p "$HF_MODULES_CACHE"

pkg=$(python3 -c 'import os, aiter; print(os.path.dirname(aiter.__file__))')
mkdir -p "$pkg/ops/triton/gluon"
cp -a /src_aiter/aiter/ops/triton/gluon/. "$pkg/ops/triton/gluon/" 2>/dev/null || true
cp /work/mla_decode_gluon_pr3402_final.py "$pkg/ops/triton/gluon/mla_decode_gluon.py"

python3 /work/scripts/patch_vllm_gluon_dcp.py
python3 /work/scripts/patch_rocm_force_aiter_mla.py
if [ "${PATCH_TORCH_MERGE_ATTN:-0}" = "1" ]; then
  python3 /work/scripts/patch_vllm_torch_merge_attn_states.py
fi

python3 -c 'import triton; from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl; print("PATCHED_TRITON", triton.__version__, triton.__file__, flush=True); print("PATCHED_AITER_MLA_CAN_LSE", AiterMLAImpl.can_return_lse_for_decode, flush=True)'

extra_args=()
if [ -n "${MOE_BACKEND:-}" ]; then
  extra_args+=(--moe-backend "$MOE_BACKEND")
fi
if [ "${ENABLE_SPEC:-1}" = "1" ]; then
  extra_args+=(--speculative-config "{\"model\":\"${DRAFT}\",\"method\":\"eagle3\",\"num_speculative_tokens\":3,\"max_model_len\":${MAX_MODEL_LEN}}")
fi

exec vllm serve "$TARGET" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --attention-backend ROCM_AITER_MLA \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 2 \
  --dcp-kv-cache-interleave-size 1 \
  --dcp-comm-backend ag_rs \
  --block-size 16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --mm-encoder-tp-mode data \
  --no-enable-prefix-caching \
  "${extra_args[@]}"
SH
chmod +x "$RUN/start.sh"

docker rm -f "$NAME" >/dev/null 2>&1 || true
: > "$RUN/server.log"

docker run -d --name "$NAME" --entrypoint bash \
  --network host --ipc host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e TARGET="$TARGET" \
  -e DRAFT="$DRAFT" \
  -e PORT="$PORT" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e ENABLE_SPEC="$ENABLE_SPEC" \
  -e GLOBAL_ATOM_TRITON="${GLOBAL_ATOM_TRITON:-0}" \
  -e MOE_BACKEND="${MOE_BACKEND:-}" \
  -e VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-}" \
  -e VLLM_ROCM_USE_AITER_MOE="${VLLM_ROCM_USE_AITER_MOE:-}" \
  -e PATCH_TORCH_MERGE_ATTN="${PATCH_TORCH_MERGE_ATTN:-0}" \
  -e HF_HOME=/data/hf-hub-cache \
  -e HF_HUB_CACHE=/data/hf-hub-cache \
  -e HF_MODULES_CACHE=/tmp/transformers_modules \
  -e AITER_GLUON_TRITON_PATH=/atom_triton \
  -v "$RUN:$RUN" \
  -v "$SCRIPT_DIR:/work/scripts:ro" \
  -v "$WORK/mla_decode_gluon_pr3402_final.py:/work/mla_decode_gluon_pr3402_final.py:ro" \
  -v "$WORK/atom_triton_site:/atom_triton:ro" \
  -v "$HOST_HF_CACHE:/data/hf-hub-cache:ro" \
  -v "$HOST_AITER_SRC:/src_aiter:ro" \
  "$IMG" \
  "$RUN/start.sh" > "$RUN/cid.txt"

(docker logs -f "$NAME" > "$RUN/server.log" 2>&1 & echo $! > "$RUN/log_tail.pid")

sleep 8
docker ps --filter name="$NAME" --format '{{.Names}} {{.Status}}'
tail -120 "$RUN/server.log"
