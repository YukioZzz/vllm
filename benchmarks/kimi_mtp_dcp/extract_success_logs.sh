#!/usr/bin/env bash
set -euo pipefail

LOG="${LOG:-/tmp/kimi_k25_eagle31_mla_dcp_atom_aiter_moe/server.log}"

grep -nE 'PATCHED_TRITON|PATCHED_AITER|patched /usr/local.*merge_attn_states|AITER_MXFP4_MXFP4|Eagle3DeepseekV2|_mla_decode_gluon|_correct_attn_cp_out_kernel|SpecDecoding metrics|RuntimeError|Traceback|ERROR' "$LOG" | tail -220 || true
echo "--- containers ---"
docker ps --no-trunc
echo "--- gpu ---"
rocm-smi --showpidgpus
