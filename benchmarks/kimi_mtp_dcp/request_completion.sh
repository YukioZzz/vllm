#!/usr/bin/env bash
set -euo pipefail

RUN="${RUN:-/tmp/kimi_k25_eagle3_dcp}"
MODEL="${MODEL:-/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/snapshots/6b0ab7ed538724ea46517351234660bdf36e2d73}"
PORT="${PORT:-18085}"
PROMPT_REPEAT="${PROMPT_REPEAT:-40}"
MAX_TOKENS="${MAX_TOKENS:-32}"

python3 - <<'PY' > "$RUN/request.json"
import json
import os

prompt = (
    "Write a concise numbered checklist for validating speculative decoding accuracy. "
    * int(os.environ.get("PROMPT_REPEAT", "40"))
)
print(json.dumps({
    "model": "/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/snapshots/6b0ab7ed538724ea46517351234660bdf36e2d73",
    "prompt": prompt,
    "max_tokens": int(os.environ.get("MAX_TOKENS", "32")),
    "temperature": 0.0,
}))
PY

curl -sS --max-time 360 "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d @"$RUN/request.json" | tee "$RUN/one_completion.json"

echo
echo "--- metrics ---"
curl -sS "http://127.0.0.1:${PORT}/metrics" \
  | grep -E 'spec_decode_num_drafts_total|spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total|spec_decode_num_accepted_tokens_per_pos_total|request_success_total|prompt_tokens_total|generation_tokens_total' \
  | sed -n '1,180p' || true

echo
echo "--- recent errors / kernels ---"
grep -nE 'ERROR|AssertionError|RuntimeError|Traceback|_mla_decode_gluon|_correct_attn_cp_out_kernel|merge_attn_states_kernel|SpecDecoding' "$RUN/server.log" | tail -180 || true
