#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN="${RUN:-/tmp/kimi_k25_eagle31_mla_dcp_atom_aiter_moe}"
export PORT="${PORT:-18089}"
export WORK="${WORK:-/tmp/kimi_mtp_dcp_verify}"
export PROMPT_REPEAT="${PROMPT_REPEAT:-96}"
export MAX_TOKENS="${MAX_TOKENS:-32}"

bash "$SCRIPT_DIR/request_completion.sh"
