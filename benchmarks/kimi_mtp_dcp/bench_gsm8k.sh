#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WORK="${WORK:-/tmp/kimi_mtp_dcp_verify}"
export RUN="${RUN:-/tmp/kimi_mtp_dcp_verify}"
export PORT="${PORT:-18089}"
export LABEL="${LABEL:-gsm8k}"
export N="${N:-64}"
export OFFSET="${OFFSET:-0}"
export CONCURRENCY="${CONCURRENCY:-8}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export PAD_REPEAT="${PAD_REPEAT:-0}"
export N_SHOTS="${N_SHOTS:-0}"
export ALIGN_TOKENS="${ALIGN_TOKENS:-1}"

python3 "$SCRIPT_DIR/bench_gsm8k.py"
