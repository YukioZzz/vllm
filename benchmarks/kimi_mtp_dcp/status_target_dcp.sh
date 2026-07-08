#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN="${RUN:-/tmp/kimi_k25_target_dcp_atom_aiter_moe}"
export NAME="${NAME:-kimi_k25_target_dcp_atom_aiter_moe}"
export WORK="${WORK:-/tmp/kimi_mtp_dcp_verify}"

bash "$SCRIPT_DIR/status_server.sh"
