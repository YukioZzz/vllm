#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-kimi_k25_eagle3_dcp}"
RUN="${RUN:-/tmp/kimi_k25_eagle3_dcp}"

date
docker ps --filter name="$NAME" --format '{{.Names}} {{.Image}} {{.Status}}'
echo "--- log tail ---"
tail -240 "$RUN/server.log" || true
echo "--- gpu ---"
rocm-smi --showuse --showmemuse --showpidgpus | tail -100 || true
