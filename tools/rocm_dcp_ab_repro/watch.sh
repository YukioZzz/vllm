#!/usr/bin/env bash
# Compact progress snapshot for a running arm.
#   usage: pit2_watch.sh <arm> [node] [tail_lines]
set -u
ARM="${1:?arm}"
NODE="${2:-pit2-p03-g02}"
N="${3:-25}"

timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" "ARM='$ARM' N='$N' bash -s" <<'REMOTE'
set -u
NAME="k3ab-$ARM"
echo "=== $(date -u '+%H:%M:%SZ') container ==="
docker ps -a --filter "name=$NAME" --format '{{.Names}} | {{.Status}}' 2>/dev/null || echo "(none)"

echo
echo "=== health ==="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
echo "  /health = $code"

echo
echo "=== gpu vram ==="
rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1{printf "  %s used=%.1fGiB\n",$1,$3/1073741824}'

echo
echo "=== milestones seen so far ==="
docker logs "$NAME" 2>&1 | grep -aiE \
  'ab mode|ab supported|native supp|Loading safetensors|loading weights took|Using .*backend|attention backend|DCP|decode context|speculative|Capturing CUDA graph|graph capturing finished|GPU KV cache size|maximum concurrency|Application startup complete|Traceback|Error|ERROR|assert|Killed|fault' \
  | tail -30

echo
echo "=== last $N lines ==="
docker logs --tail "$N" "$NAME" 2>&1 | tail -"$N"
REMOTE
