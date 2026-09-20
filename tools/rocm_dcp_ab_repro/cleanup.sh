#!/usr/bin/env bash
set -u
NODE="${1:-pit2-p03-g02}"
timeout 600 ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" "bash -s" <<'REMOTE'
set -u
echo "host=$(hostname) $(date -u '+%H:%M:%SZ')"
for c in $(docker ps -aq --filter 'name=^k3ab-'); do
  docker rm -f "$c" >/dev/null 2>&1 || true
done
echo "containers left:"
docker ps --format '{{.Names}} {{.Status}}' | head -5 || echo "(none running)"
if [ -d /tmp/k3-local ]; then
  du -sh /tmp/k3-local 2>/dev/null || true
fi
sleep 8
rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 && $1 ~ /card/{printf "  %s used=%.1fGiB\n",$1,$3/1073741824}'
df -h /tmp | tail -1
echo "cleaned"
REMOTE
