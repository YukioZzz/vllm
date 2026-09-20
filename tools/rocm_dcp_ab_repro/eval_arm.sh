#!/usr/bin/env bash
# Wait for an arm to come up, then score it and take both perf shapes.
#   usage: eval_arm.sh <arm> [node]
set -u
ARM="${1:?arm}"
NODE="${2:-pit2-p03-g02}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "waiting for $ARM on $NODE to report healthy"
ok=0
for i in $(seq 1 90); do
  code=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
    "curl -s -o /dev/null -w '%{http_code}' --max-time 8 http://127.0.0.1:8000/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ok=1; echo "healthy after ~$((i*10))s"; break; fi
  alive=$(timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
    "docker inspect -f '{{.State.Running}}' k3ab-$ARM 2>/dev/null" 2>/dev/null || echo unknown)
  if [ "$alive" = "false" ]; then
    echo "FAILED: container k3ab-$ARM exited before becoming healthy"
    timeout 60 ssh -o BatchMode=yes "$NODE" "docker logs --tail 60 k3ab-$ARM 2>&1"
    exit 4
  fi
  sleep 10
done
[ "$ok" = "1" ] || { echo "TIMEOUT waiting for health"; exit 5; }

bash "$HERE/gsm8k.sh" "$ARM" "$NODE" 200
bash "$HERE/perf.sh"  "$ARM" "$NODE" 16 2048  256
bash "$HERE/perf.sh"  "$ARM" "$NODE" 16 2048  256
bash "$HERE/perf.sh"  "$ARM" "$NODE" 8  16384 256
bash "$HERE/perf.sh"  "$ARM" "$NODE" 8  16384 256
echo "EVAL_DONE $ARM"
