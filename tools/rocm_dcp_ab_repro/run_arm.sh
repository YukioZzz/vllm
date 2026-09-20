#!/usr/bin/env bash
# Launch one K3 TP8/DCP8 + DSpark arm on a pit2 node and hold it for eval.
#   usage: pit2_run_arm.sh <arm: off|mask0|gluon> [node]
# Runs inside the node via ssh; the container stays up under a known name.
set -u
# An "-fp8" suffix (off-fp8, mask0-fp8) switches the KV cache to fp8; the
# verify-path env stays keyed off the base arm name.
ARM="${1:?arm required: off|mask0|gluon, optional -fp8 suffix}"
NODE="${2:-pit2-p03-g02}"
# fp8 KV pushes graph capture to ~24 GiB, so 0.85 leaves the A/B arms with a
# negative KV budget. Comparable arms must share whatever value is used.
UTIL="${3:-0.85}"
MAXSEQS="${4:-256}"
# Verify query length is 1 + K, so K selects the qlen the persistent path sees.
NSPEC="${5:-3}"

timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" \
  "ARM='$ARM' UTIL='$UTIL' MAXSEQS='$MAXSEQS' NSPEC='$NSPEC' bash -s" <<'REMOTE'
set -u
IMG=vllm/vllm-openai-rocm:nightly
SRC="/home/$(whoami)/k3ab/vllm"
# Eight TP workers each read the whole checkpoint, so NFS would serve
# 8 x 1.5 TB and collapse to ~200 MB/s. Prefer the node-local stage when
# pit2_stage_local.sh has laid it down.
if [ -f /tmp/k3-local/Kimi-K3/.stage_complete ]; then
  MODEL=/tmp/k3-local/Kimi-K3
  DRAFT=/tmp/k3-local/Kimi-K3-DSpark
  echo "using node-local weights"
else
  MODEL=/share_nfs/models/moonshotai/Kimi-K3
  DRAFT=/share_nfs/models/Inferact/Kimi-K3-DSpark
  echo "WARNING: using NFS weights; expect a multi-hour load"
fi
RUNROOT="/home/$(whoami)/k3ab/runs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$RUNROOT/${STAMP}-${ARM}"
NAME="k3ab-$ARM"

case "$ARM" in
  *-fp8) BASE="${ARM%-fp8}"; KVFLAG="--kv-cache-dtype fp8" ;;
  *)     BASE="$ARM";        KVFLAG="" ;;
esac
case "$BASE" in
  off)    NATIVE=1; AB=off   ;;
  mask0)  NATIVE=0; AB=mask0 ;;
  gluon)  NATIVE=0; AB=gluon ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac

mkdir -p "$RUN"
echo "RUN=$RUN"
echo "ARM=$ARM NATIVE_DCP_VERIFY=$NATIVE AB_VERIFY=$AB KV='${KVFLAG:-bf16}' UTIL=$UTIL MAXSEQS=$MAXSEQS K=$NSPEC qlen=$((1+NSPEC))"

# Arms are mutually exclusive: the env is read once per process, so switching
# arms means a fresh server. Retire every sibling before measuring occupancy,
# or the guard trips on our own outgoing run.
for prev in $(docker ps -aq --filter 'name=^k3ab-'); do
  docker rm -f "$prev" >/dev/null 2>&1 || true
done
for _ in $(seq 1 30); do
  held=$(rocm-smi --showmeminfo vram --csv 2>/dev/null \
    | awk -F, 'NR>1 && $3+0 > 2147483648 {n++} END{print n+0}')
  [ "${held:-0}" -eq 0 ] && break
  sleep 5
done

GRP=""
for g in video render; do
  gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && GRP="$GRP --group-add $gid"
done

# Be a good neighbour: refuse to launch if someone else already has the GPUs.
BUSY=$(rocm-smi --showmeminfo vram --csv 2>/dev/null \
  | awk -F, 'NR>1 && $3+0 > 2147483648 {n++} END{print n+0}')
OTHER=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -v "^k3ab-" | head -5)
if [ "${BUSY:-0}" -gt 0 ]; then
  echo "ABORT: $BUSY GPU(s) already hold >2GiB on $(hostname); not stealing the node."
  echo "  other containers: ${OTHER:-none}"
  exit 9
fi

# Single node: RCCL has no business probing the fabric, and a silent network
# plugin retry is what a rendezvous hang looks like from the outside.
IBDEV=""
[ -e /dev/infiniband ] && IBDEV="--device=/dev/infiniband"

cat > "$RUN/entry.sh" <<'ENTRY'
#!/usr/bin/env bash
set -u
VP=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
echo "[entry] overlaying A/B verify onto $VP"
cp /mysrc/vllm/v1/attention/backends/mla/rocm_aiter_mla.py \
   "$VP/v1/attention/backends/mla/rocm_aiter_mla.py"
cp /mysrc/vllm/model_executor/layers/attention/mla_attention.py \
   "$VP/model_executor/layers/attention/mla_attention.py"
python3 -c "
from vllm.v1.attention.backends.mla import rocm_aiter_mla as m
print('[entry] ab mode      =', m._ab_dcp_verify_mode())
print('[entry] ab supported =', m._ab_dcp_verify_supported(8,1))
print('[entry] native supp  =', m._native_dcp_verify_supported(8,1))
"
echo "[entry] launching server"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Kimi-K3 \
  --served-model-name Kimi-K3 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --max-num-seqs "${MAX_SEQS:-256}" \
  ${KV_CACHE_FLAG:-} \
  --speculative-config '{"method":"dspark","model":"/draft","num_speculative_tokens":'"${NUM_SPEC:-3}"'}' \
  --host 0.0.0.0 --port 8000 \
  --no-enable-log-requests
ENTRY
chmod +x "$RUN/entry.sh"

docker run -d --name "$NAME" --network host \
  --device=/dev/kfd --device=/dev/dri $IBDEV $GRP \
  --cap-add=SYS_PTRACE \
  --ipc=host --shm-size=64g --security-opt seccomp=unconfined \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN \
  -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e GLOO_SOCKET_IFNAME=lo \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_AITER_NATIVE_DCP_VERIFY="$NATIVE" \
  -e VLLM_ROCM_AITER_DCP_AB_VERIFY="$AB" \
  -e KV_CACHE_FLAG="$KVFLAG" \
  -e GPU_UTIL="$UTIL" \
  -e MAX_SEQS="$MAXSEQS" \
  -e NUM_SPEC="$NSPEC" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HSA_NO_SCRATCH_RECLAIM=0 \
  -e PYTHONUNBUFFERED=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "$SRC:/mysrc:ro" \
  -v "$MODEL:/models/Kimi-K3:ro" \
  -v "$DRAFT:/draft:ro" \
  -v "$RUN:/run_out" \
  --entrypoint bash "$IMG" /run_out/entry.sh > "$RUN/container_id" 2>&1

echo "container=$(cat "$RUN/container_id" | cut -c1-12)"
ln -sfn "$RUN" "$RUNROOT/current-$ARM"
echo "logs: docker logs -f $NAME"
REMOTE
