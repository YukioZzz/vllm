#!/usr/bin/env bash
set -euo pipefail

WORK="${WORK:-/tmp/kimi_mtp_dcp_verify}"
ATOM="${ATOM:-rocm/atom-dev:vllm-v0.19.0-nightly_20260508_perf_prebuild}"

mkdir -p "$WORK"

echo "--- pull atom image if missing ---"
if ! docker image inspect "$ATOM" >/dev/null 2>&1; then
  docker pull "$ATOM"
fi

echo "--- export atom Triton ---"
rm -rf "$WORK/atom_triton_site"
mkdir -p "$WORK/atom_triton_site"
docker run --rm --entrypoint bash \
  -v "$WORK/atom_triton_site:/out" \
  "$ATOM" \
  -lc 'cp -a /opt/venv/lib/python3.12/site-packages/triton /out/; cp -a /opt/venv/lib/python3.12/site-packages/triton-*.dist-info /out/ 2>/dev/null || true; python3 -c "import triton; print(triton.__version__, triton.__file__)"'

echo "--- fetch PR3402 final mla_decode_gluon ---"
curl -fL \
  https://raw.githubusercontent.com/ROCm/aiter/85241dac6e42ff72442b335a79b59b157ca7e180/aiter/ops/triton/gluon/mla_decode_gluon.py \
  -o "$WORK/mla_decode_gluon_pr3402_final.py"

ls -ld "$WORK/atom_triton_site/triton" "$WORK/mla_decode_gluon_pr3402_final.py"
