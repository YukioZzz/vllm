# Kimi EAGLE3.1 MLA DCP validation helpers

This directory contains ROCm validation helpers used to compare Kimi-K2.5-MXFP4
target-only decoding against `lightseekorg/kimi-k2.6-eagle3.1-mla` EAGLE3
speculative decoding with DCP enabled. The scripts are intentionally isolated from
vLLM core code and are meant for bring-up / regression validation.

## What is covered

- Kimi-K2.5-MXFP4 target serving with `ROCM_AITER_MLA`.
- Optional EAGLE3.1 MLA draft model via vLLM `--speculative-config`.
- DCP=2 validation using a Gluon MLA decode `return_lse=True` patch.
- AITER MXFP4 MoE selection for Kimi under ROCm.
- GSM8K 20-shot exact-match and throughput smoke benchmarking.

## Provider model

`patch_vllm_gluon_dcp.py` installs a small MLA DCP LSE provider module next to
vLLM's ROCm MLA backend and injects only a minimal hook into
`AiterMLAImpl.forward_mqa`.

The current provider is selected by:

```bash
export VLLM_ROCM_MLA_DCP_LSE_PROVIDER=gluon
```

Only the `gluon` provider is implemented today. The split is intentional: future
Kimi / DeepSeek / DSV MLA variants can add another provider, such as an AITER
ASM/CK provider with native LSE return, without growing model-specific logic in
the backend patch.

## Important caveats

This is not a production path. Current validation uses global atom Triton and a
Torch fallback for `merge_attn_states` to exercise the DCP LSE path before native
AITER MLA decode returns LSE.

The preferred production direction is to return LSE directly from the AITER
ASM/CK MLA decode/verify path and remove the Gluon/Torch fallback requirements.

## Typical workflow

Set a runtime work directory and host HF cache. The scripts mount the host HF
cache into containers at `/data/hf-hub-cache`.

```bash
export WORK=/it-share/yichaozhu/aiter_mtp_dcp_verify_g07
export HOST_HF_CACHE=/it-share/hf_cache
export HOST_AITER_SRC=$WORK/empty_aiter
export IMG=vllm/vllm-openai-rocm:nightly-09663abde0f50944a8d5ea30120666024b503faa
```

Prepare the validation runtime and draft model:

```bash
bash benchmarks/kimi_mtp_dcp/prepare_runtime.sh
HF_CACHE=$HOST_HF_CACHE DOCKER_NETWORK_ARGS="--network host" \
  bash benchmarks/kimi_mtp_dcp/prepare_kimi_eagle31_mla.sh
HF_CACHE=$HOST_HF_CACHE bash benchmarks/kimi_mtp_dcp/fix_kimi_blobs_root.sh
```

Start target-only baseline:

```bash
RUN=$WORK/run_baseline NAME=kimi_k25_target_dcp \
TARGET=/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/snapshots/<snapshot> \
bash benchmarks/kimi_mtp_dcp/start_target_dcp.sh
```

Start EAGLE3.1 MLA DCP:

```bash
RUN=$WORK/run_enabled NAME=kimi_k25_eagle31_mla_dcp \
TARGET=/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/snapshots/<snapshot> \
DRAFT=/data/hf-hub-cache/models--lightseekorg--kimi-k2.6-eagle3.1-mla/snapshots/<snapshot> \
bash benchmarks/kimi_mtp_dcp/start_eagle31_mla_dcp.sh
```

Run GSM8K 20-shot smoke:

```bash
RUN=$WORK/run_enabled PORT=18089 MODEL=/data/hf-hub-cache/models--amd--Kimi-K2.5-MXFP4/snapshots/<snapshot> \
LABEL=gsm8k_20shot_enabled N=8 CONCURRENCY=1 MAX_TOKENS=256 N_SHOTS=20 \
bash benchmarks/kimi_mtp_dcp/bench_gsm8k.sh
```
