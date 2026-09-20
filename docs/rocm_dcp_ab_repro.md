# ROCm AITER DCP verify arms (CPRR / mask0 / gluon) — reproduce

Branch: [`yichaozhu/rocm-aiter-dcp-cprr`](https://github.com/YukioZzz/vllm/tree/yichaozhu/rocm-aiter-dcp-cprr)
on fork [`YukioZzz/vllm`](https://github.com/YukioZzz/vllm).

Three DSpark **target-verify** routes under TP8/DCP8 (`cp_interleave=1`):

| Arm | Env | What it does |
|---|---|---|
| **off** (native CPRR) | `VLLM_ROCM_AITER_NATIVE_DCP_VERIFY=1`<br>`VLLM_ROCM_AITER_DCP_AB_VERIFY=off` | One AITER MLA decode with in-kernel round-robin causal mask |
| **mask0** | `NATIVE=0`<br>`VLLM_ROCM_AITER_DCP_AB_VERIFY=mask0` | A/B split: Stage A non-causal prefix (same MLA entry, `causal=False`); Stage B gluon 4-D MTP window |
| **gluon** | `NATIVE=0`<br>`VLLM_ROCM_AITER_DCP_AB_VERIFY=gluon` | A/B split: Stage A **flattened** gluon (one row per query token); Stage B same MTP window |

Tip of the five DCP commits (as of this doc): `9e616f00fe` … `41a987d9d3`.

## Image / hardware

- **Image:** `vllm/vllm-openai-rocm:nightly` — a floating tag, so pin the digest we
  actually measured on:

  ```bash
  docker pull vllm/vllm-openai-rocm@sha256:5550994c1874ef331c6aed3ea27ae1efb7f4cfd17a2f18f5466c2837b9bf5467
  ```

  | field | value |
  |---|---|
  | digest | `sha256:5550994c1874ef331c6aed3ea27ae1efb7f4cfd17a2f18f5466c2837b9bf5467` |
  | image id | `5550994c1874` (49 GB) |
  | created | `2026-09-18T05:16:56Z` |
  | in-image vLLM | `0.3.1.dev85+gdee37d891.rocm723` |

  This build already ships AITER with the CPRR four-arg `mla_decode_fwd`
  (`g_kv_indptr`, `cp_world_size`, `cp_rank`, `causal`), gluon with
  `return_lse` / `use_2d_view` / `min_kv_seq_len`, and the segmented Triton MLA
  (`aiter.ops.triton.attention.mla`). AITER itself reports no version string in
  this image, so gate on signatures rather than a version number.
- **GPU:** AMD **MI355X (gfx950)**, 8 GPUs. Gluon Stage B is gfx950-only; do not expect A/B on MI300X without a different Stage B.
- **Model:** Kimi-K3 + DSpark draft (paths below are PIT2 NFS; substitute your own).

Overlay **only** these two files from the branch into the image site-packages (or install the branch editable):

- `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`
- `vllm/model_executor/layers/attention/mla_attention.py`

## Server recipe (matched across arms)

```bash
export VLLM_ROCM_USE_AITER=1
export HSA_NO_SCRATCH_RECLAIM=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export SAFETENSORS_FAST_GPU=1
export HF_HUB_OFFLINE=1

# pick one arm:
export VLLM_ROCM_AITER_NATIVE_DCP_VERIFY=1   # off
export VLLM_ROCM_AITER_DCP_AB_VERIFY=off
#   or NATIVE=0 + AB=mask0 / gluon

python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Kimi-K3 \
  --served-model-name Kimi-K3 \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 256 \
  --speculative-config '{"method":"dspark","model":"/draft","num_speculative_tokens":3}' \
  --host 0.0.0.0 --port 8000 \
  --no-enable-log-requests
```

**fp8 KV:** add `--kv-cache-dtype fp8`. A/B arms need more headroom than CPRR
(graph ~24 GiB vs ~8 GiB bf16); use e.g. `--gpu-memory-utilization 0.90 --max-num-seqs 64`
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if A/B OOMs at 0.85/256.

**Weights tip:** 8 TP workers each read the full ~1.5 TB checkpoint. Stage to
node-local disk first if NFS cannot sustain ~8× streaming.

## What we scored (for calibration)

Same recipe, bf16 KV, `max_num_seqs=256`, DSpark K=3, GSM8K 200×5-shot greedy:

- Accuracy: all three arms **196/200 = 0.98** (identical running accuracy every 25 items).
- Short (c16 / ISL 2048 / OSL 256): CPRR ≈ gluon ≈ 870 tok/s; mask0 ≈ 700 tok/s.
- Long (c8 / ISL 16384 / OSL 256): CPRR ≈ 673 > mask0 ≈ 642 > gluon ≈ 613 tok/s.

fp8 (matched `util=0.90`, `max_num_seqs=64`): CPRR still fastest; A/B needs the lower concurrency.

## Cluster helper scripts (optional)

Under this tree: `tools/rocm_dcp_ab_repro/`. They assume a jump host that can
`ssh` to an idle MI355X node and a checkout at `$HOME/k3ab/vllm` on that node.

```text
tools/rocm_dcp_ab_repro/
  run_arm.sh      # docker launch + overlay; arms: off|mask0|gluon[+ -fp8]
  eval_arm.sh     # wait healthy → GSM8K 200 → short+long perf
  gsm8k.sh        # accuracy only
  perf.sh         # fixed-shape tok/s + accept_len
  cleanup.sh      # remove k3ab-* containers
```

Example (from a host that can reach the GPU node):

```bash
bash tools/rocm_dcp_ab_repro/run_arm.sh off     pit2-p03-g02
bash tools/rocm_dcp_ab_repro/eval_arm.sh off    pit2-p03-g02
bash tools/rocm_dcp_ab_repro/run_arm.sh mask0   pit2-p03-g02
bash tools/rocm_dcp_ab_repro/eval_arm.sh mask0  pit2-p03-g02
bash tools/rocm_dcp_ab_repro/run_arm.sh gluon   pit2-p03-g02
bash tools/rocm_dcp_ab_repro/eval_arm.sh gluon  pit2-p03-g02
bash tools/rocm_dcp_ab_repro/cleanup.sh         pit2-p03-g02
```

Edit `MODEL` / `DRAFT` / `IMG` inside `run_arm.sh` if your paths differ from PIT2
(`/share_nfs/models/moonshotai/Kimi-K3`, `/share_nfs/models/Inferact/Kimi-K3-DSpark`).

## Env semantics (gotchas)

- `VLLM_ROCM_AITER_NATIVE_DCP_VERIFY` is read under `@lru_cache` — set it **before** the first MLA call (i.e. process start). Default is off (`0`); set `1` for the CPRR arm.
- `VLLM_ROCM_AITER_DCP_AB_VERIFY` ∈ `{off, mask0, gluon}`. When not `off`, native CPRR verify is not selected for the target block.
- Both native CPRR and A/B require `decode_context_parallel_size > 1` and `cp_kv_cache_interleave_size == 1`.
