# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm MoE utility kernels.

Provides a Triton kernel that upscales packed MXFP4 (``float4_e2m1fn_x2``)
hidden states with ``float8_e8m0fnu`` per-block-32 scales back to a regular
FP16/BF16/FP32 tensor.  This is needed when the MoRI dispatcher transfers
MXFP4-quantized activations to reduce communication bandwidth, but the
downstream fused-MoE expert kernel expects unpacked BF16/FP16 inputs.

Ported from SGLang (PR #19757 + #24879) for vLLM's MoRI EP backend.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def upscale_fp4x2_block32_kernel(
    A_u8_ptr,  # *uint8  (view from float4_e2m1fn_x2)
    S_u8_ptr,  # *uint8  (view from float8_e8m0fnu), shape (M, N_fp4/32)
    Out_ptr,  # *fp16/fp32/bf16, shape (M, N_fp4)
    N_FP4: tl.constexpr,
    recv_token_num,
    stride_am,
    stride_an,  # A strides (in uint8 elements) for (M, packed_N)
    stride_sm,
    stride_sn,  # S strides (in uint8 elements) for (M, N_FP4/32)
    stride_om,
    stride_on,  # Out strides (in output elements) for (M, N_FP4)
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,  # tl.float16 / tl.float32 / tl.bfloat16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    recv_token_num_val = tl.load(recv_token_num)
    if pid_m >= recv_token_num_val:
        return

    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_FP4

    # Load packed fp4x2 byte: each byte holds two FP4 values (low/high nibble).
    byte_idx = offs >> 1
    is_hi = (offs & 1) != 0

    a_ptrs = A_u8_ptr + pid_m * stride_am + byte_idx * stride_an
    a_byte = tl.load(a_ptrs, mask=mask, other=0).to(tl.int32)

    lo = a_byte & 0xF
    hi = (a_byte >> 4) & 0xF
    code = tl.where(is_hi, hi, lo).to(tl.int32)  # 0..15

    # Decode float4_e2m1fn  layout: [sign|exp(2)|mant(1)] with bias=1, finite-only.
    sign = (code >> 3) & 0x1
    exp = (code >> 1) & 0x3
    mant = code & 0x1

    mant_f = mant.to(tl.float32) * 0.5
    is_sub = exp == 0

    # normal: 2^(exp-bias) * (1 + mant/2), bias=1
    e_norm = (exp - 1).to(tl.float32)
    val_norm = tl.exp2(e_norm) * (1.0 + mant_f)

    # subnormal/zero: mant/2 * 2^(1-bias) = mant/2
    val_sub = mant_f

    val = tl.where(is_sub, val_sub, val_norm)
    val = tl.where(sign != 0, -val, val)

    # Per-token block-32 scale (float8_e8m0fnu in uint8):
    #   e == 0       -> 0
    #   e in 1..254  -> 2^(e - 127)
    #   e == 255     -> clamped to 254
    scale_idx = offs >> 5  # offs // 32

    s_ptrs = S_u8_ptr + pid_m * stride_sm + scale_idx * stride_sn
    e = tl.load(s_ptrs, mask=mask, other=0).to(tl.int32)

    e = tl.minimum(e, 254)
    is_zero = e == 0
    exp_s = (e - 127).to(tl.float32)
    s = tl.exp2(exp_s)
    s = tl.where(is_zero, 0.0, s)

    out = (val * s).to(OUT_DTYPE)

    out_ptrs = Out_ptr + pid_m * stride_om + offs * stride_on
    tl.store(out_ptrs, out, mask=mask)


def upscale_mxfp4(
    hidden_state: torch.Tensor,
    hidden_state_scale: torch.Tensor,
    recv_token_num: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP4 hidden states to ``output_dtype``.

    Args:
        hidden_state: ``(M, packed_N)`` ``torch.float4_e2m1fn_x2`` tensor where
            each byte stores two FP4 values (so ``N_fp4 = packed_N * 2``).
        hidden_state_scale: ``(M, N_fp4 / 32)`` ``torch.float8_e8m0fnu`` tensor
            holding one shared exponent per 32-element FP4 group.
        recv_token_num: a 0-d tensor with the actual number of valid rows
            (rows past this index are skipped, matching the MoRI dispatcher
            zero-padded layout).
        output_dtype: ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.

    Returns:
        ``(M, N_fp4)`` tensor in ``output_dtype``.
    """
    assert hidden_state.dtype == torch.float4_e2m1fn_x2, hidden_state.dtype
    assert hidden_state_scale.dtype == torch.float8_e8m0fnu, hidden_state_scale.dtype

    M, packed_N = hidden_state.shape
    N_fp4 = packed_N * 2

    assert hidden_state_scale.shape[0] == M
    assert hidden_state_scale.shape[1] == (N_fp4 // 32), (
        hidden_state_scale.shape,
        N_fp4,
    )

    # Triton does not (reliably) accept torch.float4 / float8 pointers directly,
    # so reinterpret as raw uint8.
    A_u8 = hidden_state.view(torch.uint8)
    S_u8 = hidden_state_scale.view(torch.uint8)

    out = torch.empty((M, N_fp4), dtype=output_dtype, device=hidden_state.device)

    BLOCK_N = 256
    grid = (M, triton.cdiv(N_fp4, BLOCK_N))

    if output_dtype == torch.float16:
        out_tl = tl.float16
    elif output_dtype == torch.bfloat16:
        out_tl = tl.bfloat16
    else:
        out_tl = tl.float32

    upscale_fp4x2_block32_kernel[grid](
        A_u8,
        S_u8,
        out,
        N_FP4=N_fp4,
        recv_token_num=recv_token_num,
        stride_am=A_u8.stride(0),
        stride_an=A_u8.stride(1),
        stride_sm=S_u8.stride(0),
        stride_sn=S_u8.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_N=BLOCK_N,
        OUT_DTYPE=out_tl,
        num_warps=4,
    )
    return out
