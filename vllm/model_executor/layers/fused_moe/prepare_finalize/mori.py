# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import mori
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.rocm_moe_utils import upscale_mxfp4
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Block-wise quantization group sizes: number of elements that share one
# scale factor along the hidden dimension.
FP8_BLOCK_SIZE = 128
MXFP4_BLOCK_SIZE = 32


class DispatchDtype(str, Enum):
    """Activation dtype used for MoRI dispatch (rank-to-rank send)."""

    bf16 = "bfloat16"
    fp8 = "float8_blockwise"  # per-1x128 block-scaled FP8
    fp4 = "mxfp4_blockwise"  # per-1x32 block-scaled MXFP4 (e2m1)


class CombineDtype(str, Enum):
    """Activation dtype used for MoRI combine (rank-to-rank reduce)."""

    bf16 = "bfloat16"
    fp8 = "float8_blockwise"  # MoRI block-scaled FP8 combine
    fp8_direct_cast = "float8_direct_cast"  # naive FP8 cast (lower accuracy)


def combine_dtype_to_mori_quant_type(combine_dtype: "CombineDtype") -> str:
    """Map a CombineDtype to the string MoRI expects in EpDispatchCombineConfig."""
    if combine_dtype == CombineDtype.fp8:
        return "fp8_blockwise"
    if combine_dtype == CombineDtype.fp8_direct_cast:
        return "fp8_direct_cast"
    return "none"


def pick_mori_dispatch_combine_dtypes(
    weight_quant_dtype,
    *,
    dispatch_override: str = "auto",
    combine_override: str = "auto",
) -> tuple["DispatchDtype", "CombineDtype"]:
    """Pick the MoRI EP dispatch / combine activation dtypes from the
    *post-loaded* weight quantization dtype, honoring env-style overrides.

    Mirrors SGLang #21040: detect from the WEIGHT dtype (``_w1.dtype``)
    rather than the activation quant dtype (``_a1.dtype``).  The latter is
    misleading for mixed-precision MoE schemes such as W4A8 MXFP4 where
    ``activation=fp8`` but ``weights=mxfp4`` -- we must dispatch FP4.

    Accepts both the legacy string forms (``"mxfp4"``, ``"fp8"``) and the
    corresponding torch dtypes (``torch.float4_e2m1fn_x2``,
    ``torch.float8_e4m3fn``, ``torch.float8_e4m3fnuz``).

    ``dispatch_override`` / ``combine_override`` are the env-var values
    (``auto`` / ``bf16`` / ``fp8`` / ``fp4`` / ``fp8_direct_cast``).  An
    invalid override logs a warning and the auto-detected value is kept.
    """
    fp4_dtypes: tuple = ("mxfp4",)
    if hasattr(torch, "float4_e2m1fn_x2"):
        fp4_dtypes = fp4_dtypes + (torch.float4_e2m1fn_x2,)
    fp8_dtypes: tuple = (
        "fp8",
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )

    if weight_quant_dtype in fp4_dtypes:
        dispatch_dtype = DispatchDtype.fp4
        combine_dtype = CombineDtype.fp8
    elif weight_quant_dtype in fp8_dtypes:
        dispatch_dtype = DispatchDtype.fp8
        combine_dtype = CombineDtype.bf16
    else:
        dispatch_dtype = DispatchDtype.bf16
        combine_dtype = CombineDtype.bf16

    if dispatch_override and dispatch_override.lower() != "auto":
        mapping = {
            "bf16": DispatchDtype.bf16,
            "fp8": DispatchDtype.fp8,
            "fp4": DispatchDtype.fp4,
        }
        try:
            dispatch_dtype = mapping[dispatch_override.lower()]
        except KeyError:
            logger.warning_once(
                "VLLM_MORI_DISPATCH_DTYPE=%s is not supported "
                "(use auto|bf16|fp8|fp4); ignoring.",
                dispatch_override,
            )

    if combine_override and combine_override.lower() != "auto":
        mapping_c = {
            "bf16": CombineDtype.bf16,
            "fp8": CombineDtype.fp8,
            "fp8_direct_cast": CombineDtype.fp8_direct_cast,
        }
        try:
            combine_dtype = mapping_c[combine_override.lower()]
        except KeyError:
            logger.warning_once(
                "VLLM_MORI_COMBINE_DTYPE=%s is not supported "
                "(use auto|bf16|fp8|fp8_direct_cast); ignoring.",
                combine_override,
            )

    return dispatch_dtype, combine_dtype


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using MoRI kernels.

    Supports configurable dispatch / combine activation dtypes:

    * ``DispatchDtype.bf16``: full-precision dispatch (default).
    * ``DispatchDtype.fp8``: per-1x128 block-scaled FP8 dispatch (~2x BW save).
      Uses aiter ``QuantType.per_1x128``; MoRI moves ``fp8_dtype`` payload plus
      one fp32 scale per 128-elem block.
    * ``DispatchDtype.fp4``: per-1x32 block-scaled MXFP4 dispatch (~4x BW save).
      Uses aiter ``QuantType.per_1x32``; MoRI moves ``float4_e2m1fn_x2`` payload
      plus one ``float8_e8m0fnu`` scale per 32-elem block.  Dispatched payload
      is upcast back to ``output_dtype`` via ``upscale_mxfp4`` so that
      downstream fused-MoE expert kernels see the expected unpacked layout.

    Combine quantization is delegated to MoRI itself (``CombineDtype.fp8`` =
    block-scaled FP8 combine, recommended for MXFP4 weights).
    """

    def __init__(
        self,
        mori_op: mori.ops.EpDispatchCombineOp,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_dtype: DispatchDtype = DispatchDtype.bf16,
        combine_dtype: CombineDtype = CombineDtype.bf16,
        hidden_size: int | None = None,
    ):
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dispatch_dtype = dispatch_dtype
        self.combine_dtype = combine_dtype
        self.hidden_size = hidden_size

        # Lazily-initialized aiter quant funcs; importing aiter at module load
        # would force a hard ROCm dependency on non-AMD builds.
        self._fp8_quant_func = None
        self._fp4_quant_func = None

    @property
    def use_fp8_dispatch(self) -> bool:
        # Backwards-compatible accessor for callers that still inspect the bool.
        return self.dispatch_dtype == DispatchDtype.fp8

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def _get_fp8_quant_func(self):
        if self._fp8_quant_func is None:
            from aiter import QuantType, get_hip_quant

            self._fp8_quant_func = get_hip_quant(QuantType.per_1x128)
        return self._fp8_quant_func

    def _get_fp4_quant_func(self):
        if self._fp4_quant_func is None:
            from aiter import QuantType, get_hip_quant

            self._fp4_quant_func = get_hip_quant(QuantType.per_1x32)
        return self._fp4_quant_func

    def _quant_for_dispatch(
        self,
        a1: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize ``a1`` to the configured dispatch dtype.

        Returns ``(payload, scale)`` ready to feed ``mori_op.dispatch``.  Empty
        rank inputs (``num_tokens == 0``) are converted to correctly-shaped
        empty tensors so that MoRI's metadata still matches the dispatcher
        config; aiter's quant kernels can mishandle the zero-token case.
        """
        # ``defer_input_quant`` means the expert kernel will requant internally;
        # do not redo the quantization on the input tensor in that case.
        if defer_input_quant:
            return a1, None

        num_tokens = a1.shape[0]
        device = a1.device
        hidden_size = a1.shape[-1]

        if self.dispatch_dtype == DispatchDtype.fp8:
            if num_tokens > 0:
                # NOTE: aiter is able to handle token=0 case in UTs but fails at
                # e2e: build empty tensors directly for the zero-token case.
                a1, scale = self._get_fp8_quant_func()(
                    a1, quant_dtype=current_platform.fp8_dtype()
                )
            else:
                a1 = torch.empty(
                    a1.shape, dtype=current_platform.fp8_dtype(), device=device
                )
                scale = torch.empty(
                    (0, hidden_size // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=device,
                )
            return a1, scale

        if self.dispatch_dtype == DispatchDtype.fp4:
            if num_tokens > 0:
                a1, scale = self._get_fp4_quant_func()(a1, shuffle=False)
            else:
                a1 = torch.empty(
                    (0, hidden_size // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device=device,
                )
                scale = torch.empty(
                    (0, hidden_size // MXFP4_BLOCK_SIZE),
                    dtype=torch.float8_e8m0fnu,
                    device=device,
                )
            return a1, scale

        # bf16 / unquantized dispatch path -- but if the model is FP8 block /
        # per-token quantized and the expert kernel expects pre-quantized inputs
        # we still run the FP8 quant here for parity with previous behavior.
        if quant_config.is_block_quantized:
            from aiter import QuantType, get_hip_quant

            a1, scale = get_hip_quant(QuantType.per_1x128)(
                a1, quant_dtype=current_platform.fp8_dtype()
            )
            return a1, scale
        if quant_config.is_per_act_token:
            from aiter import QuantType, get_hip_quant

            a1, scale = get_hip_quant(QuantType.per_Token)(
                a1, quant_dtype=current_platform.fp8_dtype()
            )
            return a1, scale

        return a1, None

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Quantize, dispatch, then optionally upcast back to ``a1.dtype``.

        Returns the same 5-tuple as the original implementation:
        ``(dispatch_a1, dispatch_scale, expert_tokens_meta,
        dispatch_ids, dispatch_weights)``.
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True now."
        )

        output_dtype = a1.dtype
        a1, scale = self._quant_for_dispatch(a1, quant_config, defer_input_quant)

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(a1, topk_weights, scale, topk_ids)

        # When dispatching MXFP4 payloads, downstream expert kernels in vLLM
        # expect unpacked BF16/FP16 hidden_states (and re-quantize internally
        # when needed).  Upcast the dispatched FP4 buffer back to output_dtype
        # using the Triton upscale kernel.
        if (
            dispatch_a1.dtype == torch.float4_e2m1fn_x2
            and dispatch_scale is not None
        ):
            dispatch_a1 = upscale_mxfp4(
                dispatch_a1,
                dispatch_scale,
                dispatch_recv_token_num,
                output_dtype,
            )
            dispatch_scale = None

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        num_token = output.shape[0]
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
        )[0]
        output.copy_(result[:num_token])
