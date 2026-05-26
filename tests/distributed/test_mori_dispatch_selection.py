# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoRI EP dispatch / combine dtype auto-selection.

Covers the SGLang #21040 port:

* ``pick_mori_dispatch_combine_dtypes`` reads the *weight* quant dtype
  (``_w1.dtype``) and picks the right MoRI dispatch+combine pair.
* ``VLLM_MORI_DISPATCH_DTYPE`` / ``VLLM_MORI_COMBINE_DTYPE`` env-style
  overrides win over auto-detection.
* ``combine_dtype_to_mori_quant_type`` maps the combine enum back to the
  string MoRI's ``EpDispatchCombineConfig`` expects.

The helper is intentionally a free function so the test runs without
GPUs, the ROCm ``mori`` binding, or any cluster setup.
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

# Stub ``mori`` so that ``vllm.model_executor.layers.fused_moe.prepare_finalize``
# imports cleanly on hosts without the ROCm MoRI package.  The helper under
# test only touches torch dtypes and the local Enum classes; the stub
# satisfies the unconditional ``import mori`` at the top of ``mori.py``.
if "mori" not in sys.modules:
    _stub = types.ModuleType("mori")
    _stub.__spec__ = ModuleSpec("mori", loader=None)
    _stub_ops = types.ModuleType("mori.ops")
    _stub_ops.__spec__ = ModuleSpec("mori.ops", loader=None)
    _stub_ops.EpDispatchCombineOp = type("EpDispatchCombineOp", (), {})
    _stub.ops = _stub_ops
    sys.modules["mori"] = _stub
    sys.modules["mori.ops"] = _stub_ops

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
    CombineDtype,
    DispatchDtype,
    combine_dtype_to_mori_quant_type,
    pick_mori_dispatch_combine_dtypes,
)

_FP4 = (DispatchDtype.fp4, CombineDtype.fp8)
_FP8 = (DispatchDtype.fp8, CombineDtype.bf16)
_BF16 = (DispatchDtype.bf16, CombineDtype.bf16)


# -- auto-detect table -------------------------------------------------------


@pytest.mark.parametrize(
    "weight_dtype, expected",
    [
        ("mxfp4", _FP4),
        pytest.param(
            getattr(torch, "float4_e2m1fn_x2", None),
            _FP4,
            id="float4_e2m1fn_x2",
            marks=pytest.mark.skipif(
                not hasattr(torch, "float4_e2m1fn_x2"),
                reason="torch.float4_e2m1fn_x2 not available",
            ),
        ),
        ("fp8", _FP8),
        (torch.float8_e4m3fn, _FP8),
        (torch.float8_e4m3fnuz, _FP8),
        (None, _BF16),
        (torch.bfloat16, _BF16),
        (torch.float16, _BF16),
        ("int8", _BF16),  # unrecognized -> default to bf16 dispatch
    ],
)
def test_auto_detect(weight_dtype, expected):
    assert pick_mori_dispatch_combine_dtypes(weight_dtype) == expected


# -- dispatch override -------------------------------------------------------


@pytest.mark.parametrize(
    "override, expected_dispatch",
    [
        ("auto", DispatchDtype.fp4),  # auto-detected from MXFP4 weights
        ("AUTO", DispatchDtype.fp4),  # case-insensitive
        ("bf16", DispatchDtype.bf16),
        ("fp8", DispatchDtype.fp8),
        ("fp4", DispatchDtype.fp4),
    ],
)
def test_dispatch_override_valid(override, expected_dispatch):
    dispatch, _ = pick_mori_dispatch_combine_dtypes(
        "mxfp4", dispatch_override=override
    )
    assert dispatch == expected_dispatch


def test_dispatch_override_invalid_falls_back_to_auto():
    """A nonsense override must not raise -- log + keep auto-detected value."""
    dispatch, _ = pick_mori_dispatch_combine_dtypes(
        "mxfp4", dispatch_override="garbage"
    )
    assert dispatch == DispatchDtype.fp4


# -- combine override --------------------------------------------------------


@pytest.mark.parametrize(
    "override, expected_combine",
    [
        ("auto", CombineDtype.bf16),
        ("bf16", CombineDtype.bf16),
        ("fp8", CombineDtype.fp8),
        ("fp8_direct_cast", CombineDtype.fp8_direct_cast),
    ],
)
def test_combine_override_valid(override, expected_combine):
    _, combine = pick_mori_dispatch_combine_dtypes(
        "fp8", combine_override=override
    )
    assert combine == expected_combine


def test_combine_override_invalid_falls_back_to_auto():
    _, combine = pick_mori_dispatch_combine_dtypes(
        "fp8", combine_override="totally_made_up"
    )
    assert combine == CombineDtype.bf16


def test_overrides_are_independent():
    """Dispatch + combine overrides do not influence one another."""
    dispatch, combine = pick_mori_dispatch_combine_dtypes(
        None,  # bf16/bf16 by auto-detect
        dispatch_override="fp4",
        combine_override="fp8",
    )
    assert dispatch == DispatchDtype.fp4
    assert combine == CombineDtype.fp8


# -- enum -> mori string mapping --------------------------------------------


@pytest.mark.parametrize(
    "combine_dtype, expected_str",
    [
        (CombineDtype.bf16, "none"),
        (CombineDtype.fp8, "fp8_blockwise"),
        (CombineDtype.fp8_direct_cast, "fp8_direct_cast"),
    ],
)
def test_combine_dtype_to_mori_quant_type(combine_dtype, expected_str):
    assert combine_dtype_to_mori_quant_type(combine_dtype) == expected_str
