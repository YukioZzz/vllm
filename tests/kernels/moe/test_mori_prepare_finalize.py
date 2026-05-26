# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Python unit tests for the MoRI prepare/finalize plumbing.

These tests exercise the enum + quant-type-string mapping that drives
the MoRI EP dispatcher's runtime configuration, without touching any
GPU / RDMA paths.  They still require ``mori`` to be importable because
``vllm...prepare_finalize.mori`` does ``import mori`` at module top.
"""

import pytest

pytest.importorskip("mori")

from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
    CombineDtype,
    DispatchDtype,
    MoriPrepareAndFinalize,
    combine_dtype_to_mori_quant_type,
)


def test_combine_dtype_to_mori_quant_type():
    """The string returned here is fed verbatim into MoRI's
    ``EpDispatchCombineConfig.quant_type`` field; if these strings drift,
    MoRI silently falls back to no-quant combine -> accuracy regression.
    """
    assert combine_dtype_to_mori_quant_type(CombineDtype.bf16) == "none"
    assert combine_dtype_to_mori_quant_type(CombineDtype.fp8) == "fp8_blockwise"
    assert (
        combine_dtype_to_mori_quant_type(CombineDtype.fp8_direct_cast)
        == "fp8_direct_cast"
    )


def test_dispatch_dtype_string_values_stable():
    """Hard-pin the enum string values: they surface in logs and double
    as identifiers for the MoRI dispatch activation dtype.  Renaming
    them must be an intentional, reviewed change."""
    assert DispatchDtype.bf16.value == "bfloat16"
    assert DispatchDtype.fp8.value == "float8_blockwise"
    assert DispatchDtype.fp4.value == "mxfp4_blockwise"


def test_combine_dtype_string_values_stable():
    assert CombineDtype.bf16.value == "bfloat16"
    assert CombineDtype.fp8.value == "float8_blockwise"
    assert CombineDtype.fp8_direct_cast.value == "float8_direct_cast"


@pytest.mark.parametrize(
    "dispatch_dtype, expected_use_fp8",
    [
        (DispatchDtype.bf16, False),
        (DispatchDtype.fp8, True),
        (DispatchDtype.fp4, False),
    ],
)
def test_use_fp8_dispatch_accessor(dispatch_dtype, expected_use_fp8):
    """``use_fp8_dispatch`` is a back-compat accessor for callers that
    pre-date the enum-based API.  It must stay in sync with the enum."""
    pf = MoriPrepareAndFinalize.__new__(MoriPrepareAndFinalize)
    pf.dispatch_dtype = dispatch_dtype
    assert pf.use_fp8_dispatch is expected_use_fp8
