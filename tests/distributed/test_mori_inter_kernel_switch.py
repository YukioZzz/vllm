# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoRI EP inter-node kernel auto-switch (SGLang #18437 port).

Validates that ``MoriAll2AllManager._make_all2all_kwargs`` selects the
low-latency ``InterNodeV1LL`` MoRI kernel when ``max_num_tokens_per_dp_rank``
is at or below ``VLLM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD`` and
falls back to the throughput-oriented ``InterNodeV1`` kernel otherwise.

Single-node ``IntraNode`` dispatch must be unaffected.

The test bypasses the heavy ``__init__`` (which registers a torch process
group and calls into ``mori.shmem``) by constructing the manager via
``__new__`` and only invoking ``_make_all2all_kwargs`` directly.  ``mori``
itself is stubbed so the test runs without ROCm.
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import pytest
import torch


def _install_mori_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a minimal ``mori`` stub with just the ``KernelType`` enum
    values that the manager picks between."""
    stub = types.ModuleType("mori")
    stub.__spec__ = ModuleSpec("mori", loader=None)
    stub_ops = types.ModuleType("mori.ops")
    stub_ops.__spec__ = ModuleSpec("mori.ops", loader=None)

    class KernelType:
        IntraNode = "IntraNode"
        InterNodeV1 = "InterNodeV1"
        InterNodeV1LL = "InterNodeV1LL"

    stub_ops.EpDispatchCombineKernelType = KernelType
    stub_ops.EpDispatchCombineOp = type("EpDispatchCombineOp", (), {})
    stub.ops = stub_ops
    monkeypatch.setitem(sys.modules, "mori", stub)
    monkeypatch.setitem(sys.modules, "mori.ops", stub_ops)


@pytest.fixture
def mgr(monkeypatch: pytest.MonkeyPatch):
    """A bare ``MoriAll2AllManager`` with all GPU / mori machinery stubbed."""
    _install_mori_stub(monkeypatch)
    # ``_make_all2all_kwargs`` asserts running on gfx942/gfx950 -- force it.
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)

    from vllm.distributed.device_communicators.all2all import MoriAll2AllManager

    m = MoriAll2AllManager.__new__(MoriAll2AllManager)
    return m


def _call(mgr, *, max_tokens: int):
    return mgr._make_all2all_kwargs(
        rank=0,
        num_ep_ranks=16,
        input_dtype=torch.bfloat16,
        quant_dtype=torch.bfloat16,
        token_hidden_size=4096,
        scale_dim=0,
        scale_type_size=0,
        max_num_tokens_per_dp_rank=max_tokens,
        num_local_experts=8,
        num_experts_per_token=8,
    )


def test_intra_node_unaffected_by_threshold(mgr, monkeypatch):
    """When ``internode=False`` the threshold is irrelevant: always IntraNode."""
    mgr.internode = False
    monkeypatch.setenv("VLLM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", "256")

    for n_tokens in (1, 255, 256, 257, 4096):
        kwargs = _call(mgr, max_tokens=n_tokens)
        assert kwargs["kernel_type"] == "IntraNode", n_tokens


@pytest.mark.parametrize(
    "threshold, n_tokens, expected",
    [
        # Default threshold (256, SGLang's tuned crossover).
        (256, 1, "InterNodeV1LL"),
        (256, 255, "InterNodeV1LL"),
        (256, 256, "InterNodeV1LL"),  # boundary: <= threshold -> LL
        (256, 257, "InterNodeV1"),
        (256, 4096, "InterNodeV1"),
        # Custom thresholds.
        (128, 128, "InterNodeV1LL"),
        (128, 129, "InterNodeV1"),
        # Pathological thresholds still pick something well-defined.
        (0, 1, "InterNodeV1"),
        (1_000_000, 4096, "InterNodeV1LL"),
    ],
)
def test_inter_node_switch(mgr, monkeypatch, threshold, n_tokens, expected):
    mgr.internode = True
    monkeypatch.setenv(
        "VLLM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", str(threshold)
    )
    kwargs = _call(mgr, max_tokens=n_tokens)
    assert kwargs["kernel_type"] == expected


def test_inter_node_default_threshold_is_256(mgr, monkeypatch):
    """Unset env var -> default = 256 (matches SGLang #18437)."""
    mgr.internode = True
    monkeypatch.delenv(
        "VLLM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", raising=False
    )

    assert _call(mgr, max_tokens=256)["kernel_type"] == "InterNodeV1LL"
    assert _call(mgr, max_tokens=257)["kernel_type"] == "InterNodeV1"
