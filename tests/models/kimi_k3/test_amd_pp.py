# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of the AMD decoder's real PP forward and auxiliary relay."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.models.kimi_k3.amd import linear
from vllm.models.kimi_k3.amd.kda import KimiK3DeltaAttention
from vllm.sequence import IntermediateTensors


class AddLayer(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, positions, hidden_states, residual):
        return hidden_states + self.increment, residual


def make_stage(monkeypatch, start, end, taps):
    group = SimpleNamespace(
        world_size=2, is_first_rank=start == 0, is_last_rank=end == 4
    )
    monkeypatch.setattr(linear, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    model = linear.KimiLinearModel.__new__(linear.KimiLinearModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(attn_res_block_size=None)
    model.start_layer, model.end_layer = start, end
    model.layers = torch.nn.ModuleList(AddLayer(i) for i in (1, 2, 3, 4))
    model._set_aux_hidden_state_layers(taps)
    return model


@pytest.mark.parametrize("taps", [(0, 2, 4), (1, 3, 4), ()])
def test_pp2_forward_preserves_aux_order_without_duplicate_boundary(monkeypatch, taps):
    first = make_stage(monkeypatch, 0, 2, taps)
    sent = first(None, torch.tensor([0]), None, torch.tensor([[10.0]]))
    assert isinstance(sent, IntermediateTensors)
    torch.testing.assert_close(sent["hidden_states"], torch.tensor([[13.0]]))
    last = make_stage(monkeypatch, 2, 4, taps)
    actual = last(None, torch.tensor([0]), sent)
    if taps:
        output, aux = actual
        expected = {0: 10.0, 1: 11.0, 2: 13.0, 3: 16.0, 4: 20.0}
        assert len(aux) == len(taps)
        for value, tap in zip(aux, taps, strict=True):
            torch.testing.assert_close(value, torch.tensor([[expected[tap]]]))
    else:
        output = actual
    torch.testing.assert_close(output, torch.tensor([[20.0]]))


def test_pp2_missing_upstream_feature_fails(monkeypatch):
    last = make_stage(monkeypatch, 2, 4, (0, 2, 4))
    incoming = IntermediateTensors(
        {"hidden_states": torch.tensor([[13.0]]), "residual": None}
    )
    with pytest.raises(RuntimeError, match="Missing aux_hidden_states_0"):
        last(None, torch.tensor([0]), incoming)


class AttnResLayer(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, positions, hidden_states, residual, prefix_delta):
        residual.zero_()
        delta = (
            torch.zeros_like(hidden_states) if prefix_delta is None else prefix_delta
        )
        return hidden_states, residual, delta + self.increment


def test_attn_res_pp_boundary_transports_accumulated_prefix_and_aux(monkeypatch):
    first = make_stage(monkeypatch, 0, 2, (0, 2, 4))
    first.config.attn_res_block_size = 2
    first.layers = torch.nn.ModuleList(AttnResLayer(i) for i in (1, 2, 3, 4))
    sent = first(None, torch.tensor([0]), None, torch.tensor([[10.0]]))
    torch.testing.assert_close(sent["hidden_states"], torch.tensor([[13.0]]))
    last = make_stage(monkeypatch, 2, 4, (0, 2, 4))
    last.config.attn_res_block_size = 2
    last.layers = first.layers
    last.output_attn_res_proj = SimpleNamespace(weight=torch.ones((1, 1)))
    last.output_attn_res_norm = SimpleNamespace(
        weight=torch.ones(1), variance_epsilon=1e-5
    )
    # Substitute only the GPU attention-residual kernel, not the model relay.
    monkeypatch.setattr(linear, "attn_res", lambda prefix, delta, *args: prefix + delta)
    output, aux = last(None, torch.tensor([0]), sent)
    torch.testing.assert_close(output, torch.tensor([[20.0]]))
    assert len(aux) == 3
    for tensor, value in zip(aux, (10.0, 13.0, 20.0), strict=True):
        torch.testing.assert_close(tensor, torch.tensor([[value]]))


@pytest.mark.parametrize("ssm_dtype", ["auto", "float32", "bfloat16"])
def test_layer_and_model_cache_dtypes_agree(ssm_dtype):
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto", mamba_ssm_cache_dtype=ssm_dtype
        ),
    )
    layer = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
    torch.nn.Module.__init__(layer)
    layer.model_config = config.model_config
    layer.cache_config = config.cache_config
    expected = (
        torch.bfloat16,
        torch.bfloat16 if ssm_dtype == "bfloat16" else torch.float32,
    )
    assert layer.get_state_dtype() == expected
    assert (
        linear.KimiLinearForCausalLM.get_mamba_state_dtype_from_config(config)
        == expected
    )


def test_bf16_prefill_writeback_preserves_other_cache_rows(monkeypatch):
    from vllm.models.kimi_k3.amd.ops import kda_prefill

    cache = torch.full((3, 1, 1, 1), 7.0, dtype=torch.bfloat16)
    index = torch.tensor([1], dtype=torch.int32)
    monkeypatch.setattr(
        kda_prefill, "gather_initial_states", lambda *_: torch.zeros((1, 1, 1, 1))
    )
    result = torch.tensor([[[[3.0]]]])
    monkeypatch.setattr(
        kda_prefill,
        "chunk_kda_with_fused_gate",
        lambda **_: (result, torch.tensor([[[[1.25]]]], dtype=torch.float32)),
    )
    q = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16)
    actual, final_state = kda_prefill.chunk_kda_prefill(
        q,
        q,
        q,
        q,
        q,
        torch.zeros(1),
        state_cache=cache,
        state_indices=index,
        has_initial_state=torch.tensor([False]),
    )
    torch.testing.assert_close(actual, result)
    assert final_state is None
    torch.testing.assert_close(
        cache.flatten(), torch.tensor([7.0, 1.25, 7.0], dtype=torch.bfloat16)
    )
