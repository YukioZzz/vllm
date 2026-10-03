# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise ring ownership and preposted receive shutdown with CPU transports."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker import gpu_worker


def test_ring_waits_before_reuse_and_preserves_graph_outputs(monkeypatch):
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.device = torch.device("cpu")
    worker._pp_activation_group = object()
    worker._pp_activation_stream = SimpleNamespace(wait_event=lambda _: None)
    worker._pp_activation_slots = [None, None]
    worker._pp_activation_slot_work = [[], []]
    worker._pp_activation_slot_events = [SimpleNamespace(record=lambda _: None)] * 2
    worker._pp_activation_slot_index = 0
    worker._pp_prepost_recv_enabled = True
    retained, waited = [], []

    def send(tensors, **kwargs):
        assert kwargs["send_all_gather_metadata"]
        value = tensors["hidden_states"]
        retained.append(value)
        return [SimpleNamespace(wait=lambda: waited.append(value.item()))]

    monkeypatch.setattr(
        gpu_worker, "get_pp_group", lambda: SimpleNamespace(isend_tensor_dict=send)
    )
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: object())
    graph_output = torch.tensor([11.0])
    worker._send_pp_async_activation({"hidden_states": graph_output}, {})
    graph_output.fill_(22.0)
    worker._send_pp_async_activation({"hidden_states": graph_output}, {})
    graph_output.fill_(33.0)
    assert [tensor.item() for tensor in retained] == [11.0, 22.0]
    worker._send_pp_async_activation({"hidden_states": graph_output}, {})
    assert waited == [11.0]
    assert retained[0].data_ptr() == retained[2].data_ptr()
    assert retained[1].item() == 22.0


@pytest.mark.parametrize("already_posted", [False, True])
def test_receive_shutdown_consumes_sentinel_and_joins_thread(
    monkeypatch, already_posted
):
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.device = torch.device("cpu")
    worker._pp_prepost_recv_enabled = True
    worker._pp_activation_group = object()
    worker._pp_activation_stream = object()
    worker._pp_prepost_executor = ThreadPoolExecutor(max_workers=1)
    worker._pp_prepost_future = None
    calls = []

    def receive(**kwargs):
        calls.append(kwargs["use_sender_all_gather_metadata"])
        return {"__k3_pp_prepost_stop__": True}, [], []

    group = SimpleNamespace(is_first_rank=False, irecv_tensor_dict=receive)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: group)
    monkeypatch.setattr(gpu_worker, "get_tp_group", lambda: object())
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    if already_posted:
        worker._start_pp_preposted_activation()
    worker._shutdown_pp_preposted_activation()
    assert calls == [True]
    assert worker._pp_prepost_future is None
    assert worker._pp_prepost_executor is None
