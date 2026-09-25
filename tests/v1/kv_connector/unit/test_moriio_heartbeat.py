# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat import (
    build_payload,
    terminate_with_parent,
)


def test_build_payload_preserves_router_registration_fields():
    args = Namespace(
        role="P",
        http_address="10.0.0.1:8000",
        zmq_address="host:10.0.0.1,handshake:6301,notify:61005",
        dp_size=2,
        tp_size=8,
        transfer_mode="READ",
    )

    assert build_payload(args) == {
        "type": "P",
        "http_address": "10.0.0.1:8000",
        "zmq_address": "host:10.0.0.1,handshake:6301,notify:61005",
        "dp_size": 2,
        "tp_size": 8,
        "transfer_mode": "READ",
    }


def test_heartbeat_helper_requests_parent_death_signal():
    libc = MagicMock()
    libc.prctl.return_value = 0
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
            "moriio_heartbeat.ctypes.CDLL",
            return_value=libc,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
            "moriio_heartbeat.os.getppid",
            side_effect=[123, 123],
        ),
    ):
        terminate_with_parent()

    libc.prctl.assert_called_once()


def test_worker_starts_heartbeat_in_a_separate_process():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.local_ip = "10.0.0.1"
    worker.proxy_ip = "10.0.0.2"
    worker.proxy_ping_port = 36367
    worker.request_address = "10.0.0.1:8000"
    worker.handshake_port = 6301
    worker.notify_port = 61005
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    worker.moriio_config = SimpleNamespace(dp_size=1, tp_size=8)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio."
        "moriio_connector.subprocess.Popen"
    ) as popen:
        worker._start_ping_process()

    command = popen.call_args.args[0]
    assert command[1:3] == [
        "-m",
        "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_heartbeat",
    ]
    assert "--role" in command and command[command.index("--role") + 1] == "P"
    assert popen.call_args.kwargs == {"start_new_session": True}


def test_worker_shutdown_terminates_heartbeat_process():
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    process = MagicMock()
    process.poll.return_value = None
    worker._ping_process = process

    worker.shutdown()

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2)
    assert worker._ping_process is None
