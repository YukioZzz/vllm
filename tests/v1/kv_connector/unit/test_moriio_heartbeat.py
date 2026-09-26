# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgpack
import zmq

HELPER_PATH = (
    Path(__file__).resolve().parents[4]
    / "vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_heartbeat.py"
)
_spec = importlib.util.spec_from_file_location("moriio_heartbeat", HELPER_PATH)
assert _spec is not None and _spec.loader is not None
heartbeat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heartbeat)


def test_build_payload_preserves_router_registration_fields():
    args = Namespace(
        role="P",
        http_address="10.0.0.1:8000",
        zmq_address="host:10.0.0.1,handshake:6301,notify:61005",
        dp_size=2,
        tp_size=8,
        transfer_mode="READ",
    )

    assert heartbeat.build_payload(args) == {
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
        patch.object(heartbeat.ctypes, "CDLL", return_value=libc),
        patch.object(heartbeat.os, "getppid", return_value=123),
    ):
        heartbeat.terminate_with_parent(123)

    libc.prctl.assert_called_once()


def test_worker_starts_heartbeat_in_a_separate_process():
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
        MoRIIOMode,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
        MoRIIOConnectorWorker,
    )

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
    assert Path(command[1]).name == "moriio_heartbeat.py"
    assert command[command.index("--parent-pid") + 1] == str(os.getpid())
    assert "--role" in command and command[command.index("--role") + 1] == "P"
    assert popen.call_args.kwargs == {"start_new_session": True}


def test_worker_shutdown_terminates_heartbeat_process():
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
        MoRIIOConnectorWorker,
    )

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    process = MagicMock()
    process.poll.return_value = None
    worker._ping_process = process

    worker.shutdown()

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=2)
    assert worker._ping_process is None


def _command(address, parent_pid):
    return [
        sys.executable,
        str(HELPER_PATH),
        "--parent-pid",
        str(parent_pid),
        "--proxy-address",
        address,
        "--role",
        "P",
        "--http-address",
        "127.0.0.1:8000",
        "--zmq-address",
        "test",
        "--dp-size",
        "1",
        "--tp-size",
        "8",
        "--transfer-mode",
        "READ",
        "--interval",
        "0.05",
        "--max-retries",
        "3",
    ]


def test_helper_exits_if_worker_died_before_startup():
    # A helper reparented before prctl must never advertise a dead worker.
    result = subprocess.run(_command("tcp://127.0.0.1:1", -1), timeout=5)
    assert result.returncode == -15


def test_helper_bounds_send_when_router_is_unreachable():
    command = _command("tcp://127.0.0.1:1", os.getpid())
    command[command.index("--interval") + 1] = "0"
    command[command.index("--max-retries") + 1] = "1"
    result = subprocess.run(command, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert b"Resource temporarily unavailable" in result.stderr


def test_helper_pings_while_parent_holds_gil_and_stops_on_parent_death():
    # Exercise real ZMQ and Linux process lifetime without importing torch.
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    command = _command(f"tcp://127.0.0.1:{port}", 0)
    parent_code = """
import os, subprocess, sys, time
command = __import__('json').loads(sys.argv[1])
command[command.index('--parent-pid') + 1] = str(os.getpid())
child = subprocess.Popen(command, start_new_session=True)
sys.stdin.readline()
sys.setswitchinterval(10)
end = time.monotonic() + 1.5
while time.monotonic() < end:
    pass
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, json.dumps(command)],
        stdin=subprocess.PIPE,
        text=True,
    )
    try:
        assert socket.poll(15000), "helper did not register"
        socket.recv_multipart()
        assert parent.stdin is not None
        parent.stdin.write("start\n")
        parent.stdin.flush()
        received = []
        deadline = time.monotonic() + 5
        while parent.poll() is None and time.monotonic() < deadline:
            if socket.poll(100):
                payload = msgpack.loads(socket.recv_multipart()[-1])
                assert payload["type"] == "P"
                received.append(time.monotonic())
        assert parent.poll() is not None
        assert len(received) >= 5
        assert received[-1] - received[0] >= 0.5
        parent.wait(timeout=5)
        # Drain packets queued before worker death; no new registration follows.
        time.sleep(0.2)
        while socket.poll(0):
            socket.recv_multipart()
        assert not socket.poll(300)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=5)
        socket.close(linger=0)
        context.term()
