# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight MoRIIO service-discovery heartbeat process."""

import argparse
import ctypes
import logging
import os
import signal
import time

import msgpack
import zmq

logger = logging.getLogger(__name__)


def terminate_with_parent(parent_pid: int) -> None:
    """Ask Linux to terminate this helper if its worker parent exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "type": args.role,
        "http_address": args.http_address,
        "zmq_address": args.zmq_address,
        "dp_size": args.dp_size,
        "tp_size": args.tp_size,
        "transfer_mode": args.transfer_mode,
    }


def run_heartbeat(args: argparse.Namespace) -> None:
    terminate_with_parent(args.parent_pid)
    payload = msgpack.dumps(build_payload(args))
    context = zmq.Context()
    try:
        with context.socket(zmq.DEALER) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, 1000)
            socket.connect(args.proxy_address)
            failures = 0
            while True:
                try:
                    socket.send(payload)
                    failures = 0
                except Exception:
                    failures += 1
                    logger.exception(
                        "MoRIIO discovery heartbeat failed (%d/%d)",
                        failures,
                        args.max_retries,
                    )
                    if failures >= args.max_retries:
                        raise
                time.sleep(args.interval)
    finally:
        context.destroy(linger=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--proxy-address", required=True)
    parser.add_argument("--role", choices=("P", "D"), required=True)
    parser.add_argument("--http-address", required=True)
    parser.add_argument("--zmq-address", required=True)
    parser.add_argument("--dp-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--transfer-mode", required=True)
    parser.add_argument("--interval", type=float, required=True)
    parser.add_argument("--max-retries", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_heartbeat(parse_args())
