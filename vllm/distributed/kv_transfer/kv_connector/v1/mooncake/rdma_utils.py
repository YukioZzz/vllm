# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake requester config helpers."""

from collections.abc import Mapping
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def normalize_string_override(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_requester_local_hostname(local_ip: str) -> str:
    override = normalize_string_override(envs.MOONCAKE_REQUESTER_LOCAL_HOSTNAME)
    if override is not None:
        return override
    return local_ip


# Sentinel for preferred_segment meaning "this rank's own segment", resolved after the
# store is up (the segment name embeds a dynamically chosen RPC port, so it cannot be
# written into config ahead of time). See resolve_local_preferred_segment.
LOCAL_PREFERRED_SEGMENT_SENTINELS = frozenset({"local", "self", "own"})


def is_local_preferred_segment(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in LOCAL_PREFERRED_SEGMENT_SENTINELS


def resolve_local_preferred_segment(store: Any) -> str | None:
    """This rank's own Mooncake segment name, or None if it cannot be determined.

    Pinning each rank to its own segment keeps every put and get inside the process, so
    the transfer engine takes its local-memcpy fast path instead of a TCP loopback round
    trip. That is safe because Mooncake keys are already rank-scoped (they embed
    ``tp_rank``), so no rank ever reads another rank's shard -- with a node-global
    allocation the cross-rank traffic is pure overhead. Measured on Kimi-K3 1P1D: 808 GB
    of loopback receive on the decode node in 11 minutes, peaking at 11.5 GB/s.
    """
    getter = getattr(store, "get_hostname", None)
    if getter is None:
        logger.warning(
            "preferred_segment=local requested but the Mooncake store exposes no "
            "get_hostname(); falling back to node-global allocation."
        )
        return None
    try:
        segment = normalize_string_override(getter())
    except Exception:
        logger.exception(
            "preferred_segment=local requested but get_hostname() failed; "
            "falling back to node-global allocation."
        )
        return None
    if segment is None:
        logger.warning(
            "preferred_segment=local requested but get_hostname() returned nothing; "
            "falling back to node-global allocation."
        )
    return segment


def get_configured_preferred_segment(
    extra_config: Mapping[str, Any],
) -> str | None:
    preferred_segment = normalize_string_override(extra_config.get("preferred_segment"))
    if preferred_segment is not None:
        return preferred_segment
    if extra_config.get("preferred_segment") is not None:
        raise ValueError(
            "Mooncake preferred_segment override must be a non-empty string"
        )

    env_value = normalize_string_override(envs.MOONCAKE_PREFERRED_SEGMENT)
    if env_value is not None:
        logger.info(
            "Mooncake preferred_segment from MOONCAKE_PREFERRED_SEGMENT: %s",
            env_value,
        )
        return env_value
    return None
