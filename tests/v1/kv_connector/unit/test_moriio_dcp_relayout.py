# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Heterogeneous-DCP relayout for MoRIIO: prefill dcp=1 -> decode dcp=N.

These tests are pure arithmetic over block ids, so unlike the other MoRIIO
test modules they need neither ROCm nor the ``mori`` package.

The central test is differential: it re-implements vLLM's slot-mapping owner
rule (``_compute_slot_mapping_kernel`` in ``vllm/v1/worker/block_table.py``) and
checks the routing plan against it, so the plan cannot silently drift from the
rule the attention kernels actually use.
"""

import importlib

import pytest

moriio_layout = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout"
)

build_dcp_block_pairing = moriio_layout.build_dcp_block_pairing
build_dcp_token_pairing = moriio_layout.build_dcp_token_pairing
dcp_relayout_granularity = moriio_layout.dcp_relayout_granularity
validate_moriio_heterogeneous_dcp = moriio_layout.validate_moriio_heterogeneous_dcp
DCP_GRANULARITY_BLOCK = moriio_layout.DCP_GRANULARITY_BLOCK
DCP_GRANULARITY_TOKEN = moriio_layout.DCP_GRANULARITY_TOKEN


def _owner_of_token(
    pos: int, block_size: int, dcp_size: int, interleave: int
) -> tuple[int, int, int]:
    """Port of ``_compute_slot_mapping_kernel``'s DCP arithmetic.

    Returns ``(owner_dcp_rank, local_block_index, slot_within_block)`` for the
    token at absolute position ``pos``. ``BLOCKS_PER_KV_BLOCK`` is 1 (kernel
    block size == allocation block size).
    """
    virtual_block_size = block_size * dcp_size
    virtual_block_index = pos // virtual_block_size
    virtual_block_offset = pos % virtual_block_size
    owner = (virtual_block_offset // interleave) % dcp_size
    local_offset = (virtual_block_offset // (dcp_size * interleave)) * interleave + (
        virtual_block_offset % interleave
    )
    return owner, virtual_block_index + local_offset // block_size, (
        local_offset % block_size
    )


# --------------------------------------------------------------------------- #
# The premise: with interleave == block_size a whole block has a single owner
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
def test_block_aligned_interleave_gives_each_block_one_owner(block_size, dcp_size):
    """interleave == block_size => block j is owned wholly by rank j % dcp."""
    num_blocks = 3 * dcp_size + 1
    for j in range(num_blocks):
        owners = set()
        local_indices = set()
        for offset in range(block_size):
            owner, local_block, slot = _owner_of_token(
                j * block_size + offset, block_size, dcp_size, interleave=block_size
            )
            owners.add(owner)
            local_indices.add(local_block)
            assert slot == offset
        assert owners == {j % dcp_size}
        assert local_indices == {j // dcp_size}


@pytest.mark.parametrize("dcp_size", [2, 4])
def test_interleave_one_splits_a_block_across_ranks(dcp_size):
    """Why the validator insists on interleave == block_size.

    With the default interleave of 1 a single block's tokens are spread over
    every rank, so block-granular routing would move the wrong bytes.
    """
    block_size = 16
    owners = {
        _owner_of_token(offset, block_size, dcp_size, interleave=1)[0]
        for offset in range(block_size)
    }
    assert owners == set(range(dcp_size))


# --------------------------------------------------------------------------- #
# The pairing agrees with the owner rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_prefill_blocks", [1, 5, 16, 17])
def test_pairing_matches_owner_rule(dcp_size, num_prefill_blocks):
    block_size = 64
    # Distinct, non-identity ids on both sides so a mix-up cannot pass.
    prefill_block_ids = [100 + j for j in range(num_prefill_blocks)]
    num_decode = -(-num_prefill_blocks // dcp_size)
    decode_block_ids = [500 + v for v in range(num_decode)]

    seen: list[int] = []
    for dcp_rank in range(dcp_size):
        prefill_subset, decode_subset = build_dcp_block_pairing(
            prefill_block_ids, decode_block_ids, dcp_size, dcp_rank
        )
        # Index-aligned, which is what compute_block_transfer_offsets needs.
        assert len(prefill_subset) == len(decode_subset)
        for pb, db in zip(prefill_subset, decode_subset):
            j = pb - 100
            owner, decode_slot, _ = _owner_of_token(
                j * block_size, block_size, dcp_size, interleave=block_size
            )
            assert owner == dcp_rank, "paired a block this rank does not own"
            assert db == decode_block_ids[decode_slot]
        seen.extend(prefill_subset)

    # Across all ranks every prefill block is transferred exactly once.
    assert sorted(seen) == sorted(prefill_block_ids)


def test_pairing_dcp1_is_the_plain_pairing():
    prefill_block_ids = [7, 8, 9]
    decode_block_ids = [70, 80, 90]
    assert build_dcp_block_pairing(
        prefill_block_ids, decode_block_ids, dcp_size=1, dcp_rank=0
    ) == (prefill_block_ids, decode_block_ids)


def test_pairing_respects_absolute_position():
    """A suffix transfer must be paired by absolute logical index."""
    # Logical blocks 5,6,7 with dcp=2 -> ranks 1,0,1 and decode slots 2,3,3.
    decode_block_ids = [500, 501, 502, 503]
    assert build_dcp_block_pairing(
        [15, 16, 17],
        decode_block_ids,
        dcp_size=2,
        dcp_rank=1,
        first_prefill_block_index=5,
    ) == ([15, 17], [502, 503])
    assert build_dcp_block_pairing(
        [15, 16, 17],
        decode_block_ids,
        dcp_size=2,
        dcp_rank=0,
        first_prefill_block_index=5,
    ) == ([16], [503])


def test_pairing_can_be_empty_for_a_rank():
    """Only one block to move, so the other ranks get nothing."""
    assert build_dcp_block_pairing([11], [900], dcp_size=4, dcp_rank=2) == ([], [])


def test_pairing_overrun_raises():
    with pytest.raises(ValueError, match="overrun the decode DCP allocation"):
        build_dcp_block_pairing([1, 2, 3, 4, 5], [900], dcp_size=2, dcp_rank=0)


def test_pairing_short_prefill_list_is_fine():
    """Decode allocated more than prefill sends (prefix hit / abort)."""
    prefill_subset, decode_subset = build_dcp_block_pairing(
        [1, 2], [900, 901, 902], dcp_size=2, dcp_rank=0
    )
    assert (prefill_subset, decode_subset) == ([1], [900])


@pytest.mark.parametrize("bad_dcp", [0, -1])
def test_pairing_rejects_bad_dcp_size(bad_dcp):
    with pytest.raises(ValueError, match="dcp_size must be >= 1"):
        build_dcp_block_pairing([1], [1], dcp_size=bad_dcp, dcp_rank=0)


@pytest.mark.parametrize("bad_rank", [-1, 2, 5])
def test_pairing_rejects_out_of_range_rank(bad_rank):
    with pytest.raises(ValueError, match="out of range"):
        build_dcp_block_pairing([1], [1], dcp_size=2, dcp_rank=bad_rank)


# --------------------------------------------------------------------------- #
# Token-granular pairing (interleave == 1, the regime spec decode requires)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dcp_size", [2, 4, 8])
@pytest.mark.parametrize("block_size", [4, 64])
def test_token_pairing_matches_owner_rule(dcp_size, block_size):
    """Every owned token lands where the real owner rule says it should."""
    num_prefill_blocks = 5
    prefill_block_ids = [100 + j for j in range(num_prefill_blocks)]
    num_tokens = num_prefill_blocks * block_size
    # Enough decode pages for the strided share plus slack.
    decode_block_ids = [500 + v for v in range(num_prefill_blocks + 2)]

    seen: set[int] = set()
    for dcp_rank in range(dcp_size):
        p_slots, d_slots = build_dcp_token_pairing(
            prefill_block_ids, decode_block_ids, block_size, dcp_size, dcp_rank
        )
        assert len(p_slots) == len(d_slots)
        positions = []
        for ps, ds in zip(p_slots, d_slots):
            # Recover the absolute position from the prefill slot.
            pos = (ps // block_size - 100) * block_size + ps % block_size
            owner, local_block, slot_in_block = _owner_of_token(
                pos, block_size, dcp_size, interleave=1
            )
            assert owner == dcp_rank, f"pos {pos} is not rank {dcp_rank}'s"
            assert ds == decode_block_ids[local_block] * block_size + slot_in_block
            positions.append(pos)
        # Ascending order: the destination side relies on it to coalesce.
        assert positions == sorted(positions)
        assert not seen & set(positions)
        seen |= set(positions)

    # Across all ranks every token is transferred exactly once.
    assert sorted(seen) == list(range(num_tokens))


@pytest.mark.parametrize("dcp_size", [2, 3, 4])
def test_token_pairing_destination_is_contiguous(dcp_size):
    """Consecutive owned tokens land in consecutive decode slots.

    This is what lets merge_contiguous_offsets coalesce the destination side;
    the source side is strided by dcp and cannot merge.
    """
    block_size = 8
    prefill_block_ids = list(range(10, 10 + 6))
    decode_block_ids = list(range(90, 90 + 6))
    _, d_slots = build_dcp_token_pairing(
        prefill_block_ids, decode_block_ids, block_size, dcp_size, dcp_rank=0
    )
    # Within a decode page the slots must step by exactly 1.
    for a, b in zip(d_slots, d_slots[1:]):
        same_page = a // block_size == b // block_size
        assert (b - a == 1) if same_page else (b % block_size == 0)


def test_token_pairing_respects_num_tokens():
    """A partial last block must not transfer padding."""
    block_size = 8
    p_slots, _ = build_dcp_token_pairing(
        [10, 11], [90, 91], block_size, dcp_size=2, dcp_rank=0, num_tokens=11
    )
    # rank 0 owns even positions 0,2,4,6,8,10 -> 6 tokens
    assert len(p_slots) == 6
    assert max(s % block_size + (s // block_size - 10) * block_size for s in p_slots) == 10


def test_token_pairing_respects_absolute_position():
    """A suffix transfer is owned by absolute position, not list index."""
    block_size = 4
    # Blocks 3,4 -> positions 12..19. With dcp=2, rank 1 owns odd positions.
    p_slots, _ = build_dcp_token_pairing(
        [70, 71], list(range(90, 96)), block_size, 2, 1, first_prefill_block_index=3
    )
    positions = [(s // block_size - 70) * block_size + s % block_size + 12 for s in p_slots]
    assert positions == [13, 15, 17, 19]


def test_token_pairing_overrun_raises():
    with pytest.raises(ValueError, match="overrun the decode DCP allocation"):
        build_dcp_token_pairing([1, 2, 3, 4], [90], 8, dcp_size=2, dcp_rank=0)


def test_token_pairing_rejects_bad_num_tokens():
    with pytest.raises(ValueError, match="num_tokens must fit"):
        build_dcp_token_pairing([1], [90], 8, 2, 0, num_tokens=9)


@pytest.mark.parametrize("dcp_size", [2, 4])
def test_token_pairing_live_length_moves_no_padding(dcp_size):
    """The whole point of threading the prompt length through the connector.

    A prompt that ends mid-block leaves the rest of that block uninitialized.
    Without a live length the plan covers the block's full capacity and ships
    that tail; with it, the ranks together move exactly the live tokens.
    """
    block_size = 128
    num_tokens = 1733  # 14 blocks of 128 = 1792 capacity, so 59 slots are padding
    n_pblocks = -(-num_tokens // block_size)
    prefill = [1000 + i for i in range(n_pblocks)]
    decode = [7000 + i for i in range(-(-num_tokens // (block_size * dcp_size)) + 2)]

    live = sum(
        len(build_dcp_token_pairing(
            prefill, decode, block_size, dcp_size, r, num_tokens=num_tokens)[0])
        for r in range(dcp_size)
    )
    full = sum(
        len(build_dcp_token_pairing(prefill, decode, block_size, dcp_size, r)[0])
        for r in range(dcp_size)
    )
    assert live == num_tokens
    assert full == n_pblocks * block_size
    assert full > live  # i.e. the default really does ship the padding


# --------------------------------------------------------------------------- #
# Granularity selection
# --------------------------------------------------------------------------- #


def test_granularity_selection():
    assert dcp_relayout_granularity(64, 64) == DCP_GRANULARITY_BLOCK
    assert dcp_relayout_granularity(1, 64) == DCP_GRANULARITY_TOKEN
    # block_size 1 is degenerate: both rules coincide, block wins.
    assert dcp_relayout_granularity(1, 1) == DCP_GRANULARITY_BLOCK


@pytest.mark.parametrize("interleave", [2, 16, 128])
def test_granularity_rejects_intermediate_interleave(interleave):
    with pytest.raises(ValueError, match="supports cp_kv_cache_interleave_size"):
        dcp_relayout_granularity(interleave, 64)


# --------------------------------------------------------------------------- #
# Topology gate
# --------------------------------------------------------------------------- #


def test_validate_accepts_one_to_n_mla():
    assert (
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=1,
            decode_dcp_size=4,
            is_mla=True,
            cp_kv_cache_interleave_size=64,
            block_size=64,
        )
        == DCP_GRANULARITY_BLOCK
    )


def test_validate_accepts_interleave_one_as_token_granular():
    assert (
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=1,
            decode_dcp_size=2,
            is_mla=True,
            cp_kv_cache_interleave_size=1,
            block_size=64,
        )
        == DCP_GRANULARITY_TOKEN
    )


def test_validate_spec_decode_requires_interleave_one():
    """Whole-block copies are unavailable with spec decode, so say so early."""
    with pytest.raises(ValueError, match="requires --cp-kv-cache-interleave-size 1"):
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=1,
            decode_dcp_size=2,
            is_mla=True,
            cp_kv_cache_interleave_size=64,
            block_size=64,
            has_spec_decode=True,
        )


def test_validate_spec_decode_ok_at_interleave_one():
    assert (
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=1,
            decode_dcp_size=2,
            is_mla=True,
            cp_kv_cache_interleave_size=1,
            block_size=64,
            has_spec_decode=True,
        )
        == DCP_GRANULARITY_TOKEN
    )


@pytest.mark.parametrize("dcp_size", [1, 4])
def test_validate_equal_sizes_need_no_relayout(dcp_size):
    # Equal sizes short-circuit, so even settings the 1->N path would reject
    # (non-MLA, interleave != block_size) are fine.
    validate_moriio_heterogeneous_dcp(
        prefill_dcp_size=dcp_size,
        decode_dcp_size=dcp_size,
        is_mla=False,
        cp_kv_cache_interleave_size=1,
        block_size=64,
    )


def test_validate_rejects_n_to_one():
    with pytest.raises(NotImplementedError, match="only prefill dcp=1"):
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=2,
            decode_dcp_size=1,
            is_mla=True,
            cp_kv_cache_interleave_size=64,
            block_size=64,
        )


def test_validate_rejects_n_to_m():
    with pytest.raises(NotImplementedError, match="only prefill dcp=1"):
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=2,
            decode_dcp_size=4,
            is_mla=True,
            cp_kv_cache_interleave_size=64,
            block_size=64,
        )


def test_validate_rejects_non_mla():
    with pytest.raises(NotImplementedError, match="MLA caches only"):
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=1,
            decode_dcp_size=2,
            is_mla=False,
            cp_kv_cache_interleave_size=64,
            block_size=64,
        )


@pytest.mark.parametrize("interleave", [2, 16, 128])
def test_validate_rejects_intermediate_interleave(interleave):
    """Neither whole-block nor per-token: no plan exists for these."""
    with pytest.raises(ValueError, match="supports cp_kv_cache_interleave_size"):
        validate_moriio_heterogeneous_dcp(
            prefill_dcp_size=1,
            decode_dcp_size=2,
            is_mla=True,
            cp_kv_cache_interleave_size=interleave,
            block_size=64,
        )
