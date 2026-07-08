import os
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MlaDcpLseContext:
    impl: object
    q: torch.Tensor
    kv_c_and_k_pe_cache: torch.Tensor
    attn_metadata: object
    layer: object
    mla_padded_q: torch.Tensor
    output: torch.Tensor
    unpad_output: object


class GluonMlaDcpLseProvider:
    """Validation provider for MLA decode LSE via PR3402 Gluon.

    The provider exposes the LSE contract vLLM DCP needs:
    return local attention output and local decode LSE for a rank. It is kept
    separate from the rocm_aiter_mla hook so future providers can be added
    without growing a model-specific patch in the backend.
    """

    name = "gluon"

    @staticmethod
    def ensure_triton_runtime() -> None:
        atom_triton_path = os.environ.get("AITER_GLUON_TRITON_PATH")
        if not atom_triton_path or os.environ.get("AITER_GLUON_TRITON_SWITCHED") == "1":
            return

        triton_mod = sys.modules.get("triton")
        triton_file = getattr(triton_mod, "__file__", "") if triton_mod is not None else ""
        if triton_file.startswith(atom_triton_path):
            os.environ["AITER_GLUON_TRITON_SWITCHED"] = "1"
            return

        sys.path.insert(0, atom_triton_path)
        for mod_name in list(sys.modules):
            if mod_name == "triton" or mod_name.startswith("triton."):
                del sys.modules[mod_name]
        os.environ["AITER_GLUON_TRITON_SWITCHED"] = "1"

    @staticmethod
    def _query_layout(ctx: MlaDcpLseContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_nope, q_pe = torch.split(
            ctx.mla_padded_q,
            [ctx.impl.kv_lora_rank, ctx.impl.qk_rope_head_dim],
            dim=-1,
        )
        return q_nope.to(torch.bfloat16), q_pe.to(torch.bfloat16), ctx.attn_metadata.decode

    @staticmethod
    def _infer_q_lens(
        total_q: int,
        num_reqs: int,
        qo_indptr: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if qo_indptr is not None and qo_indptr.numel() >= num_reqs + 1:
            qo_lens = (qo_indptr[1 : num_reqs + 1] - qo_indptr[:num_reqs]).to(torch.long)
        else:
            qo_lens = torch.empty(0, dtype=torch.long, device=device)

        if qo_lens.numel() == num_reqs and int(qo_lens.sum().item()) == total_q:
            return qo_lens, qo_indptr[: num_reqs + 1].to(torch.long)

        if total_q % num_reqs != 0:
            raise RuntimeError(
                "Cannot infer q lengths for MLA DCP LSE provider: "
                f"total_q={total_q}, num_reqs={num_reqs}, qo_indptr={qo_indptr}"
            )
        uniform_q = total_q // num_reqs
        qo_lens = torch.full((num_reqs,), uniform_q, dtype=torch.long, device=device)
        qo_starts = torch.arange(0, total_q + 1, uniform_q, dtype=torch.long, device=device)
        return qo_lens, qo_starts

    @staticmethod
    def _build_seq_info_and_page_table(
        ctx: MlaDcpLseContext,
        decode: object,
        total_q: int,
        qo_lens: torch.Tensor,
        qo_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        req_ids = torch.repeat_interleave(
            torch.arange(decode.seq_lens.shape[0], dtype=torch.long, device=ctx.q.device),
            qo_lens,
        )[:total_q]
        q_pos = torch.arange(total_q, dtype=torch.long, device=ctx.q.device) - qo_starts[req_ids]
        seq_info = (
            decode.seq_lens[req_ids].to(torch.long) - (qo_lens[req_ids] - 1 - q_pos)
        ).to(torch.int32)
        min_seq = int(seq_info.min().item())
        if min_seq <= 0:
            raise RuntimeError(f"MLA DCP LSE provider requires positive seq len, got {min_seq}.")

        max_seq = int(seq_info.max().item())
        page_table = torch.empty(total_q, max_seq, dtype=torch.int32, device=ctx.q.device)

        # Validation path: materialize a per-token 2-D page table from the
        # AITER metadata builder's expanded flat token indices. This keeps the
        # provider independent from vLLM block_size.
        for row in range(total_q):
            req = int(req_ids[row].item())
            end = int(seq_info[row].item())
            start = int(decode.paged_kv_indptr[req].item())
            page_table[row, :end] = decode.paged_kv_indices[start : start + end].to(torch.int32)
            if end < max_seq:
                page_table[row, end:] = 0

        # PR3402 Gluon currently asserts NUM_KV_SPLITS=max(1, 256//B), so keep
        # this explicit instead of pretending shorter-context prompts are
        # supported by lowering the split count in the caller.
        num_kv_splits = max(1, 256 // total_q)
        if min_seq < num_kv_splits:
            raise RuntimeError(
                "Gluon MLA DCP LSE provider requires min seq len >= "
                f"{num_kv_splits}, got {min_seq}."
            )
        return seq_info, page_table, min_seq, num_kv_splits

    def forward_mqa(self, ctx: MlaDcpLseContext):
        self.ensure_triton_runtime()
        from aiter.ops.triton.gluon.mla_decode_gluon import mla_decode_gluon

        q_nope, q_pe, decode = self._query_layout(ctx)
        assert decode.block_table is not None
        assert decode.seq_lens is not None
        assert decode.paged_kv_indices is not None
        assert decode.paged_kv_indptr is not None

        total_q = q_nope.shape[0]
        num_reqs = decode.seq_lens.shape[0]
        qo_lens, qo_starts = self._infer_q_lens(
            total_q, num_reqs, decode.qo_indptr, ctx.q.device
        )
        seq_info, page_table, min_seq, num_kv_splits = self._build_seq_info_and_page_table(
            ctx, decode, total_q, qo_lens, qo_starts
        )

        kv_flat = ctx.kv_c_and_k_pe_cache.reshape(-1, ctx.kv_c_and_k_pe_cache.shape[-1])
        out_chunks = []
        lse_chunks = []
        for h0 in range(0, q_nope.shape[1], 16):
            h1 = min(h0 + 16, q_nope.shape[1])
            partial_o = torch.empty(
                total_q,
                h1 - h0,
                num_kv_splits,
                ctx.impl.kv_lora_rank,
                dtype=torch.bfloat16,
                device=ctx.q.device,
            )
            partial_o, partial_lse = mla_decode_gluon(
                q_nope[:, h0:h1, :],
                q_pe[:, h0:h1, :],
                kv_flat,
                partial_o,
                page_table,
                seq_info,
                sm_scale=ctx.impl.scale,
                k_pe=None,
                kv_pe_offset=ctx.impl.kv_lora_rank,
                use_2d_view=True,
                kv_scale=float(getattr(ctx.layer, "_k_scale_float", 1.0)),
                min_kv_seq_len=min_seq,
                return_lse=True,
            )
            local_lse_h = torch.logsumexp(partial_lse, dim=2)
            weights = torch.exp(partial_lse - local_lse_h.unsqueeze(2)).to(torch.float32)
            local_o_h = (
                partial_o.to(torch.float32) * weights.unsqueeze(-1)
            ).sum(dim=2).to(ctx.output.dtype)
            out_chunks.append(local_o_h)
            lse_chunks.append(local_lse_h)

        local_o = torch.cat(out_chunks, dim=1)
        local_lse = torch.cat(lse_chunks, dim=1)
        return ctx.unpad_output(ctx.impl.num_heads, local_o), local_lse


def get_mla_dcp_lse_provider():
    provider = os.environ.get("VLLM_ROCM_MLA_DCP_LSE_PROVIDER", "gluon").lower()
    if provider == "gluon":
        return GluonMlaDcpLseProvider()
    raise NotImplementedError(f"Unsupported MLA DCP LSE provider: {provider}")


def forward_mqa_with_lse(
    impl,
    q,
    kv_c_and_k_pe_cache,
    attn_metadata,
    layer,
    mla_padded_q,
    output,
    unpad_output,
):
    ctx = MlaDcpLseContext(
        impl=impl,
        q=q,
        kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
        attn_metadata=attn_metadata,
        layer=layer,
        mla_padded_q=mla_padded_q,
        output=output,
        unpad_output=unpad_output,
    )
    return get_mla_dcp_lse_provider().forward_mqa(ctx)
