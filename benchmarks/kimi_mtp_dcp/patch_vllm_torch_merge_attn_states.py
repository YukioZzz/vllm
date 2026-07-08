import importlib
from pathlib import Path


p = Path(importlib.import_module("vllm.v1.attention.ops.merge_attn_states").__file__)
s = p.read_text()

marker = "AITER_MTP_DCP_TORCH_MERGE"
if marker not in s:
    old = """    else:
        from vllm.v1.attention.ops.triton_merge_attn_states import (
            merge_attn_states,
        )

        return merge_attn_states(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
"""
    new = f"""    else:
        # {marker}: validation fallback for ROCm atom-Triton incompatibility.
        num_tokens = output.shape[0]
        if prefill_tokens_with_context is None:
            prefill_tokens_with_context = num_tokens
        merge_tokens = min(prefill_tokens_with_context, num_tokens)

        if merge_tokens < num_tokens:
            output[merge_tokens:].copy_(suffix_output[merge_tokens:].to(output.dtype))
            if output_lse is not None:
                output_lse[:, merge_tokens:].copy_(suffix_lse[:, merge_tokens:])

        if merge_tokens > 0:
            p_lse = prefix_lse[:, :merge_tokens].transpose(0, 1).to(torch.float32)
            s_lse = suffix_lse[:, :merge_tokens].transpose(0, 1).to(torch.float32)
            p_lse = torch.where(torch.isposinf(p_lse), torch.full_like(p_lse, -float("inf")), p_lse)
            s_lse = torch.where(torch.isposinf(s_lse), torch.full_like(s_lse, -float("inf")), s_lse)
            max_lse = torch.maximum(p_lse, s_lse)
            p_se = torch.exp(p_lse - max_lse)
            s_se = torch.exp(s_lse - max_lse)
            out_se = p_se + s_se
            merged = (
                prefix_output[:merge_tokens].to(torch.float32) * (p_se / out_se).unsqueeze(-1)
                + suffix_output[:merge_tokens].to(torch.float32) * (s_se / out_se).unsqueeze(-1)
            )
            if output_scale is not None:
                merged = merged * (1.0 / output_scale)
                finfo = torch.finfo(output.dtype)
                merged = torch.clamp(merged, finfo.min, finfo.max)
            output[:merge_tokens].copy_(merged.to(output.dtype))
            if output_lse is not None:
                output_lse[:, :merge_tokens].copy_((torch.log(out_se) + max_lse).transpose(0, 1))

        return None
"""
    if old not in s:
        raise SystemExit("merge_attn_states patch anchor not found")
    s = s.replace(old, new, 1)
    p.write_text(s)

print("patched", p)
