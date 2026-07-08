import importlib
import shutil
from pathlib import Path


backend_module = importlib.import_module("vllm.v1.attention.backends.mla.rocm_aiter_mla")
backend_path = Path(backend_module.__file__)
provider_src = Path(__file__).with_name("mla_dcp_lse_provider.py")
provider_dst = backend_path.with_name("mla_dcp_lse_provider.py")
shutil.copyfile(provider_src, provider_dst)

s = backend_path.read_text()


def ensure_top_import(text: str, import_line: str) -> str:
    top = "\n".join(text.splitlines()[:120])
    if import_line in top:
        return text

    lines = text.splitlines(keepends=True)
    insert_at = 0
    for idx, line in enumerate(lines[:120]):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_at = idx + 1
    lines.insert(insert_at, import_line + "\n")
    return "".join(lines)


s = ensure_top_import(
    s,
    "from vllm.v1.attention.backends.mla.mla_dcp_lse_provider import forward_mqa_with_lse",
)

if "can_return_lse_for_decode: bool = True" not in s:
    s = s.replace(
        "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):\n",
        "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):\n"
        "    can_return_lse_for_decode: bool = True\n",
        1,
    )

old = """        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # Build kwargs for mla_decode_fwd. Pass persistent metadata only
"""

new = """        if self.need_to_return_lse_for_decode:
            return forward_mqa_with_lse(
                self,
                q,
                kv_c_and_k_pe_cache,
                attn_metadata,
                layer,
                mla_padded_q,
                o,
                AiterMLAHelper.get_mla_unpadded_o,
            )

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # Build kwargs for mla_decode_fwd. Pass persistent metadata only
"""

if "forward_mqa_with_lse(" not in s:
    if old not in s:
        raise SystemExit("patch anchor not found")
    s = s.replace(old, new, 1)

backend_path.write_text(s)
print("patched", backend_path)
print("installed provider", provider_dst)
