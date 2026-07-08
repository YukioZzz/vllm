import importlib
from pathlib import Path


p = Path(importlib.import_module("vllm.platforms.rocm").__file__)
s = p.read_text()

old = """    if use_mla:
        if rocm_aiter_ops.is_mla_enabled():
            return [
                AttentionBackendEnum.ROCM_AITER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
            ]
        else:
            return [
                AttentionBackendEnum.TRITON_MLA,
            ]
"""

new = """    if use_mla:
        # Validation patch: force dense MLA users, including native MTP drafter,
        # through ROCM_AITER_MLA so the DCP LSE-return path can be exercised.
        return [
            AttentionBackendEnum.ROCM_AITER_MLA,
            AttentionBackendEnum.TRITON_MLA,
            AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
        ]
"""

if old not in s:
    raise SystemExit("rocm priority anchor not found")

p.write_text(s.replace(old, new, 1))
print("patched", p)
