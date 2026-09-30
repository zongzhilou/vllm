# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.machinery
import os
import sys
import types

# cute/ 里的文件用的是 `flash_attn.cute.*` 这套导入，两种形态都要让它们解析：
# ① 符号链接模式（构建时给了 VLLM_FLASH_ATTN_SRC_DIR）：cute/ 指向真实源码树；
# ② 随 wheel 装下来的自带快照（cute/ 是真目录，且与 requirements 里钉的
#    nvidia-cutlass-dsl / quack-kernels 是配套的那一版）。
# 已经装了 flash_attn / fa4 的话不动它（下面那句 not in sys.modules）。
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.isdir(_cute_dir) and "flash_attn" not in sys.modules:
    _fa_mod = types.ModuleType("flash_attn")
    _fa_mod.__path__ = [os.path.dirname(os.path.realpath(_cute_dir))]
    _fa_mod.__package__ = "flash_attn"
    _fa_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", None, is_package=True
    )
    _fa_mod.__spec__.submodule_search_locations = _fa_mod.__path__
    sys.modules["flash_attn"] = _fa_mod

from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    compile_flash_attn_varlen_func_from_specs,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )

__all__ = [
    "compile_flash_attn_varlen_func_from_specs",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
