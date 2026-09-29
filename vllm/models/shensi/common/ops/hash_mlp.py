# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def hash_scale(out: torch.Tensor, deepemb: torch.Tensor) -> torch.Tensor:
    return out * deepemb
