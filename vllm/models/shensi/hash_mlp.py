# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)

from .common.ops.hash_mlp import hash_scale


class ClampedSwiGLU(nn.Module):
    def __init__(self, swiglu_limit: float | None) -> None:
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.act_fn = (
            SiluAndMul() if swiglu_limit is None else SiluAndMulWithClamp(swiglu_limit)
        )

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        return self.act_fn(gate_up)


class ShensiHashMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        vocab_size: int,
        swiglu_limit: float | None,
        mlp_bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        config = get_current_vllm_config_or_none()
        quant_config = None if config is None else config.quant_config
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.deepemb = nn.Embedding(vocab_size, hidden_size)
        self.act_fn = ClampedSwiGLU(swiglu_limit)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gated = self.act_fn(gate_up)
        out, _ = self.down_proj(gated)
        return hash_scale(out, self.deepemb(input_ids))
