# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear

from .common.ops.attn_res import (
    attn_res_route,
    attn_res_update,
    scale_by_rms,
)


class ShensiAttentionResidual(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float,
        rank: int,
        num_layers: int,
        read_heads: int = 8,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_eps = rms_norm_eps
        self.rank = rank
        self.read_heads = read_heads
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, has_weight=False)
        self.q_a_proj = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_b_proj = ReplicatedLinear(
            rank,
            hidden_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_b_proj",
        )
        self.g_a_proj = ReplicatedLinear(
            hidden_size,
            rank,
            bias=True,
            params_dtype=torch.float32,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ReplicatedLinear(
            rank,
            3 * hidden_size,
            bias=True,
            params_dtype=torch.float32,
            prefix=f"{prefix}.g_b_proj",
        )
        self.k_a_proj = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_a_proj",
        )
        self.k_b_proj = ReplicatedLinear(
            rank,
            hidden_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_b_proj",
        )
        self.g_scale = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.t = nn.Parameter(
            torch.linspace(0.0, 1.0, hidden_size, dtype=torch.float32)
            * math.log(2.0 * num_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        hidden_size = self.hidden_size
        with torch.no_grad():
            nn.init.zeros_(self.g_a_proj.weight)
            nn.init.zeros_(self.g_b_proj.weight)
            self.g_b_proj.bias[:hidden_size] = 2.0
            self.g_b_proj.bias[hidden_size : 2 * hidden_size] = -2.0
            self.g_b_proj.bias[2 * hidden_size :] = -2.0
            nn.init.zeros_(self.q_a_proj.weight)
            nn.init.zeros_(self.q_b_proj.weight)
            nn.init.kaiming_normal_(
                self.k_a_proj.weight, mode="fan_in", nonlinearity="linear"
            )
            nn.init.kaiming_normal_(
                self.k_b_proj.weight, mode="fan_in", nonlinearity="linear"
            )

    def forward(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self.norm(prefix + (delta if delta is not None else 0.0))
        updated = attn_res_update(
            state,
            prefix,
            delta,
            self.g_a_proj.weight.float(),
            self.g_a_proj.bias.float(),
            self.g_b_proj.weight.float(),
            self.g_b_proj.bias.float(),
            self.k_a_proj.weight.float(),
            self.k_b_proj.weight.float(),
            self.g_scale.float(),
            self.t.float(),
        )
        if num_blocks > 0:
            routed = attn_res_route(
                blocks,
                updated,
                F.linear(
                    F.linear(state, self.q_a_proj.weight.float()),
                    self.q_b_proj.weight.float(),
                ),
                self.norm_eps,
                num_blocks,
                self.read_heads,
                self.g_scale[3].float(),
            )
        else:
            routed = torch.zeros_like(updated)
        output = updated + routed
        if output_norm_weight is not None:
            output = scale_by_rms(output, output_norm_weight.float(), self.norm_eps)
        return output.to(out_dtype), updated.to(out_dtype)
