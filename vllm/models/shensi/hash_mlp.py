# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config_or_none
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP


class ShensiHashMLP(DeepseekV4MLP):
    """DeepseekV4MLP whose gated activation is scaled by a deep embedding.

    The leading Shensi layers are token-id gated dense MLPs: the token's deep
    embedding scales the low-rank bottleneck between the activation and the
    down projection. The scale is a single elementwise multiply — there is no
    kernel to fuse it with — so it stays inline rather than behind an op.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        vocab_size: int,
        swiglu_limit: float | None,
        prefix: str = "",
    ) -> None:
        config = get_current_vllm_config_or_none()
        quant_config = None if config is None else config.quant_config
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
            swiglu_limit=swiglu_limit,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.deepemb = nn.Embedding(vocab_size, intermediate_size)
        nn.init.ones_(self.deepemb.weight)

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gated = self.act_fn(gate_up)
        gated = gated * self.deepemb(input_ids)
        out, _ = self.down_proj(gated)
        return out
