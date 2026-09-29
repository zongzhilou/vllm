# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.transformers_utils.configs.shensi import ShensiConfig

from .common.ops.xhc import (
    xhc_gate,
    xhc_gate_collapse,
    xhc_gram_schmidt,
    xhc_route_select,
    xhc_write_back,
)

_INIT_STD = float(ShensiConfig.initializer_range)


class ShensiHyperConnection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        active_streams: int,
        fixed_streams: int,
        conv_kernels: tuple[int, ...] | list[int] | None,
        rms_norm_eps: float,
        is_mlp: bool,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hc_mult = hc_mult
        self.active_streams = active_streams
        self.fixed_streams = fixed_streams
        self.routed_streams = active_streams - fixed_streams
        assert self.routed_streams >= 0, (active_streams, fixed_streams)
        self.norm_eps = rms_norm_eps
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps, has_weight=False)
        self.pre_proj = ReplicatedLinear(
            hc_mult * hidden_size,
            hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.pre_proj",
        )
        self.pre_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.pre_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.route_norm = LayerNorm(hc_mult * hidden_size, eps=1e-5)
        self.route_proj = ReplicatedLinear(
            hc_mult * hidden_size,
            hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.route_proj",
        )
        self.route_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.route_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.is_mlp = is_mlp
        self.kr = (len(conv_kernels) + 1) if (is_mlp and conv_kernels) else 1
        if is_mlp and conv_kernels:
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden_size,
                        hidden_size,
                        kernel_size,
                        padding=kernel_size - 1,
                        groups=hidden_size,
                        bias=False,
                        dtype=torch.float32,
                    )
                    for kernel_size in conv_kernels
                ]
            )
        else:
            self.temporal_convs = nn.ModuleList()
        self.post_proj = ReplicatedLinear(
            active_streams * hidden_size,
            active_streams * self.kr,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.post_proj",
        )
        self.post_base = nn.Parameter(
            torch.empty(active_streams * self.kr, dtype=torch.float32)
        )
        self.post_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.pre_proj.weight, mean=0.0, std=_INIT_STD)
            nn.init.zeros_(self.pre_base)
            nn.init.constant_(self.pre_scale, 0.01)
            nn.init.normal_(self.route_proj.weight, mean=0.0, std=_INIT_STD)
            nn.init.zeros_(self.route_base)
            nn.init.ones_(self.route_scale)
            nn.init.normal_(self.post_proj.weight, mean=0.0, std=_INIT_STD)
            nn.init.zeros_(self.post_base)
            nn.init.constant_(self.post_scale, 0.01)
            for conv in self.temporal_convs:
                nn.init.normal_(conv.weight, mean=0.0, std=_INIT_STD)

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_streams.flatten(start_dim=1).float())
        return xhc_gate_collapse(
            flat,
            self.pre_proj.weight.float(),
            self.pre_scale.float(),
            self.pre_base.float(),
            hidden_streams,
            hidden_streams.dtype,
        )

    def write_back(
        self, hidden_streams: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        hc = hidden_streams.shape[1]
        hidden_size = hidden_streams.shape[2]
        flat = self.route_norm(
            hidden_streams.flatten(start_dim=1).to(self.route_norm.weight.dtype)
        ).float()
        route_scores = xhc_gate(
            flat,
            self.route_proj.weight.float(),
            self.route_scale.float(),
            self.route_base.float(),
        )
        active_idx, p = xhc_route_select(
            route_scores, hc, self.fixed_streams, self.routed_streams
        )
        active = hidden_streams.gather(
            1, active_idx.unsqueeze(-1).expand(-1, -1, hidden_size)
        ).float()
        active_flat = self.input_norm(active.flatten(start_dim=1))
        if self.is_mlp and len(self.temporal_convs) > 0:
            x = (
                sublayer_output.transpose(0, 1)
                .unsqueeze(0)
                .to(self.temporal_convs[0].weight.dtype)
            )
            num_tokens = sublayer_output.shape[0]
            conv_outs = [
                conv(x)[..., :num_tokens].squeeze(0).transpose(0, 1)
                for conv in self.temporal_convs
            ]
            out_aug = xhc_gram_schmidt(
                sublayer_output, conv_outs, self.norm_eps
            ).float()
        else:
            out_aug = sublayer_output.float().unsqueeze(1)
        return xhc_write_back(
            hidden_streams,
            active_idx,
            p,
            active_flat,
            self.post_proj.weight.float(),
            self.post_scale.float(),
            self.post_base.float(),
            out_aug,
            self.kr,
        )


class ShensiHyperHead(nn.Module):
    def __init__(
        self, hidden_size: int, hc_mult: int, rms_norm_eps: float, prefix: str = ""
    ) -> None:
        super().__init__()
        self.input_norm = RMSNorm(hidden_size, eps=rms_norm_eps, has_weight=False)
        self.hc_proj = ReplicatedLinear(
            hc_mult * hidden_size,
            hc_mult,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.hc_proj",
        )
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.hc_proj.weight, mean=0.0, std=_INIT_STD)
            nn.init.zeros_(self.hc_base)
            nn.init.ones_(self.hc_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(start_dim=1).float())
        return xhc_gate_collapse(
            flat,
            self.hc_proj.weight.float(),
            self.hc_scale.float(),
            self.hc_base.float(),
            x,
            x.dtype,
        )
