# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    vllm_topk_softplus_sqrt,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.shensi import ShensiConfig

_INIT_STD = float(ShensiConfig.initializer_range)


def _current_quant_config():
    config = get_current_vllm_config_or_none()
    return None if config is None else config.quant_config


def select_moe_backend(vllm_config: VllmConfig) -> str:
    if vllm_config.kernel_config.moe_backend == "auto":
        vllm_config.kernel_config.moe_backend = "triton"
    return vllm_config.kernel_config.moe_backend


def unregister_moe_runner(experts: nn.Module) -> None:
    layer_name = getattr(experts, "layer_name", None)
    if layer_name is None:
        return
    compilation_config = get_current_vllm_config().compilation_config
    compilation_config.static_forward_context.pop(layer_name, None)
    if layer_name in compilation_config.static_all_moe_layers:
        compilation_config.static_all_moe_layers.remove(layer_name)


class ShensiTopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        scoring_func: str,
        routed_scaling_factor: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert scoring_func == "sqrtsoftplus", scoring_func
        self.hidden_dim = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.linear = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            quant_config=_current_quant_config(),
            prefix=prefix,
        )
        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    def reset_parameters(self) -> None:
        nn.init.normal_(self.linear.weight, mean=0.0, std=_INIT_STD)

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits, _ = self.linear(hidden_states.reshape(-1, self.hidden_dim))
        return logits

    def select_experts(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gating = logits.float() if logits.dtype != torch.float32 else logits
        num_tokens = gating.shape[0]
        topk_weights = torch.empty(
            num_tokens, self.top_k, dtype=torch.float32, device=gating.device
        )
        topk_ids = torch.empty(
            num_tokens, self.top_k, dtype=torch.int32, device=gating.device
        )
        token_expert_indices = torch.empty(
            num_tokens, self.top_k, dtype=torch.int32, device=gating.device
        )
        vllm_topk_softplus_sqrt(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating,
            True,
            None,
            None,
            None,
            self.routed_scaling_factor,
        )
        return topk_weights, topk_ids.long()

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router_logits(hidden_states)
        weights, indices = self.select_experts(logits)
        return logits, weights, indices


class ShensiFusedMoeRouter(BaseRouter):
    def __init__(self, gate: ShensiTopKRouter) -> None:
        super().__init__(
            top_k=gate.top_k, global_num_experts=gate.num_experts, eplb_state=None
        )
        self.gate = gate

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func="sqrtsoftplus",
            top_k=self.top_k,
            renormalize=True,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.gate.select_experts(router_logits)


class ShensiRoutedOutputTransform(nn.Module):
    def __init__(self, norm: nn.Module, up_proj: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        out, _ = self.up_proj(self.norm(latent))
        return out


class ShensiSparseMoeBlockMixin:
    def __init__(
        self,
        hidden_size: int,
        routed_hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        scoring_func: str,
        routed_scaling_factor: float,
        rms_norm_eps: float,
        swiglu_limit: float | None,
        mlp_bias: bool = False,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.routed_hidden_size = routed_hidden_size
        self.gate = ShensiTopKRouter(
            hidden_size,
            num_experts,
            top_k,
            scoring_func,
            routed_scaling_factor,
            prefix=f"{prefix}.gate",
        )
        vllm_config = get_current_vllm_config()
        if current_platform.is_cuda():
            select_moe_backend(vllm_config)
        quant_config = vllm_config.quant_config
        self.routed_expert_down_proj = ReplicatedLinear(
            hidden_size,
            routed_hidden_size,
            bias=mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_norm = RMSNorm(
            routed_hidden_size, eps=rms_norm_eps, dtype=torch.float32
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            routed_hidden_size,
            hidden_size,
            bias=mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        self.routed_output_transform = ShensiRoutedOutputTransform(
            self.routed_expert_norm, self.routed_expert_up_proj
        )
        self.experts = FusedMoEFactory(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=routed_hidden_size,
            intermediate_size=moe_intermediate_size,
            renormalize=True,
            routed_scaling_factor=routed_scaling_factor,
            quant_config=quant_config,
            activation="silu",
            swiglu_limit=swiglu_limit,
            has_bias=mlp_bias,
            ckpt_names=("gate_up_proj", "down_proj", "up_proj"),
            router=ShensiFusedMoeRouter(self.gate),
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_output_transform,
            prefix=prefix,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, self.hidden_size)
        logits = self.gate.router_logits(flat)
        routed = self.experts(hidden_states=flat, router_logits=logits)
        return routed.view_as(hidden_states)
