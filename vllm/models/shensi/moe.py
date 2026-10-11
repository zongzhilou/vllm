# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MoE
from vllm.platforms import current_platform


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


class ShensiRoutedOutputTransform(nn.Module):
    def __init__(self, norm: nn.Module, up_proj: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.up_proj = up_proj

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        out, _ = self.up_proj(self.norm(latent))
        return out


class ShensiSparseMoeBlock(DeepseekV4MoE):
    """Shensi MoE: DeepseekV4MoE with a low-rank routed bottleneck.

    Everything else (gate, default sqrtsoftplus router, expert construction,
    EPLB wiring, forward) comes from the base. The deltas are:

    - the routed experts operate on ``routed_expert_hidden_size``: a down
      projection feeds them and a norm + up projection project back, while
      the gate still reads the pre-bottleneck hidden states;
    - the checkpoint names the expert weights
      ``gate_up_proj`` / ``down_proj`` / ``up_proj``;
    - Shensi ships no shared experts and no hash-routed layers (the leading
      token-gated dense layers are plain MLPs, not MoE).
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "", **kwargs) -> None:
        if current_platform.is_cuda():
            select_moe_backend(vllm_config)
        config = vllm_config.model_config.hf_config
        if not config.n_shared_experts:
            # The base only skips shared experts when the field is None; a
            # Shensi checkpoint config may carry 0 for "no shared experts".
            config.n_shared_experts = None
        super().__init__(vllm_config, prefix, num_hash_layers=0, **kwargs)
        if self.use_native_mega_moe:
            raise NotImplementedError("Shensi does not support the MegaMoE backend.")

    def _init_fused_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        quant_config,
        prefix: str,
    ) -> None:
        # The bottleneck projections are created here (not in __init__) so the
        # module assignments happen after nn.Module.__init__ resets the tree.
        self.routed_hidden_size = config.routed_expert_hidden_size
        self.routed_expert_down_proj = ReplicatedLinear(
            config.hidden_size,
            self.routed_hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_norm = RMSNorm(
            self.routed_hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.routed_expert_up_proj = ReplicatedLinear(
            self.routed_hidden_size,
            config.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )
        self.routed_output_transform = ShensiRoutedOutputTransform(
            self.routed_expert_norm, self.routed_expert_up_proj
        )

        parallel_config = vllm_config.parallel_config
        ep_group = get_ep_group()
        ep_size = ep_group.world_size
        ep_rank = ep_group.rank_in_group

        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // ep_size
        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.physical_expert_start = self.experts_start_idx
        self.physical_expert_end = self.experts_end_idx

        self.experts = FusedMoEFactory(
            reduce_results=self.reduce_results,
            shared_experts=None,
            # sqrtsoftplus without an e_score_correction_bias resolves to the
            # plain FusedTopK router, which does not implement sqrtsoftplus;
            # the bias router handles that scoring function and works with no
            # bias (its sqrtsoftplus kernel takes ``e_score_correction_bias``).
            router=FusedTopKBiasRouter(
                top_k=self.n_activated_experts,
                global_num_experts=self.n_routed_experts,
                e_score_correction_bias=None,
                renormalize=config.norm_topk_prob,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
            ),
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=self.routed_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            ckpt_names=("gate_up_proj", "down_proj", "up_proj"),
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=None,
            hash_indices_table=None,
            swiglu_limit=self.swiglu_limit,
            has_bias=config.mlp_bias,
            router_logits_dtype=torch.float32,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=eplb_config.num_redundant_experts,
            is_sequence_parallel=False,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_output_transform,
        )

    def _forward_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # The gate reads the pre-bottleneck hidden states while the experts run
        # on the down-projected latent (the runner applies
        # ``routed_input_transform``), so the logits are computed here instead
        # of binding the gate to the runner.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        return final_hidden_states.view(hidden_states.shape)
