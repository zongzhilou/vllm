# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import ClassVar, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.registry import AttentionBackendEnum

_COMPRESS_RATIO_TO_LAYER_TYPE = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _select_shensi_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):
        raise ValueError(
            f"{backend.name} is not a DeepSeek V4 attention backend. "
            "Use FLASHINFER_MLA_SPARSE_DSV4 for Shensi FlashInfer sparse MLA."
        )
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLAAttention,
    )
    from vllm.models.shensi.attention import (
        ShensiFlashInferSM120Attention,
        ShensiFlashMLAAttention,
    )

    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        if device_capability is not None and device_capability.major == 12:
            return ShensiFlashInferSM120Attention
        return DeepseekV4FlashInferMLAAttention
    if backend in (
        AttentionBackendEnum.FLASHMLA_SPARSE,
        AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
    ):
        return ShensiFlashMLAAttention
    if device_capability is not None and device_capability.major == 12:
        return ShensiFlashInferSM120Attention
    return ShensiFlashMLAAttention


def _attn_res_block_layer_types(config) -> list[str]:
    num_hash_layers = getattr(config, "num_hash_layers", None)
    if num_hash_layers is None:
        mlp_layer_types = getattr(config, "mlp_layer_types", None)
        num_hash_layers = (
            mlp_layer_types.count("hash_moe") if mlp_layer_types is not None else 0
        )
    block_size = config.attn_res_block_size
    return [
        "block_write_layer"
        if i == 0 or (i - num_hash_layers) % block_size == 0
        else "block_read_layer"
        for i in range(config.num_hidden_layers)
    ]


class ShensiUnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(
            x.dtype
        )


class ShensiHyperConnection(nn.Module):
    def __init__(self, config, prefix: str, is_mlp: bool = False):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.active_streams = config.hc_active_streams
        self.fixed_streams = config.hc_fixed_streams
        self.routed_streams = self.active_streams - self.fixed_streams

        self.input_norm = ShensiUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.route_norm = nn.LayerNorm(self.hc_mult * config.hidden_size)

        self.pre_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult))
        self.pre_scale = nn.Parameter(torch.empty(1))

        self.route_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.route_base = nn.Parameter(torch.empty(self.hc_mult))
        self.route_scale = nn.Parameter(torch.empty(1))

        self.is_mlp = is_mlp
        self.kr = (len(config.hc_conv_kernels) + 1) if self.is_mlp else 1
        if self.is_mlp:
            self.conv_kernels = config.hc_conv_kernels
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        ks,
                        padding=ks - 1,
                        groups=config.hidden_size,
                        bias=False,
                    )
                    for ks in self.conv_kernels
                ]
            )

        self.post_fn = nn.Parameter(
            torch.empty(
                self.active_streams * self.kr, self.active_streams * config.hidden_size
            )
        )
        self.post_base = nn.Parameter(torch.empty(self.active_streams * self.kr))
        self.post_scale = nn.Parameter(torch.empty(1))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        if current_platform.is_cuda():
            from vllm.models.shensi.ops.hc import hc_collapse

            return hc_collapse(
                hidden_streams,
                self.pre_fn,
                self.pre_scale,
                self.pre_base,
                self.input_norm,
            )
        flat = self.input_norm(hidden_streams.flatten(start_dim=1).float())
        pre = torch.sigmoid(
            F.linear(flat, self.pre_fn.float()) * self.pre_scale.float()
            + self.pre_base.float()
        )
        collapsed = (
            (pre.unsqueeze(-1) * hidden_streams).sum(dim=1).to(hidden_streams.dtype)
        )
        return collapsed

    def write_back(
        self, hidden_streams: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        T, hc, H = hidden_streams.shape

        flat = self.route_norm(
            hidden_streams.flatten(start_dim=1).to(self.route_norm.weight.dtype)
        ).float()
        route_scores = torch.sigmoid(
            F.linear(flat, self.route_fn.float()) * self.route_scale.float()
            + self.route_base.float()
        )
        fixed_mask = torch.arange(hc, device=route_scores.device) < self.fixed_streams
        route_scores = route_scores.masked_fill(fixed_mask.view(1, -1), float("-inf"))
        fixed_idx = torch.arange(self.fixed_streams, device=hidden_streams.device)
        fixed_idx = fixed_idx.view(1, -1).expand(T, -1)
        routed_idx = route_scores.topk(self.routed_streams, dim=-1).indices
        active_idx = torch.cat([fixed_idx, routed_idx], dim=-1)
        p = torch.cat(
            [
                torch.ones_like(fixed_idx, dtype=route_scores.dtype),
                route_scores.gather(-1, routed_idx),
            ],
            dim=-1,
        )

        if self.is_mlp:
            x = (
                sublayer_output.transpose(0, 1)
                .unsqueeze(0)
                .to(self.temporal_convs[0].weight.dtype)
            )
            conv_outs = [conv(x)[..., :T] for conv in self.temporal_convs]
            ortho = []
            prevs = [x]
            for g in conv_outs:
                v = g
                for prev in prevs:
                    denom = (
                        (prev * prev)
                        .sum(dim=1, keepdim=True)
                        .clamp_min(self.input_norm.eps)
                    )
                    v = v - ((prev * v).sum(dim=1, keepdim=True) / denom) * prev
                ortho.append(v)
                prevs.append(v)
            aug_flat = torch.cat([x] + ortho, dim=1)
            out_aug = (
                aug_flat.view(self.kr, H, T).transpose(0, 2).transpose(1, 2).float()
            )
        else:
            out_aug = sublayer_output.float().unsqueeze(-2)

        active_streams = hidden_streams.gather(
            1, active_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        post = 2 * torch.sigmoid(
            F.linear(
                self.input_norm(active_streams.flatten(start_dim=1).float()),
                self.post_fn.float(),
            ).view(T, self.active_streams, self.kr)
            * self.post_scale.float()
            + self.post_base.float().view(self.active_streams, self.kr)
        )

        delta = torch.einsum("tkr,trh->tkh", post, out_aug) * p.unsqueeze(-1)
        updated_active = delta.to(hidden_streams.dtype)
        return hidden_streams.scatter(
            1, active_idx.unsqueeze(-1).expand(-1, -1, H), updated_active
        )


class ShensiAttentionResidual(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        # Layout: [decay(H), erase(H), write(H), bias(3H), k(H), q(H)] = 8H
        self.proj_weight = nn.Parameter(
            torch.empty(8 * config.hidden_size, config.hidden_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = hidden_states
        blocks = residual
        T, hc, D = prefix_sum.shape
        flat_prefix = prefix_sum.reshape(-1, D)
        flat_delta = delta.reshape(-1, D) if delta is not None else None
        if flat_delta is not None:
            state = self.norm((flat_prefix + flat_delta).float())
        else:
            state = self.norm(flat_prefix.float())
        proj = F.linear(state, self.proj_weight.float())
        if current_platform.is_cuda():
            from vllm.models.shensi.ops.attn_res import attn_res

            output = attn_res(
                flat_prefix,
                flat_delta,
                blocks.reshape(-1, blocks.shape[2], D),
                proj,
                output_norm_weight,
                num_blocks=num_blocks,
                eps=self.norm.eps,
                output_norm_eps=self.norm.eps,
            )
            return (
                output.to(prefix_sum.dtype).reshape(T, hc, D),
                flat_prefix.to(prefix_sum.dtype).reshape(T, hc, D),
                blocks,
            )
        # eager fallback
        decay, erase, write = torch.sigmoid(proj[:, : 3 * D].reshape(-1, 3, D)).unbind(
            1
        )
        forgotten = decay * flat_prefix.float()
        khat = F.normalize(proj[:, 6 * D : 7 * D], dim=-1)
        r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
        updated = (
            forgotten
            - khat * r
            + write * (flat_delta.float() if flat_delta is not None else 0.0)
        )
        if num_blocks > 0:
            values = torch.cat(
                [
                    blocks.reshape(-1, blocks.shape[2], D)[..., :num_blocks, :].float(),
                    updated.unsqueeze(-2),
                ],
                dim=-2,
            )
            reciprocal_std = torch.rsqrt(values.square().mean(dim=-1) + self.norm.eps)
            logits = (values * proj[:, 7 * D :].unsqueeze(-2)).sum(
                dim=-1
            ) * reciprocal_std
            scores = F.softmax(logits, dim=-1)
            routed = scores.unsqueeze(-1).mul(values).sum(dim=-2)
        else:
            routed = torch.zeros_like(updated)
        output = updated + routed
        if output_norm_weight is not None:
            output = (
                output
                * torch.rsqrt(
                    output.square().mean(dim=-1, keepdim=True) + self.norm.eps
                )
                * output_norm_weight.float()
            )
        updated_out = updated.to(prefix_sum.dtype).reshape(T, hc, D)
        return output.to(prefix_sum.dtype).reshape(T, hc, D), updated_out, blocks


class ShensiMLP(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.is_hash = True
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.routed_expert_hidden_size] * 2,
            bias=config.mlp_bias,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act_fn = SiluAndMulWithClamp(config.swiglu_limit)
        self.down_proj = RowParallelLinear(
            config.routed_expert_hidden_size,
            config.hidden_size,
            bias=config.mlp_bias,
            prefix=f"{prefix}.down_proj",
        )
        self.deepemb = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x * self.deepemb(input_ids.to(self.deepemb.weight.device)).to(x.dtype)


class ShensiRoutedOutput(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.routed_expert_norm = RMSNorm(
            config.routed_expert_hidden_size, config.rms_norm_eps
        )
        self.routed_expert_up_proj = nn.Linear(
            config.routed_expert_hidden_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.routed_expert_up_proj(self.routed_expert_norm(x))


class ShensiSparseMoeBlock(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.is_hash = False
        # HF semantics: the router scores the FULL hidden state (before the
        # routed-expert down projection), so the gate stays outside the runner
        # and is applied here on the un-transformed hidden states.
        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.routed_expert_hidden_size = config.routed_expert_hidden_size
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size, config.routed_expert_hidden_size, bias=False
        )
        self.routed_output = ShensiRoutedOutput(
            config, prefix=f"{prefix}.routed_output"
        )
        self.experts = FusedMoEFactory(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.routed_expert_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            quant_config=None,
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "sqrtsoftplus"),
            routed_scaling_factor=config.routed_scaling_factor,
            swiglu_limit=config.swiglu_limit,
            routed_input_transform=self.routed_expert_down_proj,
            routed_output_transform=self.routed_output,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return self.experts(hidden_states, router_logits)


class ShensiDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        block_layer_types: list[str],
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        config: object | None = None,
        layer_type_index: int | None = None,
    ):
        super().__init__()
        # Draft models (DSpark) build layers from their own config while the
        # vllm_config still points at the target; None falls back to the target.
        # The layer types are indexed by `layer_type_index` (the draft-relative
        # layer number) because draft layers carry offset prefixes.
        if config is None:
            config = vllm_config.model_config.hf_config.get_text_config()
        layer_idx = extract_layer_index(prefix)
        type_idx = layer_idx if layer_type_index is None else layer_type_index
        self.layer_idx = layer_idx
        self.is_block_write_layer = block_layer_types[type_idx] == "block_write_layer"
        self.prev_valid_blocks = sum(
            1
            for layer_type in block_layer_types[:type_idx]
            if layer_type == "block_write_layer"
        )
        self.block_write_idx = self.prev_valid_blocks

        self.self_attn = _select_shensi_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        mlp_layer_types = getattr(config, "mlp_layer_types", None) or []
        if mlp_layer_types:
            is_hash = mlp_layer_types[type_idx] == "hash_moe"
        else:
            is_hash = type_idx < getattr(config, "num_hash_layers", 0)
        self.mlp = (
            ShensiMLP(config, prefix=f"{prefix}.mlp")
            if is_hash
            else ShensiSparseMoeBlock(config, prefix=f"{prefix}.mlp")
        )

        self.attn_hc = ShensiHyperConnection(
            config, prefix=f"{prefix}.attn_hc", is_mlp=False
        )
        self.ffn_hc = ShensiHyperConnection(
            config, prefix=f"{prefix}.ffn_hc", is_mlp=True
        )
        self.self_attention_attn_res = ShensiAttentionResidual(
            config, prefix=f"{prefix}.self_attention_attn_res"
        )
        self.mlp_attn_res = ShensiAttentionResidual(
            config, prefix=f"{prefix}.mlp_attn_res"
        )

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = hidden_states - prefix_sum if hidden_states is not None else None

        # Retrieve from the previous blocks' slots before the local attention runs.
        hidden_states, prefix_sum, residual = self.self_attention_attn_res(
            delta,
            residual,
            prefix_sum,
            output_norm_weight=self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
        )
        if self.is_block_write_layer:
            # Deposit the layer-entry state into this block's slot, then restart the
            # prefix so the attention output below seeds the rest of the block.
            residual = torch.cat(
                [
                    residual[..., : self.block_write_idx, :],
                    (hidden_states if hidden_states is not None else prefix_sum)
                    .to(residual.dtype)
                    .unsqueeze(-2),
                    residual[..., self.block_write_idx + 1 :, :],
                ],
                dim=-2,
            )
            prefix_sum = None

        collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(positions, collapsed, None)
        hidden_states = self.attn_hc.write_back(hidden_states, attn_output)

        # The prefix accumulates the attention output; write layers restart from it.
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        # MLP-side lookup consumes the attention-inclusive prefix, including the
        # just-written slot on write layers.
        mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer
        hidden_states, prefix_sum, residual = self.mlp_attn_res(
            prefix_sum,
            residual,
            prefix_sum,
            output_norm_weight=self.post_attention_layernorm.weight,
            num_blocks=mlp_valid_blocks,
        )

        collapsed = self.ffn_hc(hidden_states)
        mlp_output = (
            self.mlp(collapsed, input_ids=input_ids)
            if self.mlp.is_hash
            else self.mlp(collapsed)
        )
        hidden_states = self.ffn_hc.write_back(hidden_states, mlp_output)

        prefix_sum = prefix_sum + hidden_states
        return hidden_states, prefix_sum, residual


class ShensiHyperHead(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = ShensiUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.hc_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_mult * config.hidden_size)
        )
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.models.shensi.ops.hc import hc_collapse

        return hc_collapse(x, self.hc_fn, self.hc_scale, self.hc_base, self.input_norm)


class ShensiModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Multimodal wrappers (Shensi-VL) pass the composite config; the
        # language stack lives on its text_config.
        config = vllm_config.model_config.hf_config.get_text_config()
        self.config = config
        # DeepSeek-V4 components (attention / compressor) read the hf_config
        # off the vllm_config; hand them a text-only view so multimodal
        # wrappers (Shensi-VL) work without touching those files.
        text_vllm_config = vllm_config.with_hf_config(config)

        if not hasattr(config, "compress_ratios") or config.compress_ratios is None:
            config.compress_ratios = [
                _COMPRESS_RATIO_TO_LAYER_TYPE.get(t, 1)
                for t in getattr(config, "layer_types", [])
            ]
        if not hasattr(config, "qk_rope_head_dim"):
            config.qk_rope_head_dim = int(
                config.head_dim * getattr(config, "partial_rotary_factor", 1.0)
            )

        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.block_layer_types = _attn_res_block_layer_types(config)
        self.num_attn_res_blocks = self.block_layer_types.count("block_write_layer")

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ShensiDecoderLayer(
                text_vllm_config,
                prefix=prefix,
                block_layer_types=self.block_layer_types,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        self._share_pool_expert_weights()

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.hc_head = ShensiHyperHead(config, prefix=f"{prefix}.hc_head")
        self.output_attn_res = ShensiAttentionResidual(
            config, prefix=f"{prefix}.output_attn_res"
        )

    def _share_pool_expert_weights(self) -> None:
        write_layers = [
            i for i, t in enumerate(self.block_layer_types) if t == "block_write_layer"
        ]
        shared: dict[int, list] = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer.mlp.is_hash:
                continue
            owners = [w for w in write_layers if w <= layer_idx]
            if not owners:
                continue
            block_id = max(owners)
            pool = shared.setdefault(block_id, [layer.mlp, [layer_idx]])
            if pool[0] is not layer.mlp:
                pool[1].append(layer_idx)
            for idx in pool[1][1:]:
                self.layers[idx].mlp.gate = pool[0].gate
                self.layers[idx].mlp.experts = pool[0].experts
                self.layers[idx].mlp.routed_expert_down_proj = pool[
                    0
                ].routed_expert_down_proj
                self.layers[idx].mlp.routed_output = pool[0].routed_output

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.hidden_size),
                    dtype=dtype,
                    device=device,
                )
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        hidden_states = (
            hidden_states.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        )

        block_residual = hidden_states.new_zeros(
            hidden_states.size(0),
            hidden_states.size(1),
            self.num_attn_res_blocks,
            hidden_states.size(2),
        )
        prefix_sum = hidden_states
        hidden_states = None
        residual = block_residual

        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer], start=self.start_layer
        ):
            hidden_states, prefix_sum, residual = layer(
                hidden_states,
                residual,
                prefix_sum,
                positions,
                input_ids,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # Collapse the hc streams: the DSpark drafter fuses per-layer
                # target hidden states into its context input.
                aux_hidden_states.append(hidden_states.mean(dim=1))

        hidden_states, _, _ = self.output_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            output_norm_weight=None,
            num_blocks=self.num_attn_res_blocks,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(self.hc_head(hidden_states))
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_shensi_weights(self, weights)


def _load_shensi_weights(
    module: nn.Module, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Shared weight loader for Shensi stacks (target and DSpark drafts).

    Resolves the fused / renamed / expert-sharded checkpoint names against
    ``module.named_parameters()``; both name spaces use relative prefixes.
    """
    stacked_params_mapping = [
        ("self_attn.fused_wqa_wkv", "self_attn.q_a_proj", 0),
        ("self_attn.fused_wqa_wkv", "self_attn.kv_proj", 1),
        (
            "self_attn.compressor.fused_wkv_wgate",
            "self_attn.compressor.kv_proj",
            0,
        ),
        (
            "self_attn.compressor.fused_wkv_wgate",
            "self_attn.compressor.gate_proj",
            1,
        ),
        (
            "self_attn.compressor.indexer.compressor.fused_wkv_wgate",
            "self_attn.compressor.indexer.kv_proj",
            0,
        ),
        (
            "self_attn.compressor.indexer.compressor.fused_wkv_wgate",
            "self_attn.compressor.indexer.gate_proj",
            1,
        ),
        ("mlp.gate_up_proj", "mlp.gate_proj", 0),
        ("mlp.gate_up_proj", "mlp.up_proj", 1),
    ]
    expert_mapping = [
        ("mlp.experts.w13_weight", "mlp.experts.gate_up_proj", 0, "w1"),
        ("mlp.experts.w13_weight", "mlp.experts.gate_up_proj", 0, "w3"),
        ("mlp.experts.w2_weight", "mlp.experts.down_proj", 0, "w2"),
    ]
    renames = {
        "self_attn.q_a_norm": "self_attn.q_norm",
        "self_attn.q_b_proj": "self_attn.wq_b",
        "self_attn.o_a_proj": "self_attn.wo_a",
        "self_attn.o_b_proj": "self_attn.wo_b",
        "self_attn.sinks": "self_attn.attn_sink",
        "self_attn.compressor.position_bias": "self_attn.compressor.ape",
        "self_attn.compressor.kv_norm": "self_attn.compressor.norm",
        "self_attn.compressor.indexer.position_bias": (
            "self_attn.indexer.compressor.ape"
        ),
        "self_attn.compressor.indexer.kv_norm": ("self_attn.indexer.compressor.norm"),
        "self_attn.compressor.indexer.q_b_proj": ("self_attn.compressor.indexer.wq_b"),
        "self_attn.compressor.indexer.scorer.weights_proj": (
            "self_attn.compressor.indexer.weights_proj"
        ),
        "mlp.routed_expert_norm": "mlp.routed_output.routed_expert_norm",
        "mlp.routed_expert_up_proj": "mlp.routed_output.routed_expert_up_proj",
    }

    params_dict = dict(module.named_parameters())
    loaded_params: set[str] = set()
    hidden = module.config.hidden_size
    for name, loaded_weight in weights:
        stacked = False
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            target_name = name.replace(weight_name, param_name)
            if target_name not in params_dict:
                continue
            param = params_dict[target_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(target_name)
            stacked = True
            break
        if stacked:
            continue
        if ".experts." in name:
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in name:
                    continue
                target_name = name.replace(weight_name, param_name)
                if target_name not in params_dict:
                    continue
                param = params_dict[target_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, target_name, shard_id, expert_id)
                loaded_params.add(target_name)
            continue
        if name.endswith("_attn_res.gate_proj.weight"):
            param = params_dict[
                name.replace("_attn_res.gate_proj.weight", "_attn_res.proj_weight")
            ]
            with torch.no_grad():
                param[: 3 * hidden].copy_(loaded_weight)
            loaded_params.add(
                name.replace("_attn_res.gate_proj.weight", "_attn_res.proj_weight")
            )
            continue
        if name.endswith("_attn_res.gate_proj.bias"):
            param = params_dict[
                name.replace("_attn_res.gate_proj.bias", "_attn_res.proj_weight")
            ]
            with torch.no_grad():
                # The fused proj_weight stores the gate bias as a per-row
                # (3H, H) block; broadcast the (3H,) bias across columns.
                param[3 * hidden : 6 * hidden].copy_(loaded_weight.unsqueeze(1))
            loaded_params.add(
                name.replace("_attn_res.gate_proj.bias", "_attn_res.proj_weight")
            )
            continue
        if name.endswith("_attn_res.q_proj"):
            param = params_dict[
                name.replace("_attn_res.q_proj", "_attn_res.proj_weight")
            ]
            with torch.no_grad():
                param[7 * hidden :].copy_(loaded_weight)
            loaded_params.add(name.replace("_attn_res.q_proj", "_attn_res.proj_weight"))
            continue
        if name.endswith("_attn_res.k_proj"):
            param = params_dict[
                name.replace("_attn_res.k_proj", "_attn_res.proj_weight")
            ]
            with torch.no_grad():
                param[6 * hidden : 7 * hidden].copy_(loaded_weight)
            loaded_params.add(name.replace("_attn_res.k_proj", "_attn_res.proj_weight"))
            continue
        for src, dst in renames.items():
            if src in name:
                name = name.replace(src, dst)
                break
        if name not in params_dict:
            continue
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        loaded_params.add(name)

    return loaded_params


class ShensiForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    supports_eagle3: ClassVar[Literal[True]] = True
    model_cls = ShensiModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.num_moe_layers = sum(
            1 for layer in self.model.layers if not layer.mlp.is_hash
        )
        self.num_expert_groups = 1
        self.num_logical_experts = config.n_routed_experts
        self.num_physical_experts = config.n_routed_experts
        self.num_local_physical_experts = config.n_routed_experts
        self.num_routed_experts = config.n_routed_experts
        self.num_shared_experts = 0
        self.num_redundant_experts = 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint nests the backbone under "model." while the inner
        # model's parameter names are relative; the lm_head lives on this
        # outer module. Resolve both here before delegating.
        stripped = []
        for name, tensor in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            stripped.append((name, tensor))
        loaded: set[str] = set()
        lm_head_weights = [
            (name, tensor) for name, tensor in stripped if name == "lm_head.weight"
        ]
        if lm_head_weights:
            weight_loader = getattr(
                self.lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(self.lm_head.weight, lm_head_weights[0][1])
            loaded.add("lm_head.weight")
        inner_weights = [
            (name, tensor) for name, tensor in stripped if name != "lm_head.weight"
        ]
        loaded.update(
            "model." + name for name in self.model.load_weights(iter(inner_weights))
        )
        return loaded
