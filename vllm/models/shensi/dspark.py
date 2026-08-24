# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shensi DSpark draft model for speculative decoding.

The drafter shares the Shensi attention / HyperConnection / AttnRes stack with
the target and fuses per-layer target hidden states into its context input via
``context_proj`` + ``context_norm``, mirroring the K3 and DeepSeek-V4 DSpark
drafters. Non-causal drafting uses the same sparse-attention mechanism as the
DeepSeek-V4 DSpark (future query tokens are included in the top-k indices).
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.deepseek_v4.nvidia.dspark import _insert_context_kv

from .model import (
    ShensiAttentionResidual,
    ShensiDecoderLayer,
    ShensiHyperHead,
    _load_shensi_weights,
)


class DSparkShensiModel(nn.Module):
    """Shensi DSpark draft backbone: target-fused context + draft layers."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.quant_config = get_draft_quant_config(vllm_config)
        self.hc_mult = config.hc_mult
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.block_layer_types = config.attn_res_block_layer_types
        self.num_attn_res_blocks = self.block_layer_types.count("block_write_layer")

        # The frozen target embedding is aliased after the draft checkpoint loads.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        # Context fusion: combine the target's per-layer hidden states.
        target_hidden_size = (
            getattr(config, "target_hidden_size", None) or config.hidden_size
        )
        num_target_layers = getattr(config, "num_target_layers", None) or len(
            getattr(config, "target_layer_ids", [])
        )
        self.context_proj = ReplicatedLinear(
            target_hidden_size * num_target_layers,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Draft layers reuse the target's decoder-layer stack, numbered past the
        # target layers so the KV-cache layer names do not collide.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        # Hand the DSV4 components a text-only vllm_config derived from the
        # draft config (the ambient vllm_config may point at the target).
        draft_vllm_config = vllm_config.with_hf_config(config)
        self.layers = nn.ModuleList(
            [
                ShensiDecoderLayer(
                    draft_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{start_layer_id + i}"),
                    block_layer_types=self.block_layer_types,
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=aux_stream_list,
                    config=config,
                    layer_type_index=i,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final norm + hc head, and the Markov + confidence heads.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = ShensiHyperHead(config, prefix=maybe_prefix(prefix, "hc_head"))
        self.output_attn_res = ShensiAttentionResidual(
            config, prefix=maybe_prefix(prefix, "output_attn_res")
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_rank = (
            getattr(config, "dspark_markov_rank", None)
            or getattr(config, "markov_rank", None)
            or 64
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", False):
            with_markov = getattr(config, "confidence_head_with_markov", True)
            input_dim = config.hidden_size
            if with_markov:
                input_dim += markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=with_markov,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """context_x = context_norm(context_proj(concat of target hidden states))."""
        return self.context_norm(self.context_proj(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        # Expand to hc_mult streams for the hyper-connections ([T, H] -> [T, hc, H]).
        hidden_states = (
            inputs_embeds.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        )
        block_residual = hidden_states.new_zeros(
            hidden_states.size(0),
            hidden_states.size(1),
            self.num_attn_res_blocks,
            hidden_states.size(-1),
        )
        prefix_sum = hidden_states
        hidden_states = None
        residual = block_residual

        for layer in self.layers:
            hidden_states, prefix_sum, residual = layer(
                hidden_states,
                residual,
                prefix_sum,
                positions,
                input_ids,
            )
        hidden_states, _, _ = self.output_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            output_norm_weight=None,
            num_blocks=self.num_attn_res_blocks,
        )
        # hc_head reduces the hc copies, then the final norm.
        return self.norm(self.hc_head(hidden_states))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Reuse the target's fused / renamed / expert-sharded name resolution;
        # the AutoWeightsLoader has already stripped the "model." prefix.
        return _load_shensi_weights(self, weights)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Each layer derives its context KV from the same projected target hidden
        ``main_x`` via its own fused wkv + kv_norm + RoPE + quant, then writes
        it at the layer's context slots (mirrors the DeepSeek-V4 DSpark).
        """
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.self_attn
            qr_kv, _ = attn.fused_wqa_wkv(main_x)
            kv = qr_kv[..., attn.q_lora_rank :]
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)


class DSparkShensiForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    hf_to_vllm_mapper = WeightsMapper(
        # confidence_head is training-only. The frozen target embedding and LM
        # head are shared after this draft-specific checkpoint is loaded.
        orig_to_new_substr={
            "confidence_head": None,
            "embed_tokens": None,
            "lm_head": None,
        },
        orig_to_new_prefix={"": "model."},
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        # Relative prefix: the AutoWeightsLoader strips "model." before calling
        # the inner model's load_weights, and the draft layers' KV-cache names
        # must not collide with the target's ("model.layers.*").
        self.model = DSparkShensiModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix="",
        )

        # Assigned by load_dspark_model from the target. Keeping no placeholder
        # avoids a transient full-vocabulary allocation.
        self.lm_head: nn.Module | None = None
        draft_vocab_size = (
            getattr(self.config, "draft_vocab_size", None) or self.config.vocab_size
        )
        self.logits_processor = LogitsProcessor(
            draft_vocab_size, scale=getattr(self.config, "logit_scale", 1.0)
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
