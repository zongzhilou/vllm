# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.nvidia.dspark import (
    DSparkDeepseekV4ForCausalLM,
    DSparkDeepseekV4Model,
    _insert_context_kv,
)

from ..xhc import ShensiHyperHead
from .model import (
    ShensiDecoderLayer,
    _make_shensi_weights_mapper,
    select_attn_cls,
)

logger = init_logger(__name__)
_REQUIRED_DSPARK_FIELDS = ("dspark_target_layer_ids", "dspark_markov_rank")


def _stacked_params_mapping() -> list[tuple[str, str, int]]:
    return [
        ("self_attn.fused_wqa_wkv", "self_attn.q_a_proj", 0),
        ("self_attn.fused_wqa_wkv", "self_attn.kv_proj", 1),
    ]


def _require_dspark_config(config) -> None:
    missing = [
        field
        for field in _REQUIRED_DSPARK_FIELDS
        if getattr(config, field, None) is None
    ]
    if missing:
        raise ValueError(
            "Shensi DSpark needs "
            f"{', '.join(missing)} in the model config, but the checkpoint "
            "does not declare it. Shensi checkpoints are published without "
            "the DSpark fields and the weights do not exist yet; enable DSpark "
            "only for a checkpoint that ships both."
        )


class ShensiDSparkModel(DSparkDeepseekV4Model):
    _layer_cls: ClassVar[type[nn.Module]] = ShensiDecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        attn_cls: type[nn.Module],
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        _require_dspark_config(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.use_sequence_parallel = False
        self.num_dspark_layers = getattr(config, "n_mtp_layers", None) or 3
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.main_proj = ReplicatedLinear(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
        )
        self.main_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        self.layers = nn.ModuleList(
            [
                self._layer_cls(
                    vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                    attn_cls=attn_cls,
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        self.context_wkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.head_dim] * self.num_dspark_layers,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "context_wkv_proj"),
            disable_tp=True,
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.hc_head = ShensiHyperHead(
            config.hidden_size, config.hc_mult, config.rms_norm_eps
        )
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = DSparkConfidenceHead(
                config.hidden_size + config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.main_norm(self.main_proj(aux_hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        all_kv = self.context_wkv_proj(main_x).view(
            main_x.shape[0], self.num_dspark_layers, self.config.head_dim
        )
        for i, (layer, kv) in enumerate(
            zip(self.layers, all_kv.unbind(1), strict=True)
        ):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            attn = layer.self_attn
            kv = attn.kv_norm(kv)
            if slot_mapping is None:
                continue
            _insert_context_kv(attn, kv, context_positions, slot_mapping)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = (
            inputs_embeds.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        )
        residual = hidden_states.new_empty(
            hidden_states.shape[0], self.hc_mult, 1, hidden_states.shape[-1]
        )
        for layer in self.layers:
            hidden_states, _, residual = layer(
                None, hidden_states, residual, positions, input_ids
            )
        return self.hc_head(hidden_states)


class ShensiDSparkForCausalLM(DSparkDeepseekV4ForCausalLM):
    _layer_cls: ClassVar[type[nn.Module]] = ShensiDecoderLayer
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    hf_to_vllm_mapper = _make_shensi_weights_mapper()
    model_cls = ShensiDSparkModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = self.model_cls(
            vllm_config=vllm_config,
            attn_cls=select_attn_cls(vllm_config),
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

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
        assert self.model.confidence_head is not None
        return torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_confidence_head = False
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        num_local_heads = self.config.num_attention_heads // tp_size
        head_start = num_local_heads * tp_rank
        for name, loaded_weight in _duplicate_context_kv_weights(
            weights, len(self.model.layers)
        ):
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            if "confidence_head." in mapped:
                loaded_confidence_head = True
            name = self.hf_to_vllm_mapper.map_name(mapped)
            param = params.get(name)
            if param is None:
                continue
            if name.startswith("model.context_wkv_proj."):
                param.weight_loader(param, loaded_weight, loaded_weight.shard_id)
            elif name.endswith("attn_sink"):
                narrow = loaded_weight[head_start : head_start + num_local_heads]
                param.data[: narrow.shape[0]].copy_(narrow.to(param.dtype))
            else:
                for param_name, shard_name, shard_id in _stacked_params_mapping():
                    if shard_name not in name:
                        continue
                    stacked_name = name.replace(shard_name, param_name)
                    stacked_param = params.get(stacked_name)
                    if stacked_param is not None:
                        stacked_param.weight_loader(
                            stacked_param, loaded_weight, shard_id
                        )
                        name = stacked_name
                    break
                else:
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        if self.model.confidence_head is not None and not loaded_confidence_head:
            self.model.confidence_head = None
        self.process_weights_after_loading()
        logger.info_once(
            "Shensi DSpark draft model loaded: %d params", len(loaded_params)
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        pass

    def _remap_dspark_name(self, name: str) -> str | None:
        if name.startswith("context_wkv_proj."):
            return f"model.{name}"
        if not name.startswith("mtp."):
            return None
        remainder = name.split(".", 2)[2]
        if remainder.startswith("confidence_head.") and (
            self.model.confidence_head is None
        ):
            return None
        head_prefixes = (
            "norm.",
            "hc_head.",
            "markov_head.",
            "confidence_head.",
        )
        if remainder.startswith(("main_proj.", "main_norm.", *head_prefixes)):
            return f"model.{remainder}"
        return f"model.layers.{name.split('.', 2)[1]}.{remainder}"


def _duplicate_context_kv_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_layers: int
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, weight in weights:
        yield name, weight
        parts = name.split(".")
        if len(parts) < 5 or parts[0] != "mtp" or parts[2] != "self_attn":
            continue
        if parts[3] != "kv_proj":
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        if layer_idx >= num_layers:
            continue
        stacked_weight = weight.detach()
        stacked_weight.shard_id = layer_idx
        yield f"context_wkv_proj.{'.'.join(parts[4:])}", stacked_weight
