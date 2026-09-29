# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.mtp_validation import (
    is_mtp_completeness_check_enabled,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.models.deepseek_v4.nvidia.mtp import DeepSeekV4MTP

from ..xhc import ShensiHyperHead
from .model import (
    ShensiDecoderLayer,
    _make_shensi_weights_mapper,
    default_aux_streams,
    select_attn_cls,
)

logger = init_logger(__name__)
_STAGE_LEVEL_PREFIXES = (
    "enorm",
    "hnorm",
    "e_proj",
    "h_proj",
    "hc_head.",
    "shared_head.",
)


def _stacked_params_mapping() -> list[tuple[str, str, int]]:
    return [
        ("self_attn.fused_wqa_wkv", "self_attn.q_a_proj", 0),
        ("self_attn.fused_wqa_wkv", "self_attn.kv_proj", 1),
    ]


class ShensiMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        attn_cls: type[nn.Module],
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.enorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.hnorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.e_proj",
        )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.h_proj",
        )
        self.hc_head = ShensiHyperHead(
            config.hidden_size, config.hc_mult, config.rms_norm_eps
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        self.mtp_block = ShensiDecoderLayer(
            vllm_config,
            prefix=prefix,
            attn_cls=attn_cls,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        previous_hidden_states = previous_hidden_states.view(
            -1, self.hc_mult, self.config.hidden_size
        )
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.h_proj(previous_hidden_states) + self.e_proj(
            inputs_embeds
        ).unsqueeze(-2)
        residual = hidden_states.new_empty(
            hidden_states.shape[0], self.hc_mult, 1, hidden_states.shape[-1]
        )
        hidden_states, _, _ = self.mtp_block(
            None, hidden_states, residual, positions, input_ids
        )
        return hidden_states.flatten(start_dim=1)


class ShensiMultiTokenPredictor(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        attn_cls: type[nn.Module],
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers <= 0:
            raise ValueError(
                "Shensi MTP requires num_nextn_predict_layers > 0 in the "
                "checkpoint config; the Shensi checkpoints published so far "
                "set it to 0 and ship no mtp.* weights."
            )
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        aux_stream_list = default_aux_streams()
        self.layers = nn.ModuleDict(
            {
                str(idx): ShensiMultiTokenPredictorLayer(
                    vllm_config,
                    self.topk_indices_buffer,
                    f"{prefix}.layers.{idx}",
                    attn_cls=attn_cls,
                    aux_stream_list=aux_stream_list,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = mtp_layer.hc_head(hidden_states)
        hidden_states = mtp_layer.shared_head.norm(hidden_states)
        return self.logits_processor(mtp_layer.shared_head.head, hidden_states)


class ShensiMTP(DeepSeekV4MTP):
    hf_to_vllm_mapper = _make_shensi_weights_mapper()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = ShensiMultiTokenPredictor(
            vllm_config=vllm_config,
            attn_cls=select_attn_cls(vllm_config),
            prefix=maybe_prefix(prefix, "model"),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded_params: set[str] = set()
        skipped_ckpt_keys: list[str] = []
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        num_local_heads = self.config.num_attention_heads // tp_size
        head_start = num_local_heads * tp_rank
        for name, loaded_weight in weights:
            name = self._to_absolute_name(name)
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self._rewrite_spec_layer_name(spec_layer, name)
            mapped_name = self.hf_to_vllm_mapper.map_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            param = params.get(name)
            if param is None:
                skipped_ckpt_keys.append(name)
                continue
            if name.endswith("attn_sink"):
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
        loaded_layers: set[int] = set()
        for param_name in loaded_params:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, param_name)
            if spec_layer is not None:
                loaded_layers.add(spec_layer)
        for layer_idx in range(
            self.model.mtp_start_layer_idx,
            self.model.mtp_start_layer_idx + self.model.num_mtp_layers,
        ):
            if layer_idx not in loaded_layers and is_mtp_completeness_check_enabled():
                raise ValueError(
                    f"MTP speculative decoding layer {layer_idx} weights are "
                    "missing from the checkpoint. Use a checkpoint that "
                    "includes the MTP stages, or disable speculative decoding."
                )
        missing_params = sorted(p for p in params if p not in loaded_params)
        report_path = os.environ.get("VLLM_SHENSI_LOAD_REPORT", "")
        if report_path:
            report = {
                "mtp_start_layer_idx": self.model.mtp_start_layer_idx,
                "num_mtp_layers": self.model.num_mtp_layers,
                "num_params": len(params),
                "num_loaded": len(loaded_params),
                "loaded_params": sorted(loaded_params),
                "missing_params": missing_params,
                "skipped_ckpt_keys": sorted(set(skipped_ckpt_keys)),
            }
            try:
                out = report_path.replace(".json", "") + ".mtp.json"
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                with open(out, "w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=1, ensure_ascii=False)
            except OSError as exc:  # pragma: no cover - diagnostics only
                logger.warning("Shensi MTP: could not write load report: %r", exc)
        logger.info_once("Shensi MTP draft model loaded: %d params", len(loaded_params))
        logger.info(
            "Shensi MTP draft loading: %d/%d parameters from the checkpoint "
            "(missing=%d skipped=%d)%s",
            len(loaded_params),
            len(params),
            len(missing_params),
            len(set(skipped_ckpt_keys)),
            f"; missing: {missing_params[:8]}" if missing_params else "",
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        pass

    def _to_absolute_name(self, name: str) -> str:
        for part in name.split("."):
            try:
                stage = int(part)
            except ValueError:
                continue
            return name.replace(
                f"mtp.{stage}.",
                f"model.layers.{self.config.num_hidden_layers + stage}.",
                1,
            )
        return name

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        prefix = f"model.layers.{spec_layer}."
        if not name.startswith(prefix):
            return name
        remainder = name[len(prefix) :]
        if remainder.startswith("embed_tokens"):
            return name.replace(prefix, "model.", 1)
        if remainder.startswith(_STAGE_LEVEL_PREFIXES):
            return name
        return name.replace(prefix, f"{prefix}mtp_block.", 1)
