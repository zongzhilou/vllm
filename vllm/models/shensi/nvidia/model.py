# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    WeightsMapper,
    extract_layer_index,
    get_spec_layer_idx_from_weight_name,
    make_layers,
    maybe_prefix,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    DeepseekV4MoE,
    _select_dsv4_attn_cls,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from ..attn_res import ShensiAttentionResidual
from ..common.ops.attn_res import write_residual_block
from ..hash_mlp import ShensiHashMLP
from ..moe import ShensiSparseMoeBlockMixin, unregister_moe_runner
from ..xhc import ShensiHyperConnection, ShensiHyperHead

logger = init_logger(__name__)


def select_attn_cls(vllm_config: VllmConfig) -> type[nn.Module]:
    return _select_dsv4_attn_cls(vllm_config)


def default_aux_streams() -> "list | None":
    if current_platform.is_cpu() or current_platform.is_xpu():
        return None
    return [torch.cuda.Stream() for _ in range(3)]


def _make_shensi_weights_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_regex={
            re.compile(
                r"^(.+)\.self_attn\.compressor\.indexer\.q_b_proj\.weight$"
            ): r"\1.self_attn.indexer.wq_b.weight",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.indexer\.scorer\.weights_proj\.weight$"
            ): r"\1.self_attn.indexer.weights_proj.weight",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.indexer\.kv_norm\.weight$"
            ): r"\1.self_attn.indexer.compressor.norm.weight",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.indexer\.position_bias$"
            ): r"\1.self_attn.indexer.compressor.ape",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.indexer\.kv_proj\.weight$"
            ): r"\1.self_attn.indexer.kv_proj.weight",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.indexer\.gate_proj\.weight$"
            ): r"\1.self_attn.indexer.gate_proj.weight",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.kv_norm\.weight$"
            ): r"\1.self_attn.compressor.norm.weight",
            re.compile(
                r"^(.+)\.self_attn\.compressor\.position_bias$"
            ): r"\1.self_attn.compressor.ape",
            re.compile(
                r"^(.+)\.self_attn\.q_a_norm\.weight$"
            ): r"\1.self_attn.q_norm.weight",
            re.compile(
                r"^(.+)\.self_attn\.q_b_proj\.weight$"
            ): r"\1.self_attn.wq_b.weight",
            re.compile(
                r"^(.+)\.self_attn\.o_a_proj\.weight$"
            ): r"\1.self_attn.wo_a.weight",
            re.compile(
                r"^(.+)\.self_attn\.o_b_proj\.weight$"
            ): r"\1.self_attn.wo_b.weight",
            re.compile(r"^(.+)\.self_attn\.sinks$"): r"\1.self_attn.attn_sink",
            re.compile(
                r"^(.+)\.(?P<hc>attn_hc|ffn_hc)\.pre_fn$"
            ): r"\1.\g<hc>.pre_proj.weight",
            re.compile(
                r"^(.+)\.(?P<hc>attn_hc|ffn_hc)\.route_fn$"
            ): r"\1.\g<hc>.route_proj.weight",
            re.compile(
                r"^(.+)\.(?P<hc>attn_hc|ffn_hc)\.post_fn$"
            ): r"\1.\g<hc>.post_proj.weight",
            re.compile(r"^(.+)\.hc_fn$"): r"\1.hc_proj.weight",
            re.compile(
                r"^(.+)\.(?P<slot>self_attention_attn_res|mlp_attn_res|output_attn_res)\.q_proj$"
            ): r"\1.\g<slot>.q_proj.weight",
            re.compile(
                r"^(.+)\.(?P<slot>self_attention_attn_res|mlp_attn_res|output_attn_res)\.k_proj$"
            ): r"\1.\g<slot>.k_proj.weight",
            re.compile(r"^(.+)\.mlp\.gate\.weight$"): r"\1.mlp.gate.linear.weight",
        },
        orig_to_new_suffix={
            ".mlp.experts.gate_up_proj": ".mlp.experts.routed_experts.w13_weight",
            ".mlp.experts.down_proj": ".mlp.experts.routed_experts.w2_weight",
        },
    )


def _stacked_params_mapping() -> list[tuple[str, str, int]]:
    return [
        ("attn.fused_wqa_wkv", "attn.q_a_proj", 0),
        ("attn.fused_wqa_wkv", "attn.kv_proj", 1),
        ("compressor.fused_wkv_wgate", "compressor.kv_proj", 0),
        ("compressor.fused_wkv_wgate", "compressor.gate_proj", 1),
        ("indexer.compressor.fused_wkv_wgate", "indexer.kv_proj", 0),
        ("indexer.compressor.fused_wkv_wgate", "indexer.gate_proj", 1),
        ("mlp.gate_up_proj", "mlp.gate_proj", 0),
        ("mlp.gate_up_proj", "mlp.up_proj", 1),
    ]


class ShensiSparseMoeBlock(ShensiSparseMoeBlockMixin, DeepseekV4MoE):  # type: ignore[misc]
    pass


class ShensiDecoderLayer(DeepseekV4DecoderLayer):
    moe_cls = ShensiSparseMoeBlock

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        attn_cls: type[nn.Module],
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.layer_type = self._resolve_layer_type(config, self.layer_idx)
        self.is_hash = self.layer_type == "hash_moe"
        block_types = config.attn_res_block_layer_types
        if self.layer_idx < config.num_hidden_layers:
            self.is_block_write_layer = (
                block_types[self.layer_idx] == "block_write_layer"
            )
            self.prev_valid_blocks = sum(
                1
                for entry in block_types[: self.layer_idx]
                if entry == "block_write_layer"
            )
        else:
            self.is_block_write_layer = True
            self.prev_valid_blocks = 0
        self.self_attn = attn_cls(
            vllm_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.attn_hc = ShensiHyperConnection(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            active_streams=config.hc_active_streams,
            fixed_streams=config.hc_fixed_streams,
            conv_kernels=config.hc_conv_kernels,
            rms_norm_eps=config.rms_norm_eps,
            is_mlp=False,
            prefix=f"{prefix}.attn_hc",
        )
        self.ffn_hc = ShensiHyperConnection(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            active_streams=config.hc_active_streams,
            fixed_streams=config.hc_fixed_streams,
            conv_kernels=config.hc_conv_kernels,
            rms_norm_eps=config.rms_norm_eps,
            is_mlp=True,
            prefix=f"{prefix}.ffn_hc",
        )
        self.self_attention_attn_res = ShensiAttentionResidual(
            config.hidden_size,
            config.rms_norm_eps,
            config.routed_expert_hidden_size,
            config.num_hidden_layers,
            prefix=f"{prefix}.self_attention_attn_res",
        )
        self.mlp_attn_res = ShensiAttentionResidual(
            config.hidden_size,
            config.rms_norm_eps,
            config.routed_expert_hidden_size,
            config.num_hidden_layers,
            prefix=f"{prefix}.mlp_attn_res",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=torch.float32
        )
        if self.is_hash:
            self.mlp = ShensiHashMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.routed_expert_hidden_size,
                vocab_size=config.vocab_size,
                swiglu_limit=config.swiglu_limit,
                mlp_bias=config.mlp_bias,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = self.moe_cls(
                hidden_size=config.hidden_size,
                routed_hidden_size=config.routed_expert_hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                scoring_func=config.scoring_func,
                routed_scaling_factor=config.routed_scaling_factor,
                rms_norm_eps=config.rms_norm_eps,
                swiglu_limit=config.swiglu_limit,
                mlp_bias=config.mlp_bias,
                prefix=f"{prefix}.mlp",
            )

    @staticmethod
    def _resolve_layer_type(config, layer_idx: int) -> str:
        if layer_idx < config.num_hidden_layers:
            return config.mlp_layer_types[layer_idx]
        return config.mlp_layer_types[-1]

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor | None,
        prefix_sum: torch.Tensor,
        residual: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = None if hidden_states is None else hidden_states - prefix_sum
        if self.is_block_write_layer:
            residual = write_residual_block(
                prefix_sum, residual, self.prev_valid_blocks
            )
        hidden_states, prefix_sum = self.self_attention_attn_res(
            prefix_sum,
            delta,
            residual,
            output_norm_weight=self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
        )
        if self.is_block_write_layer:
            prefix_sum = None
        collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(positions, collapsed)
        hidden_states = self.attn_hc.write_back(hidden_states, attn_output)
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        hidden_states, prefix_sum = self.mlp_attn_res(
            prefix_sum,
            prefix_sum,
            residual,
            output_norm_weight=self.post_attention_layernorm.weight,
            num_blocks=self.prev_valid_blocks + int(self.is_block_write_layer),
        )
        collapsed = self.ffn_hc(hidden_states)
        if self.is_hash:
            mlp_output = self.mlp(collapsed, input_ids)
        else:
            mlp_output = self.mlp(collapsed)
        hidden_states = self.ffn_hc.write_back(hidden_states, mlp_output)
        prefix_sum = prefix_sum + hidden_states
        return hidden_states, prefix_sum, residual


class ShensiModel(DeepseekV4Model):
    _select_attn_cls = staticmethod(select_attn_cls)
    _layer_cls = ShensiDecoderLayer

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hidden_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.num_attn_res_blocks = config.attn_res_block_layer_types.count(
            "block_write_layer"
        )
        if get_pp_group().world_size != 1:
            raise NotImplementedError(
                "Shensi 暂不支持流水并行：超连接流 ([T, hc_mult, H]) 与 AttnRes "
                "bank 需要跨 stage。"
            )
        if get_tensor_model_parallel_world_size() != 1:
            raise NotImplementedError(
                "Shensi 暂不支持张量并行：超连接、AttnRes 与低秩专家均为复制语义。"
            )
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        aux_stream_list = default_aux_streams()
        attn_cls = self._select_attn_cls(vllm_config)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: self._layer_cls(
                vllm_config,
                prefix=prefix,
                attn_cls=attn_cls,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=self.rms_norm_eps, dtype=torch.float32
        )
        self.hc_head = ShensiHyperHead(
            config.hidden_size, config.hc_mult, config.rms_norm_eps
        )
        self.output_attn_res = ShensiAttentionResidual(
            config.hidden_size,
            config.rms_norm_eps,
            config.routed_expert_hidden_size,
            config.num_hidden_layers,
        )
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hidden_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None
        self.tie_moe_groups()

    def tie_moe_groups(self) -> None:
        block_types = self.config.attn_res_block_layer_types
        write_layers = [
            idx for idx, entry in enumerate(block_types) if entry == "block_write_layer"
        ]
        self.moe_group_of_layer: dict[int, int] = {}
        self.moe_groups: dict[int, int] = {}
        owners: dict[int, ShensiSparseMoeBlockMixin] = {}
        for idx, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            if not isinstance(mlp, ShensiSparseMoeBlockMixin):
                continue
            block_id = max(w for w in write_layers if w <= idx)
            self.moe_group_of_layer[idx] = block_id
            if block_id not in owners:
                owners[block_id] = mlp
                self.moe_groups[block_id] = idx
                continue
            owner_mlp = owners[block_id]
            if owner_mlp is not mlp:
                unregister_moe_runner(mlp.experts)
                mlp.gate = owner_mlp.gate
                mlp.experts = owner_mlp.experts

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

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert intermediate_tensors is None, "Shensi requires PP=1"
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        streams = hidden_states.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        residual = streams.new_empty(
            streams.shape[0], self.hc_mult, self.num_attn_res_blocks, streams.shape[-1]
        )
        prefix_sum = streams
        hidden_states = None
        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, prefix_sum, residual = layer(
                hidden_states, prefix_sum, residual, positions, input_ids
            )
            if idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))
        hidden_states, _ = self.output_attn_res(
            prefix_sum,
            hidden_states,
            residual,
            output_norm_weight=None,
            num_blocks=self.num_attn_res_blocks,
        )
        if self._mtp_hidden_buffer is not None:
            num_tokens = prefix_sum.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(prefix_sum.flatten(start_dim=1))
        hidden_states = self.hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


def _skip_mtp_weights(
    weights: Iterable[tuple[str, torch.Tensor]], config
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        if name.startswith("mtp."):
            continue
        if get_spec_layer_idx_from_weight_name(config, name) is not None:
            continue
        yield name, tensor


class ShensiForCausalLM(DeepseekV4ForCausalLM):
    model_cls = ShensiModel
    hf_to_vllm_mapper = _make_shensi_weights_mapper()
    packed_modules_mapping = {
        "fused_wqa_wkv": ["q_a_proj", "kv_proj"],
        "fused_wkv_wgate": ["kv_proj", "gate_proj"],
    }
    lora_skip_prefixes = ["mtp."]

    def set_moe_parameters(self) -> None:
        config = self.config
        self.num_expert_groups = 1
        self.num_moe_layers = sum(1 for t in config.mlp_layer_types if t == "moe")
        self.num_logical_experts = config.n_routed_experts
        self.num_physical_experts = config.n_routed_experts
        self.num_local_physical_experts = config.n_routed_experts
        self.num_routed_experts = config.n_routed_experts
        self.num_shared_experts = 0
        self.num_redundant_experts = 0
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_layers: list[torch.nn.Module] = []
        self.moe_mlp_layers: list[torch.nn.Module] = []
        self.expert_weights: list[torch.Tensor] = []

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = _stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        num_local_heads = self.config.num_attention_heads // tp_size
        head_start = num_local_heads * tp_rank
        for name, loaded_weight in _skip_mtp_weights(weights, self.config):
            mapped_name = self.hf_to_vllm_mapper.map_name(name)
            if mapped_name is None:
                continue
            name = mapped_name
            if name.endswith("attn_sink"):
                param = params_dict[name]
                narrow = loaded_weight[head_start : head_start + num_local_heads]
                param.data[: narrow.shape[0]].copy_(narrow.to(param.dtype))
                loaded_params.add(name)
                continue
            if name.endswith(("w13_weight", "w2_weight")):
                param = params_dict[name]
                param.data.copy_(loaded_weight.to(param.dtype))
                loaded_params.add(name)
                continue
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        missing_params = sorted(p for p in params_dict if p not in loaded_params)
        logger.info_once(
            "Shensi: loaded %d parameters, %d missing",
            len(loaded_params),
            len(missing_params),
        )
        return loaded_params

    def process_weights_after_loading(self) -> None:
        pass
