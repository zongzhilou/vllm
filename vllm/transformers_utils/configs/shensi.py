# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters

SHENSI_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)
SHENSI_MLP_LAYER_TYPES = ("hash_moe", "moe")
_COMPRESS_RATIO_TO_LAYER_TYPE = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


@strict
class ShensiConfig(PreTrainedConfig):
    model_type = "shensi"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }
    n_shared_experts = 0
    hc_eps = 1.0e-6
    hc_sinkhorn_iters = 20
    vocab_size: int = 129280
    hidden_size: int = 2560
    moe_intermediate_size: int = 1280
    num_hidden_layers: int = 35
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 640
    default_partial_rotary_factor = 64 / 512
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float | int = 10000.0
    layer_types: list[str] | None = None
    compress_rates: dict | None = None
    default_compress_rates = {
        "compressed_sparse_attention": 4,
        "heavily_compressed_attention": 128,
    }
    compress_rope_theta: float | int = 160000.0
    hc_mult: int = 16
    mlp_layer_types: list[str] | None = None
    default_num_hash_layers = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 4
    o_lora_rank: int = 640
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    routed_expert_hidden_size: int | None = 640
    hc_active_streams: int | None = 4
    hc_fixed_streams: int | None = 2
    hc_conv_kernels: tuple[int, ...] | list[int] | None = (4, 8, 12)
    attn_res_block_size: int | None = 4
    attn_res_read_heads: int = 8
    _rope_type_labels = ("main", "compress")

    def validate_layer_type(self):
        if self.num_hidden_layers is None:
            return
        for name, types, allowed in (
            ("layer_types", self.layer_types, SHENSI_LAYER_TYPES),
            ("mlp_layer_types", self.mlp_layer_types, SHENSI_MLP_LAYER_TYPES),
        ):
            if types is None:
                continue
            if len(types) != self.num_hidden_layers:
                raise ValueError(
                    f"`num_hidden_layers` ({self.num_hidden_layers}) must equal "
                    f"`len({name})` ({len(types)})."
                )
            bad = [t for t in types if t not in allowed]
            if bad:
                raise ValueError(
                    f"`{name}` entries must be one of {allowed} for Shensi; got {bad}."
                )

    def __post_init__(self, **kwargs):
        legacy_compress_ratios = kwargs.pop("compress_ratios", None)
        legacy_compress_rate_csa = kwargs.pop("compress_rate_csa", None)
        legacy_compress_rate_hca = kwargs.pop("compress_rate_hca", None)
        legacy_num_hash_layers = kwargs.pop("num_hash_layers", None)
        legacy_qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)
        super().__post_init__(**kwargs)
        num_layers = self.num_hidden_layers
        if self.compress_rates is None:
            self.compress_rates = dict(self.default_compress_rates)
        if legacy_compress_rate_csa is not None:
            self.compress_rates["compressed_sparse_attention"] = (
                legacy_compress_rate_csa
            )
        if legacy_compress_rate_hca is not None:
            self.compress_rates["heavily_compressed_attention"] = (
                legacy_compress_rate_hca
            )
        if self.layer_types is None and legacy_compress_ratios is not None:
            self.layer_types = [
                _COMPRESS_RATIO_TO_LAYER_TYPE[ratio] for ratio in legacy_compress_ratios
            ]
        if self.layer_types is None:
            interleave = [
                "compressed_sparse_attention"
                if i % 2
                else "heavily_compressed_attention"
                for i in range(max(num_layers - 2, 0))
            ]
            self.layer_types = ["heavily_compressed_attention"] * min(
                num_layers, 2
            ) + interleave
        self.layer_types = list(self.layer_types[:num_layers])
        if self.mlp_layer_types is None:
            num_hash = (
                legacy_num_hash_layers
                if legacy_num_hash_layers is not None
                else self.default_num_hash_layers
            )
            self.mlp_layer_types = ["hash_moe"] * min(num_layers, num_hash) + [
                "moe"
            ] * max(0, num_layers - num_hash)
        self.mlp_layer_types = list(self.mlp_layer_types[:num_layers])
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = (
                legacy_qk_rope_head_dim / self.head_dim
                if legacy_qk_rope_head_dim is not None
                else self.default_partial_rotary_factor
            )
        self.qk_rope_head_dim = int(self.head_dim * self.partial_rotary_factor)
        rope_parameters = self.rope_parameters or {}
        if isinstance(rope_parameters.get("main"), dict) and isinstance(
            rope_parameters.get("compress"), dict
        ):
            self.rope_parameters_by_label = {
                "main": rope_parameters["main"],
                "compress": rope_parameters["compress"],
            }
            flat = dict(self.rope_parameters_by_label["compress"])
            flat.setdefault("rope_type", "default")
            self.rope_parameters = flat
        else:
            yarn = {
                key: value
                for key, value in rope_parameters.items()
                if key not in ("main", "compress")
            }
            main = {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress = {
                **yarn,
                "rope_theta": self.compress_rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
            }
            compress.setdefault("rope_type", "default")
            if compress["rope_type"] == "yarn":
                compress.setdefault("attention_factor", 1.0)
            self.rope_parameters_by_label = {"main": main, "compress": compress}
            flat = dict(compress)
            flat["rope_type"] = "yarn" if compress["rope_type"] == "yarn" else "default"
            self.rope_parameters = flat

    @property
    def compress_ratios(self) -> list[int]:
        rates = self.compress_rates or {}
        return [int(rates.get(layer_type, 0)) for layer_type in self.layer_types]

    @property
    def num_hash_layers(self) -> int:
        return sum(1 for layer_type in self.mlp_layer_types if layer_type == "hash_moe")

    @property
    def qk_nope_head_dim(self) -> int:
        return int(self.head_dim - self.qk_rope_head_dim)

    @property
    def attn_res_block_layer_types(self) -> list[str]:
        num_hash = self.mlp_layer_types.count("hash_moe")
        return [
            "block_write_layer"
            if i == 0
            or (i >= num_hash and (i - num_hash) % self.attn_res_block_size == 0)
            else "block_read_layer"
            for i in range(self.num_hidden_layers)
        ]


__all__ = ["ShensiConfig"]
