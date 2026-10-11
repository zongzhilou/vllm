# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from huggingface_hub.dataclasses import strict
from transformers import DeepseekV4Config

SHENSI_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)
SHENSI_MLP_LAYER_TYPES = ("hash_moe", "moe")


@strict
class ShensiConfig(DeepseekV4Config):
    """Configuration for the Shensi sparse-attention MoE family.

    Shensi reuses the DeepSeek-V4 config surface — field set, legacy-key
    folding, rope-parameter nesting and validation all come from
    :class:`~transformers.DeepseekV4Config` — and only overrides the defaults
    that differ plus the fields that are Shensi-specific:

    - ``routed_expert_hidden_size``: rank of the routed experts' low-rank
      bottleneck.
    - ``erc_loss_alpha`` / ``erc_loss_coef``: expert-router coupling (ERC)
      loss settings (arXiv 2512.23447).
    - ``attn_res_block_size``: attention-residual block length; every
      ``attn_res_block_size`` layers form a block whose history the later
      layers read.
    """

    model_type = "shensi"

    hidden_size: int = 1280
    moe_intermediate_size: int = 800
    num_hidden_layers: int = 19
    num_attention_heads: int = 16
    q_lora_rank: int = 320
    o_groups: int = 2
    o_lora_rank: int = 320
    n_shared_experts: int | None = 0
    routed_expert_hidden_size: int | None = 320

    # ERC
    erc_loss_coef: float = 1.0
    erc_loss_alpha: float = 0.5

    # AttnRes
    attn_res_block_size: int | None = 2

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # The parent nests ``rope_parameters`` under the ``main`` / ``compress``
        # labels. vLLM consumes a flat ``rope_parameters`` (the compress branch,
        # which is what the rope pipeline standardizes), so export that beside
        # the labeled variants. Read through ``getattr`` like the parent's own
        # ``__post_init__``: the field's ``TypedDict`` union cannot be inferred
        # across the ``follow_imports = "silent"`` boundary of the installed
        # transformers package.
        rope_parameters = getattr(self, "rope_parameters", None)
        assert isinstance(rope_parameters, dict), rope_parameters
        assert isinstance(rope_parameters.get("main"), dict)
        assert isinstance(rope_parameters.get("compress"), dict)
        self.rope_parameters_by_label = {
            "main": rope_parameters["main"],
            "compress": rope_parameters["compress"],
        }
        flat = dict(self.rope_parameters_by_label["compress"])
        flat.setdefault("rope_type", "default")
        self.rope_parameters = flat

    def validate_layer_type(self):
        """Same checks as DeepSeek-V4 with the Shensi error messages."""
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

    @property
    def compress_ratios(self) -> list[int]:
        assert self.layer_types is not None
        rates = self.compress_rates or {}
        return [int(rates.get(layer_type, 0)) for layer_type in self.layer_types]

    @property
    def num_hash_layers(self) -> int:
        assert self.mlp_layer_types is not None
        return sum(layer_type == "hash_moe" for layer_type in self.mlp_layer_types)

    @property
    def qk_nope_head_dim(self) -> int:
        return int(self.head_dim - self.qk_rope_head_dim)

    @property
    def attn_res_block_layer_types(self) -> list[str]:
        assert self.mlp_layer_types is not None
        assert self.attn_res_block_size is not None
        num_hash = self.mlp_layer_types.count("hash_moe")
        return [
            "block_write_layer"
            if i == 0
            or (i >= num_hash and (i - num_hash) % self.attn_res_block_size == 0)
            else "block_read_layer"
            for i in range(self.num_hidden_layers)
        ]


__all__ = ["ShensiConfig"]
