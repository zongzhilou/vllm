# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from huggingface_hub.errors import StrictDataclassClassValidationError

from vllm.transformers_utils.config import (
    _CONFIG_REGISTRY,
    is_rope_parameters_nested,
)
from vllm.transformers_utils.configs import ShensiConfig

from . import TINY_SHENSI_KWARGS


def _tiny_config() -> ShensiConfig:
    return ShensiConfig(**TINY_SHENSI_KWARGS)


def test_config_is_registered():
    assert _CONFIG_REGISTRY["shensi"] is ShensiConfig
    assert ShensiConfig.model_type == "shensi"


def test_derived_fields():
    config = _tiny_config()
    assert config.layer_types == TINY_SHENSI_KWARGS["layer_types"]
    assert config.compress_ratios == [4, 128, 4, 128]
    assert config.num_hash_layers == 1
    assert config.qk_rope_head_dim == 16
    assert config.qk_nope_head_dim == 48
    assert config.attn_res_block_layer_types == [
        "block_write_layer",
        "block_write_layer",
        "block_read_layer",
        "block_write_layer",
    ]


def test_deepseek_v4_only_fields_have_fallbacks():
    config = _tiny_config()
    assert config.n_shared_experts == 0
    assert config.hc_eps == 1.0e-6
    assert config.hc_sinkhorn_iters == 20


def test_rope_parameters_export_is_flat_with_labels_kept_side_by_side():
    config = _tiny_config()
    assert set(config.rope_parameters_by_label) == {"main", "compress"}
    assert config.rope_parameters_by_label["main"]["rope_theta"] == 10000.0
    assert config.rope_parameters_by_label["compress"]["rope_theta"] == 160000.0
    assert config.rope_parameters["rope_theta"] == 160000.0
    assert "rope_type" in config.rope_parameters
    assert not is_rope_parameters_nested(config.rope_parameters)


def test_vllm_rope_pipeline_accepts_the_export():
    config = _tiny_config()
    config.standardize_rope_params()
    config.validate_rope()
    rope_parameters = config.rope_parameters
    if not is_rope_parameters_nested(rope_parameters):
        rope_parameters = {"": rope_parameters}
    assert all(rp["rope_type"] == "default" for rp in rope_parameters.values())


def test_legacy_keys_are_folded_into_the_modern_fields():
    config = ShensiConfig(
        num_hidden_layers=4,
        hidden_size=64,
        compress_ratios=[0, 4, 128, 4],
        num_hash_layers=1,
        qk_rope_head_dim=16,
        head_dim=64,
    )
    assert config.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    assert config.compress_ratios[0] == 0
    assert config.mlp_layer_types == ["hash_moe", "moe", "moe", "moe"]
    assert config.partial_rotary_factor == pytest.approx(0.25)


def test_layer_type_validation():
    with pytest.raises(StrictDataclassClassValidationError, match="must be one of"):
        ShensiConfig(**{**TINY_SHENSI_KWARGS, "layer_types": ["full_attention"] * 4})
    with pytest.raises(StrictDataclassClassValidationError, match="must equal"):
        ShensiConfig(**{**TINY_SHENSI_KWARGS, "mlp_layer_types": ["moe"]})


def test_matches_transformers_config():
    transformers_shensi = pytest.importorskip(
        "transformers.models.shensi.configuration_shensi"
    )
    reference = transformers_shensi.ShensiConfig(**TINY_SHENSI_KWARGS)
    config = _tiny_config()
    for field in (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "q_lora_rank",
        "o_groups",
        "o_lora_rank",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "sliding_window",
        "hc_mult",
        "hc_active_streams",
        "hc_fixed_streams",
        "attn_res_block_size",
        "routed_expert_hidden_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
        "scoring_func",
        "routed_scaling_factor",
        "swiglu_limit",
        "rms_norm_eps",
        "max_position_embeddings",
        "rope_theta",
        "compress_rope_theta",
        "num_nextn_predict_layers",
        "vocab_size",
        "partial_rotary_factor",
        "qk_rope_head_dim",
        "erc_loss_alpha",
        "erc_loss_coef",
        "layer_types",
        "mlp_layer_types",
        "compress_rates",
    ):
        assert getattr(config, field) == getattr(reference, field), field
    assert config.attn_res_block_layer_types == reference.attn_res_block_layer_types
    assert config.rope_parameters_by_label == reference.rope_parameters


def test_shensi_family_knobs_match_reference():
    """VLLM 的配置类要覆盖 HF 侧的全部 shensi 家族 knob：@strict 会丢掉没声明的字段。"""
    import transformers.models.shensi.configuration_shensi as ref

    knobs = (
        "attn_res",
        "hc_",
        "_o_",
        "o_",
        "index",
        "compress",
        "scoring",
        "swiglu",
        "sliding",
        "mlp_layer_types",
        "layer_types",
        "q_lora",
    )
    hf = {k for k in vars(ref.ShensiConfig) if not k.startswith("_")}
    hf |= set(getattr(ref.ShensiConfig, "__annotations__", {}))
    ours = {k for k in vars(ShensiConfig) if not k.startswith("_")}
    ours |= set(getattr(ShensiConfig, "__annotations__", {}))
    missing = sorted(k for k in hf if any(t in k for t in knobs) and k not in ours)
    assert not missing, missing
