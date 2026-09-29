# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest

from vllm.models.shensi.hash_mlp import ShensiHashMLP
from vllm.models.shensi.nvidia.model import (
    ShensiModel,
    ShensiSparseMoeBlock,
    _make_shensi_weights_mapper,
    _stacked_params_mapping,
)
from vllm.transformers_utils.configs import ShensiConfig

from . import TINY_SHENSI_KWARGS

_RENAME_CASES = [
    (
        "model.layers.0.self_attn.compressor.indexer.q_b_proj.weight",
        "model.layers.0.self_attn.indexer.wq_b.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.compressor.indexer.scorer.weights_proj.weight",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.compressor.indexer.kv_proj.weight",
        "model.layers.0.self_attn.indexer.compressor.fused_wkv_wgate.weight",
        0,
    ),
    (
        "model.layers.0.self_attn.compressor.indexer.gate_proj.weight",
        "model.layers.0.self_attn.indexer.compressor.fused_wkv_wgate.weight",
        1,
    ),
    (
        "model.layers.0.self_attn.compressor.indexer.kv_norm.weight",
        "model.layers.0.self_attn.indexer.compressor.norm.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.compressor.indexer.position_bias",
        "model.layers.0.self_attn.indexer.compressor.ape",
        None,
    ),
    (
        "model.layers.0.self_attn.compressor.kv_proj.weight",
        "model.layers.0.self_attn.compressor.fused_wkv_wgate.weight",
        0,
    ),
    (
        "model.layers.0.self_attn.compressor.gate_proj.weight",
        "model.layers.0.self_attn.compressor.fused_wkv_wgate.weight",
        1,
    ),
    (
        "model.layers.0.self_attn.compressor.kv_norm.weight",
        "model.layers.0.self_attn.compressor.norm.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.compressor.position_bias",
        "model.layers.0.self_attn.compressor.ape",
        None,
    ),
    (
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.fused_wqa_wkv.weight",
        0,
    ),
    (
        "model.layers.0.self_attn.kv_proj.weight",
        "model.layers.0.self_attn.fused_wqa_wkv.weight",
        1,
    ),
    (
        "model.layers.0.self_attn.q_a_norm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.wq_b.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.kv_norm.weight",
        "model.layers.0.self_attn.kv_norm.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.o_a_proj.weight",
        "model.layers.0.self_attn.wo_a.weight",
        None,
    ),
    (
        "model.layers.0.self_attn.o_b_proj.weight",
        "model.layers.0.self_attn.wo_b.weight",
        None,
    ),
    ("model.layers.0.self_attn.sinks", "model.layers.0.self_attn.attn_sink", "sink"),
    ("model.layers.0.attn_hc.pre_fn", "model.layers.0.attn_hc.pre_proj.weight", None),
    (
        "model.layers.0.ffn_hc.temporal_convs.0.weight",
        "model.layers.0.ffn_hc.temporal_convs.0.weight",
        None,
    ),
    (
        "model.layers.0.self_attention_attn_res.gate_proj.weight",
        "model.layers.0.self_attention_attn_res.gate_proj.weight",
        None,
    ),
    ("model.hc_head.hc_fn", "model.hc_head.hc_proj.weight", None),
    ("lm_head.weight", "lm_head.weight", None),
]
_WEIGHTS_MAPPER = _make_shensi_weights_mapper()


def _apply_stacked(name: str) -> tuple[str, object]:
    if name.endswith("attn_sink"):
        return name, "sink"
    for param_name, shard_name, shard_id in _stacked_params_mapping():
        if shard_name in name:
            return name.replace(shard_name, param_name), shard_id
    return name, None


@pytest.mark.parametrize("checkpoint_key,expected,shard", _RENAME_CASES)
def test_weight_name_mapping(checkpoint_key: str, expected: str, shard):
    mapped, shard_id = _apply_stacked(_WEIGHTS_MAPPER.map_name(checkpoint_key))
    assert (mapped, shard_id) == (expected, shard)


def test_every_deepseek_v4_only_weight_is_renamed():
    trunk_keys = [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_proj.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.o_a_proj.weight",
        "model.layers.0.self_attn.o_b_proj.weight",
        "model.layers.0.self_attn.sinks",
        "model.layers.0.self_attn.compressor.kv_proj.weight",
        "model.layers.0.self_attn.compressor.gate_proj.weight",
        "model.layers.0.self_attn.compressor.position_bias",
        "model.layers.0.self_attn.compressor.kv_norm.weight",
        "model.layers.0.self_attn.compressor.indexer.kv_proj.weight",
        "model.layers.0.self_attn.compressor.indexer.gate_proj.weight",
        "model.layers.0.self_attn.compressor.indexer.q_b_proj.weight",
        "model.layers.0.self_attn.compressor.indexer.scorer.weights_proj.weight",
        "model.layers.0.self_attn.compressor.indexer.position_bias",
        "model.layers.0.self_attn.compressor.indexer.kv_norm.weight",
    ]
    for key in trunk_keys:
        mapped, _ = _apply_stacked(_WEIGHTS_MAPPER.map_name(key))
        assert mapped != key, key


def test_moe_groups_share_gate_and_experts(default_vllm_config, dist_init):
    config = ShensiConfig(**TINY_SHENSI_KWARGS)
    layers = []
    for layer_idx in range(config.num_hidden_layers):
        if config.mlp_layer_types[layer_idx] == "hash_moe":
            mlp = ShensiHashMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.routed_expert_hidden_size,
                vocab_size=config.vocab_size,
                swiglu_limit=config.swiglu_limit,
            )
        else:
            mlp = ShensiSparseMoeBlock(
                hidden_size=config.hidden_size,
                routed_hidden_size=config.routed_expert_hidden_size,
                moe_intermediate_size=config.moe_intermediate_size,
                num_experts=config.n_routed_experts,
                top_k=config.num_experts_per_tok,
                scoring_func=config.scoring_func,
                routed_scaling_factor=config.routed_scaling_factor,
                rms_norm_eps=config.rms_norm_eps,
                swiglu_limit=config.swiglu_limit,
                prefix=f"layers.{layer_idx}.mlp",
            )
        layers.append(SimpleNamespace(mlp=mlp))
    stub_model = SimpleNamespace(config=config, layers=layers)
    ShensiModel.tie_moe_groups(stub_model)
    assert stub_model.moe_group_of_layer == {1: 1, 2: 1, 3: 3}
    assert stub_model.moe_groups == {1: 1, 3: 3}
    assert layers[2].mlp.gate is layers[1].mlp.gate
    assert layers[2].mlp.experts is layers[1].mlp.experts
    names = {name for name, _ in layers[1].mlp.named_parameters()}
    assert "gate.linear.weight" in names
    assert "experts.routed_experts.w13_weight" in names
    assert "experts.routed_experts.w2_weight" in names


@pytest.mark.skipif(
    not os.environ.get("VLLM_SHENSI_CKPT"),
    reason="set VLLM_SHENSI_CKPT to a Shensi checkpoint directory to run this",
)
def test_weight_loading_closure():
    from vllm import LLM, SamplingParams

    llm = LLM(
        os.environ["VLLM_SHENSI_CKPT"],
        enforce_eager=True,
        max_model_len=64,
        gpu_memory_utilization=0.6,
    )
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=4))
    assert outputs
