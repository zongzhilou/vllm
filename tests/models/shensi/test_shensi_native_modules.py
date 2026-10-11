# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.shensi.attn_res import ShensiAttentionResidual
from vllm.models.shensi.hash_mlp import ShensiHashMLP
from vllm.models.shensi.moe import ShensiSparseMoeBlock
from vllm.models.shensi.nvidia.model import (
    _make_shensi_weights_mapper,
    _stacked_params_mapping,
)
from vllm.platforms import current_platform

from . import TINY_SHENSI_KWARGS

REFERENCE = pytest.importorskip("transformers.models.shensi.modeling_shensi")
ATOL = 1e-6
RTOL = 1e-5
MOE_ATOL = 1e-5
MOE_RTOL = 1e-4
_MOE_PREFIX = "model.layers.1.mlp"


@pytest.fixture
def config():
    return REFERENCE.ShensiConfig(**TINY_SHENSI_KWARGS)


def _seed_parameters(module: torch.nn.Module, std: float = 0.02) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, std)


def _copy_weights(target: torch.nn.Module, source: torch.nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name in list(target_state):
        if name.endswith("linear.weight") and name not in source_state:
            legacy = name[: -len("linear.weight")] + "weight"
            if legacy in source_state:
                source_state[name] = source_state.pop(legacy)
    if "gate_up_proj.weight" in target_state and "gate_proj.weight" in source_state:
        source_state["gate_up_proj.weight"] = torch.cat(
            [
                source_state.pop("gate_proj.weight"),
                source_state.pop("up_proj.weight"),
            ],
            dim=0,
        )
    for legacy, current in (
        ("pre_fn", "pre_proj.weight"),
        ("route_fn", "route_proj.weight"),
        ("post_fn", "post_proj.weight"),
        ("hc_fn", "hc_proj.weight"),
    ):
        for name in list(target_state):
            if not name.endswith(current):
                continue
            src = name[: -len(current)] + legacy
            if src in source_state:
                source_state[name] = source_state.pop(src)
    for name in list(target_state):
        if not name.endswith(".weight") or name in source_state:
            continue
        legacy = name[: -len(".weight")]
        if legacy in source_state:
            source_state[name] = source_state.pop(legacy)
    assert set(source_state) == set(target_state), sorted(
        set(source_state) ^ set(target_state)
    )
    target.load_state_dict(source_state)


@pytest.fixture(autouse=True)
def _vllm_layer_context(default_vllm_config, dist_init):
    with set_current_vllm_config(default_vllm_config):
        yield default_vllm_config


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the attention-residual kernels are Triton kernels",
)
@pytest.mark.parametrize("num_blocks", [0, 1, 2])
@pytest.mark.parametrize("with_delta", [True, False])
def test_attention_residual_matches_reference(
    config, num_blocks: int, with_delta: bool
):
    torch.manual_seed(0)
    reference = REFERENCE.ShensiAttentionResidual(config)
    _seed_parameters(reference)
    module = ShensiAttentionResidual(
        config.hidden_size,
        config.rms_norm_eps,
        config.routed_expert_hidden_size,
        config.num_hidden_layers,
    )
    _copy_weights(module, reference)
    module = module.to("cuda")
    reference = reference.to("cuda")
    num_tokens = 4
    prefix = torch.randn(num_tokens, config.hidden_size, device="cuda")
    delta = (
        torch.randn(num_tokens, config.hidden_size, device="cuda")
        if with_delta
        else None
    )
    reserved_blocks = 2
    blocks = torch.randn(num_tokens, reserved_blocks, config.hidden_size, device="cuda")
    output_norm_weight = torch.randn(config.hidden_size, device="cuda")
    # The read path samples its value rows from the global RNG, so both sides need the
    # same RNG state for a tight comparison.
    torch.manual_seed(7)
    output, updated = module(prefix, delta, blocks, output_norm_weight, num_blocks)
    torch.manual_seed(7)
    output_ref, updated_ref = reference(
        prefix.unsqueeze(0),
        None if delta is None else delta.unsqueeze(0),
        blocks.unsqueeze(0),
        output_norm_weight,
        num_blocks,
    )
    torch.testing.assert_close(output, output_ref[0], atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(updated, updated_ref[0], atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the attention-residual kernels are Triton kernels",
)
def test_write_residual_block_matches_reference(config):
    from vllm.models.shensi.common.ops import write_residual_block

    torch.manual_seed(0)
    num_tokens, num_blocks = 4, 3
    prefix = torch.randn(
        num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn(
        num_tokens, num_blocks, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    original = residual.clone()
    block_idx = 1
    written = write_residual_block(prefix, residual, block_idx)
    # The op writes in place and hands the bank back.
    assert written is residual
    expected = torch.cat(
        [original[:, :block_idx], prefix.unsqueeze(1), original[:, block_idx + 1 :]],
        dim=1,
    )
    assert torch.equal(written, expected)


def test_rms_norms_match_reference():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    reference_weighted = REFERENCE.ShensiRMSNorm(16, eps=1e-6)
    reference_weighted.weight.data.copy_(torch.rand(16) + 0.5)
    weighted = RMSNorm(16, eps=1e-6, dtype=torch.float32)
    weighted.load_state_dict(reference_weighted.state_dict())
    torch.testing.assert_close(weighted(x), reference_weighted(x), atol=ATOL, rtol=RTOL)
    unweighted = RMSNorm(1, eps=1e-6, has_weight=False)
    reference_unweighted = REFERENCE.ShensiUnweightedRMSNorm(eps=1e-6)
    torch.testing.assert_close(
        unweighted(x), reference_unweighted(x), atol=ATOL, rtol=RTOL
    )


def test_rms_norm_keeps_the_fp32_weight_in_bf16():
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    reference = REFERENCE.ShensiRMSNorm(16, eps=1e-6)
    reference.weight.data.copy_(torch.rand(16) + 0.5)
    module = RMSNorm(16, eps=1e-6, dtype=torch.float32)
    module.load_state_dict(reference.state_dict())
    assert module.weight.dtype == torch.float32
    out = module(x)
    assert out.dtype == x.dtype, "the norm must return the activation dtype"
    torch.testing.assert_close(out, reference(x).to(x.dtype), atol=2e-2, rtol=2e-2)


def test_hash_mlp_matches_reference(config, default_vllm_config):
    torch.manual_seed(0)
    reference = REFERENCE.ShensiHashMLP(config)
    _seed_parameters(reference)
    module = ShensiHashMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.routed_expert_hidden_size,
        vocab_size=config.vocab_size,
        swiglu_limit=config.swiglu_limit,
    )
    _copy_weights(module, reference)
    hidden_states = torch.randn(4, config.hidden_size) * 20.0
    input_ids = torch.tensor([1, 5, 7, 3])
    output = module(hidden_states, input_ids)
    output_ref = reference(hidden_states.unsqueeze(0), input_ids.unsqueeze(0))
    torch.testing.assert_close(output, output_ref[0], atol=MOE_ATOL, rtol=MOE_RTOL)
    with torch.no_grad():
        module.deepemb.weight.zero_()
    torch.testing.assert_close(
        module(hidden_states, input_ids), torch.zeros_like(hidden_states)
    )


def _shensi_vllm_config():
    # The blocks under test read the vLLM-flavoured config (which, unlike the
    # HF reference config, carries the DeepSeek-V4 fallback fields).
    from vllm.transformers_utils.configs import ShensiConfig

    config = ShensiConfig(**TINY_SHENSI_KWARGS)
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))
    return vllm_config.with_hf_config(config, architectures=["ShensiForCausalLM"])


def _make_sparse_moe_block(vllm_config) -> ShensiSparseMoeBlock:
    return ShensiSparseMoeBlock(vllm_config, prefix=_MOE_PREFIX)


def _load_reference_moe_weights(block: ShensiSparseMoeBlock, reference: nn.Module):
    host = nn.Module()
    host.model = nn.Module()
    layer = nn.Module()
    layer.mlp = block
    host.model.layers = nn.ModuleList([nn.Identity(), layer])
    mapper = _make_shensi_weights_mapper()
    params = dict(host.named_parameters())
    loaded = set()
    for name, tensor in reference.state_dict().items():
        mapped = mapper.map_name(f"{_MOE_PREFIX}.{name}")
        for param_name, shard_name, shard_id in _stacked_params_mapping():
            if shard_name in mapped:
                mapped = mapped.replace(shard_name, param_name)
                tensor = (tensor, shard_id)
                break
        param = params.get(mapped)
        if param is None:
            continue
        if mapped.endswith(("w13_weight", "w2_weight")):
            param.data.copy_(tensor.to(param.dtype))
        elif isinstance(tensor, tuple):
            param.weight_loader(param, tensor[0], tensor[1])
        else:
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
        loaded.add(mapped)
    assert loaded == {f"{_MOE_PREFIX}.{name}" for name, _ in block.named_parameters()}


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused router kernel and the fused MoE op are CUDA-only",
)
def test_sparse_moe_block_matches_reference(config, default_vllm_config, dist_init):
    torch.manual_seed(0)
    reference = REFERENCE.ShensiSparseMoeBlock(config, layer_idx=1)
    _seed_parameters(reference)
    vllm_config = _shensi_vllm_config()
    with set_current_vllm_config(vllm_config):
        block = _make_sparse_moe_block(vllm_config)
    _load_reference_moe_weights(block, reference)
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    _processed = 0
    for _m in block.modules():
        _qm = getattr(_m, "quant_method", None)
        if _qm is not None and hasattr(_qm, "process_weights_after_loading"):
            _qm.process_weights_after_loading(_m)
            _processed += 1
    assert _processed, "未找到带 quant_method 的模块（MoE 内核不会被构建）"
    block = block.to("cuda").to(torch.bfloat16)
    reference = reference.to("cuda").to(torch.bfloat16)
    hidden_states = torch.randn(1, 4, config.hidden_size, device="cuda") * 5.0
    hidden_states = hidden_states.to(torch.bfloat16)
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=hidden_states.numel() // config.hidden_size,
    ):
        out = block(hidden_states[0])
    torch.testing.assert_close(
        out,
        reference(hidden_states)[0],
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused router kernel is CUDA-only",
)
def test_gate_routing_matches_torch_reference(config, default_vllm_config, dist_init):
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )

    torch.manual_seed(0)
    vllm_config = _shensi_vllm_config()
    with set_current_vllm_config(vllm_config):
        block = _make_sparse_moe_block(vllm_config).to("cuda")
    # Well-separated logits so the top-k selection has no near-ties (the fused
    # kernel and torch may legitimately break exact ties differently).
    with torch.no_grad():
        block.gate.weight.data.normal_()
    hidden_states = torch.randn(8, config.hidden_size, device="cuda") * 3.0
    logits, _ = block.gate(hidden_states)
    weights, indices = fused_topk_bias(
        hidden_states=hidden_states,
        gating_output=logits,
        scoring_func=config.scoring_func,
        e_score_correction_bias=None,
        topk=config.num_experts_per_tok,
        renormalize=config.norm_topk_prob,
        indices_type=torch.int32,
        routed_scaling_factor=config.routed_scaling_factor,
    )
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    indices_ref = torch.topk(
        scores, config.num_experts_per_tok, dim=-1, sorted=False
    ).indices
    weights_ref = scores.gather(1, indices_ref)
    weights_ref = (
        weights_ref / (weights_ref.sum(dim=-1, keepdim=True) + 1e-20)
    ) * config.routed_scaling_factor
    torch.testing.assert_close(indices.long(), indices_ref)
    torch.testing.assert_close(weights, weights_ref, atol=1e-4, rtol=1e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused MTP input-norm kernels are CUDA-only",
)
def test_mtp_fused_norms_match_masked_reference(config):
    """The MTP path reuses deepseek_v4's fused norm kernels. Verify the Shensi
    calling convention: a single hidden slot and the DeepSeek-standard
    position-0 mask on the embed norm."""
    from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
        _FUSED_MTP_INPUT_RMSNORM_KERNEL,
        _MTP_SHARED_HEAD_RMSNORM_KERNEL,
    )

    torch.manual_seed(0)
    num_tokens, hidden = 6, config.hidden_size
    eps = config.rms_norm_eps
    inputs_embeds = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    previous = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    enorm_weight = torch.rand(hidden, device="cuda", dtype=torch.float32) + 0.5
    hnorm_weight = torch.rand(hidden, device="cuda", dtype=torch.float32) + 0.5

    def ref_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        reciprocal_std = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
        return (xf * reciprocal_std * weight).to(x.dtype)

    enorm_out, hnorm_out = _FUSED_MTP_INPUT_RMSNORM_KERNEL(
        inputs_embeds,
        positions,
        previous.view(-1, 1, hidden),
        enorm_weight,
        hnorm_weight,
        eps,
        1,
    )
    masked = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    torch.testing.assert_close(
        enorm_out, ref_norm(masked, enorm_weight), atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(
        hnorm_out.view(-1, hidden),
        ref_norm(previous, hnorm_weight),
        atol=2e-3,
        rtol=1e-2,
    )

    normed = _MTP_SHARED_HEAD_RMSNORM_KERNEL(previous, hnorm_weight, eps)
    torch.testing.assert_close(
        normed, ref_norm(previous, hnorm_weight), atol=2e-3, rtol=1e-2
    )
