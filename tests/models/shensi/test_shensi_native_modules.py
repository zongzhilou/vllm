# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn

from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.shensi.attn_res import ShensiAttentionResidual
from vllm.models.shensi.hash_mlp import ShensiHashMLP
from vllm.models.shensi.moe import ShensiTopKRouter
from vllm.models.shensi.nvidia.model import (
    ShensiSparseMoeBlock,
    _make_shensi_weights_mapper,
    _stacked_params_mapping,
)
from vllm.models.shensi.xhc import (
    ShensiHyperConnection,
    ShensiHyperHead,
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


def _make_hyper_connection(config, is_mlp: bool) -> ShensiHyperConnection:
    return ShensiHyperConnection(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        active_streams=config.hc_active_streams,
        fixed_streams=config.hc_fixed_streams,
        conv_kernels=config.hc_conv_kernels,
        rms_norm_eps=config.rms_norm_eps,
        is_mlp=is_mlp,
    )


@pytest.mark.parametrize("is_mlp", [False, True])
def test_hyper_connection_matches_reference(config, is_mlp: bool):
    torch.manual_seed(0)
    reference = REFERENCE.ShensiHyperConnection(config, is_mlp=is_mlp)
    _seed_parameters(reference)
    module = _make_hyper_connection(config, is_mlp)
    _copy_weights(module, reference)
    num_tokens = 5
    streams = torch.randn(num_tokens, config.hc_mult, config.hidden_size)
    sublayer_output = torch.randn(num_tokens, config.hidden_size)
    collapsed = module(streams)
    collapsed_ref = reference(streams.unsqueeze(0))[0]
    torch.testing.assert_close(collapsed, collapsed_ref, atol=ATOL, rtol=RTOL)
    written = module.write_back(streams, sublayer_output)
    written_ref = reference.write_back(
        streams.unsqueeze(0), sublayer_output.unsqueeze(0)
    )[0]
    torch.testing.assert_close(written, written_ref, atol=ATOL, rtol=RTOL)


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
    num_tokens = 4
    prefix = torch.randn(num_tokens, config.hidden_size)
    delta = torch.randn(num_tokens, config.hidden_size) if with_delta else None
    reserved_blocks = 2
    blocks = torch.randn(num_tokens, reserved_blocks, config.hidden_size)
    output_norm_weight = torch.randn(config.hidden_size)
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


def test_hyper_head_matches_reference(config):
    torch.manual_seed(0)
    reference = REFERENCE.ShensiHyperHead(config)
    _seed_parameters(reference)
    module = ShensiHyperHead(config.hidden_size, config.hc_mult, config.rms_norm_eps)
    _copy_weights(module, reference)
    streams = torch.randn(3, config.hc_mult, config.hidden_size)
    torch.testing.assert_close(
        module(streams), reference(streams.unsqueeze(0))[0], atol=ATOL, rtol=RTOL
    )


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
        mlp_bias=config.mlp_bias,
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


def _make_sparse_moe_block(config) -> ShensiSparseMoeBlock:
    return ShensiSparseMoeBlock(
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
        prefix=_MOE_PREFIX,
    )


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
    with set_current_vllm_config(default_vllm_config):
        block = _make_sparse_moe_block(config)
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
        default_vllm_config,
        num_tokens=hidden_states.numel() // config.hidden_size,
    ):
        out = block(hidden_states[0])
    torch.testing.assert_close(
        out,
        reference(hidden_states)[0],
        atol=2e-2,
        rtol=2e-2,
    )
    flat = hidden_states.reshape(-1, config.hidden_size)
    _, weights, indices = block.gate(flat)
    _, weights_ref, indices_ref = reference.gate(hidden_states)
    torch.testing.assert_close(indices, indices_ref)
    torch.testing.assert_close(
        weights.float(), weights_ref.float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused router kernel is CUDA-only",
)
def test_router_fused_kernel_matches_torch_reference(config):
    torch.manual_seed(0)
    module = ShensiTopKRouter(
        config.hidden_size,
        config.n_routed_experts,
        config.num_experts_per_tok,
        config.scoring_func,
        config.routed_scaling_factor,
    ).to("cuda")
    hidden_states = torch.randn(8, config.hidden_size, device="cuda") * 3.0
    logits = torch.nn.functional.linear(hidden_states, module.weight)
    _, weights, indices = module(hidden_states)
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    indices_ref = torch.topk(
        scores, config.num_experts_per_tok, dim=-1, sorted=False
    ).indices
    weights_ref = scores.gather(1, indices_ref)
    weights_ref = (
        weights_ref / (weights_ref.sum(dim=-1, keepdim=True) + 1e-20)
    ) * config.routed_scaling_factor
    torch.testing.assert_close(indices, indices_ref)
    torch.testing.assert_close(weights, weights_ref, atol=1e-6, rtol=1e-6)


def test_mega_hc_write_back_matches_reference() -> None:
    from vllm.models.shensi.nvidia.ops.mega_hc import (
        can_use,
        mega_hc_write_back,
    )

    torch.manual_seed(0)
    num_tokens, hc_mult, hidden_size, active, kr = 7, 4, 128, 2, 2
    streams = torch.randn(
        num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    active_idx = torch.cat(
        [
            torch.zeros(num_tokens, 1, dtype=torch.long, device="cuda"),
            torch.randint(1, hc_mult, (num_tokens, 1), device="cuda"),
        ],
        dim=1,
    )
    p = torch.rand(num_tokens, active, device="cuda", dtype=torch.float32)
    active_flat = torch.randn(num_tokens, hidden_size, device="cuda") * 0.1
    post_weight = torch.randn(active * kr, hidden_size, device="cuda") * 0.1
    post_scale = torch.full((1,), 0.01, device="cuda")
    post_base = torch.zeros(active * kr, device="cuda")
    out_aug = torch.randn(
        num_tokens, kr, hidden_size, device="cuda", dtype=torch.float32
    )
    if not can_use(
        streams,
        active_idx,
        p,
        active_flat,
        post_weight,
        post_scale,
        post_base,
        out_aug,
        kr,
    ):
        pytest.skip("mega xHC backend unavailable on this platform")
    got = mega_hc_write_back(
        streams,
        active_idx,
        p,
        active_flat,
        post_weight,
        post_scale,
        post_base,
        out_aug,
        kr,
    )
    post = 2 * torch.sigmoid(
        torch.nn.functional.linear(active_flat, post_weight) * post_scale + post_base
    ).view(num_tokens, active, kr)
    delta = (torch.einsum("tkr,trh->tkh", post, out_aug) * p.unsqueeze(-1)).to(
        streams.dtype
    )
    ref = streams.scatter(
        1, active_idx.unsqueeze(-1).expand(-1, -1, hidden_size), delta
    )
    assert torch.equal(got, ref), "mega xHC write-back diverged from the torch path"
