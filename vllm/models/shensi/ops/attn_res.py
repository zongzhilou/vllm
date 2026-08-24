# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_L": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_L": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_L": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_L": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_L": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_L": 8}, num_warps=8, num_stages=3),
    ],
    key=["num_blocks", "hidden_size"],
)
@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    proj_ptr,
    out_norm_ptr,
    output_ptr,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_block_r,
    stride_proj_m,
    stride_output_m,
    num_blocks,
    hidden_size,
    eps,
    output_norm_eps,
    HAS_DELTA,
    APPLY_OUTPUT_NORM,
    BLOCK_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < hidden_size

    prefix = tl.load(prefix_ptr + row * stride_prefix_m + d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row * stride_delta_m + d, mask=d_mask, other=0.0
        ).to(tl.float32)
    else:
        delta = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # proj layout: [decay(H), erase(H), write(H), bias(3H), k(H), q(H)]
    decay = tl.sigmoid(
        tl.load(proj_ptr + row * stride_proj_m + d, mask=d_mask, other=0.0).to(
            tl.float32
        )
    )
    erase = tl.sigmoid(
        tl.load(
            proj_ptr + row * stride_proj_m + hidden_size + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
    )
    write = tl.sigmoid(
        tl.load(
            proj_ptr + row * stride_proj_m + 2 * hidden_size + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
    )
    k_part = tl.load(
        proj_ptr + row * stride_proj_m + 6 * hidden_size + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    q_part = tl.load(
        proj_ptr + row * stride_proj_m + 7 * hidden_size + d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    forgotten = decay * prefix
    # L2-normalize like HF's F.normalize (divide by sqrt(sum(x^2))); the
    # mean-normalized form would scale khat (and r) by sqrt(D).
    khat = k_part * tl.rsqrt(tl.sum(k_part * k_part, axis=0) + eps)
    r = tl.sum(khat * erase * forgotten, axis=0)
    updated = forgotten - khat * r + write * delta

    if num_blocks == 0:
        routed = tl.zeros((BLOCK_D,), dtype=tl.float32)
    else:
        max_logit = tl.full((), float("-inf"), tl.float32)
        denominator = tl.zeros((), tl.float32)
        routed = tl.zeros((BLOCK_D,), tl.float32)
        num_sources = num_blocks + 1
        for source_tile in range(tl.cdiv(num_sources, BLOCK_L)):
            source_offsets = source_tile * BLOCK_L + tl.arange(0, BLOCK_L)
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            block_ptrs = (
                blocks_ptr
                + row * stride_block_m
                + source_offsets[:, None] * stride_block_r
                + d[None, :]
            )
            updated_ptrs = prefix_ptr + row * stride_prefix_m + d[None, :]
            value_ptrs = tl.where(is_prefix[:, None], updated_ptrs, block_ptrs)
            values = tl.load(
                value_ptrs,
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            values = tl.where(is_prefix[:, None], updated[None, :], values)
            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = tl.sum(values * q_part[None, :], axis=1) * reciprocal_std
            scores = tl.where(source_mask, logits, -float("inf"))

            new_max_logit = tl.maximum(max_logit, tl.max(scores, axis=0))
            old_scale = tl.exp(max_logit - new_max_logit)
            block_scales = tl.exp(scores - new_max_logit)
            denominator = denominator * old_scale + tl.sum(block_scales, axis=0)
            routed = routed * old_scale + tl.sum(block_scales[:, None] * values, axis=0)
            max_logit = new_max_logit
        routed /= denominator

    output = updated + routed
    if APPLY_OUTPUT_NORM:
        out_norm = tl.load(out_norm_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        output = (
            output
            * tl.rsqrt(tl.sum(output * output, axis=0) / hidden_size + output_norm_eps)
            * out_norm
        )

    tl.store(prefix_ptr + row * stride_prefix_m + d, updated, mask=d_mask)
    tl.store(output_ptr + row * stride_output_m + d, output, mask=d_mask)


def get_attn_res_triton_warmup_profiles(
    max_blocks: int,
) -> tuple[tuple[int, int], ...]:
    return tuple((num_blocks, -1) for num_blocks in range(1, max_blocks + 1))


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    proj: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix.shape
    if not current_platform.is_cuda():
        raise NotImplementedError
    output = prefix.new_empty(prefix.shape)
    # Triton requires real pointer arguments even when the corresponding flag
    # is false, so substitute dummy buffers for missing delta / norm weight.
    has_delta = delta is not None
    if delta is None:
        delta = prefix.new_zeros(prefix.shape)
    has_output_norm = output_norm_weight is not None
    if output_norm_weight is None:
        output_norm_weight = prefix.new_zeros(hidden_size)
    _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        proj,
        output_norm_weight,
        output,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        proj.stride(0),
        output.stride(0),
        num_blocks,
        hidden_size,
        eps,
        output_norm_eps,
        HAS_DELTA=has_delta,
        APPLY_OUTPUT_NORM=has_output_norm,
        BLOCK_D=triton.next_power_of_2(hidden_size),
    )
    return output


def shensi_triton_warmup(worker) -> None:
    config = worker.model_config.hf_text_config
    if not hasattr(config, "attn_res_block_size"):
        return
    if not current_platform.is_cuda():
        return

    block_size = int(config.attn_res_block_size)
    hidden_size = int(config.hidden_size)
    max_blocks = (int(config.num_hidden_layers) + block_size - 1) // block_size
    if max_blocks < 1:
        return

    dtype = worker.model_config.dtype
    device = torch.device("cuda")
    eps = float(config.rms_norm_eps)
    # attn_res operates on flat [T, D] tensors with an implicit single stream.
    prefix = torch.zeros((1, hidden_size), dtype=dtype, device=device)
    delta = torch.zeros_like(prefix)
    blocks = torch.zeros((1, max_blocks, hidden_size), dtype=dtype, device=device)
    proj = torch.zeros((1, 8 * hidden_size), dtype=dtype, device=device)
    output_norm_weight = torch.zeros(hidden_size, dtype=dtype, device=device)

    for num_blocks, _ in get_attn_res_triton_warmup_profiles(max_blocks):
        attn_res(
            prefix,
            delta,
            blocks,
            proj,
            output_norm_weight,
            num_blocks=num_blocks,
            eps=eps,
            output_norm_eps=eps,
        )

    print("DBG shensi warmup running", flush=True)
    # Warm the autotuned HC collapse kernel (its autotune must not run while a
    # CUDA graph is capturing).
    from vllm.models.shensi.ops.hc import hc_collapse

    hc_mult = int(config.hc_mult)
    from vllm.models.shensi.model import ShensiUnweightedRMSNorm

    streams = torch.randn(1, hc_mult, hidden_size, dtype=dtype, device=device)
    hc_collapse(
        streams,
        torch.randn(hc_mult, hc_mult * hidden_size, dtype=torch.float32, device=device),
        torch.randn(1, dtype=torch.float32, device=device),
        torch.randn(hc_mult, dtype=torch.float32, device=device),
        ShensiUnweightedRMSNorm(eps=float(config.rms_norm_eps)),
    )
