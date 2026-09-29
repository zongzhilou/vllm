# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform

_MAX_ACTIVE_STREAMS = 8
_MAX_KR = 4


@triton.jit
def _mega_hc_write_back_kernel(
    streams_ptr,
    out_ptr,
    idx_ptr,
    p_ptr,
    flat_ptr,
    post_w_ptr,
    post_scale_ptr,
    post_base_ptr,
    aug_ptr,
    H,
    K,
    stride_st,
    stride_sr,
    stride_sh,
    stride_ot,
    stride_or,
    stride_oh,
    stride_it,
    stride_ia,
    stride_pt,
    stride_pa,
    stride_ft,
    stride_fk,
    stride_pwt,
    stride_pwk,
    stride_aut,
    stride_aur,
    stride_auh,
    A: tl.constexpr,
    KR: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token = tl.program_id(0)
    stream = tl.program_id(1)
    slot = tl.full((), -1, tl.int32)
    for a in tl.static_range(A):
        idx = tl.load(idx_ptr + token * stride_it + a * stride_ia).to(tl.int32)
        slot = tl.where(idx == stream, a, slot)
    is_active = slot >= 0
    slot_safe = tl.where(is_active, slot, 0)
    post_scale = tl.load(post_scale_ptr)
    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < K
    flat = tl.load(
        flat_ptr + token * stride_ft + offs_k * stride_fk, mask=kmask, other=0.0
    )
    scale = tl.load(p_ptr + token * stride_pt + slot_safe * stride_pa)
    offs_h = tl.arange(0, BLOCK_H)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + offs_h
        hmask = h < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for r in tl.static_range(KR):
            row = slot_safe * KR + r
            w = tl.load(
                post_w_ptr + row * stride_pwt + offs_k * stride_pwk,
                mask=kmask,
                other=0.0,
            )
            base = tl.load(post_base_ptr + row)
            logit = tl.sum(flat * w, axis=0) * post_scale + base
            post = 2.0 * tl.sigmoid(logit)
            aug = tl.load(
                aug_ptr + token * stride_aut + r * stride_aur + h * stride_auh,
                mask=hmask,
                other=0.0,
            )
            acc += post * aug
        delta = acc * scale
        base_v = tl.load(
            streams_ptr + token * stride_st + stream * stride_sr + h * stride_sh,
            mask=hmask,
            other=0.0,
        )
        value = tl.where(is_active, delta, base_v.to(tl.float32))
        tl.store(
            out_ptr + token * stride_ot + stream * stride_or + h * stride_oh,
            value.to(out_ptr.dtype.element_ty),
            mask=hmask,
        )


def can_use(
    streams: torch.Tensor,
    active_idx: torch.Tensor,
    p: torch.Tensor,
    flat: torch.Tensor,
    post_weight: torch.Tensor,
    post_scale: torch.Tensor,
    post_base: torch.Tensor,
    out_aug: torch.Tensor,
    kr: int,
) -> bool:
    if not current_platform.is_cuda():
        return False
    if streams.dim() != 3 or out_aug.dim() != 3:
        return False
    num_tokens, hc_mult, hidden_size = streams.shape
    active = active_idx.shape[-1] if active_idx.dim() == 2 else 0
    hidden = flat.shape[-1] if flat.dim() == 2 else 0
    return (
        num_tokens > 0
        and active_idx.shape == (num_tokens, active)
        and p.shape == (num_tokens, active)
        and flat.shape == (num_tokens, hidden)
        and post_weight.shape == (active * kr, hidden)
        and post_scale.numel() == 1
        and post_base.numel() == active * kr
        and out_aug.shape == (num_tokens, kr, hidden_size)
        and 1 <= active <= _MAX_ACTIVE_STREAMS
        and 1 <= kr <= _MAX_KR
        and 1 <= hc_mult <= 64
        and 1 <= hidden <= 4096
        and out_aug.dtype == torch.float32
        and flat.dtype == torch.float32
        and post_weight.dtype == torch.float32
        and post_scale.dtype == torch.float32
        and post_base.dtype == torch.float32
        and p.dtype == torch.float32
        and streams.dtype in (torch.bfloat16, torch.float16)
        and active_idx.dtype in (torch.int32, torch.int64)
        and all(
            t.is_cuda
            for t in (
                streams,
                active_idx,
                p,
                flat,
                post_weight,
                post_scale,
                post_base,
                out_aug,
            )
        )
    )


def mega_hc_write_back(
    streams: torch.Tensor,
    active_idx: torch.Tensor,
    p: torch.Tensor,
    flat: torch.Tensor,
    post_weight: torch.Tensor,
    post_scale: torch.Tensor,
    post_base: torch.Tensor,
    out_aug: torch.Tensor,
    kr: int,
) -> torch.Tensor:
    streams = streams.contiguous()
    active_idx = active_idx.contiguous()
    p = p.contiguous()
    flat = flat.contiguous()
    post_weight = post_weight.contiguous()
    post_scale = post_scale.contiguous()
    post_base = post_base.contiguous()
    out_aug = out_aug.contiguous()
    num_tokens, hc_mult, hidden_size = streams.shape
    hidden = flat.shape[-1]
    out = torch.empty_like(streams)
    _mega_hc_write_back_kernel[(num_tokens, hc_mult)](
        streams,
        out,
        active_idx,
        p,
        flat,
        post_weight,
        post_scale,
        post_base,
        out_aug,
        hidden_size,
        hidden,
        streams.stride(0),
        streams.stride(1),
        streams.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        active_idx.stride(0),
        active_idx.stride(1),
        p.stride(0),
        p.stride(1),
        flat.stride(0),
        flat.stride(1),
        post_weight.stride(0),
        post_weight.stride(1),
        out_aug.stride(0),
        out_aug.stride(1),
        out_aug.stride(2),
        A=active_idx.shape[-1],
        KR=kr,
        BLOCK_H=min(triton.next_power_of_2(hidden_size), 1024),
        BLOCK_K=triton.next_power_of_2(hidden),
    )
    return out
