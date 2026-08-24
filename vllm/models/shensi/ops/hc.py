# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 4096}, num_warps=8, num_stages=3),
    ],
    key=["hidden_size", "hc_mult"],
)
@triton.jit
def _hc_collapse_fused_kernel(
    hidden_ptr,
    pre_logits_ptr,
    pre_base_ptr,
    output_ptr,
    pre_scale_ptr,
    eps,
    stride_hidden_m,
    stride_hidden_h,
    stride_hidden_d,
    stride_logits_m,
    stride_logits_h,
    hc_mult,
    hidden_size,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused HC collapse: sqrsum + RMSNorm + sigmoid + weighted sum.

    cuBLAS matvec is done OUTSIDE on raw (unnormalized) hidden_streams.
    RMSNorm is a scalar multiplier per token, so it can be applied AFTER
    the matvec: sigmoid(raw_matvec * rms_inv * scale + base).

    One program per token. Two passes:
      Pass 1: sqrsum over hc streams -> rms_inv
      Pass 2: sigmoid(pre_logits * rms_inv * scale + base) * hidden summed
    """
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < hidden_size

    res_base = token * stride_hidden_m

    # Pass 1: sqrsum for RMSNorm
    sq = tl.zeros((), dtype=tl.float32)
    for h in range(BLOCK_H):
        vals = tl.load(
            hidden_ptr + res_base + h * stride_hidden_h + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        sq += tl.sum(vals * vals)
    rms_inv = tl.rsqrt(sq / (hc_mult * hidden_size) + eps)

    # Pass 2: sigmoid(raw_matvec * rms_inv * scale + base) + weighted sum
    out = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for h in range(BLOCK_H):
        raw_logit = tl.load(
            pre_logits_ptr + token * stride_logits_m + h * stride_logits_h
        ).to(tl.float32)
        pre_base = tl.load(pre_base_ptr + h).to(tl.float32)
        pre_scale = tl.load(pre_scale_ptr)
        pre_h = tl.sigmoid(raw_logit * rms_inv * pre_scale + pre_base)
        vals = tl.load(
            hidden_ptr + res_base + h * stride_hidden_h + d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        out += pre_h * vals

    tl.store(
        output_ptr + token * hidden_size + d,
        out.to(hidden_ptr.dtype.element_ty),
        mask=d_mask,
    )


def hc_collapse(
    hidden_streams: torch.Tensor,
    pre_fn: torch.Tensor,
    pre_scale: torch.Tensor,
    pre_base: torch.Tensor,
    norm_fn: "torch.nn.Module",
) -> torch.Tensor:
    """HC collapse: norm + matvec + sigmoid + weighted sum.

    CUDA path: cuBLAS matvec on raw data + fused Triton kernel (sqrsum + norm
    + sigmoid + weighted sum in one kernel, 2 launches instead of 3).
    Non-CUDA: eager fallback (3 ops).
    """
    T, hc, H = hidden_streams.shape
    if not current_platform.is_cuda():
        flat = norm_fn(hidden_streams.flatten(1).float())
        pre_logits = F.linear(flat, pre_fn.float())
        pre = torch.sigmoid(pre_logits * pre_scale + pre_base)
        return (pre.unsqueeze(-1) * hidden_streams).sum(dim=1).to(hidden_streams.dtype)

    # cuBLAS matvec on raw (unnormalized) hidden_streams
    hidden_flat = hidden_streams.reshape(T, hc * H)
    pre_logits = F.linear(hidden_flat.float(), pre_fn.float())  # [T, hc]

    output = hidden_streams.new_empty(T, H)
    _hc_collapse_fused_kernel[(T,)](
        hidden_streams,
        pre_logits,
        pre_base.float(),
        output,
        pre_scale.float(),
        norm_fn.eps,
        hidden_streams.stride(0),
        hidden_streams.stride(1),
        hidden_streams.stride(2),
        pre_logits.stride(0),
        pre_logits.stride(1),
        hc_mult=hc,
        hidden_size=H,
        BLOCK_H=triton.next_power_of_2(hc),
    )
    return output.to(hidden_streams.dtype)
