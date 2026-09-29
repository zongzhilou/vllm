# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def xhc_gate(
    flat: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
) -> torch.Tensor:
    return torch.sigmoid(F.linear(flat, weight) * scale + base)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def xhc_gate_collapse(
    flat: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    streams: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    pre = torch.sigmoid(F.linear(flat, weight) * scale + base)
    return (pre.unsqueeze(-1) * streams).sum(dim=1).to(out_dtype)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def xhc_post_gate(
    flat: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    active_streams: int,
    kr: int,
) -> torch.Tensor:
    logits = F.linear(flat, weight).view(flat.shape[0], active_streams, kr)
    return 2 * torch.sigmoid(logits * scale + base.view(active_streams, kr))


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def xhc_route_select(
    route_scores: torch.Tensor, hc: int, fixed_streams: int, routed_streams: int
) -> tuple[torch.Tensor, torch.Tensor]:
    fixed_mask = torch.arange(hc, device=route_scores.device) < fixed_streams
    masked = route_scores.masked_fill(fixed_mask.view(1, hc), float("-inf"))
    routed_idx = masked.topk(routed_streams, dim=-1).indices
    active_idx = torch.cat(
        [
            torch.arange(fixed_streams, device=route_scores.device)
            .view(1, -1)
            .expand(route_scores.shape[0], -1),
            routed_idx,
        ],
        dim=-1,
    )
    p = torch.cat(
        [
            torch.ones_like(routed_idx, dtype=route_scores.dtype),
            route_scores.gather(-1, routed_idx),
        ],
        dim=-1,
    )
    return active_idx, p


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def xhc_gram_schmidt(
    base: torch.Tensor, conv_outs: list[torch.Tensor], eps: float
) -> torch.Tensor:
    ortho: list[torch.Tensor] = []
    prevs: list[torch.Tensor] = [base]
    for conv_out in conv_outs:
        value = conv_out
        for prev in prevs:
            denom = (prev * prev).sum(dim=-1, keepdim=True).clamp_min(eps)
            value = value - ((prev * value).sum(dim=-1, keepdim=True) / denom) * prev
        ortho.append(value)
        prevs.append(value)
    return torch.stack([base, *ortho], dim=1)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def xhc_write_delta(
    post: torch.Tensor, out_aug: torch.Tensor, p: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    delta = torch.einsum("tkr,trh->tkh", post, out_aug) * p.unsqueeze(-1)
    return delta.to(out_dtype)


def xhc_write_back(
    streams: torch.Tensor,
    active_idx: torch.Tensor,
    p: torch.Tensor,
    active_flat: torch.Tensor,
    post_weight: torch.Tensor,
    post_scale: torch.Tensor,
    post_base: torch.Tensor,
    out_aug: torch.Tensor,
    kr: int,
) -> torch.Tensor:
    if current_platform.is_cuda():
        from ...nvidia.ops.mega_hc import can_use, mega_hc_write_back

        if can_use(
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
            return mega_hc_write_back(
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
    hidden_size = streams.shape[2]
    post = xhc_post_gate(
        active_flat, post_weight, post_scale, post_base, p.shape[-1], kr
    )
    updated = xhc_write_delta(post, out_aug, p, streams.dtype)
    return streams.scatter(
        1, active_idx.unsqueeze(-1).expand(-1, -1, hidden_size), updated
    )
