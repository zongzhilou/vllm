# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def attn_res_update(
    state: torch.Tensor,
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    g_a_weight: torch.Tensor,
    g_a_bias: torch.Tensor,
    g_b_weight: torch.Tensor,
    g_b_bias: torch.Tensor,
    k_a_weight: torch.Tensor,
    k_b_weight: torch.Tensor,
    g_scale: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    gates = F.linear(F.linear(state, g_a_weight, g_a_bias), g_b_weight, g_b_bias)
    r_decay, r_erase, r_write = gates.reshape(*state.shape[:-1], 3, -1).unbind(-2)
    decay_scale, erase_scale, write_scale, _ = g_scale.unbind()
    decay = torch.exp(-F.softplus(r_decay) * decay_scale * t.exp())
    erase = F.softplus(r_erase) * erase_scale
    write = 1.0 + torch.tanh(r_write) * write_scale
    # 没有 delta（首层）时 k 投影的输入与加法项都退回 state / 0，与 HF、mcore 两侧一致
    khat = F.normalize(
        F.linear(F.linear(state if delta is None else delta, k_a_weight), k_b_weight),
        dim=-1,
    )
    m = decay * prefix + write * (0.0 if delta is None else delta)
    lam = erase.mean(dim=-1, keepdim=True).clamp(min=-0.5)
    return m - (lam / (1.0 + lam)) * khat * (khat * m).sum(dim=-1, keepdim=True)


@torch._dynamo.disable
def _per_head_whiten(
    flat_v: torch.Tensor, flat_q: torch.Tensor, read_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐 head（块对角）ZCA 白化；分解类算子不进 inductor 的融合区，保证与 HF、mcore 数值一致。"""
    dim = flat_v.shape[-1]
    head_dim = dim // read_heads
    V = flat_v.reshape(-1, read_heads, head_dim)
    Q = flat_q.reshape(-1, read_heads, head_dim)
    n = V.shape[0]
    cov = torch.einsum("nhd,nhe->hde", V, V) / n
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)
    ridge = max(head_dim, n) * torch.finfo(cov.dtype).eps * scale
    cov.diagonal(dim1=-2, dim2=-1).add_(ridge.unsqueeze(-1))
    evals, evecs = torch.linalg.eigh(cov)
    floor = (evals[..., -1:] * head_dim * torch.finfo(cov.dtype).eps).clamp_min(
        torch.finfo(cov.dtype).tiny
    )
    whiten = (
        evecs
        @ torch.diag_embed(torch.rsqrt(evals.clamp_min(floor)))
        @ evecs.transpose(-1, -2)
    )
    return (
        torch.einsum("nhd,hde->nhe", V, whiten),
        torch.einsum("nhd,hde->nhe", Q, whiten),
    )


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def attn_res_route(
    blocks: torch.Tensor,
    updated: torch.Tensor,
    query: torch.Tensor,
    eps: float,
    num_blocks: int,
    read_heads: int,
    read_scale: torch.Tensor,
) -> torch.Tensor:
    values = torch.cat(
        [blocks[..., :num_blocks, :].float(), updated.unsqueeze(-2)], dim=-2
    )
    with torch.no_grad():
        v, q = _per_head_whiten(
            values.reshape(-1, values.shape[-1]),
            query.reshape(-1, query.shape[-1]),
            read_heads,
        )
    v = v.view(*values.shape[:-1], read_heads, -1)
    q = q.view(*query.shape[:-1], read_heads, -1)
    logits = (v * q.unsqueeze(q.dim() - 2)).sum(dim=-1) * torch.rsqrt(
        v.square().mean(dim=-1) + eps
    )
    s = torch.logsumexp(logits, dim=-2, keepdim=True)
    scores = torch.exp(logits - F.softplus(s))
    return (scores.unsqueeze(-1) * values.view_as(v)).sum(dim=-3).reshape(
        *values.shape[:-2], values.shape[-1]
    ) * read_scale


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def scale_by_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    reciprocal_std = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return x * reciprocal_std * weight


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def write_residual_block(
    prefix_sum: torch.Tensor, residual: torch.Tensor, block_idx: int
) -> torch.Tensor:
    written = prefix_sum.to(residual.dtype).unsqueeze(-2)
    return torch.cat(
        [residual[..., :block_idx, :], written, residual[..., block_idx + 1 :, :]],
        dim=-2,
    )
