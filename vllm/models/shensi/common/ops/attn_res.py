# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

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
    r_decay, r_erase, r_write = (
        F.linear(F.linear(state, g_a_weight, g_a_bias), g_b_weight, g_b_bias)
        .reshape(*state.shape[:-1], 3, -1)
        .unbind(-2)
    )
    decay_scale, erase_scale, write_scale, _ = g_scale.unbind()
    # decay_scale 前向取非负，梯度照常回传（直接 clamp 在边界上梯度为零）
    decay_scale = decay_scale + (decay_scale.clamp(min=0.0) - decay_scale).detach()
    decay = torch.exp(F.softplus(r_decay) * (-decay_scale * t.exp()))
    erase = F.softplus(r_erase) * erase_scale
    write = 1.0 + torch.tanh(r_write) * write_scale
    # 没有 delta（首层）时 k 投影的输入与加法项都退回 state / 0，与 HF、mcore 两侧一致
    khat = F.normalize(
        F.linear(F.linear(state if delta is None else delta, k_a_weight), k_b_weight),
        dim=-1,
    )
    m = (
        torch.addcmul(decay * prefix, write, delta)
        if delta is not None
        else decay * prefix
    )
    lam = erase.mean(dim=-1, keepdim=True).clamp(min=-0.5)
    return torch.addcmul(
        m,
        khat,
        torch.einsum("...d,...d->...", khat, m).unsqueeze(-1) * lam / (1.0 + lam),
        value=-1.0,
    )


@torch._dynamo.disable
def _spike_metric_query(
    blocks: torch.Tensor,
    updated: torch.Tensor,
    query: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    """对角 + 高于 MP 边沿的尖峰度量，秩由数据决定；采样与 SVD 不进 inductor 的融合区，
    保证与 HF、mcore 数值一致（采样与原随机 SVD 的 RNG 消耗顺序也必须一致）。"""
    H = updated.shape[-1]
    M = updated.numel() // H
    # 定形预算：K = 2H 个采样 value 行
    K = min(2 * H, M * num_blocks)
    ti = torch.randint(M, (K,), device=blocks.device)
    ji = torch.randint(num_blocks, (K,), device=blocks.device)
    blk = blocks[..., :num_blocks, :]
    rows = blk.detach()[(*torch.unravel_index(ti, blocks.shape[:-2]), ji)].float()

    # 逐坐标二阶矩，按证据量 H/(K+H) 收缩向 1；下限保证死坐标有限，无需调 eps
    m2 = torch.einsum("kd,kd->d", rows, rows) / K
    s = torch.lerp(m2, m2.new_ones(()), H / (K + H)).clamp_min(1e-20).rsqrt()
    z = rows * s

    # 一次 SVD + Marchenko-Pastur 边沿：只保留统计上可估的方向；
    # BBP 去偏恢复总体尖峰强度
    _, S, V = torch.svd_lowrank(z, q=min(64, K, H), niter=4)
    ell = S.square() / K
    keep = ell > (1.0 + math.sqrt(H / K)) ** 2
    b = ell[keep] - 1.0 - H / K
    theta = 0.5 * (b + (b.square() - 4.0 * H / K).clamp_min(0.0).sqrt())
    U = V[:, keep] * (theta / (1.0 + theta)).unsqueeze(0) * s.unsqueeze(-1)

    # 度量全部折进 query：C^-1 ~ s^2 (I - U U^T)，values 只被打分与检索，从不被度量触碰
    return s * (query - (query @ U) @ U.T)


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def attn_res_route(
    blocks: torch.Tensor,
    updated: torch.Tensor,
    query: torch.Tensor,
    num_blocks: int,
    read_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    blk = blocks[..., :num_blocks, :]
    q = _spike_metric_query(blocks, updated, query, num_blocks)

    # 打分与检索：仅有的两遍扫 values
    logits = torch.cat(
        (
            (blk @ q.to(out_dtype).unsqueeze(-1)).squeeze(-1).float(),
            torch.einsum("...d,...d->...", updated, q).unsqueeze(-1),
        ),
        dim=-1,
    )
    # exp(logit - softplus(lse)) == sigmoid(lse) * softmax(logit)；
    # 分数和始终小于 1，残差恒主导读出
    w = torch.softmax(logits, dim=-1)
    g = torch.sigmoid(torch.logsumexp(logits, dim=-1, keepdim=True)) * read_scale
    return torch.addcmul(
        (w[..., :num_blocks].to(out_dtype).unsqueeze(-2) @ blk).squeeze(-2).float(),
        w[..., num_blocks:] * g,
        updated,
    )


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
