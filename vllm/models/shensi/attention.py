# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shensi attention classes.

Thin subclasses of the DeepSeek-V4 attention stack. The only addition is a
torch fallback for the fused fp8 o-projection so non-quantized checkpoints
(e.g. dummy profiling with bf16 weights) can run the FlashMLA / FlashInfer
backends; fp8 checkpoints take the unmodified DSV4 path.
"""

import torch

from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention


def _eager_o_proj(
    attn: DeepseekV4FlashMLAAttention, o: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Torch fallback for the fused fp8 o-projection (non-fp8 weights).

    Mirrors the fused kernel's inverse RoPE: the trailing ``rope_dim`` entries
    of each head are rotated back with the cos/sin cache, the partner entry
    being the adjacent index. The grouped wo_a contraction is written as the
    same ``bgd,grd->bgr`` einsum the fused path uses.
    """
    nope_dim, rope_dim = attn.nope_head_dim, attn.rope_head_dim
    rope = o[..., nope_dim:]
    cos_sin = attn.rotary_emb.cos_sin_cache[positions]
    half = rope_dim // 2
    cos = cos_sin[..., :half]
    sin = cos_sin[..., half:]
    partner = (
        rope.reshape(*rope.shape[:-1], half, 2)
        .flip(-1)
        .reshape(*rope.shape[:-1], rope_dim)
    )
    idx = torch.arange(rope_dim, device=o.device)
    cs_idx = (idx >> 1) % half
    cos_v = cos[..., cs_idx].unsqueeze(1)
    sin_v = sin[..., cs_idx].unsqueeze(1)
    x_add = rope * cos_v + partner * sin_v
    x_sub = rope * cos_v - partner * sin_v
    rotated = torch.where(idx % 2 == 0, x_add, x_sub)
    o = torch.cat([o[..., :nope_dim], rotated], dim=-1)

    num_tokens, _, head_dim = o.shape
    groups = attn.n_local_groups
    heads_per_group = attn.n_local_heads // groups
    o = o.reshape(num_tokens, groups, heads_per_group * head_dim)
    weight = attn.wo_a.weight.reshape(groups, attn.o_lora_rank, -1)
    out_a = torch.einsum("bgd,grd->bgr", o.float(), weight.float())
    out_a = out_a.to(attn.wo_b.weight.dtype)
    return attn.wo_b(out_a.reshape(num_tokens, groups * attn.o_lora_rank))


class ShensiFlashMLAAttention(DeepseekV4FlashMLAAttention):
    """FlashMLA sparse MLA attention layer for Shensi (CUDA)."""

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype != torch.float8_e4m3fn:
            return _eager_o_proj(self, o, positions)
        return super()._o_proj(o, positions)


class ShensiFlashInferSM120Attention(DeepseekV4FlashInferSM120Attention):
    """FlashInfer SM120 sparse MLA attention layer for Shensi (CUDA)."""

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.wo_a.weight.dtype != torch.float8_e4m3fn:
            return _eager_o_proj(self, o, positions)
        return super()._o_proj(o, positions)
