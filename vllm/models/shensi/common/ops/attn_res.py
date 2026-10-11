# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shensi attention-residual ops.

The gated state update, the block-routing softmax and the residual-bank write
run as Triton kernels owned by ``VllmTritonJitKernel``, so they are compiled
during the JIT-kernel warmup phase instead of being ``torch.compile``'d on
first use. The spectral query metric stays in torch: ``svd_lowrank`` and its
RNG draw cannot live inside a kernel, and keeping both outside any fusion
region is what makes the numerics match the HF and Megatron-Core
implementations.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonWarmupTensor,
    VllmTritonJitKernel,
)
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.math_utils import next_power_of_2

# The routing softmax runs over at most ``num_blocks + 1`` sources; one more
# than the widest bank a Shensi checkpoint can build (a block-write layer at
# most every second layer, then the updated state).
_MAX_SOURCES = 32


@triton.jit
def _softplus(x):
    """``torch.nn.functional.softplus`` with ``beta=1, threshold=20``."""
    return tl.where(x <= 20.0, tldevice.log1p(tl.exp(x)), x)


@triton.jit
def _tanh(x):
    # Matches the in-tree triton activation convention (vllm/kernels/triton/
    # activation.py).
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


class AttnResUpdateKernel(VllmTritonJitKernel["AttnResUpdateKernel.CompileKey"]):
    """Gated decay/erase/write update of the attention-residual state.

    Consumes the two projection outputs (``g_b`` over the gate bottleneck and
    ``k_b`` over the key bottleneck) and produces the updated state. The
    projections themselves run on the model's vLLM-native linear layers.
    """

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        hidden: int
        block_size: int
        has_delta: bool

    @staticmethod
    @triton.jit
    def kernel(
        g_ptr,
        kraw_ptr,
        prefix_ptr,
        delta_ptr,
        g_scale_ptr,
        t_ptr,
        output_ptr,
        HIDDEN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_DELTA: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < HIDDEN
        row_in = row * HIDDEN
        row_g = row * 3 * HIDDEN

        g_decay = tl.load(g_ptr + row_g + offs, mask=mask, other=0.0)
        g_erase = tl.load(g_ptr + row_g + HIDDEN + offs, mask=mask, other=0.0)
        g_write = tl.load(g_ptr + row_g + 2 * HIDDEN + offs, mask=mask, other=0.0)
        kraw = tl.load(kraw_ptr + row_in + offs, mask=mask, other=0.0)
        prefix = tl.load(prefix_ptr + row_in + offs, mask=mask, other=0.0)
        t = tl.load(t_ptr + offs, mask=mask, other=0.0)

        # The decay scale stays non-negative in the forward; the training graph
        # passes the gradient through (here: a plain clamp).
        decay_scale = tl.maximum(tl.load(g_scale_ptr), 0.0)
        erase_scale = tl.load(g_scale_ptr + 1)
        write_scale = tl.load(g_scale_ptr + 2)

        decay = tl.exp(_softplus(g_decay) * (-decay_scale * tl.exp(t)))
        erase = _softplus(g_erase) * erase_scale
        write = 1.0 + _tanh(g_write) * write_scale

        if HAS_DELTA:
            delta = tl.load(delta_ptr + row_in + offs, mask=mask, other=0.0)
            m = decay * prefix + write * delta
        else:
            m = decay * prefix

        # F.normalize(kraw, dim=-1)
        norm = tl.sqrt(tl.sum(kraw * kraw, axis=0))
        khat = kraw / tl.maximum(norm, 1e-12)

        lam = tl.maximum(tl.sum(erase, axis=0) / HIDDEN, -0.5)
        correction = (tl.sum(khat * m, axis=0) * lam) / (1.0 + lam)
        output = m - khat * correction
        tl.store(output_ptr + row_in + offs, output, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
        has_delta: bool,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            hidden=hidden,
            block_size=next_power_of_2(hidden),
            has_delta=has_delta,
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        hidden = int(getattr(hf_config, "hidden_size", 0) or 0)
        if hidden <= 0:
            return []

        keys = self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            hidden=hidden,
            has_delta=True,
        )
        keys += self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            hidden=hidden,
            has_delta=False,
        )
        return keys

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        hidden = compile_key.hidden
        return dict(
            g=TritonWarmupTensor(torch.float32, shape=(1, 3 * hidden)),
            kraw=TritonWarmupTensor(torch.float32, shape=(1, hidden)),
            prefix=TritonWarmupTensor(torch.float32, shape=(1, hidden)),
            delta=(
                TritonWarmupTensor(torch.float32, shape=(1, hidden))
                if compile_key.has_delta
                else None
            ),
            g_scale=TritonWarmupTensor(torch.float32, shape=(4,)),
            t=TritonWarmupTensor(torch.float32, shape=(hidden,)),
            _outputs=(TritonWarmupTensor(torch.float32, shape=(1, hidden)),),
        )

    @kernel_launcher
    def __call__(
        self,
        g: torch.Tensor,
        kraw: torch.Tensor,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        g_scale: torch.Tensor,
        t: torch.Tensor,
        *,
        _outputs: tuple[Any, ...] | None = None,
    ) -> LaunchSpec:
        num_tokens = prefix.shape[0]
        hidden = prefix.shape[1]
        has_delta = delta is not None
        if _outputs is None:
            _outputs = (torch.empty_like(prefix),)
        (output,) = _outputs
        block_size = next_power_of_2(hidden)
        grid = None if num_tokens == 0 else (num_tokens,)
        return (
            grid,
            dict(
                output_ptr=output,
                # A dummy pointer when there is no delta: the kernel never
                # loads it under ``HAS_DELTA`` False.
                delta_ptr=delta if has_delta else prefix,
                HIDDEN=hidden,
                BLOCK_SIZE=block_size,
                HAS_DELTA=has_delta,
                num_warps=8 if block_size >= 2048 else 4,
            ),
            _outputs,
        )


class AttnResRouteKernel(VllmTritonJitKernel["AttnResRouteKernel.CompileKey"]):
    """Softmax read over the residual bank plus the updated state.

    Scores the completed blocks and the updated state with the (already
    spectral-debiased) query, softmaxes the ``num_blocks + 1`` sources, and
    mixes them with the sigmoid-gated state term. Mirrors the reference
    rounding: the query and the per-block scores are rounded to the activation
    dtype before the fp32 softmax, the block mix rounds once after the sum,
    and the state term stays fp32.
    """

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        hidden: int
        block_size: int

    @staticmethod
    @triton.jit
    def kernel(
        blocks_ptr,
        updated_ptr,
        q_ptr,
        g_scale_ptr,
        output_ptr,
        num_blocks,
        blocks_per_row,
        HIDDEN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_SOURCES: tl.constexpr,
    ):
        activation_dtype: tl.constexpr = blocks_ptr.dtype.element_ty
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < HIDDEN
        row_in = row * HIDDEN
        row_blocks = row * blocks_per_row * HIDDEN
        source_idx = tl.arange(0, MAX_SOURCES)

        q = tl.load(q_ptr + row_in + offs, mask=mask, other=0.0)
        q_blocks = q.to(activation_dtype).to(tl.float32)

        # Score the completed blocks (activation-dtype products, one rounding
        # per source like the reference matmul) and the updated state (fp32).
        logits = tl.zeros((MAX_SOURCES,), dtype=tl.float32)
        for source in range(0, num_blocks):
            block = tl.load(
                blocks_ptr + row_blocks + source * HIDDEN + offs,
                mask=mask,
                other=0.0,
            )
            score = tl.sum(block * q_blocks, axis=0)
            score = score.to(activation_dtype).to(tl.float32)
            logits += tl.where(source_idx == source, score, 0.0)
        updated = tl.load(updated_ptr + row_in + offs, mask=mask, other=0.0)
        logits += tl.where(source_idx == num_blocks, tl.sum(updated * q, axis=0), 0.0)

        scores = tl.where(source_idx <= num_blocks, logits, float("-inf"))
        row_max = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - row_max)
        denominator = tl.sum(exp_scores, axis=0)
        weights = exp_scores / denominator
        logsumexp = row_max + tl.log(denominator)
        gate = (1.0 / (1.0 + tl.exp(-logsumexp))) * tl.load(g_scale_ptr + 3)

        # Mix the blocks with weights rounded to the activation dtype (the
        # reference matmul casts them), then round the summed mix once.
        mixed = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for source in range(0, num_blocks):
            block = tl.load(
                blocks_ptr + row_blocks + source * HIDDEN + offs,
                mask=mask,
                other=0.0,
            )
            weight = tl.sum(tl.where(source_idx == source, weights, 0.0), axis=0)
            mixed += weight.to(activation_dtype).to(tl.float32) * block
        mixed = mixed.to(activation_dtype).to(tl.float32)

        updated_weight = tl.sum(
            tl.where(source_idx == num_blocks, weights, 0.0), axis=0
        )
        output = mixed + (updated_weight * gate) * updated
        tl.store(output_ptr + row_in + offs, output, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            hidden=hidden,
            block_size=next_power_of_2(hidden),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        hidden = int(getattr(hf_config, "hidden_size", 0) or 0)
        if hidden <= 0:
            return []

        return self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            hidden=hidden,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        hidden = compile_key.hidden
        return dict(
            blocks=TritonWarmupTensor(compile_key.dtype, shape=(1, 1, hidden)),
            updated=TritonWarmupTensor(torch.float32, shape=(1, hidden)),
            q=TritonWarmupTensor(torch.float32, shape=(1, hidden)),
            g_scale=TritonWarmupTensor(torch.float32, shape=(4,)),
            num_blocks=1,
            _outputs=(TritonWarmupTensor(torch.float32, shape=(1, hidden)),),
        )

    @kernel_launcher
    def __call__(
        self,
        blocks: torch.Tensor,
        updated: torch.Tensor,
        q: torch.Tensor,
        g_scale: torch.Tensor,
        num_blocks: int,
        *,
        _outputs: tuple[Any, ...] | None = None,
    ) -> LaunchSpec:
        num_tokens = blocks.shape[0]
        hidden = blocks.shape[2]
        if _outputs is None:
            _outputs = (torch.empty_like(updated),)
        (output,) = _outputs
        block_size = next_power_of_2(hidden)
        grid = None if num_tokens == 0 else (num_tokens,)
        return (
            grid,
            dict(
                output_ptr=output,
                blocks_per_row=blocks.shape[1],
                HIDDEN=hidden,
                BLOCK_SIZE=block_size,
                MAX_SOURCES=_MAX_SOURCES,
                num_warps=8 if block_size >= 2048 else 4,
            ),
            _outputs,
        )


class AttnResWriteBlockKernel(
    VllmTritonJitKernel["AttnResWriteBlockKernel.CompileKey"]
):
    """Write the current state into the residual bank, in place."""

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        hidden: int
        block_size: int

    @staticmethod
    @triton.jit
    def kernel(
        prefix_ptr,
        blocks_ptr,
        block_idx,
        blocks_per_row,
        HIDDEN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        activation_dtype: tl.constexpr = blocks_ptr.dtype.element_ty
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < HIDDEN

        row_in = row * HIDDEN
        value = tl.load(prefix_ptr + row_in + offs, mask=mask, other=0.0)
        tl.store(
            blocks_ptr + row * blocks_per_row * HIDDEN + block_idx * HIDDEN + offs,
            value.to(activation_dtype),
            mask=mask,
        )

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        hidden: int,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            hidden=hidden,
            block_size=next_power_of_2(hidden),
        )

    def get_warmup_keys(self, vllm_config: Any) -> list[CompileKey]:
        hf_config = vllm_config.model_config.hf_config
        hidden = int(getattr(hf_config, "hidden_size", 0) or 0)
        if hidden <= 0:
            return []

        return self._trace_dispatch(self.dispatch)(
            dtype=vllm_config.model_config.dtype,
            hidden=hidden,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        hidden = compile_key.hidden
        return dict(
            prefix=TritonWarmupTensor(compile_key.dtype, shape=(1, hidden)),
            blocks=TritonWarmupTensor(compile_key.dtype, shape=(1, 1, hidden)),
            block_idx=0,
        )

    @kernel_launcher
    def __call__(
        self,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        block_idx: int,
    ) -> LaunchSpec:
        num_tokens = prefix.shape[0]
        hidden = prefix.shape[1]
        block_size = next_power_of_2(hidden)
        grid = None if num_tokens == 0 else (num_tokens,)
        return (
            grid,
            dict(
                blocks_per_row=blocks.shape[1],
                HIDDEN=hidden,
                BLOCK_SIZE=block_size,
                num_warps=8 if block_size >= 2048 else 4,
            ),
        )


_ATTN_RES_UPDATE_KERNEL = AttnResUpdateKernel()
_ATTN_RES_ROUTE_KERNEL = AttnResRouteKernel()
_ATTN_RES_WRITE_BLOCK_KERNEL = AttnResWriteBlockKernel()


@torch._dynamo.disable
def _spike_metric_query(
    blocks: torch.Tensor,
    updated: torch.Tensor,
    query: torch.Tensor,
    num_blocks: int,
) -> torch.Tensor:
    """Diagonal plus above-MP-edge spike metric, with the rank decided by the data.
    The sampling and the SVD stay outside the inductor fusion region so the numerics
    match the HF and Megatron-Core implementations, including the RNG draw order."""
    H = updated.shape[-1]
    M = updated.numel() // H
    # Fixed-shape budget: K = 2H sampled value rows.
    K = min(2 * H, M * num_blocks)
    ti = torch.randint(M, (K,), device=blocks.device)
    ji = torch.randint(num_blocks, (K,), device=blocks.device)
    blk = blocks[..., :num_blocks, :]
    rows = blk.detach()[(*torch.unravel_index(ti, blocks.shape[:-2]), ji)].float()

    # Per-coordinate second moments shrunk toward 1 by evidence (H / (K + H)); the
    # floor keeps dead coordinates finite.
    m2 = torch.einsum("kd,kd->d", rows, rows) / K
    s = torch.lerp(m2, m2.new_ones(()), H / (K + H)).clamp_min(1e-20).rsqrt()
    z = rows * s

    # One SVD plus the Marchenko-Pastur edge: keep only statistically estimable
    # directions, and debias the recovered spike strengths (BBP).
    _, S, V = torch.svd_lowrank(z, q=min(64, K, H), niter=4)
    ell = S.square() / K
    keep = ell > (1.0 + math.sqrt(H / K)) ** 2
    b = ell[keep] - 1.0 - H / K
    theta = 0.5 * (b + (b.square() - 4.0 * H / K).clamp_min(0.0).sqrt())
    U = V[:, keep] * (theta / (1.0 + theta)).unsqueeze(0) * s.unsqueeze(-1)

    # Fold the metric into the query (C^-1 ~ s^2 (I - U U^T)); values are only scored
    # and retrieved.
    return s * (query - (query @ U) @ U.T)


def attn_res_update(
    g: torch.Tensor,
    kraw: torch.Tensor,
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    g_scale: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Apply the gated decay/erase/write update to ``prefix``.

    ``g`` is the fp32 output of the gate bottleneck (``3 * hidden`` wide) and
    ``kraw`` the un-normalized key bottleneck output. All inputs are fp32.
    Returns the updated state in fp32.
    """
    for tensor in (g, kraw, prefix, g_scale, t):
        assert tensor.dtype == torch.float32, tensor.dtype
        assert tensor.is_contiguous()
    if delta is not None:
        assert delta.dtype == torch.float32 and delta.is_contiguous()
    (updated,) = _ATTN_RES_UPDATE_KERNEL(g, kraw, prefix, delta, g_scale, t)
    return updated


def attn_res_route(
    blocks: torch.Tensor,
    updated: torch.Tensor,
    query: torch.Tensor,
    num_blocks: int,
    g_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Read the residual bank back through the spike-metric query.

    ``blocks`` is the activation-dtype residual bank, ``updated``/``query``
    fp32, and ``out_dtype`` must equal the bank's dtype; the result is fp32.
    """
    assert 0 <= num_blocks < _MAX_SOURCES, (
        f"attn_res_route supports at most {_MAX_SOURCES - 1} residual blocks, "
        f"got {num_blocks}"
    )
    assert blocks.dtype == out_dtype, (blocks.dtype, out_dtype)
    assert blocks.is_contiguous() and updated.is_contiguous() and query.is_contiguous()
    assert g_scale.dtype == torch.float32 and g_scale.is_contiguous()
    q = _spike_metric_query(blocks, updated, query, num_blocks)
    (routed,) = _ATTN_RES_ROUTE_KERNEL(blocks, updated, q, g_scale, num_blocks)
    return routed


def write_residual_block(
    prefix_sum: torch.Tensor, residual: torch.Tensor, block_idx: int
) -> torch.Tensor:
    """Write ``prefix_sum`` into ``residual[:, block_idx, :]`` in place.

    ``residual`` is the activation-dtype residual bank owned by the model's
    forward; the value is cast to the bank's dtype on the way in.
    """
    assert residual.is_contiguous() and prefix_sum.is_contiguous()
    assert residual.shape[0] == prefix_sum.shape[0]
    if prefix_sum.shape[0] > 0:
        _ATTN_RES_WRITE_BLOCK_KERNEL(prefix_sum, residual, block_idx)
    return residual
