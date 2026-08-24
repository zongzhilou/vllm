# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shensi-VL model: Shensi text backbone + MoonViT3d-style vision tower with
# DeepRecur cross-modal blocks, and continuous-latent reasoning.
#
# The language layer reuses the vLLM Shensi components verbatim
# (``vllm.models.shensi.model``), the vision layer reuses the Kimi-K2.5 VIT
# components (``vllm.model_executor.models.kimi_k25_vit``); only the DeepRecur
# blocks (CrossBlock / LoopBlock / PoolerCrossAttention / Reasoning) are
# Shensi-VL specific.
#
# Adapted from transformers/models/shensi_vl/modeling_shensi_vl.py.

from collections import deque
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.kimi_k25 import KimiK25MediaPixelInputs
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonVision3dPatchEmbed,
    MoonViTEncoderLayer,
    Rope2DPosEmbRepeated,
    _apply_rope_input_validation,
    _make_vision_norm,
    build_image_merge_gather_idx,
    tpool_patch_merger,
    tpool_patch_merger_packed,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.models.shensi.model import (
    ShensiAttentionResidual,
    ShensiModel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

from .common.mm_preprocess import (
    ShensiVlDummyInputsBuilder,
    ShensiVlMultiModalProcessor,
    ShensiVlProcessingInfo,
)

if TYPE_CHECKING:
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
    )


def cross_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Memory-efficient cross-attention: the naive matmul materializes the
    # full (heads, q_len, kv_len) scores, which is quadratic in the
    # vision/language lengths and OOMs on long contexts. The fused kernel
    # keeps the scores on chip and is numerically equivalent. The additive
    # (-inf/0) mask becomes a bool mask so the flash kernel applies instead
    # of the math fallback.
    if attention_mask is not None:
        attention_mask = attention_mask > -1
    attn_weights = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
    )
    attn_output = attn_weights.transpose(1, 2).contiguous()
    return attn_output, None


# --- DeepRecur cross-modal exchange as a graph-splitting custom op ---------
# torch.compile / PIECEWISE graphs would otherwise freeze the exchange with
# whatever branch the capture-time dummy run took (no vision), silently
# dropping cross-attention for real multimodal requests. Registering the
# exchange as a custom op and adding it to ``splitting_ops`` keeps it outside
# every captured graph, so it always executes eagerly with live state.

_CROSS_EXCHANGE_REGISTRY: dict[int, nn.Module] = {}


def _cross_modal_exchange_op_impl(
    hidden_states: torch.Tensor,
    prefix_sum: torch.Tensor,
    vision_prefix_sum: torch.Tensor,
    gate: torch.Tensor,
    evidence_mask: torch.Tensor,
    guidance_mask: torch.Tensor,
    hidden_out: torch.Tensor,
    vision_out: torch.Tensor,
    block_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _CROSS_EXCHANGE_REGISTRY[block_index]
    out = module.run_exchange_impl(
        hidden_states,
        prefix_sum,
        vision_prefix_sum,
        gate,
        evidence_mask if evidence_mask.numel() else None,
        guidance_mask if guidance_mask.numel() else None,
    )
    hidden_out[: out[0].shape[0]].copy_(out[0])
    vision_out[: out[1].shape[0]].copy_(out[1])
    return hidden_out[: out[0].shape[0]], vision_out[: out[1].shape[0]]


def _cross_modal_exchange_fake(
    hidden_states: torch.Tensor,
    prefix_sum: torch.Tensor,
    vision_prefix_sum: torch.Tensor,
    gate: torch.Tensor,
    evidence_mask: torch.Tensor,
    guidance_mask: torch.Tensor,
    hidden_out: torch.Tensor,
    vision_out: torch.Tensor,
    block_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        hidden_out[: hidden_states.shape[0]],
        vision_out[: vision_prefix_sum.shape[0]],
    )


def _loop_evidence_op_impl(
    language_queries: torch.Tensor,
    vision_final: torch.Tensor,
    evidence_mask: torch.Tensor,
    block_index: int,
) -> torch.Tensor:
    module = _CROSS_EXCHANGE_REGISTRY[block_index]
    return module.retrieve_evidence(
        language_queries,
        vision_final,
        evidence_mask if evidence_mask.numel() else None,
    )


def _loop_evidence_fake(
    language_queries: torch.Tensor,
    vision_final: torch.Tensor,
    evidence_mask: torch.Tensor,
    block_index: int,
) -> torch.Tensor:
    return torch.empty_like(language_queries)


def _vit_attn_capture_op_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    fa_version: int | None,
) -> torch.Tensor:
    """Vision self-attention with a static (capture-time) max-seqlen bound.

    ``flash_attn_varlen_func`` needs a Python int for its kernel launch
    bounds; the vLLM vit wrappers derive it with a host sync, which CUDA
    graph capture forbids. The graph path precomputes the int before
    capture, so this op is capture-safe and numerically identical to
    ``vit_flash_attn_wrapper`` on CUDA.
    """
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    q, k, v = (x.reshape(-1, x.shape[-2], x.shape[-1]) for x in (q, k, v))
    kwargs = {} if fa_version is None else {"fa_version": fa_version}
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        **kwargs,
    )


def _vit_attn_capture_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    fa_version: int | None,
) -> torch.Tensor:
    return torch.empty_like(q)


def _cross_mask_op_impl(
    input_ids: torch.Tensor,
    patch_counts: list[int],
    query_start_loc: torch.Tensor,
    sequence_length: int,
    dtype: torch.dtype,
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat-token cross-attention masks over the packed patch set.

    Runs eagerly (graph splitting point): the per-request patch counts are
    only known at runtime, so the masks must not be frozen into a captured
    graph.
    """
    device = input_ids.device
    number_of_clips = len(patch_counts)
    min_value = torch.finfo(dtype).min
    total_kv = sum(patch_counts)

    token_indices = torch.arange(sequence_length, device=device)
    cu_seqlens_64 = query_start_loc.to(torch.int64)
    sample_ids = torch.searchsorted(cu_seqlens_64, token_indices, right=True) - 1
    token_pos = token_indices - cu_seqlens_64[sample_ids]

    # Vectorized clip assignment: consecutive placeholder tokens inside one
    # sample form a single clip, numbered in packing order.
    is_ph = input_ids == image_token_id
    prev_ph = torch.zeros_like(is_ph)
    prev_ph[1:] = is_ph[:-1]
    prev_same_sample = torch.zeros_like(is_ph)
    prev_same_sample[1:] = sample_ids[1:] == sample_ids[:-1]
    new_clip = is_ph & ~(prev_ph & prev_same_sample)

    kv_clip_ids = torch.where(
        is_ph, torch.cumsum(new_clip, dim=0) - 1, torch.full_like(is_ph, -1)
    )
    ph_idx = torch.nonzero(new_clip, as_tuple=True)[0]
    clip_to_sample = torch.full((number_of_clips,), -1, dtype=torch.long, device=device)
    clip_start_pos = torch.zeros(number_of_clips, dtype=torch.long, device=device)
    if ph_idx.numel() > 0:
        clip_to_sample.scatter_(0, kv_clip_ids[ph_idx], sample_ids[ph_idx])
        clip_start_pos.scatter_(0, kv_clip_ids[ph_idx], token_pos[ph_idx])

    patch_clip_ids = torch.arange(number_of_clips, device=device).repeat_interleave(
        torch.tensor(patch_counts, device=device)
    )
    patch_sample = clip_to_sample[patch_clip_ids]
    patch_start = clip_start_pos[patch_clip_ids]

    # Evidence mask: token i may attend to patch j iff they share a sample and
    # the clip's start is not after token i's position.
    evidence_mask = torch.full(
        (sequence_length, total_kv), min_value, dtype=dtype, device=device
    )
    visible = (sample_ids[:, None] == patch_sample[None, :]) & (
        token_pos[:, None] >= patch_start[None, :]
    )
    evidence_mask.masked_fill_(visible, 0.0)
    # Rows without any visible patch are fully released to avoid NaN softmax.
    row_has_visible = visible.any(dim=-1, keepdim=True)
    evidence_mask.masked_fill_(~row_has_visible, 0.0)

    # Guidance mask: a patch may attend back only to its own clip's tokens.
    guidance_mask = torch.full(
        (total_kv, sequence_length), min_value, dtype=dtype, device=device
    )
    same_clip = patch_clip_ids[:, None] == kv_clip_ids[None, :]
    guidance_mask.masked_fill_(same_clip, 0.0)
    return evidence_mask, guidance_mask


def _cross_mask_fake(
    input_ids: torch.Tensor,
    patch_counts: list[int],
    query_start_loc: torch.Tensor,
    sequence_length: int,
    dtype: torch.dtype,
    image_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_kv = sum(patch_counts)
    device = input_ids.device
    return (
        torch.empty(sequence_length, total_kv, dtype=dtype, device=device),
        torch.empty(total_kv, sequence_length, dtype=dtype, device=device),
    )


def _pin_loop_outputs(
    module: nn.Module, out: tuple, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Write the loop block's language outputs into stable buffers.

    The final norm piece consumes them; pinning the addresses (the
    "with_output" convention for splitting ops) keeps the piece reads current
    instead of relying on allocator address reuse.
    """
    bufs = getattr(module, "_loop_out_buffers", None)
    lang = out[0]
    if lang is not None:
        if (
            bufs is None
            or bufs[0].device != lang.device
            or bufs[0].shape[0] < lang.shape[0]
        ):
            hc = lang.shape[1]
            hidden = lang.shape[-1]
            blocks = out[2].shape[2]
            max_tokens = max(lang.shape[0], 2048)
            bufs = (
                torch.empty(
                    max_tokens, hc, hidden, device=lang.device, dtype=lang.dtype
                ),
                torch.empty(
                    max_tokens, hc, hidden, device=lang.device, dtype=lang.dtype
                ),
                torch.empty(
                    max_tokens,
                    hc,
                    blocks,
                    hidden,
                    device=lang.device,
                    dtype=lang.dtype,
                ),
            )
            module._loop_out_buffers = bufs
        for buf, t in zip(bufs, out[:3]):
            buf[: t.shape[0]].copy_(t)
        return (
            bufs[0][: out[0].shape[0]],
            bufs[1][: out[1].shape[0]],
            bufs[2][: out[2].shape[0]],
            out[3],
            out[4],
            out[5],
        )
    return out


def _loop_block_op_impl(
    hidden_states: torch.Tensor | None,
    residual: torch.Tensor,
    prefix_sum: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    vision_hidden_states: torch.Tensor | None,
    vision_residual: torch.Tensor | None,
    vision_prefix_sum: torch.Tensor | None,
    vision_cu_seqlens: torch.Tensor | None,
    vision_rope_freqs_cis: torch.Tensor | None,
    vision_max_seqlen: int | None,
    vision_sequence_lengths: torch.Tensor | None,
    evidence_mask: torch.Tensor | None,
    num_attn_res_blocks: int,
    block_index: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the DeepRecur loop block with its data-dependent while loop.

    The convergence guards read device scalars on the host, which is illegal
    inside CUDA graph capture, and the iteration count varies per step.
    Keeping the whole block outside every captured graph (via splitting_ops)
    preserves the loop semantics; the surrounding cross blocks stay captured.
    """
    module = _CROSS_EXCHANGE_REGISTRY[block_index]
    out = module.forward(
        hidden_states,
        residual,
        prefix_sum,
        positions,
        input_ids,
        vision_hidden_states=vision_hidden_states,
        vision_residual=vision_residual,
        vision_prefix_sum=vision_prefix_sum,
        vision_cu_seqlens=vision_cu_seqlens,
        vision_rope_freqs_cis=vision_rope_freqs_cis,
        vision_max_seqlen=vision_max_seqlen,
        vision_sequence_lengths=vision_sequence_lengths,
        evidence_mask=evidence_mask,
        num_attn_res_blocks=num_attn_res_blocks,
        run_vision_layers=True,
    )
    # Optional vision outputs are encoded as empty tensors for the op schema.
    device = hidden_states.device if hidden_states is not None else residual.device
    return _pin_loop_outputs(module, out, device)


def _loop_block_fake(
    hidden_states: torch.Tensor | None,
    residual: torch.Tensor,
    prefix_sum: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    vision_hidden_states: torch.Tensor | None,
    vision_residual: torch.Tensor | None,
    vision_prefix_sum: torch.Tensor | None,
    vision_cu_seqlens: torch.Tensor | None,
    vision_rope_freqs_cis: torch.Tensor | None,
    vision_max_seqlen: int | None,
    vision_sequence_lengths: torch.Tensor | None,
    evidence_mask: torch.Tensor | None,
    num_attn_res_blocks: int,
    block_index: int,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty_like(prefix_sum),
        torch.empty_like(prefix_sum),
        torch.empty_like(residual),
        torch.empty(0, device=prefix_sum.device),
        torch.empty(0, device=prefix_sum.device),
        torch.empty(0, device=prefix_sum.device),
    )


direct_register_custom_op(
    op_name="shensi_vl_cross_modal_exchange",
    op_func=_cross_modal_exchange_op_impl,
    mutates_args=[],
    fake_impl=_cross_modal_exchange_fake,
)
direct_register_custom_op(
    op_name="shensi_vl_loop_evidence",
    op_func=_loop_evidence_op_impl,
    mutates_args=[],
    fake_impl=_loop_evidence_fake,
)
direct_register_custom_op(
    op_name="shensi_vl_vit_attn",
    op_func=_vit_attn_capture_op_impl,
    mutates_args=[],
    fake_impl=_vit_attn_capture_fake,
)
direct_register_custom_op(
    op_name="shensi_vl_cross_mask",
    op_func=_cross_mask_op_impl,
    mutates_args=[],
    fake_impl=_cross_mask_fake,
)
direct_register_custom_op(
    op_name="shensi_vl_loop_block",
    op_func=_loop_block_op_impl,
    mutates_args=[],
    fake_impl=_loop_block_fake,
)


class ShensiVlVisionEncoderLayer(MoonViTEncoderLayer):
    """MoonViT encoder layer with a DeepRecur attention-residual wrapper.

    Reuses the MoonViT components (norm0/norm1, wqkv/wo, attn, mlp) and adds the
    per-block AttnRes state, mirroring the HF
    ``ShensiVlVisionEncoderLayer(Kimi_K25VisionEncoderLayer)``. The vision
    streams have no ``hc`` axis, so AttnRes sees a singleton stream axis.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            num_heads=config.num_attention_heads,
            hidden_dim=config.hidden_size,
            mlp_dim=config.intermediate_size,
            quant_config=quant_config,
            prefix=prefix,
            activation=F.gelu,
            attn_bias=getattr(config, "attn_bias", False),
            qkv_hidden_size=getattr(config, "qkv_hidden_size", None),
            norm_type=getattr(config, "norm_type", "rmsnorm"),
            mlp_type=getattr(config, "mlp_type", "mlp2"),
            linear_bias=getattr(config, "linear_bias", False),
        )
        self.is_block_write_layer = (
            config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        )
        self.prev_valid_blocks = sum(
            1
            for layer_type in config.attn_res_block_layer_types[:layer_idx]
            if layer_type == "block_write_layer"
        )
        self.block_write_idx = self.prev_valid_blocks
        self.self_attention_attn_res = ShensiAttentionResidual(
            config, prefix=f"{prefix}.self_attention_attn_res"
        )
        self.mlp_attn_res = ShensiAttentionResidual(
            config, prefix=f"{prefix}.mlp_attn_res"
        )

    def _attn_res(
        self,
        module: ShensiAttentionResidual,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # AttnRes operates on [T, hc, D]; the vision stream is [T_v, D] with a
        # single implicit stream.
        hidden = hidden_states.unsqueeze(1) if hidden_states is not None else None
        out, updated, blocks = module(
            hidden,
            residual.unsqueeze(1),
            prefix_sum.unsqueeze(1),
            output_norm_weight=output_norm_weight,
            num_blocks=num_blocks,
        )
        return out.squeeze(1), updated.squeeze(1), blocks.squeeze(1)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor | int | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ):
        if isinstance(max_seqlen, int):
            # CUDA-graph path: the kernel launch bound was precomputed as a
            # static int before capture; the vLLM vit wrappers would otherwise
            # perform a host sync inside the captured region. Numerically
            # identical to ``vit_flash_attn_wrapper`` on CUDA.
            seq_length = x.size(0)
            xqkv, _ = self.wqkv(x)
            qkv_shape = xqkv.size()[:-1] + (
                3,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            xqkv = xqkv.view(*qkv_shape)
            xq, xk, xv = torch.unbind(xqkv, dim=-3)

            _apply_rope_input_validation(xq, rope_freqs_cis)
            _apply_rope_input_validation(xk, rope_freqs_cis)
            rope_cos = rope_freqs_cis.real.contiguous()
            rope_sin = rope_freqs_cis.imag.contiguous()
            xq = self.apply_rotary_emb(xq, rope_cos, rope_sin)
            xk = self.apply_rotary_emb(xk, rope_cos, rope_sin)

            attn_out = torch.ops.vllm.shensi_vl_vit_attn(
                xq.unsqueeze(0),
                xk.unsqueeze(0),
                xv.unsqueeze(0),
                cu_seqlens,
                max_seqlen,
                self.attn.scale,
                getattr(self.attn, "_fa_version", None),
            )
            attn_out = attn_out.reshape(
                seq_length,
                self.num_attention_heads_per_partition
                * self.hidden_size_per_attention_head,
            )
            attn_out, _ = self.wo(attn_out)
            return attn_out
        return super().attention_qkvpacked(
            x,
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reference block-residual layout: the write layer deposits the layer-entry
        # state into its block slot and restarts the prefix, keeping the attention
        # output on the MLP-side lookup path (same layout as the language layers).
        entry = hidden_states if hidden_states is not None else prefix_sum
        delta = hidden_states - prefix_sum if hidden_states is not None else None

        hidden_states, prefix_sum, residual = self._attn_res(
            self.self_attention_attn_res,
            delta,
            residual,
            prefix_sum,
            output_norm_weight=self.norm0.weight,
            num_blocks=self.prev_valid_blocks,
        )
        if self.is_block_write_layer:
            residual = torch.cat(
                [
                    residual[..., : self.block_write_idx, :],
                    entry.to(residual.dtype).unsqueeze(-2),
                    residual[..., self.block_write_idx + 1 :, :],
                ],
                dim=-2,
            )
            prefix_sum = None

        attn_output = self.attention_qkvpacked(
            hidden_states,
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        hidden_states = hidden_states + attn_output
        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states

        hidden_states, prefix_sum, residual = self._attn_res(
            self.mlp_attn_res,
            prefix_sum,
            residual,
            prefix_sum,
            output_norm_weight=self.norm1.weight,
            num_blocks=self.prev_valid_blocks + self.is_block_write_layer,
        )

        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output
        prefix_sum = prefix_sum + hidden_states

        return hidden_states, prefix_sum, residual


class ShensiVlVisionModel(nn.Module):
    """Vision tower: MoonViT3d patch embed + 2D rope + DeepRecur encoder layers.

    Layer instances are owned here; the DeepRecur blocks reference slices of
    them, so the slices stay plain tuples instead of being re-registered as
    submodules.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.merge_kernel_size = tuple(
            config.merge_kernel_size
            if isinstance(config.merge_kernel_size, (list, tuple))
            else (config.merge_kernel_size, config.merge_kernel_size)
        )
        # The attention-residual modules read rms_norm_eps; the vision config
        # does not ship it. The modular reference uses 1e-5 for the vision
        # norms (ShensiVlRMSNorm / ShensiVlUnweightedRMSNorm).
        if not hasattr(config, "rms_norm_eps"):
            config.rms_norm_eps = 1.0e-5
        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=getattr(
                config, "init_pos_emb_height", config.pos_emb_height
            ),
            pos_emb_width=getattr(config, "init_pos_emb_width", config.pos_emb_width),
            pos_emb_time=getattr(config, "init_pos_emb_time", config.pos_emb_time),
            pos_emb_type=getattr(config, "pos_emb_type", "divided_fixed"),
            patch_embed_proj_bias=getattr(config, "patch_embed_proj_bias", False),
            pos_emb_interpolation_mode=getattr(
                config, "pos_emb_interpolation_mode", "bilinear"
            ),
        )
        self.rope_2d = Rope2DPosEmbRepeated(
            config.qkv_hidden_size // config.num_attention_heads, 512, 512
        )
        self.layers = nn.ModuleList(
            [
                ShensiVlVisionEncoderLayer(
                    config,
                    layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = _make_vision_norm(
            getattr(config, "norm_type", "rmsnorm"), config.hidden_size
        )
        self.num_attn_res_blocks = config.attn_res_block_layer_types.count(
            "block_write_layer"
        )
        self.output_attn_res = ShensiAttentionResidual(
            config, prefix=f"{prefix}.output_attn_res"
        )

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        device: torch.device,
        max_batch_size: int | None = None,
        max_seqlen_override: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thw_list, device=device)
        grid_thw_np = torch.tensor(grid_thw_list, dtype=torch.int32, device=device)
        lengths = grid_thw_np[:, 0] * grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = (
            torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), lengths.cumsum(0)]
            )
            .to(torch.int32)
            .contiguous()
        )
        if max_batch_size is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < max_batch_size:
                # Pad with zero-length tail sequences (the captured kernels
                # always process the full padded batch).
                cu_seqlens = torch.cat(
                    [
                        cu_seqlens,
                        torch.full(
                            (max_batch_size - num_seqs,),
                            int(cu_seqlens[-1]),
                            dtype=torch.int32,
                            device=device,
                        ),
                    ]
                )
        if max_seqlen_override is not None:
            max_seqlen = max_seqlen_override
        else:
            max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
        return {
            "rope_freqs_cis": rope_freqs_cis,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32, device=device),
        }

    def prepare_encoder_cudagraph_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int,
        max_seqlen_override: int | None = None,
        device: torch.device,
    ) -> dict[str, torch.Tensor | None]:
        """Precompute fixed-buffer metadata for image encoder CUDA graphs."""
        metadata = self.prepare_encoder_metadata(
            grid_thw_list,
            device=device,
            max_batch_size=max_batch_size,
            max_seqlen_override=max_seqlen_override,
        )
        metadata["pos_embeds"] = self.patch_embed.pos_emb.get_pos_embeds(
            grid_thw_list
        ).to(device=device)
        merge_gather_idx = build_image_merge_gather_idx(
            grid_thw_list, self.merge_kernel_size
        )
        metadata["merge_gather_idx"] = torch.from_numpy(merge_gather_idx).to(
            device=device, non_blocking=True
        )
        return metadata

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        hidden = self.config.hidden_size
        for name, loaded_weight in weights:
            if name.endswith("output_attn_res.gate_proj.weight"):
                target = name.replace(
                    "output_attn_res.gate_proj.weight",
                    "output_attn_res.proj_weight",
                )
                with torch.no_grad():
                    params_dict[target][: 3 * hidden].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith("output_attn_res.gate_proj.bias"):
                target = name.replace(
                    "output_attn_res.gate_proj.bias",
                    "output_attn_res.proj_weight",
                )
                with torch.no_grad():
                    # The fused proj_weight stores the gate bias as a per-row
                    # (3H, H) block; broadcast the (3H,) bias across columns.
                    params_dict[target][3 * hidden : 6 * hidden].copy_(
                        loaded_weight.unsqueeze(1)
                    )
                loaded_params.add(target)
                continue
            if name.endswith("output_attn_res.q_proj"):
                target = name.replace(
                    "output_attn_res.q_proj", "output_attn_res.proj_weight"
                )
                with torch.no_grad():
                    params_dict[target][7 * hidden :].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith("output_attn_res.k_proj"):
                target = name.replace(
                    "output_attn_res.k_proj", "output_attn_res.proj_weight"
                )
                with torch.no_grad():
                    params_dict[target][6 * hidden : 7 * hidden].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith(".self_attention_attn_res.gate_proj.weight"):
                target = name.replace(
                    ".self_attention_attn_res.gate_proj.weight",
                    ".self_attention_attn_res.proj_weight",
                )
                with torch.no_grad():
                    params_dict[target][: 3 * hidden].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith(".self_attention_attn_res.gate_proj.bias"):
                target = name.replace(
                    ".self_attention_attn_res.gate_proj.bias",
                    ".self_attention_attn_res.proj_weight",
                )
                with torch.no_grad():
                    # The fused proj_weight stores the gate bias as a per-row
                    # (3H, H) block; broadcast the (3H,) bias across columns.
                    params_dict[target][3 * hidden : 6 * hidden].copy_(
                        loaded_weight.unsqueeze(1)
                    )
                loaded_params.add(target)
                continue
            if name.endswith(".self_attention_attn_res.q_proj"):
                target = name.replace(
                    ".self_attention_attn_res.q_proj",
                    ".self_attention_attn_res.proj_weight",
                )
                with torch.no_grad():
                    params_dict[target][7 * hidden :].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith(".self_attention_attn_res.k_proj"):
                target = name.replace(
                    ".self_attention_attn_res.k_proj",
                    ".self_attention_attn_res.proj_weight",
                )
                with torch.no_grad():
                    params_dict[target][6 * hidden : 7 * hidden].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith(".mlp_attn_res.gate_proj.weight"):
                target = name.replace(
                    ".mlp_attn_res.gate_proj.weight", ".mlp_attn_res.proj_weight"
                )
                with torch.no_grad():
                    params_dict[target][: 3 * hidden].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith(".mlp_attn_res.gate_proj.bias"):
                target = name.replace(
                    ".mlp_attn_res.gate_proj.bias", ".mlp_attn_res.proj_weight"
                )
                with torch.no_grad():
                    # The fused proj_weight stores the gate bias as a per-row
                    # (3H, H) block; broadcast the (3H,) bias across columns.
                    params_dict[target][3 * hidden : 6 * hidden].copy_(
                        loaded_weight.unsqueeze(1)
                    )
                loaded_params.add(target)
                continue
            if name.endswith(".mlp_attn_res.q_proj"):
                target = name.replace(
                    ".mlp_attn_res.q_proj", ".mlp_attn_res.proj_weight"
                )
                with torch.no_grad():
                    params_dict[target][7 * hidden :].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            if name.endswith(".mlp_attn_res.k_proj"):
                target = name.replace(
                    ".mlp_attn_res.k_proj", ".mlp_attn_res.proj_weight"
                )
                with torch.no_grad():
                    params_dict[target][6 * hidden : 7 * hidden].copy_(loaded_weight)
                loaded_params.add(target)
                continue
            # HF names norm1/norm2 for the pre-attn / pre-mlp norms; MoonViT uses
            # norm0/norm1. MLP2 names its projections fc0/fc1; HF uses fc1/fc2.
            mapped = name
            if mapped.endswith(".norm1.weight"):
                mapped = mapped.replace(".norm1.weight", ".norm0.weight")
            elif mapped.endswith(".norm2.weight"):
                mapped = mapped.replace(".norm2.weight", ".norm1.weight")
            elif mapped.endswith(".mlp.fc1.weight"):
                mapped = mapped.replace(".mlp.fc1.weight", ".mlp.fc0.weight")
            elif mapped.endswith(".mlp.fc2.weight"):
                mapped = mapped.replace(".mlp.fc2.weight", ".mlp.fc1.weight")
            elif mapped.endswith("patch_embed.pos_emb.position_embeddings"):
                mapped = mapped.replace(
                    "patch_embed.pos_emb.position_embeddings",
                    "patch_embed.pos_emb.weight",
                )
            if mapped not in params_dict:
                continue
            param = params_dict[mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped)
        return loaded_params


class ShensiVlPoolerCrossAttention(nn.Module):
    """Cross-attention without output projection (o_proj)."""

    def __init__(self, q_dim, kv_dim, num_heads):
        super().__init__()
        self.embed_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads (got "
                f"`embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = 0.0
        self.is_causal = False
        self.num_key_value_groups = 1
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.q_proj = nn.Linear(q_dim, q_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Input shape: Batch x Time x Channel"""
        batch_size, q_seq_length, embed_dim = queries.shape
        kv_seq_length = keys.shape[1]

        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        queries = queries.view(
            batch_size, q_seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, kv_seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = values.view(
            batch_size, kv_seq_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_output, attn_weights = cross_eager_attention_forward(
            self,
            queries,
            keys,
            values,
            attention_mask,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )
        attn_output = attn_output.reshape(
            batch_size, q_seq_length, embed_dim
        ).contiguous()
        return attn_output, attn_weights

    def run_exchange_impl(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        vision_prefix_sum: torch.Tensor,
        gate: torch.Tensor,
        evidence_mask: torch.Tensor | None,
        guidance_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The language stream is flat [T, hc, H]; collapse the hc axis before
        # any cross-attention. The packed vision sequence [T_v, H] is shared
        # across the batch, and the per-token masks keep each token attending
        # only to its own sample's patches.
        language_queries = prefix_sum + hidden_states
        language_queries = language_queries.mean(dim=1)
        retrieved_evidence = self.vision_to_language_attention(
            language_queries.unsqueeze(0),
            vision_prefix_sum.unsqueeze(0),
            vision_prefix_sum.unsqueeze(0),
            attention_mask=evidence_mask,
        )[0].squeeze(0)
        # Split the fused gate along the stream dimension: `hc_mult` evidence
        # gates followed by a single guidance gate.
        evidence_gate, guidance_gate = self.gate.split([gate.size(0) - 1, 1])
        hidden_states = hidden_states + torch.tanh(evidence_gate)[
            None, :, None
        ] * retrieved_evidence.unsqueeze(1)

        retrieved_guidance = self.language_to_vision_attention(
            vision_prefix_sum.unsqueeze(0),
            language_queries.unsqueeze(0),
            language_queries.unsqueeze(0),
            attention_mask=guidance_mask,
        )[0].squeeze(0)
        vision_prefix_sum = vision_prefix_sum + torch.tanh(guidance_gate) * (
            retrieved_guidance
        )
        return hidden_states, vision_prefix_sum


class ShensiVlReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        text_config = config.text_config
        self.gate = nn.Parameter(torch.ones(text_config.hc_mult))
        self.proj = nn.Linear(text_config.hidden_size, text_config.hidden_size)
        self.update_gate = nn.Linear(
            text_config.hidden_size * 2, text_config.hidden_size
        )

    def forward(
        self,
        retrieved_evidence: torch.Tensor,
        reasoning_state: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Continuous-latent reasoning (Coconut: chain of continuous thought): the
        # evidence refines a gated recurrent state, and the state plus evidence is
        # injected back into the language stream on every loop iteration.
        evolving = self.proj(retrieved_evidence)
        update_gate = torch.sigmoid(
            self.update_gate(torch.cat([evolving, reasoning_state], dim=-1))
        )
        reasoning_state = update_gate * reasoning_state + (1 - update_gate) * evolving

        if hidden_states is not None:
            injected = retrieved_evidence + reasoning_state
            hidden_states = hidden_states + torch.tanh(self.gate)[
                None, :, None
            ] * injected.unsqueeze(1)

        return reasoning_state, hidden_states


class ShensiVlCrossBlock(nn.Module):
    def __init__(
        self,
        config,
        vision_layers: Sequence[nn.Module],
        language_layers: Sequence[nn.Module],
        block_index: int = 0,
    ):
        super().__init__()
        self.block_index = block_index
        vision_config, text_config = config.vision_config, config.text_config
        # Layer instances are owned by the towers; blocks only reference slices of
        # them, so the slices stay plain tuples instead of being re-registered as
        # submodules.
        self.vision_layers = tuple(vision_layers)
        self.language_layers = tuple(language_layers)
        self.vision_to_language_attention = ShensiVlPoolerCrossAttention(
            text_config.hidden_size,
            vision_config.hidden_size,
            config.num_cross_attention_heads,
        )
        self.language_to_vision_attention = ShensiVlPoolerCrossAttention(
            vision_config.hidden_size,
            text_config.hidden_size,
            config.num_cross_attention_heads,
        )
        # Fused gates: the leading `hc_mult` entries gate the evidence injection per
        # stream, the trailing entry gates the guidance write-back.
        self.gate = nn.Parameter(torch.zeros(text_config.hc_mult + 1))

    def _run_vision_layers(
        self,
        vision_hidden_states,
        vision_residual,
        vision_prefix_sum,
        vision_cu_seqlens,
        vision_rope_freqs_cis,
        vision_max_seqlen,
        vision_sequence_lengths=None,
    ):
        if vision_hidden_states is not None:
            for layer in self.vision_layers:
                vision_hidden_states, vision_prefix_sum, vision_residual = layer(
                    vision_hidden_states,
                    residual=vision_residual,
                    prefix_sum=vision_prefix_sum,
                    cu_seqlens=vision_cu_seqlens,
                    rope_freqs_cis=vision_rope_freqs_cis,
                    max_seqlen=vision_max_seqlen,
                    sequence_lengths=vision_sequence_lengths,
                )
        return vision_hidden_states, vision_prefix_sum, vision_residual

    def _cross_modal_exchange(
        self,
        hidden_states,
        prefix_sum,
        vision_prefix_sum,
        evidence_mask,
        guidance_mask,
    ):
        if vision_prefix_sum is None:
            return hidden_states, vision_prefix_sum

        # Dispatch through the registered custom op (a graph splitting point):
        # the exchange must execute eagerly at runtime -- the captured pieces
        # would otherwise read the masks at their capture-time addresses, and
        # the fresh per-step masks from the cross-mask op would be stale. The
        # outputs are written into stable pre-allocated buffers (the
        # "with_output" convention) so the downstream captured pieces always
        # read current data instead of relying on allocator address reuse.
        empty = torch.empty(0, device=prefix_sum.device)
        bufs = getattr(self, "_exchange_out_buffers", None)
        need = (
            bufs is None
            or bufs[0].device != hidden_states.device
            or bufs[0].shape[0] < hidden_states.shape[0]
            or bufs[1].shape[0] < vision_prefix_sum.shape[0]
        )
        if need:
            hc = hidden_states.shape[1]
            hidden = hidden_states.shape[-1]
            max_tokens = max(
                hidden_states.shape[0],
                getattr(self, "_vision_state_max_patches", 2048),
            )
            max_v = max(
                vision_prefix_sum.shape[0],
                getattr(self, "_vision_state_max_patches", 2048),
            )
            bufs = (
                torch.empty(
                    max_tokens,
                    hc,
                    hidden,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
                torch.empty(
                    max_v,
                    vision_prefix_sum.shape[-1],
                    device=vision_prefix_sum.device,
                    dtype=vision_prefix_sum.dtype,
                ),
            )
            self._exchange_out_buffers = bufs
        return torch.ops.vllm.shensi_vl_cross_modal_exchange(
            hidden_states,
            prefix_sum,
            vision_prefix_sum,
            self.gate,
            evidence_mask if evidence_mask is not None else empty,
            guidance_mask if guidance_mask is not None else empty,
            bufs[0],
            bufs[1],
            self.block_index,
        )

    def run_exchange_impl(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        vision_prefix_sum: torch.Tensor,
        gate: torch.Tensor,
        evidence_mask: torch.Tensor | None,
        guidance_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The language stream is flat [T, hc, H]; collapse the hc axis before any
        # cross-attention. The packed vision sequence [T_v, H] is shared across the
        # batch, and the per-token masks below keep each token attending only to its
        # own sample's patches.
        language_queries = prefix_sum + (
            hidden_states if hidden_states is not None else prefix_sum
        )
        language_queries = language_queries.mean(dim=1)
        retrieved_evidence = self.vision_to_language_attention(
            language_queries.unsqueeze(0),
            vision_prefix_sum.unsqueeze(0),
            vision_prefix_sum.unsqueeze(0),
            attention_mask=evidence_mask,
        )[0].squeeze(0)
        if hidden_states is not None:
            # Split the fused gate along the stream dimension: `hc_mult` evidence
            # gates followed by a single guidance gate.
            evidence_gate, guidance_gate = self.gate.split([self.gate.size(0) - 1, 1])
            hidden_states = hidden_states + torch.tanh(evidence_gate)[
                None, :, None
            ] * retrieved_evidence.unsqueeze(1)

        retrieved_guidance = self.language_to_vision_attention(
            vision_prefix_sum.unsqueeze(0),
            language_queries.unsqueeze(0),
            language_queries.unsqueeze(0),
            attention_mask=guidance_mask,
        )[0].squeeze(0)
        # Each patch only sees its own clip's tokens through the guidance mask, so
        # no per-row scatter is needed on the flat layout.
        vision_prefix_sum = vision_prefix_sum + torch.tanh(guidance_gate) * (
            retrieved_guidance
        )

        return hidden_states, vision_prefix_sum

    def forward(
        self,
        hidden_states,
        residual,
        prefix_sum,
        positions,
        input_ids,
        vision_hidden_states=None,
        vision_residual=None,
        vision_prefix_sum=None,
        vision_cu_seqlens=None,
        vision_rope_freqs_cis=None,
        vision_max_seqlen=None,
        vision_sequence_lengths=None,
        evidence_mask=None,
        guidance_mask=None,
        num_attn_res_blocks: int = 0,
        run_vision_layers=True,
    ):
        # `num_attn_res_blocks` is only consumed by the loop block; accepted
        # here so the block loop can pass a uniform input set.
        del num_attn_res_blocks
        if run_vision_layers:
            vision_hidden_states, vision_prefix_sum, vision_residual = (
                self._run_vision_layers(
                    vision_hidden_states=vision_hidden_states,
                    vision_residual=vision_residual,
                    vision_prefix_sum=vision_prefix_sum,
                    vision_cu_seqlens=vision_cu_seqlens,
                    vision_rope_freqs_cis=vision_rope_freqs_cis,
                    vision_max_seqlen=vision_max_seqlen,
                    vision_sequence_lengths=vision_sequence_lengths,
                )
            )

        for layer in self.language_layers:
            hidden_states, prefix_sum, residual = layer(
                hidden_states,
                residual,
                prefix_sum,
                positions,
                input_ids,
            )

        hidden_states, vision_prefix_sum = self._cross_modal_exchange(
            hidden_states,
            prefix_sum,
            vision_prefix_sum,
            evidence_mask,
            guidance_mask,
        )

        return (
            hidden_states,
            prefix_sum,
            residual,
            vision_hidden_states,
            vision_prefix_sum,
            vision_residual,
        )


class ShensiVlLoopBlock(nn.Module):
    def __init__(
        self,
        config,
        vision_layers: Sequence[nn.Module],
        language_layers: Sequence[nn.Module],
        vision_output_attn_res: nn.Module,
        output_attn_res: nn.Module,
        vision_final_norm: nn.Module | None = None,
        block_index: int = 0,
    ):
        super().__init__()
        self.block_index = block_index
        vision_config, text_config = config.vision_config, config.text_config
        # Layer instances are owned by the towers; blocks only reference slices of
        # them, so the slices stay plain tuples instead of being re-registered as
        # submodules.
        self.vision_layers = tuple(vision_layers)
        self.language_layers = tuple(language_layers)
        self.vision_to_language_attention = ShensiVlPoolerCrossAttention(
            text_config.hidden_size,
            vision_config.hidden_size,
            config.num_cross_attention_heads,
        )
        object.__setattr__(self, "vision_output_attn_res", vision_output_attn_res)
        object.__setattr__(self, "output_attn_res", output_attn_res)
        # The final vision AttnRes applies the tower's final layernorm as its
        # output norm (modular reference: output_norm_weight=final_layernorm).
        object.__setattr__(self, "vision_final_norm", vision_final_norm)
        self.reasoning = ShensiVlReasoning(config)

    def _run_vision_layers(
        self,
        vision_hidden_states,
        vision_residual,
        vision_prefix_sum,
        vision_cu_seqlens,
        vision_rope_freqs_cis,
        vision_max_seqlen,
        vision_sequence_lengths=None,
    ):
        if vision_hidden_states is not None:
            for layer in self.vision_layers:
                vision_hidden_states, vision_prefix_sum, vision_residual = layer(
                    vision_hidden_states,
                    residual=vision_residual,
                    prefix_sum=vision_prefix_sum,
                    cu_seqlens=vision_cu_seqlens,
                    rope_freqs_cis=vision_rope_freqs_cis,
                    max_seqlen=vision_max_seqlen,
                    sequence_lengths=vision_sequence_lengths,
                )
        return vision_hidden_states, vision_prefix_sum, vision_residual

    def _vision_attn_res(
        self,
        module: ShensiAttentionResidual,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # AttnRes operates on [T, hc, D]; the vision stream is [T_v, D] with a
        # single implicit stream. The vision tower's final layernorm is applied
        # as the output norm, matching the modular reference.
        hidden = hidden_states.unsqueeze(1) if hidden_states is not None else None
        final_norm = getattr(self, "vision_final_norm", None)
        out, updated, blocks = module(
            hidden,
            residual.unsqueeze(1),
            prefix_sum.unsqueeze(1),
            output_norm_weight=final_norm.weight if final_norm is not None else None,
            num_blocks=num_blocks,
        )
        return out.squeeze(1), updated.squeeze(1), blocks.squeeze(1)

    def run_exchange_impl(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        vision_prefix_sum: torch.Tensor,
        gate: torch.Tensor,
        evidence_mask: torch.Tensor | None,
        guidance_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("the loop block only retrieves evidence")

    def retrieve_evidence(
        self,
        language_queries: torch.Tensor,
        vision_final: torch.Tensor,
        evidence_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.vision_to_language_attention(
            language_queries.unsqueeze(0),
            vision_final.unsqueeze(0),
            vision_final.unsqueeze(0),
            attention_mask=evidence_mask,
        )[0].squeeze(0)

    def _run_damped_language_layers(
        self,
        hidden_states,
        residual,
        prefix_sum,
        damping,
        positions,
        input_ids,
    ):
        # Damp the hyper-connection write-backs with `damping` for this loop
        # iteration. The weight-tied layers are shared with the other iterations, so
        # the temporary write-back override is restored in the `finally` block below.
        damped_connections = []
        if damping < 1.0:
            for layer in self.language_layers:
                for connection in (layer.attn_hc, layer.ffn_hc):
                    original_write_back = connection.write_back
                    damped_connections.append((connection, original_write_back))

                    def _damped_write_back(
                        hidden, output, _original=original_write_back
                    ):
                        return _original(hidden, damping * output)

                    connection.write_back = _damped_write_back
        try:
            for layer in self.language_layers:
                hidden_states, prefix_sum, residual = layer(
                    hidden_states,
                    residual,
                    prefix_sum,
                    positions,
                    input_ids,
                )
        finally:
            for connection, original_write_back in damped_connections:
                connection.write_back = original_write_back
        return hidden_states, prefix_sum, residual

    def forward(
        self,
        hidden_states,
        residual,
        prefix_sum,
        positions,
        input_ids,
        vision_hidden_states=None,
        vision_residual=None,
        vision_prefix_sum=None,
        vision_cu_seqlens=None,
        vision_rope_freqs_cis=None,
        vision_max_seqlen=None,
        vision_sequence_lengths=None,
        evidence_mask=None,
        num_attn_res_blocks: int = 0,
        run_vision_layers=True,
    ):
        if run_vision_layers:
            vision_hidden_states, vision_prefix_sum, vision_residual = (
                self._run_vision_layers(
                    vision_hidden_states=vision_hidden_states,
                    vision_residual=vision_residual,
                    vision_prefix_sum=vision_prefix_sum,
                    vision_cu_seqlens=vision_cu_seqlens,
                    vision_rope_freqs_cis=vision_rope_freqs_cis,
                    vision_max_seqlen=vision_max_seqlen,
                    vision_sequence_lengths=vision_sequence_lengths,
                )
            )

        vision_final = (
            self._vision_attn_res(
                self.vision_output_attn_res,
                vision_hidden_states,
                vision_residual,
                vision_prefix_sum,
                num_attn_res_blocks,
            )[0]
            if vision_hidden_states is not None
            else None
        )

        initial_queries = (
            prefix_sum + (hidden_states if hidden_states is not None else prefix_sum)
        ).mean(dim=1)
        reasoning_state = torch.zeros_like(initial_queries)

        previous = None
        first_delta = None
        hidden_delta = None
        first_reasoning_delta = None
        reasoning_delta = None
        previous_reasoning_state = None
        delta_history = deque(maxlen=8)
        orbit_history = deque(maxlen=8)
        orbit_counts = deque(maxlen=8)
        while True:
            # Fixed damped write-back: each iteration refines the state by a
            # geometrically decaying amount, so the hidden delta below shrinks by a
            # constant factor per iteration; once it has dropped three orders of
            # magnitude from the first iteration the iterate is at the loop's fixed
            # point -- that stall is the stop. Two joint guards cover non-contractive
            # dynamics: the reasoning state settling while the stream still steps at
            # half its initial rate (a cycle under a settled thought), and the orbit
            # returning near a recent iterate three times within the last eight
            # iterations (limit-cycle lock-in).
            damping = 0.25
            converged = first_delta is not None and hidden_delta <= 1e-3 * first_delta
            thought_settled = (
                first_reasoning_delta is not None
                and reasoning_delta <= 1e-1 * first_reasoning_delta
            )
            stream_stalled = delta_history and max(delta_history) >= 0.5 * first_delta
            stop = (
                converged
                or (thought_settled and stream_stalled)
                or sum(orbit_counts) >= 3
            )
            _spin = getattr(self, "_spin_count", 0) + 1
            self._spin_count = _spin

            # Reasoning runs on every iteration: evidence is the retrieved visual
            # memory when vision is present, otherwise the language state itself
            # self-references.
            language_queries = (
                prefix_sum
                + (hidden_states if hidden_states is not None else prefix_sum)
            ).mean(dim=1)
            if vision_final is not None:
                empty = torch.empty(0, device=language_queries.device)
                retrieved_evidence = torch.ops.vllm.shensi_vl_loop_evidence(
                    language_queries,
                    vision_final,
                    evidence_mask if evidence_mask is not None else empty,
                    self.block_index,
                )
            else:
                retrieved_evidence = language_queries

            reasoning_state, hidden_states = self.reasoning(
                retrieved_evidence, reasoning_state, hidden_states
            )

            hidden_states, prefix_sum, residual = self._run_damped_language_layers(
                hidden_states,
                residual,
                prefix_sum,
                damping,
                positions,
                input_ids,
            )
            if previous is not None:
                hidden_delta = (hidden_states - previous).abs().max()
                if first_delta is None:
                    first_delta = hidden_delta
                delta_history.append(hidden_delta)
            previous = hidden_states
            if previous_reasoning_state is not None:
                reasoning_delta = (
                    (reasoning_state - previous_reasoning_state).abs().max()
                )
                if first_reasoning_delta is None:
                    first_reasoning_delta = reasoning_delta
            previous_reasoning_state = reasoning_state
            if orbit_history:
                orbit_return = min(
                    (hidden_states.detach() - past).abs().max()
                    for past in orbit_history
                )
                orbit_history.append(hidden_states.detach())
                orbit_counts.append(1 if orbit_return <= 1e-1 * first_delta else 0)
            else:
                orbit_history.append(hidden_states.detach())
            # Hard iteration cap: the convergence guards compare device
            # deltas, which never fire on NaN iterates (e.g. corrupted graph
            # replay); a fixed bound keeps the block from spinning forever.
            if stop or _spin >= 32:
                break

        hidden_states, _, _ = self.output_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            output_norm_weight=None,
            num_blocks=num_attn_res_blocks,
        )

        return (
            hidden_states,
            prefix_sum,
            residual,
            vision_hidden_states,
            vision_prefix_sum,
            vision_residual,
        )


class ShensiVlLanguageModel(nn.Module):
    """Language backbone: reuses the vLLM Shensi model verbatim.

    The DeepRecur blocks slice ``layers`` and drive the layer loop themselves;
    ``ShensiModel.forward`` is not used here.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config.get_text_config()
        self.model = ShensiModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    @property
    def embed_tokens(self) -> nn.Module:
        return self.model.embed_tokens

    @property
    def layers(self):
        return self.model.layers

    @property
    def norm(self) -> nn.Module:
        return self.model.norm

    @property
    def hc_head(self) -> nn.Module:
        return self.model.hc_head

    @property
    def output_attn_res(self) -> nn.Module:
        return self.model.output_attn_res

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The checkpoint's language stack is "model.language_model.layers.*":
        # the AutoWeightsLoader has stripped "language_model.", and the inner
        # ShensiModel's parameters are relative, so drop the remaining "model."
        # segment before delegating, and re-prefix the reported names so the
        # loader's initialization check matches "model.layers.*".
        stripped = (
            (name[len("model.") :] if name.startswith("model.") else name, tensor)
            for name, tensor in weights
        )
        loaded = self.model.load_weights(stripped)
        return {"model." + name for name in loaded}


@MULTIMODAL_REGISTRY.register_processor(
    ShensiVlMultiModalProcessor,
    info=ShensiVlProcessingInfo,
    dummy_inputs=ShensiVlDummyInputsBuilder,
)
class ShensiVlForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
):
    """Shensi-VL: vision-language model with DeepRecur blocks."""

    supports_eagle3: ClassVar[Literal[True]] = True
    supports_encoder_cudagraph: ClassVar[Literal[True]] = True

    # -- encoder CUDA graph (SupportsEncoderCudaGraph) -----------------------
    # The captured region is the DeepRecur round-0 vision pass: patch embed,
    # the first block's vision layers (which allocate the AttnRes state), the
    # packed merge and the projector. The DeepRecur state (packed patches,
    # prefix sum, AttnRes slots) is written into pre-allocated instance
    # buffers inside the graph; replay refreshes those buffers and the
    # language blocks read them through ``self._vision_state``.

    def _vision_state_buffers(
        self,
        num_patches: int,
        num_seqs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Persistent DeepRecur state + metadata slots.

        The captured decoder pieces and the encoder graph reference these
        addresses; replay refreshes the *contents*
        (``prepare_encoder_cudagraph_replay_buffers`` for the metadata, the
        encoder graph's trailing copies for the hidden state), so the buffers
        must never be reallocated while a graph is live. Allocation is
        grow-only and happens at capture-preparation time (or on the first
        eager run), so eager mode is unaffected.
        """
        buffers = getattr(self, "_vision_state_slots", None)
        need_realloc = (
            buffers is None
            or buffers[0].device != device
            or buffers[0].dtype != dtype
            or buffers[0].shape[0] < num_patches
            or buffers[3].shape[0] < num_seqs + 2
        )
        if need_realloc:
            # Allocate at the scheduler's upper bounds so the addresses never
            # change once a CUDA graph has been captured against them (the
            # decoder pieces capture before the encoder graph preparation).
            num_patches = max(
                num_patches, getattr(self, "_vision_state_max_patches", num_patches)
            )
            num_seqs = max(num_seqs, getattr(self, "_vision_state_max_seqs", num_seqs))
            hidden = self.config.vision_config.hidden_size
            vision_config = self.config.vision_config
            rope_dim = (
                vision_config.qkv_hidden_size // vision_config.num_attention_heads
            ) // 2
            blocks = self.num_attn_res_blocks

            def alloc(*shape: int, dt: torch.dtype | None = None) -> torch.Tensor:
                return torch.zeros(*shape, device=device, dtype=dt or dtype)

            buffers = (
                alloc(num_patches, hidden),
                alloc(num_patches, hidden),
                alloc(num_patches, blocks, hidden),
                alloc(num_seqs + 2, dt=torch.int32),  # cu_seqlens (+pad seqs)
                alloc(num_patches, rope_dim, dt=torch.complex64),  # freqs_cis
                alloc(num_seqs, dt=torch.int32),  # sequence_lengths
            )
            self._vision_state_slots = buffers
        return buffers

    def get_encoder_cudagraph_config(self) -> "EncoderCudaGraphConfig":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        pad_totals = getattr(self, "_encoder_cudagraph_pad_totals", {})

        def pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
            # Varlen attention requires cu_seqlens[-1] to equal the number of
            # rows actually passed in; complete the tail with one padding
            # sequence covering rows the real batch does not fill.
            total = pad_totals.get(dst.data_ptr())
            n = min(src.shape[0], dst.shape[0])
            dst[:n].copy_(src[:n])
            dst[n:] = total if total is not None else src[-1]

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "pixel_values",
                "pos_embeds",
                "rope_freqs_cis",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
            ],
            out_hidden_size=self.config.text_config.hidden_size,
            padding_logics={"cu_seqlens": pad_cu_seqlens},
        )

    def get_input_modality(self, mm_kwargs: dict[str, object]) -> str:
        return "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        kh, kw = self.vision_tower.merge_kernel_size
        specs = []
        for _t, h, w in self._get_image_grid_thw_list(mm_kwargs):
            specs.append(
                EncoderItemSpec(
                    input_size=h * w,
                    output_tokens=(h // kh) * (w // kw),
                )
            )
        return specs

    def _get_image_grid_thw_list(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid = mm_kwargs["image_grid_thw"]
        if isinstance(grid, torch.Tensor):
            return [[int(x) for x in row] for row in grid.tolist()]
        return [[int(x) for x in row] for row in grid]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thw_list = self._get_image_grid_thw_list(mm_kwargs)
        pixel_values = mm_kwargs["pixel_values"]

        if len(indices) == 0:
            return {
                "pixel_values": pixel_values[:0],
                "image_grid_thw": pixel_values.new_zeros((0, 3), dtype=torch.long),
            }

        patch_counts = [h * w for _t, h, w in grid_thw_list]
        cum = [0]
        for pc in patch_counts:
            cum.append(cum[-1] + pc)

        selected_pv = torch.cat(
            [pixel_values[cum[i] : cum[i + 1]] for i in indices], dim=0
        )
        selected_grid = torch.tensor(
            [grid_thw_list[i] for i in indices],
            dtype=torch.long,
            device=pixel_values.device,
        )
        return {"pixel_values": selected_pv, "image_grid_thw": selected_grid}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
    ) -> "EncoderCudaGraphCaptureInputs":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphCaptureInputs

        kh, kw = self.vision_tower.merge_kernel_size
        per_item_out = (token_budget + max_batch_size - 1) // max_batch_size

        rope = self.vision_tower.rope_2d
        max_wo = rope.max_width // kw
        wo = min(per_item_out, max_wo)
        ho = (per_item_out + wo - 1) // wo
        assert ho * kh <= rope.max_height, (
            f"per_item_out={per_item_out} exceeds RoPE grid capacity "
            f"(max {(rope.max_height // kh) * (rope.max_width // kw)} tokens)"
        )

        grid_thw_list = [[1, ho * kh, wo * kw] for _ in range(max_batch_size)]
        ps = self.vision_tower.patch_size
        if isinstance(ps, int):
            ps = (ps, ps)
        total_patches = max_batch_size * ho * kh * wo * kw
        dummy_pixel_values = torch.zeros(
            total_patches, 3, ps[0], ps[1], device=device, dtype=dtype
        )

        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thw_list,
            max_batch_size=max_batch_size + 1,
            max_seqlen_override=total_patches,
            device=device,
        )

        # Pre-allocate the persistent DeepRecur state buffers for this
        # path/budget; the captured kernels reference these addresses and the
        # replay preparation refreshes their contents. Also stash the host
        # values the captured forward must not re-read from device tensors
        # (grid list and the max-seqlen kernel launch bound).
        self._vision_state_buffers(total_patches, max_batch_size + 2, device, dtype)
        self._encoder_cudagraph_active_path = path
        self._encoder_cudagraph_total_patches = total_patches
        self._encoder_cudagraph_capture_grid = grid_thw_list
        self._encoder_cudagraph_grid_list = grid_thw_list
        max_seqlen = metadata["max_seqlen"]
        self._encoder_cudagraph_max_seqlen_int = (
            int(max_seqlen.item())
            if isinstance(max_seqlen, torch.Tensor)
            else int(max_seqlen)
        )

        values: dict[str, torch.Tensor] = {"pixel_values": dummy_pixel_values}
        values.update({k: v for k, v in metadata.items() if v is not None})

        cu_seqlens = values.get("cu_seqlens")
        if cu_seqlens is not None:
            pad_totals = getattr(self, "_encoder_cudagraph_pad_totals", None)
            if pad_totals is None:
                pad_totals = {}
                self._encoder_cudagraph_pad_totals = pad_totals
            pad_totals[cu_seqlens.data_ptr()] = total_patches

        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> "EncoderCudaGraphReplayBuffers":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphReplayBuffers

        grid_thw_list = self._get_image_grid_thw_list(mm_kwargs)
        pixel_values = mm_kwargs["pixel_values"]
        real_patches = sum(h * w for _t, h, w in grid_thw_list)
        num_seqs = len(grid_thw_list)

        metadata = self.vision_tower.prepare_encoder_cudagraph_metadata(
            grid_thw_list,
            max_batch_size=None,
            device=pixel_values.device,
        )

        # Refresh the persistent state/metadata slots before the graphs
        # replay; the captured kernels reference these addresses, so only the
        # contents may change. The hidden-state slots are refreshed by the
        # encoder graph's own trailing copies.
        buffers = self._vision_state_buffers(
            real_patches, num_seqs, pixel_values.device, pixel_values.dtype
        )
        cu_seqlens = metadata["cu_seqlens"]
        n_cu = 0
        if cu_seqlens is not None:
            n_cu = min(cu_seqlens.shape[0], buffers[3].shape[0])
            buffers[3][:n_cu].copy_(cu_seqlens[:n_cu])
            if n_cu < buffers[3].shape[0]:
                # Zero-length tail sequences cover the captured batch slots.
                buffers[3][n_cu:] = int(cu_seqlens[-1])
        rope = metadata["rope_freqs_cis"]
        if rope is not None:
            buffers[4][: rope.shape[0]].copy_(rope)
        seq_lens = metadata.get("sequence_lengths")
        if seq_lens is not None:
            buffers[5][: seq_lens.shape[0]].copy_(seq_lens)
        max_seqlen = metadata["max_seqlen"]
        max_seqlen_int = (
            int(max_seqlen.item())
            if isinstance(max_seqlen, torch.Tensor)
            else int(max_seqlen)
        )
        self._vision_state = {
            "hidden_states": buffers[0][:real_patches],
            "prefix_sum": buffers[1][:real_patches],
            "residual": buffers[2][:real_patches],
            "cu_seqlens": buffers[3][:n_cu],
            "rope_freqs_cis": (
                buffers[4][: rope.shape[0]] if rope is not None else None
            ),
            "max_seqlen": max_seqlen_int,
            "sequence_lengths": (
                buffers[5][: seq_lens.shape[0]] if seq_lens is not None else None
            ),
            "patch_counts": [h * w for _t, h, w in grid_thw_list],
        }

        values: dict[str, torch.Tensor | None] = {"pixel_values": pixel_values}
        values.update(metadata)
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self, inputs: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        pixel_values = inputs.pop("pixel_values")
        metadata = inputs
        if "max_seqlen" in metadata:
            # Static launch bound computed at capture-preparation time; the
            # tensor form would require a host sync inside the capture.
            metadata = {
                **metadata,
                "max_seqlen": self._encoder_cudagraph_max_seqlen_int,
            }

        # The grid tensor is not passed into the captured forward: rebuilding
        # it from the host list would be a H2D copy inside the capture. The
        # stashed grid list (set at capture preparation) drives the layout.
        embeddings = self.embed_multimodal_impl(
            pixel_values=pixel_values,
            image_grid_thw=None,
            encoder_metadata=metadata,
        )
        if isinstance(embeddings, list):
            embeddings = embeddings[0]
        return embeddings.view(-1, self.config.text_config.hidden_size)

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        image_input = self._parse_and_validate_image_input(**mm_kwargs)
        assert image_input is not None
        embeddings = self.embed_multimodal_impl(
            pixel_values=image_input["pixel_values"],
            image_grid_thw=image_input["grid_thws"],
        )
        return torch.cat([e.view(-1, e.shape[-1]) for e in embeddings], dim=0)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = ShensiVlVisionModel(
                config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if not hasattr(config.vision_config, "mm_projector_type"):
                config.vision_config.mm_projector_type = "patchmergerv2"
            # The Kimi-K2.5 projector derives its output dim from
            # text_hidden_size (falling back to mm_hidden_size); Shensi-VL
            # names it via text_config.
            if not hasattr(config.vision_config, "text_hidden_size"):
                config.vision_config.text_hidden_size = config.text_config.hidden_size
            if not hasattr(config.vision_config, "mm_hidden_size"):
                config.vision_config.mm_hidden_size = config.vision_config.hidden_size
            if not hasattr(config.vision_config, "projector_ln_eps"):
                config.vision_config.projector_ln_eps = 1.0e-5
            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = ShensiVlLanguageModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
            if get_pp_group().is_last_rank:
                self.lm_head = ParallelLMHead(
                    config.text_config.vocab_size,
                    config.text_config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                # The HF checkpoint ties the language-model output head to the
                # input embeddings ("model.language_model.embed_tokens.weight"
                # via `_tied_weights_keys`), so it has no lm_head tensor of its
                # own. Mirror the tie here or the head stays zero-initialized.
                if getattr(config, "tie_word_embeddings", False):
                    self.lm_head = self.lm_head.tie_weights(
                        self.language_model.embed_tokens
                    )
            else:
                self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.text_config.vocab_size)
        self.make_empty_intermediate_tensors = None

        # The op implementation needs the active pooler module; set the handle
        # and register the DeepRecur ops as graph splitting points so the
        # captured graphs never freeze the cross-modal branch or the
        # data-dependent loop, and the per-request cross masks stay eager.
        compilation_config = vllm_config.compilation_config
        for op_name in (
            "vllm::shensi_vl_cross_modal_exchange",
            "vllm::shensi_vl_loop_evidence",
            "vllm::shensi_vl_cross_mask",
            "vllm::shensi_vl_loop_block",
        ):
            if compilation_config.splitting_ops is None:
                compilation_config.splitting_ops = [op_name]
            elif op_name not in compilation_config.splitting_ops:
                compilation_config.splitting_ops.append(op_name)

        # Upper bounds for the persistent DeepRecur state buffers: the
        # scheduler never processes more encoder patches or sequences per step
        # than these, so the buffer addresses stay fixed for the process.
        scheduler_config = vllm_config.scheduler_config
        self._vision_state_max_patches = max(
            scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )
        self._vision_state_max_seqs = scheduler_config.max_num_seqs

        # The HF checkpoint nests everything under "model." while the vLLM
        # layout is flat; the language stack also carries an inner ShensiModel
        # ("language_model.model.layers.*"). Redirect with a prefix mapper.
        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"model.": ""},
            orig_to_new_substr={
                "language_model.layers": "language_model.model.layers",
            },
        )

        vision_config, text_config = config.vision_config, config.text_config
        if vision_config.attn_res_block_layer_types.count(
            "block_write_layer"
        ) != text_config.attn_res_block_layer_types.count("block_write_layer"):
            raise ValueError(
                "DeepRecur requires the vision and language stacks to have the "
                "same number of attention-residual blocks."
            )
        self.num_attn_res_blocks = vision_config.attn_res_block_layer_types.count(
            "block_write_layer"
        )

        def split_blocks(layers, layer_types):
            write_indices = [
                i for i, t in enumerate(layer_types) if t == "block_write_layer"
            ]
            return [
                list(layers[start:end])
                for start, end in zip(write_indices, write_indices[1:] + [len(layers)])
            ]

        vision_chunks = split_blocks(
            self.vision_tower.layers, vision_config.attn_res_block_layer_types
        )
        language_chunks = split_blocks(
            self.language_model.layers, text_config.attn_res_block_layer_types
        )
        if not (len(vision_chunks) == len(language_chunks) == self.num_attn_res_blocks):
            raise ValueError(
                "The vision and language layer chunks must each match the "
                "number of attention-residual blocks."
            )
        self.blocks = nn.ModuleList(
            [
                ShensiVlCrossBlock(
                    config,
                    vision_chunks[index],
                    language_chunks[index],
                    block_index=index,
                )
                for index in range(self.num_attn_res_blocks - 1)
            ]
            + [
                ShensiVlLoopBlock(
                    config,
                    vision_chunks[-1],
                    language_chunks[-1],
                    self.vision_tower.output_attn_res,
                    self.language_model.output_attn_res,
                    vision_final_norm=self.vision_tower.final_layernorm,
                    block_index=self.num_attn_res_blocks - 1,
                )
            ]
        )
        # Register the blocks for the DeepRecur custom ops (graph splitting
        # points; the op dispatches by block index so each block uses its own
        # pooler weights).
        for block in self.blocks:
            _CROSS_EXCHANGE_REGISTRY[block.block_index] = block

    # -- multimodal (multimodal.md: validate -> process -> embeddings) -------

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> KimiK25MediaPixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if image_grid_thw is None:
            image_grid_thw = kwargs.pop("grid_thw", None)
        if pixel_values is None:
            return None

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(cast(list[torch.Tensor], pixel_values), dim=0)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError(
                "pixel_values must be a tensor or a list of tensors, "
                f"got {type(pixel_values)}"
            )

        target_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(target_dtype)
        assert isinstance(image_grid_thw, torch.Tensor), (
            f"expect image_grid_thw to be a tensor, got {type(image_grid_thw)}"
        )
        image_grid_thw = image_grid_thw.reshape(-1, image_grid_thw.shape[-1])
        assert image_grid_thw.ndim == 2 and image_grid_thw.size(1) == 3, (
            f"unexpected shape for image_grid_thw: {image_grid_thw.shape}"
        )

        return KimiK25MediaPixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            grid_thws=image_grid_thw,
        )

    def _process_image_input(
        self, image_input: KimiK25MediaPixelInputs
    ) -> MultiModalEmbeddings:
        # DeepRecur's visual state (packed patches + AttnRes slots) is computed
        # alongside the round-0 embeddings and cached on this model for the
        # language blocks' cross-modal exchange.
        return self.embed_multimodal_impl(
            pixel_values=image_input["pixel_values"],
            image_grid_thw=image_input["grid_thws"],
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    # -- DeepRecur vision pipeline (round 0 + state caching) ------------------

    def embed_multimodal_impl(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
    ) -> MultiModalEmbeddings:
        """Run the round-0 vision stack and project patches into text space.

        vLLM scatters the returned per-image embeddings into the placeholder
        positions of ``inputs_embeds`` before calling ``forward``; the visual
        DeepRecur state (packed patches, residual slots and encoder metadata)
        is cached here for the language blocks' cross-modal exchange.
        """
        if pixel_values is None:
            return []
        grid_list = getattr(self, "_encoder_cudagraph_grid_list", None)
        if image_grid_thw is None and grid_list is None:
            return []

        vision_tower = self.vision_tower
        # The grid list is read on the host; inside a captured encoder graph
        # the grid tensor is a fixed capture buffer, so use the list stashed
        # at capture-preparation time instead of syncing.
        grid_thw_list = grid_list if grid_list is not None else image_grid_thw.tolist()
        if encoder_metadata is not None and "pos_embeds" in encoder_metadata:
            vision_hidden_states = vision_tower.patch_embed(
                pixel_values, None, pos_embeds=encoder_metadata["pos_embeds"]
            )
        else:
            vision_hidden_states = vision_tower.patch_embed(pixel_values, grid_thw_list)
        if encoder_metadata is None:
            encoder_metadata = vision_tower.prepare_encoder_metadata(
                grid_thw_list, device=vision_hidden_states.device
            )
        max_seqlen = encoder_metadata["max_seqlen"]
        if isinstance(max_seqlen, torch.Tensor):
            # Eager path: reduce once on the host. The captured path receives
            # the precomputed int from ``encoder_cudagraph_forward`` so no
            # sync happens inside the capture.
            max_seqlen = int(max_seqlen.item())
        vision_residual = vision_hidden_states.new_zeros(
            vision_hidden_states.size(0),
            self.num_attn_res_blocks,
            vision_hidden_states.size(-1),
        )
        vision_prefix_sum = vision_hidden_states
        vision_hidden_states = None  # first vision layer sees delta=None

        # Round 0: the first block's vision layers, before placeholder filling.
        for layer in self.blocks[0].vision_layers:
            vision_hidden_states, vision_prefix_sum, vision_residual = layer(
                vision_hidden_states,
                residual=vision_residual,
                prefix_sum=vision_prefix_sum,
                cu_seqlens=encoder_metadata["cu_seqlens"],
                rope_freqs_cis=encoder_metadata["rope_freqs_cis"],
                max_seqlen=max_seqlen,
                sequence_lengths=encoder_metadata.get("sequence_lengths"),
            )

        vision_config = self.config.vision_config
        kernel_height, kernel_width = vision_config.merge_kernel_size
        merge_gather_idx = (
            encoder_metadata.get("merge_gather_idx")
            if encoder_metadata is not None
            else None
        )
        if merge_gather_idx is not None:
            # Image-only packed merge (CUDA graph path): a single gather.
            merged = tpool_patch_merger_packed(vision_hidden_states, merge_gather_idx)
            projected = [self.mm_projector(merged).to(vision_prefix_sum.dtype)]
        else:
            pooled_images = tpool_patch_merger(
                vision_hidden_states,
                image_grid_thw,
                merge_kernel_size=(kernel_height, kernel_width),
            )
            projected = [
                self.mm_projector(pooled).to(vision_prefix_sum.dtype)
                for pooled in pooled_images
            ]

        image_patch_counts = [t * h * w for t, h, w in grid_thw_list]
        # Persist the round-0 state at stable addresses: the captured decoder
        # pieces and the encoder graph reference these slots, and replay only
        # refreshes the contents (the encoder graph's recorded copies for the
        # hidden state, ``prepare_encoder_cudagraph_replay_buffers`` for the
        # metadata).
        num_patches = vision_hidden_states.shape[0]
        num_seqs = len(grid_thw_list)
        buffers = self._vision_state_buffers(
            num_patches,
            num_seqs,
            vision_hidden_states.device,
            vision_hidden_states.dtype,
        )
        buffers[0][:num_patches].copy_(vision_hidden_states)
        buffers[1][:num_patches].copy_(vision_prefix_sum)
        buffers[2][:num_patches].copy_(vision_residual)
        cu_seqlens = encoder_metadata["cu_seqlens"]
        if cu_seqlens is not None:
            buffers[3][: cu_seqlens.shape[0]].copy_(cu_seqlens)
        rope_freqs_cis = encoder_metadata["rope_freqs_cis"]
        if rope_freqs_cis is not None:
            buffers[4][: rope_freqs_cis.shape[0]].copy_(rope_freqs_cis)
        sequence_lengths = encoder_metadata.get("sequence_lengths")
        if sequence_lengths is not None:
            buffers[5][: sequence_lengths.shape[0]].copy_(sequence_lengths)
        self._vision_state = {
            "hidden_states": buffers[0][:num_patches],
            "prefix_sum": buffers[1][:num_patches],
            "residual": buffers[2][:num_patches],
            "cu_seqlens": (
                buffers[3][: cu_seqlens.shape[0]] if cu_seqlens is not None else None
            ),
            "rope_freqs_cis": (
                buffers[4][: rope_freqs_cis.shape[0]]
                if rope_freqs_cis is not None
                else None
            ),
            "max_seqlen": max_seqlen,
            "sequence_lengths": (
                buffers[5][: sequence_lengths.shape[0]]
                if sequence_lengths is not None
                else None
            ),
            "patch_counts": image_patch_counts,
        }
        return projected

    def forward_language(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # ``inputs_embeds`` already carries the projected visual features at the
        # multimodal placeholder positions (scattered by the vLLM runner from
        # ``embed_multimodal``'s output).
        if input_ids is None:
            # The hash-MoE layers need token ids for their deep-embedding
            # lookup; placeholder ids are zero, mirroring the HF convention.
            input_ids = torch.zeros_like(positions)
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        sequence_length = inputs_embeds.shape[0]
        embed_dtype = inputs_embeds.dtype

        # The cached DeepRecur vision state persists from the request's
        # prefill (the encoder only runs when the batch carries images);
        # decode steps keep consuming it. Text-only steps must not reuse a
        # previous request's patches: the runner calls ``embed_multimodal``
        # only for image batches, so the state is only refreshed on image
        # steps; mixed text/image batching is handled by the scheduler
        # running image prefills through the encoder runner per batch.
        vision_state = getattr(self, "_vision_state", None)
        vision_hidden_states = vision_residual = vision_prefix_sum = None
        vision_cu_seqlens = vision_rope_freqs_cis = vision_max_seqlen = None
        vision_sequence_lengths = None
        vision_evidence_mask = vision_guidance_mask = None

        if vision_state is not None:
            vision_hidden_states = vision_state["hidden_states"]
            vision_prefix_sum = vision_state["prefix_sum"]
            vision_residual = vision_state["residual"]
            vision_cu_seqlens = vision_state["cu_seqlens"]
            vision_rope_freqs_cis = vision_state["rope_freqs_cis"]
            vision_max_seqlen = vision_state["max_seqlen"]
            vision_sequence_lengths = vision_state.get("sequence_lengths")
            # Cross-attention masks over the packed patch set. Clip ids
            # increment in packing order; the language token layout is
            # recovered from the attention metadata's cumulative seq lengths.
            # Runs eagerly (splitting op): the masks depend on the per-request
            # patch counts, so they must not be frozen into a captured graph.
            attn_metadata = get_forward_context().attn_metadata
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            if query_start_loc is not None:
                vision_evidence_mask, vision_guidance_mask = (
                    torch.ops.vllm.shensi_vl_cross_mask(
                        input_ids,
                        vision_state["patch_counts"],
                        query_start_loc,
                        sequence_length,
                        embed_dtype,
                        self.config.image_token_id,
                    )
                )

        # Language state: expand the hc streams and initialise the per-block
        # residual slots, then run the DeepRecur blocks in place of the language
        # model's own layer loop.
        language_model = self.language_model
        num_streams = self.config.text_config.hc_mult
        hidden_states = (
            inputs_embeds.unsqueeze(1).expand(-1, num_streams, -1).contiguous()
        )
        block_residual = hidden_states.new_zeros(
            hidden_states.size(0),
            num_streams,
            self.num_attn_res_blocks,
            hidden_states.size(-1),
        )
        prefix_sum = hidden_states
        hidden_states = None
        residual = block_residual

        # Run the DeepRecur blocks. Block 0's vision layers already ran in
        # embed_multimodal, so it receives the full state but skips them; later
        # blocks run theirs on the evolved state.
        for block_index, block in enumerate(self.blocks):
            block_inputs = {
                "hidden_states": hidden_states,
                "residual": residual,
                "prefix_sum": prefix_sum,
                "positions": positions,
                "input_ids": input_ids,
                "vision_hidden_states": vision_hidden_states,
                "vision_prefix_sum": vision_prefix_sum,
                "vision_residual": vision_residual,
                "vision_cu_seqlens": vision_cu_seqlens,
                "vision_rope_freqs_cis": vision_rope_freqs_cis,
                "vision_max_seqlen": vision_max_seqlen,
                "vision_sequence_lengths": vision_sequence_lengths,
                "evidence_mask": vision_evidence_mask,
                "num_attn_res_blocks": self.num_attn_res_blocks,
                "run_vision_layers": (block_index > 0),
            }
            if isinstance(block, ShensiVlCrossBlock):
                # Guidance exchange exists only in cross blocks; the loop block
                # consumes finalized evidence and never writes back into the vision
                # state.
                block_inputs.update(guidance_mask=vision_guidance_mask)
            if isinstance(block, ShensiVlLoopBlock):
                # The loop block's convergence guards read device scalars on
                # the host and its iteration count varies per step; run it
                # eagerly through a splitting op so the surrounding cross
                # blocks stay captured.
                (
                    hidden_states,
                    prefix_sum,
                    residual,
                    vision_hidden_states,
                    vision_prefix_sum,
                    vision_residual,
                ) = torch.ops.vllm.shensi_vl_loop_block(
                    hidden_states,
                    residual,
                    prefix_sum,
                    positions,
                    input_ids,
                    vision_hidden_states,
                    vision_residual,
                    vision_prefix_sum,
                    vision_cu_seqlens,
                    vision_rope_freqs_cis,
                    vision_max_seqlen,
                    vision_sequence_lengths,
                    vision_evidence_mask,
                    self.num_attn_res_blocks,
                    block.block_index,
                )
            else:
                (
                    hidden_states,
                    prefix_sum,
                    residual,
                    vision_hidden_states,
                    vision_prefix_sum,
                    vision_residual,
                ) = block(**block_inputs)

        # DSpark drafting: expose the final language-stream state (collapsed over
        # the hc axis) as the auxiliary hidden state the drafter fuses. The
        # language model's aux layers are set by the EAGLE-3 machinery, which the
        # DSpark speculator reuses.
        aux_hidden_states: list[torch.Tensor] = []
        if self.language_model.model.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states.mean(dim=1))

        hidden_states = language_model.norm(language_model.hc_head(hidden_states))
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The HF checkpoint nests everything under "model." and names the
        # projector projections fc1/fc2 (vLLM: linear_1/linear_2). Language
        # model, vision tower and DeepRecur blocks are delegated to the
        # AutoWeightsLoader, which strips enclosing prefixes before calling
        # the children's load_weights methods.
        renamed = []
        for name, tensor in weights:
            if "mm_projector.fc1" in name:
                name = name.replace("mm_projector.fc1", "mm_projector.linear_1", 1)
            elif "mm_projector.fc2" in name:
                name = name.replace("mm_projector.fc2", "mm_projector.linear_2", 1)
            renamed.append((name, tensor))
        loader = AutoWeightsLoader(self)
        return loader.load_weights(renamed, mapper=self.hf_to_vllm_mapper)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Pass the multimodal kwargs through: forward_language consults them
        # to decide whether the cached DeepRecur vision state applies to this
        # step (text-only steps must not reuse a previous request's patches).
        return self.forward_language(input_ids, positions, inputs_embeds, **kwargs)

    def compute_logits(
        self, hidden_states: torch.Tensor, **kwargs
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
