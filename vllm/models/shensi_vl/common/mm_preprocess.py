# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Shensi-VL multimodal preprocessing."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions, ImageDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    cached_encode,
)
from vllm.transformers_utils.processor import cached_get_image_processor

logger = None  # noqa: F841


class ShensiVlProcessingInfo(BaseProcessingInfo):
    """Processing info for Shensi-VL (image-only, grid_thw style)."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__(ctx)
        self.hf_config = self.get_hf_config()
        self.image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )
        self.image_token_id = self.hf_config.image_token_id
        # Number of post-merge media tokens for one image: the processor emits
        # (h / patch) x (w / patch) patches and each merge_kernel_size block
        # collapses into one token.
        vision_cfg = self.hf_config.vision_config
        patch_size = getattr(vision_cfg, "patch_size", 14)
        merge = getattr(vision_cfg, "merge_kernel_size", (2, 2))
        kh, kw = (merge, merge) if isinstance(merge, int) else (merge[0], merge[1])

        def media_tokens_calculator(media: dict) -> int:
            image = media["image"]
            if isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                width, height = image.size
            return (height // patch_size // kh) * (width // patch_size // kw)

        self.media_tokens_calculator = media_tokens_calculator

    def get_hf_processor(self, **kwargs: object):
        processor = self.image_processor

        # The vLLM multimodal context always injects `truncation=False`, which
        # the strict Kimi-K25 processor kwargs reject; strip it here.
        def _call(**kw: object):
            kw.pop("truncation", None)
            return processor(**kw)

        return _call

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        # One image yields (size / patch / merge)^2 tokens; the processor
        # resizes inputs to 448x448, matching the dummy input size below.
        vision_cfg = self.hf_config.vision_config
        patch = getattr(vision_cfg, "patch_size", 14)
        merge = getattr(vision_cfg, "merge_kernel_size", (2, 2))
        kh, kw = (merge, merge) if isinstance(merge, int) else (merge[0], merge[1])
        side = 448
        tokens = (side // patch // kh) * (side // patch // kw)
        return {"image": min(tokens, seq_len)}


class ShensiVlDummyInputsBuilder(BaseDummyInputsBuilder[ShensiVlProcessingInfo]):
    """Builds image-based dummy inputs for Shensi-VL profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = cast(
            ImageDummyOptions | None,
            mm_options.get("image") if mm_options else None,
        )
        return {
            "image": self._get_dummy_images(
                width=448,
                height=448,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class ShensiVlMultiModalProcessor(BaseMultiModalProcessor[ShensiVlProcessingInfo]):
    """Image multi-modal processor for Shensi-VL."""

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # The fused Kimi processor emits "grid_thw"; the upstream
        # transformers processor emits "image_grid_thw".
        grid_thws = hf_inputs.get(
            "grid_thw", hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        )
        grid_sizes = grid_thws.prod(-1)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", grid_sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        image_placeholder = "<image>"

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image = images.get(item_idx)
            if image is None:
                raise ValueError(f"Missing image data at index {item_idx}")
            num_media_token = self.info.media_tokens_calculator(
                {"type": "image", "image": image}
            )
            pads = "<image>" * num_media_token
            return PromptUpdateDetails.select_token_id(
                cached_encode(tokenizer, pads, add_special_tokens=False),
                self.info.image_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=cached_encode(
                    tokenizer, image_placeholder, add_special_tokens=False
                ),
                replacement=get_replacement,
            ),
        ]
