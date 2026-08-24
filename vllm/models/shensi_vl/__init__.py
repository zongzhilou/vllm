# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shensi-VL model — vision-language variant of Shensi with DeepRecur blocks."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dspark import DSparkShensiVlForCausalLM
    from .model import ShensiVlForConditionalGeneration
else:
    from .dspark import DSparkShensiVlForCausalLM
    from .model import ShensiVlForConditionalGeneration

__all__ = ["DSparkShensiVlForCausalLM", "ShensiVlForConditionalGeneration"]
