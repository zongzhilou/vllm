# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shensi-VL DSpark draft model for speculative decoding.

Drafting is text-only: the drafter never sees pixels and shares the Shensi-VL
target's language embedding / LM head via ``load_dspark_model``.
"""

from vllm.models.shensi.dspark import DSparkShensiForCausalLM

__all__ = ["DSparkShensiVlForCausalLM"]


class DSparkShensiVlForCausalLM(DSparkShensiForCausalLM):
    """DSpark drafter for the Shensi-VL target (text-only Shensi drafter)."""
