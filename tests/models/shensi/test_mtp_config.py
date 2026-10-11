# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from transformers import PreTrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs import ShensiConfig

from . import TINY_SHENSI_KWARGS


def _shensi_config(**overrides) -> ShensiConfig:
    return ShensiConfig(**{**TINY_SHENSI_KWARGS, **overrides})


def test_mtp_override_uses_the_shensi_architecture():
    config = SpeculativeConfig.hf_config_override(
        _shensi_config(architectures=["ShensiForCausalLM"], num_nextn_predict_layers=2)
    )
    assert config.model_type == "shensi_mtp"
    assert config.architectures == ["ShensiMTPModel"]
    assert config.n_predict == 2


def test_mtp_override_rejects_checkpoints_without_mtp_layers():
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        SpeculativeConfig.hf_config_override(
            _shensi_config(
                architectures=["ShensiForCausalLM"], num_nextn_predict_layers=0
            )
        )


def test_mtp_override_keeps_the_deepseek_path():
    config = PreTrainedConfig(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        num_nextn_predict_layers=1,
    )
    config = SpeculativeConfig.hf_config_override(config)
    assert config.model_type == "deepseek_mtp"
    assert config.architectures == ["DeepSeekV4MTPModel"]


def test_dspark_contract_requires_the_training_side_fields():
    from vllm.models.shensi.nvidia.dspark import _require_dspark_config

    with pytest.raises(ValueError, match="dspark_target_layer_ids"):
        _require_dspark_config(_shensi_config())
    config = _shensi_config(dspark_target_layer_ids=[1, 2], dspark_markov_rank=8)
    _require_dspark_config(config)
