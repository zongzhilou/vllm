# SPDX-License-Identifier: Apache-2.0
# Generate dummy model directories (config.json only) for local benchmarking.

import json
import os

from transformers import AutoConfig

ROOT = os.path.join(os.path.dirname(__file__), "..", "dummy_models")
os.makedirs(ROOT, exist_ok=True)


def dump_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"wrote {path}")


# --- Shensi (text-only) ---
shensi_cfg = AutoConfig.for_model("shensi")
# Small shape for a single 16 GB GPU.
shensi_cfg.num_hidden_layers = 4
shensi_cfg.hidden_size = 1024
shensi_cfg.head_dim = 64
shensi_cfg.intermediate_size = 4096
shensi_cfg.vocab_size = 4096
shensi_cfg.architectures = ["ShensiForCausalLM"]
dump_config(shensi_cfg, os.path.join(ROOT, "shensi", "config.json"))

# --- Shensi-VL ---
vl_cfg = AutoConfig.for_model("shensi_vl")
vl_cfg.architectures = ["ShensiVlForConditionalGeneration"]
text_cfg = vl_cfg.text_config
text_cfg.num_hidden_layers = 4
text_cfg.hidden_size = 1024
text_cfg.head_dim = 64
text_cfg.intermediate_size = 4096
text_cfg.vocab_size = 4096
vis_cfg = vl_cfg.vision_config
vis_cfg.num_hidden_layers = 4
vis_cfg.hidden_size = 512
vis_cfg.intermediate_size = 2048
vis_cfg.qkv_hidden_size = 768
dump_config(vl_cfg, os.path.join(ROOT, "shensi_vl", "config.json"))

print("done")
