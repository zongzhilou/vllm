# SPDX-License-Identifier: Apache-2.0
"""HF transformers throughput baseline for Shensi / Shensi-VL.

Mirrors the vLLM benchmark_throughput workload (input_len / output_len /
num_prompts) so the speedup numbers are directly comparable. Usage:

    .venv/bin/python scripts/bench_hf_shensi.py --model shensi --input-len 128 \
        --output-len 128 --num-prompts 8 --dtype float16
"""

import argparse
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="shensi", choices=["shensi", "shensi_vl"])
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--no-image", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    cfg = AutoConfig.for_model(args.model)
    if args.model == "shensi":
        cfg.num_hidden_layers = 4
        cfg.hidden_size = 1024
        cfg.head_dim = 64
        cfg.intermediate_size = 4096
        cfg.vocab_size = 4096
    else:
        cfg.text_config.num_hidden_layers = 4
        cfg.text_config.hidden_size = 1024
        cfg.text_config.head_dim = 64
        cfg.text_config.intermediate_size = 4096
        cfg.text_config.vocab_size = 4096
        cfg.vision_config.num_hidden_layers = 4
        cfg.vision_config.hidden_size = 512
        cfg.vision_config.intermediate_size = 2048
        cfg.vision_config.qkv_hidden_size = 768

    if args.model == "shensi_vl":
        model = AutoModelForImageTextToText.from_config(cfg).cuda().to(dtype)
    else:
        model = AutoModelForCausalLM.from_config(cfg).cuda().to(dtype)
    model.eval()

    vocab_size = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size
    input_ids = torch.randint(5, vocab_size, (args.num_prompts, args.input_len)).cuda()
    kwargs = {}
    if args.model == "shensi_vl" and not args.no_image:
        # One 112x112 image per prompt (8x8 patches, 14px patch): the vision
        # tower consumes flattened patches, one (3, 14, 14) crop per row, with
        # grid_thw in patch units.
        patch = cfg.vision_config.patch_size
        kwargs["pixel_values"] = (
            torch.randn(args.num_prompts * 64, 3, patch, patch).cuda().to(dtype)
        )
        kwargs["image_grid_thw"] = torch.tensor([[1, 8, 8]] * args.num_prompts).cuda()

    with torch.inference_mode():
        # Warmup.
        model.generate(input_ids[:1], max_new_tokens=8, do_sample=False, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        total = 0
        for i in range(args.num_prompts):
            out = model.generate(
                input_ids[i : i + 1],
                max_new_tokens=args.output_len,
                do_sample=False,
                **{k: v[i : i + 1] for k, v in kwargs.items()},
            )
            total += out.shape[1] - args.input_len
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    tokens_per_sec = total / elapsed
    print(
        f"HF {args.model} dtype={args.dtype} prompts={args.num_prompts} "
        f"in={args.input_len} out={args.output_len}: generated {total} tokens in "
        f"{elapsed:.2f}s -> {tokens_per_sec:.2f} tokens/s"
    )


if __name__ == "__main__":
    main()
