# SPDX-License-Identifier: Apache-2.0
"""vLLM throughput benchmark for Shensi / Shensi-VL (dummy checkpoints).

Mirrors scripts/bench_hf_shensi.py (input_len / output_len / num_prompts) so
the speedup numbers are directly comparable. Usage:

    .venv/bin/python scripts/bench_vllm_shensi.py --model shensi --input-len 128 \
        --output-len 128 --num-prompts 8
"""

import argparse
import time

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="shensi", choices=["shensi", "shensi_vl"])
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cudagraph", action="store_true")
    parser.add_argument("--image", action="store_true")
    args = parser.parse_args()

    model_path = f"dummy_models/{args.model}"
    compilation_config = None
    if args.cudagraph and args.model == "shensi_vl":
        # Enable the multimodal encoder CUDA graph (DeepRecur round-0 vision
        # pass) in addition to the text decoder graphs. The DeepRecur loop
        # block has a data-dependent while loop (host convergence guards), so
        # it runs eagerly as a graph splitting point and FULL graphs (which
        # freeze control flow) must be disabled. Piecewise input copying keeps
        # the captured pieces reading stable buffers (the splitting ops emit
        # fresh tensors each step).
        from vllm.config import CompilationConfig, CUDAGraphMode

        compilation_config = CompilationConfig(
            cudagraph_mm_encoder=True,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_copy_inputs=True,
        )
    llm = LLM(
        model=model_path,
        load_format="dummy",
        dtype=args.dtype,
        max_model_len=512,
        enforce_eager=not args.cudagraph,
        compilation_config=compilation_config,
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=2048,
        max_num_seqs=args.num_prompts,
    )

    # Each "token1 word1 " pair tokenizes to 2 tokens in the dummy tokenizer,
    # so input_len / 2 pairs yield a prompt of exactly input_len tokens.
    base = "token1 word1 " * (args.input_len // 2)
    if args.model == "shensi_vl" and args.image:
        # One 112x112 image per prompt (8x8 patches).
        import numpy as np

        image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        prompts = [
            {"prompt": "<image>" + base, "multi_modal_data": {"image": image}}
        ] * args.num_prompts
    else:
        # Text-only workload by default (mirrors the HF baseline script).
        prompts = [base] * args.num_prompts

    params = SamplingParams(max_tokens=args.output_len, temperature=0.0)
    # Warmup.
    llm.generate(prompts[:1], params)
    start = time.perf_counter()
    outs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    tokens_per_sec = total / elapsed
    print(
        f"vLLM {args.model} dtype={args.dtype} prompts={args.num_prompts} "
        f"in={args.input_len} out={args.output_len}: generated {total} tokens in "
        f"{elapsed:.2f}s -> {tokens_per_sec:.2f} tokens/s"
    )


if __name__ == "__main__":
    main()
