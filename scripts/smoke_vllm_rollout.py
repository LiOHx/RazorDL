#!/usr/bin/env python3
"""Manual vLLM rollout smoke test for GRPO/OPD.

Requires a Python environment with vLLM installed (the project `.venv` does
not include it).  Example:

    python scripts/smoke_vllm_rollout.py --model /path/to/Qwen3.5-0.8B

The script loads the model through RazorDL's ``GRPOVLLMRollout`` wrapper,
generates one short completion, and verifies the response token/mask shapes.
It does not exercise weight sync; that is covered separately by the GRPO/OPD
training smoke tests once a real training run is available.
"""

import argparse

from razordl.ops.model.vllm_rollout import (
    GRPOVLLMRollout,
    InferenceConfig,
    SamplingParamsConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--prompt-token-ids", default="10,20,30,40,50")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = GRPOVLLMRollout(
        InferenceConfig(
            model=args.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=4,
            max_num_batched_tokens=args.max_model_len,
            distributed_executor_backend="mp",
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
        ),
        SamplingParamsConfig(
            n=1,
            temperature=0.7,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_tokens,
            detokenize=False,
            seed=args.seed,
        ),
    )
    prompt = [int(token) for token in args.prompt_token_ids.split(",") if token.strip()]
    result = engine.generate([prompt], seed=args.seed)

    print("response_token_ids:", result["response_token_ids"])
    print("response_mask:", result["response_mask"])
    assert result["response_token_ids"], "vLLM returned no completions"
    assert len(result["response_token_ids"]) == len(result["response_mask"])
    print("VLLM SMOKE OK")


if __name__ == "__main__":
    main()
