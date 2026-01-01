from nanovllm_jax.llm import LLM, SamplingParams
from nanovllm_jax.utils.logging_utils import setup_logging
import os
import warnings

# Suppress XLA warnings about missing SoL config
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow/XLA warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    llm = LLM(
        args.model_path,
        max_model_len=2048,
        tensor_parallel_size=args.tp_size,
    )

    # Single-prompt test
    outputs = llm.generate(
        ["你好，介绍一下你自己。"],
        SamplingParams(temperature=0.6, max_tokens=100),
    )
    print("Single prompt output:\n", outputs[0]["text"])

    # Multi-prompt test (same length prompts to trigger batched prefill/decode)
    multi_prompts = ["中国的首都在哪里", "列出100以内的质数", "解释一下什么是量子力学"]
    multi_outputs = llm.generate(
        multi_prompts,
        SamplingParams(temperature=0.6, max_tokens=100),
    )
    print("\nMulti-prompt outputs:")
    for i, out in enumerate(multi_outputs):
        print(f"Prompt {i}: {multi_prompts[i]}")
        print(
            f"Response:{out["text"]}!",
        )
        print("-")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from NANO_VLLM_LOG_LEVEL env var or INFO)",
    )
    parser.add_argument(
        "--debug-modules",
        type=str,
        default=None,
        help="Comma-separated list of modules to enable DEBUG logging (e.g., nanovllm_jax.layers.attention)",
    )
    args = parser.parse_args()

    # Setup logging
    debug_modules = args.debug_modules.split(",") if args.debug_modules else None
    setup_logging(level=args.log_level, enable_debug_modules=debug_modules)

    main(args)
