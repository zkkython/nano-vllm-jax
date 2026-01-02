import os
import time
from random import randint, seed
import warnings

from nanovllm_jax.llm import LLM
from nanovllm_jax.sampling_params import SamplingParams

# Suppress XLA warnings about missing SoL config
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow/XLA warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    seed(0)
    num_seqs = 50
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096, tensor_parallel_size=8)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
