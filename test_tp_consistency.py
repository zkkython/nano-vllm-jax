"""测试 Tensor Parallel 一致性

比较 tp_size=1 和 tp_size=2 的输出是否一致
"""

from nanovllm_jax.llm import LLM, SamplingParams
import logging

# 配置日志输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def test_consistency(model_path: str):
    """测试不同 TP size 的输出一致性"""

    prompt = "你好，介绍一下你自己。"
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=50
    )  # temperature=0 确保确定性输出

    print("=" * 80)
    print("测试 TP Size = 1")
    print("=" * 80)
    llm1 = LLM(model_path, max_model_len=2048, tensor_parallel_size=1)
    output1 = llm1.generate([prompt], sampling_params, use_tqdm=False)
    print(f"\nTP=1 输出:\n{output1[0]['text']}\n")
    print(f"Token IDs: {output1[0]['token_ids'][:20]}...")  # 显示前20个token

    print("\n" + "=" * 80)
    print("测试 TP Size = 2")
    print("=" * 80)
    llm2 = LLM(model_path, max_model_len=2048, tensor_parallel_size=2)
    output2 = llm2.generate([prompt], sampling_params, use_tqdm=False)
    print(f"\nTP=2 输出:\n{output2[0]['text']}\n")
    print(f"Token IDs: {output2[0]['token_ids'][:20]}...")  # 显示前20个token

    # 比较输出
    print("\n" + "=" * 80)
    print("一致性检查")
    print("=" * 80)

    tokens1 = output1[0]["token_ids"]
    tokens2 = output2[0]["token_ids"]

    if tokens1 == tokens2:
        print("✅ 成功！两种配置的输出完全一致")
    else:
        print("❌ 失败！输出不一致")
        print(f"\nTP=1 tokens ({len(tokens1)}): {tokens1[:30]}")
        print(f"TP=2 tokens ({len(tokens2)}): {tokens2[:30]}")

        # 找出第一个不同的位置
        min_len = min(len(tokens1), len(tokens2))
        for i in range(min_len):
            if tokens1[i] != tokens2[i]:
                print(f"\n第一个不同的token在位置 {i}:")
                print(f"  TP=1: {tokens1[i]}")
                print(f"  TP=2: {tokens2[i]}")
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="模型路径",
    )
    args = parser.parse_args()

    test_consistency(args.model_path)
