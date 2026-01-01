"""测试 Attention 层的 Tensor Parallel

模拟真实的 Q/K/V 投影和 attention 计算
"""

import warnings
import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from nanovllm_jax.utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


# Suppress XLA warnings about missing SoL config
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow/XLA warnings
warnings.filterwarnings("ignore", category=UserWarning)


def test_qkv_projection_tp():
    """测试 Q/K/V 投影的 TP"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    hidden_size = 8
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 2

    # 输入
    x = jnp.ones((1, hidden_size), dtype=jnp.float32)

    # 权重
    q_weight = jnp.ones((hidden_size, num_heads * head_dim), dtype=jnp.float32)
    k_weight = jnp.ones((hidden_size, num_kv_heads * head_dim), dtype=jnp.float32) * 2
    o_weight = jnp.ones((num_heads * head_dim, hidden_size), dtype=jnp.float32)

    with mesh:
        # Shard 权重
        q_w_sharded = jax.device_put(q_weight, NamedSharding(mesh, P(None, "tensor")))
        k_w_sharded = jax.device_put(k_weight, NamedSharding(mesh, P(None, "tensor")))
        o_w_sharded = jax.device_put(o_weight, NamedSharding(mesh, P("tensor", None)))

        logger.info("=== Q/K Projection ===")
        x_rep = jax.lax.with_sharding_constraint(x, P(None, None))

        # Q 投影
        q = jnp.dot(x_rep, q_w_sharded)
        logger.info(f"Q after projection: shape={q.shape}, sharding={q.sharding}")
        q = jax.lax.with_sharding_constraint(q, P(None, "tensor"))

        # K 投影
        k = jnp.dot(x_rep, k_w_sharded)
        logger.info(f"K after projection: shape={k.shape}, sharding={k.sharding}")
        k = jax.lax.with_sharding_constraint(k, P(None, "tensor"))

        # Reshape
        q_reshaped = q.reshape(1, num_heads, head_dim)
        k_reshaped = k.reshape(1, num_kv_heads, head_dim)
        logger.info(
            f"Q after reshape: shape={q_reshaped.shape}, sharding={q_reshaped.sharding}"
        )
        logger.info(
            f"K after reshape: shape={k_reshaped.shape}, sharding={k_reshaped.sharding}"
        )

        # 添加 sharding constraint
        q_reshaped = jax.lax.with_sharding_constraint(
            q_reshaped, P(None, "tensor", None)
        )
        k_reshaped = jax.lax.with_sharding_constraint(
            k_reshaped, P(None, "tensor", None)
        )
        logger.info(
            f"Q after constraint: shape={q_reshaped.shape}, sharding={q_reshaped.sharding}"
        )
        logger.info(
            f"K after constraint: shape={k_reshaped.shape}, sharding={k_reshaped.sharding}"
        )

        # GQA repeat
        repeat_factor = num_heads // num_kv_heads
        k_repeated = jnp.repeat(k_reshaped, repeat_factor, axis=1)
        logger.info(
            f"K after repeat: shape={k_repeated.shape}, sharding={k_repeated.sharding}"
        )
        logger.info(f"K values: {k_repeated[0, :, 0]}")  # 应该全是16（8*2）

        # 简单的 attention（不做 softmax，只看数值）
        # Q: [1, 4, 2]（每个设备2个heads）
        # K: [1, 4, 2]（repeat后）
        scores = jnp.einsum("...qhd,...khd->...hqk", q_reshaped, k_repeated)
        logger.info(f"Scores: shape={scores.shape}, sharding={scores.sharding}")
        logger.info(f"Scores values:\n{scores}")

        # Flatten back
        attn_out = q_reshaped.reshape(1, -1)  # 假设 attention 输出就是 q
        logger.info(
            f"Attn out (flattened): shape={attn_out.shape}, sharding={attn_out.sharding}"
        )

        attn_out = jax.lax.with_sharding_constraint(attn_out, P(None, "tensor"))
        logger.info(
            f"Attn out (after constraint): shape={attn_out.shape}, sharding={attn_out.sharding}"
        )

        # O projection (row parallel)
        out = jnp.dot(attn_out, o_w_sharded)
        logger.info(
            f"Final out (before all-reduce): shape={out.shape}, sharding={out.sharding}"
        )
        logger.info(f"Values: {out}")

        out = jax.lax.with_sharding_constraint(out, P(None, None))
        logger.info(
            f"Final out (after all-reduce): shape={out.shape}, sharding={out.sharding}"
        )
        logger.info(f"Values: {out}")

        # 期望值：attn_out 每个元素=8，o_weight=1，所以 out=8*8=64


def test_with_real_values():
    """使用不同的值测试，确保不是巧合"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    hidden_size = 4
    intermediate_size = 8

    # 使用不同的值
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    w1 = jnp.arange(hidden_size * intermediate_size, dtype=jnp.float32).reshape(
        hidden_size, intermediate_size
    )
    w2 = jnp.arange(intermediate_size * hidden_size, dtype=jnp.float32).reshape(
        intermediate_size, hidden_size
    )

    with mesh:
        w1_sharded = jax.device_put(w1, NamedSharding(mesh, P(None, "tensor")))
        w2_sharded = jax.device_put(w2, NamedSharding(mesh, P("tensor", None)))

        logger.info("\n=== Test with Real Values ===")
        logger.info(f"Input: {x}")
        logger.info(f"W1:\n{w1}")
        logger.info(f"W2:\n{w2}")

        x_rep = jax.lax.with_sharding_constraint(x, P(None, None))
        h = jnp.dot(x_rep, w1_sharded)
        h = jax.lax.with_sharding_constraint(h, P(None, "tensor"))

        logger.info(f"After W1: {h}")

        y = jnp.dot(h, w2_sharded)
        logger.info(f"After W2 (before all-reduce): {y}")

        y = jax.lax.with_sharding_constraint(y, P(None, None))
        logger.info(f"After W2 (after all-reduce): {y}")

        # 手动计算期望值
        expected = jnp.dot(jnp.dot(x, w1), w2)
        logger.info(f"Expected (full precision): {expected}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    logger.info("=" * 80)
    logger.info("测试 Q/K/V 投影和 Attention")
    logger.info("=" * 80)
    test_qkv_projection_tp()

    logger.info("\n" + "=" * 80)
    logger.info("测试真实数值")
    logger.info("=" * 80)
    test_with_real_values()
