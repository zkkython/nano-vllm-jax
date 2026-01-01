"""详细调试 Tensor Parallel 的 sharding

检查每一层的权重和激活的 sharding
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_qkv_projection():
    """测试 Q/K/V 投影的 sharding"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    logger.info("=" * 80)
    logger.info("测试 Q/K/V 投影的 Tensor Parallel")
    logger.info("=" * 80)

    hidden_size = 8
    num_heads = 4
    num_kv_heads = 2  # GQA: 2 KV heads, 4 Q heads
    head_dim = 2

    # 输入是 replicated 的
    x = jnp.ones((1, hidden_size), dtype=jnp.float32)

    # Q 权重: [hidden_size, num_heads * head_dim]，column-parallel
    q_weight = jnp.ones((hidden_size, num_heads * head_dim), dtype=jnp.float32)
    # K/V 权重: [hidden_size, num_kv_heads * head_dim]，column-parallel
    k_weight = jnp.ones((hidden_size, num_kv_heads * head_dim), dtype=jnp.float32) * 2

    with mesh:
        # 权重 sharding
        q_sharding = NamedSharding(mesh, P(None, "tensor"))
        k_sharding = NamedSharding(mesh, P(None, "tensor"))

        q_weight_sharded = jax.device_put(q_weight, q_sharding)
        k_weight_sharded = jax.device_put(k_weight, k_sharding)

        logger.info(f"Input shape: {x.shape}")
        logger.info(
            f"Q weight shape: {q_weight.shape}, sharding: {q_weight_sharded.sharding}"
        )
        logger.info(
            f"K weight shape: {k_weight.shape}, sharding: {k_weight_sharded.sharding}"
        )

        # 计算 Q, K
        x_replicated = jax.lax.with_sharding_constraint(x, P(None, None))
        q = jnp.dot(x_replicated, q_weight_sharded)
        k = jnp.dot(x_replicated, k_weight_sharded)

        logger.info(f"\nAfter projection:")
        logger.info(f"Q shape: {q.shape}, sharding: {q.sharding}")
        logger.info(f"K shape: {k.shape}, sharding: {k.sharding}")

        # Reshape 到 [batch, num_tokens, num_heads, head_dim]
        q_reshaped = q.reshape(1, 1, num_heads, head_dim)
        k_reshaped = k.reshape(1, 1, num_kv_heads, head_dim)

        logger.info(f"\nAfter reshape:")
        logger.info(f"Q shape: {q_reshaped.shape}, sharding: {q_reshaped.sharding}")
        logger.info(f"K shape: {k_reshaped.shape}, sharding: {k_reshaped.sharding}")

        # 如果要做 attention，K 需要被 repeat（GQA）
        # 在 tensor parallel 中，这个 repeat 很关键！
        repeat_factor = num_heads // num_kv_heads
        k_repeated = jnp.repeat(k_reshaped, repeat_factor, axis=2)

        logger.info(f"\nAfter repeat (GQA):")
        logger.info(f"K shape: {k_repeated.shape}, sharding: {k_repeated.sharding}")

        # 计算 attention
        # Q: [1, 1, 4, 2], K: [1, 1, 4, 2]
        scores = jnp.einsum("...qhd,...khd->...hqk", q_reshaped, k_repeated)
        logger.info(f"\nAttention scores:")
        logger.info(f"Scores shape: {scores.shape}, sharding: {scores.sharding}")
        logger.info(f"Scores value:\n{scores}")

        # 期望值：因为 q_weight 和 k_weight 都是 1 和 2，
        # 在 column-parallel 下，每个设备计算部分 heads
        # 最终 scores 应该在所有设备上一致


def test_output_projection():
    """测试输出投影的 sharding"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    logger.info("\n" + "=" * 80)
    logger.info("测试输出投影的 Tensor Parallel")
    logger.info("=" * 80)

    hidden_size = 8
    num_heads = 4
    head_dim = 2

    # Attention 输出：每个设备有部分 heads
    # [batch, seq, num_heads, head_dim]
    attn_out = jnp.ones((1, 1, num_heads, head_dim), dtype=jnp.float32)
    attn_out_flat = attn_out.reshape(1, num_heads * head_dim)

    # O 权重: [num_heads * head_dim, hidden_size]，row-parallel
    o_weight = jnp.ones((num_heads * head_dim, hidden_size), dtype=jnp.float32)

    with mesh:
        # Attention 输出在 heads 维度分片
        attn_sharding = NamedSharding(mesh, P(None, "tensor"))
        attn_out_flat_sharded = jax.device_put(attn_out_flat, attn_sharding)

        # O 权重在输入维度（heads*head_dim）分片
        o_sharding = NamedSharding(mesh, P("tensor", None))
        o_weight_sharded = jax.device_put(o_weight, o_sharding)

        logger.info(
            f"Attn output shape: {attn_out_flat.shape}, sharding: {attn_out_flat_sharded.sharding}"
        )
        logger.info(
            f"O weight shape: {o_weight.shape}, sharding: {o_weight_sharded.sharding}"
        )

        # 计算输出投影
        out = jnp.dot(attn_out_flat_sharded, o_weight_sharded)
        logger.info(f"\nBefore all-reduce:")
        logger.info(f"Output shape: {out.shape}, sharding: {out.sharding}")
        logger.info(f"Output value: {out}")

        # 添加 all-reduce constraint
        out_replicated = jax.lax.with_sharding_constraint(out, P(None, None))
        logger.info(f"\nAfter all-reduce:")
        logger.info(
            f"Output shape: {out_replicated.shape}, sharding: {out_replicated.sharding}"
        )
        logger.info(f"Output value: {out_replicated}")

        # 期望值：在 row-parallel 下，每个设备计算部分输出
        # all-reduce 后应该得到完整结果


if __name__ == "__main__":
    test_qkv_projection()
    test_output_projection()
