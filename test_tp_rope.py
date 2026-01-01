"""测试 RoPE 在 Tensor Parallel 下是否正确"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_rope_with_tp():
    """测试 RoPE 操作在 TP 下的行为"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    num_tokens = 4
    num_heads = 4
    head_dim = 8

    # 模拟 Q/K 张量（在 heads 维度分片）
    q = jnp.ones((num_tokens, num_heads, head_dim), dtype=jnp.float32)

    # 模拟 positions
    positions = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    # 模拟 cos/sin（这些应该是 replicated 的）
    cos = jnp.arange(num_tokens * head_dim // 2, dtype=jnp.float32).reshape(
        num_tokens, head_dim // 2
    )
    sin = (
        jnp.arange(num_tokens * head_dim // 2, dtype=jnp.float32).reshape(
            num_tokens, head_dim // 2
        )
        * 0.1
    )

    with mesh:
        # Shard Q
        q_sharded = jax.device_put(q, NamedSharding(mesh, P(None, "tensor", None)))
        logger.info(f"Q sharding: {q_sharded.sharding}")

        # Positions 和 cos/sin 应该是 replicated
        positions_rep = jax.lax.with_sharding_constraint(positions, P(None))
        cos_rep = jax.lax.with_sharding_constraint(cos, P(None, None))
        sin_rep = jax.lax.with_sharding_constraint(sin, P(None, None))

        logger.info(f"Positions sharding: {positions_rep.sharding}")
        logger.info(f"Cos sharding: {cos_rep.sharding}")

        # 模拟索引操作
        cos_indexed = cos_rep[positions_rep]
        sin_indexed = sin_rep[positions_rep]

        logger.info(f"Cos indexed sharding: {cos_indexed.sharding}")
        logger.info(f"Cos indexed shape: {cos_indexed.shape}")
        logger.info(f"Cos indexed values:\n{cos_indexed}")

        # 模拟 RoPE 应用（简化版 - 只是乘法）
        # 在真实代码中，这里会有更复杂的操作
        q_rotated = q_sharded[:, :, : head_dim // 2] * cos_indexed[:, None, :]

        logger.info(f"Q rotated sharding: {q_rotated.sharding}")
        logger.info(f"Q rotated shape: {q_rotated.shape}")


def test_index_with_sharded_array():
    """测试用 replicated 索引访问 sharded 数组"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    # 创建一个 sharded 数组
    arr = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    indices = jnp.array([0, 2, 4, 6], dtype=jnp.int32)

    with mesh:
        # Shard 数组
        arr_sharded = jax.device_put(arr, NamedSharding(mesh, P("tensor", None)))
        indices_rep = jax.lax.with_sharding_constraint(indices, P(None))

        logger.info(f"\n=== Index Sharded Array ===")
        logger.info(f"Array sharding: {arr_sharded.sharding}")
        logger.info(f"Indices sharding: {indices_rep.sharding}")

        # 索引操作
        result = arr_sharded[indices_rep]

        logger.info(f"Result sharding: {result.sharding}")
        logger.info(f"Result:\n{result}")

        # 期望：每个设备只有自己那部分数据
        # device 0: rows [0, 1, 2, 3] of arr
        # device 1: rows [4, 5, 6, 7] of arr
        # 但索引 [0, 2, 4, 6] 需要从两个设备读取！


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("测试 RoPE 在 TP 下的行为")
    logger.info("=" * 80)
    test_rope_with_tp()

    logger.info("\n" + "=" * 80)
    logger.info("测试用 replicated 索引访问 sharded 数组")
    logger.info("=" * 80)
    test_index_with_sharded_array()
