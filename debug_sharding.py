"""调试 Sharding 问题

测试 with_sharding_constraint 是否需要 jit
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, Mesh, NamedSharding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_without_jit():
    """测试不使用 jit 的情况"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    logger.info(f"Using devices: {devices}")

    with mesh:
        # 创建一个简单的权重矩阵，切分在 tensor 维度
        weight = jnp.ones((8, 4), dtype=jnp.float32)
        weight_sharded = jax.device_put(weight, NamedSharding(mesh, P("tensor", None)))

        logger.info(f"Weight sharding: {weight_sharded.sharding}")

        # 输入数据 replicated
        x = jnp.ones((2, 8), dtype=jnp.float32)
        x_replicated = jax.lax.with_sharding_constraint(x, P(None, None))

        # 矩阵乘法
        y = jnp.dot(x_replicated, weight_sharded)
        logger.info(f"Output sharding (before constraint): {y.sharding}")

        # 添加 constraint 触发 all-reduce
        y_replicated = jax.lax.with_sharding_constraint(y, P(None, None))
        logger.info(f"Output sharding (after constraint): {y_replicated.sharding}")

        logger.info(f"Output value:\n{y_replicated}")


def test_with_jit():
    """测试使用 jit 的情况"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    logger.info(f"Using devices: {devices}")

    # 创建一个简单的权重矩阵，切分在 tensor 维度
    weight = jnp.ones((8, 4), dtype=jnp.float32)
    weight_sharded = jax.device_put(weight, NamedSharding(mesh, P("tensor", None)))

    @jax.jit
    def compute(x, w):
        x = jax.lax.with_sharding_constraint(x, P(None, None))
        y = jnp.dot(x, w)
        # ❌ 不能在 JIT 内部访问 .sharding
        # logger.info(f"Inside jit - output sharding: {y.sharding}")

        # Row-parallel 后需要 all-reduce
        y = jax.lax.with_sharding_constraint(y, P(None, None))
        return y

    with mesh:
        x = jnp.ones((2, 8), dtype=jnp.float32)
        y = compute(x, weight_sharded)

        # ✅ 在 JIT 外部可以访问 .sharding
        logger.info(f"Final output sharding: {y.sharding}")
        logger.info(f"Output value:\n{y}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("测试 WITHOUT JIT")
    logger.info("=" * 80)
    try:
        test_without_jit()
    except Exception as e:
        logger.error(f"Error: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("测试 WITH JIT")
    logger.info("=" * 80)
    try:
        test_with_jit()
    except Exception as e:
        logger.error(f"Error: {e}")
