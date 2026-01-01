"""端到端测试 Tensor Parallel

直接测试简化版的 Transformer layer
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_linear_tp():
    """测试最简单的线性层 TP"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    hidden_size = 8
    intermediate_size = 16

    # 输入（replicated）
    x = jnp.ones((1, hidden_size), dtype=jnp.float32)

    # Column parallel 权重
    w1 = jnp.ones((hidden_size, intermediate_size), dtype=jnp.float32)
    # Row parallel 权重
    w2 = jnp.ones((intermediate_size, hidden_size), dtype=jnp.float32)

    with mesh:
        # Shard 权重
        w1_sharded = jax.device_put(w1, NamedSharding(mesh, P(None, "tensor")))
        w2_sharded = jax.device_put(w2, NamedSharding(mesh, P("tensor", None)))

        logger.info(f"Input: {x.shape}")
        logger.info(f"W1 sharding: {w1_sharded.sharding}")
        logger.info(f"W2 sharding: {w2_sharded.sharding}")

        # 第一层：column parallel
        x_rep = jax.lax.with_sharding_constraint(x, P(None, None))
        h = jnp.dot(x_rep, w1_sharded)
        logger.info(f"After W1 (before constraint): {h.shape}, sharding: {h.sharding}")

        # Column parallel 输出在最后一维分片
        h = jax.lax.with_sharding_constraint(h, P(None, "tensor"))
        logger.info(f"After W1 (after constraint): {h.shape}, sharding: {h.sharding}")

        # 第二层：row parallel
        y = jnp.dot(h, w2_sharded)
        logger.info(f"After W2 (before all-reduce): {y.shape}, sharding: {y.sharding}")
        logger.info(f"Value (before all-reduce): {y}")

        # Row parallel 需要 all-reduce
        y = jax.lax.with_sharding_constraint(y, P(None, None))
        logger.info(f"After W2 (after all-reduce): {y.shape}, sharding: {y.sharding}")
        logger.info(f"Value (after all-reduce): {y}")

        # 期望值：x是全1，w1是全1，所以h每个元素=8
        # w2是全1，所以y每个元素=16*8=128（all-reduce后）
        logger.info(f"Expected: 128, Got: {y[0,0]}")


def test_with_jit():
    """测试JIT编译版本"""
    devices = jax.devices()[:2]
    mesh = Mesh(devices, ("tensor",))

    hidden_size = 8
    intermediate_size = 16

    w1 = jnp.ones((hidden_size, intermediate_size), dtype=jnp.float32)
    w2 = jnp.ones((intermediate_size, hidden_size), dtype=jnp.float32)

    @jax.jit
    def forward(x, w1, w2):
        x = jax.lax.with_sharding_constraint(x, P(None, None))
        h = jnp.dot(x, w1)
        h = jax.lax.with_sharding_constraint(h, P(None, "tensor"))
        y = jnp.dot(h, w2)
        y = jax.lax.with_sharding_constraint(y, P(None, None))
        return y

    with mesh:
        w1_sharded = jax.device_put(w1, NamedSharding(mesh, P(None, "tensor")))
        w2_sharded = jax.device_put(w2, NamedSharding(mesh, P("tensor", None)))

        x = jnp.ones((1, hidden_size), dtype=jnp.float32)
        y = forward(x, w1_sharded, w2_sharded)

        logger.info(f"\n=== WITH JIT ===")
        logger.info(f"Output sharding: {y.sharding}")
        logger.info(f"Output value: {y}")
        logger.info(f"Expected: 128, Got: {y[0,0]}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("测试简单线性层 TP（无JIT）")
    logger.info("=" * 80)
    test_simple_linear_tp()

    logger.info("\n" + "=" * 80)
    logger.info("测试简单线性层 TP（有JIT）")
    logger.info("=" * 80)
    test_with_jit()
