import jax
from flax import nnx
import jax.numpy as jnp
from collections.abc import Sequence
from jax.sharding import PartitionSpec as P
import logging

logger = logging.getLogger(__name__)


class Linear(nnx.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        use_bias: bool = False,
        kernel_axes: Sequence[str] | None = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.use_bias = use_bias
        self.kernel_axes = kernel_axes

        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), kernel_axes)(
                jax.random.PRNGKey(0), (in_features, out_features), dtype
            )
        )

        if use_bias:
            if kernel_axes is not None:
                bias_axes = (kernel_axes[-1],)
            else:
                bias_axes = None
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.zeros_init(), bias_axes)(
                    jax.random.PRNGKey(0), (out_features,), dtype
                )
            )
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.weight.value
        y = jnp.dot(x, w)

        # 添加 sharding constraint 来触发自动 all-reduce
        # 这需要在 mesh 上下文中执行（在 model_runner.run_model 中设置）
        if self.kernel_axes is not None:
            # Column parallel: (None, "tensor") -> 输出最后一维分片
            # Row parallel: ("tensor", None) -> 输出需要 all-reduce，即全部 None
            if self.kernel_axes[0] == "tensor":
                # Row parallel: 输出应该是 replicated（触发 all-reduce）
                logger.debug(f"Linear: Row-parallel, applying all-reduce constraint")
                # 为所有维度设置 None
                output_spec = tuple([None] * y.ndim)
                y = jax.lax.with_sharding_constraint(y, P(*output_spec))
            else:
                # Column parallel: 输出在最后一维保持分片
                logger.debug(f"Linear: Column-parallel, keeping sharding on last dim")
                # 只有最后一维是 "tensor"，其他都是 None
                output_spec = tuple([None] * (y.ndim - 1) + [self.kernel_axes[-1]])
                y = jax.lax.with_sharding_constraint(y, P(*output_spec))

        if self.bias is not None:
            y = y + self.bias.value
        return y.astype(self.dtype)
