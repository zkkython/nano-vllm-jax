import jax
from flax import nnx
import jax.numpy as jnp
from collections.abc import Sequence


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

        w_shape = (in_features, out_features)
        self.weight = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), w_shape, dtype)
            * (1.0 / jnp.sqrt(in_features))
        )
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
        if self.bias is not None:
            y = y + self.bias.value
        return y.astype(self.dtype)
