import jax
from flax import nnx
import jax.numpy as jnp


class Linear(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
        use_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.use_bias = use_bias

        w_shape = (out_features, in_features)
        self.weight = nnx.Param(
            jax.random.normal(jax.random.PRNGKey(0), w_shape, dtype)
            * (1.0 / jnp.sqrt(in_features))
        )
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_features,), dtype=dtype))
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.weight.value
        y = x @ w.T
        if self.bias is not None:
            y = y + self.bias.value
        return y.astype(self.dtype)
