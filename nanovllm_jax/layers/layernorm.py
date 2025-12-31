from flax import nnx
import jax.numpy as jnp
import jax


class RMSNorm(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x_f = x.astype(jnp.float32)
        var = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
        x_norm = x_f * jax.lax.rsqrt(var + self.eps)
        x_norm = x_norm.astype(self.dtype) * self.weight.value
        return x_norm.astype(orig_dtype)
