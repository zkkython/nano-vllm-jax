import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional


class Sampler(nnx.Module):
    """Sampling layer for token generation."""

    def __init__(self, rngs: Optional[nnx.Rngs] = None):
        """Initialize sampler with optional RNG state.

        Args:
            rngs: Optional NNX Rngs object for random number generation.
                  If None, a default RNG with seed 42 will be used.
        """
        self.rngs = rngs if rngs is not None else nnx.Rngs(42)

    def __call__(
        self,
        logits: jnp.ndarray,
        temperatures: Optional[jnp.ndarray] = None,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Sample tokens from logits.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size).
            temperatures: Optional temperature values for scaling logits.
            rng: Optional JAX PRNG key. If provided, this overrides the module's RNG.

        Returns:
            Sampled token IDs of shape (batch_size,).
        """
        if rng is None:
            # Use the module's RNG state
            rng = self.rngs.default()

        if temperatures is not None and temperatures.size > 0:
            # Apply temperature scaling
            # Reshape temperatures to match logits batch dimension
            temperatures = jnp.reshape(temperatures, (-1, 1))
            # Only apply to the batch size that matches
            batch_size = min(logits.shape[0], temperatures.shape[0])
            logits = logits.at[:batch_size].set(
                logits[:batch_size] / temperatures[:batch_size]
            )

        # Sample from the distribution
        token_ids = jax.random.categorical(rng, logits, axis=-1)

        return token_ids
