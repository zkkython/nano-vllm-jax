from flax.typing import PromoteDtypeFn
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes
from flax.nnx.nn.linear import default_embed_init

init_fn = nnx.initializers.uniform()


class Embed(nnx.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      embedding_init: embedding initializer.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: nnx.Rngs | None = None,
    ):
        """
        Sets up the embedding parameters for the model.

        This method initializes the embedding parameters with logical partitioning.
        The embedding is represented as a parameter with the specified shape and data type.

        Args:
            num_embeddings: Number of embeddings in the vocabulary.
            features: Number of feature dimensions for each embedding.
            dtype: Data type for computations (forward pass, attend operations).
                   If None, uses the same dtype as the embedding parameter.
            param_dtype: Data type for storing the embedding parameters in memory.
                        Controls memory usage and precision of stored weights.
            promote_dtype: Function to handle dtype promotion during mixed-precision
                          computations between query/embedding tensors.
            rngs: Random number generator state for parameter initialization.
        """
        self.embedding = nnx.Param(
            default_embed_init(
                jax.random.PRNGKey(0), (num_embeddings, features), param_dtype
            )
        )

        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.value.dtype
        self.promote_dtype = promote_dtype

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = self.promote_dtype(
            (self.embedding.value,), dtype=self.dtype, inexact=False
        )
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, inputs.shape + (self.features,))
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: jax.Array) -> jax.Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        query, embedding = self.promote_dtype(
            (query, self.embedding.value), dtype=self.dtype
        )
        return jnp.dot(query, embedding.T)


class ParallelLMHead(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.rngs = rngs

        self.weight = nnx.Param(
            init_fn(jax.random.PRNGKey(0), (vocab_size, hidden_size), dtype)
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = hidden_states.astype(self.dtype)
        return hidden_states @ self.weight.value.T
