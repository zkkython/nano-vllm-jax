"""Qwen3 model implementation in JAX using Flax NNX.

This file contains attention implementations:


1. SelfAttentionVarlen - Uses varlen attention from attention.py
   - Supports variable-length sequences with flattened format
   - Uses external KV cache with block-based management
   - Required for model_runner.py integration

Usage Example with Varlen Attention:

```python
from nanovllm_jax.models.qwen3 import SelfAttentionVarlen
from nanovllm_jax.utils.context import set_context
import jax.numpy as jnp

# Initialize attention layer
attn = SelfAttentionVarlen(config, dtype=jnp.bfloat16, block_size=256)

# Prepare flattened sequences (e.g., 2 sequences of length 10 and 15)
hidden_states = jnp.zeros((25, 768))  # [total_tokens, hidden_size]
positions = jnp.arange(25)  # position indices

# Set context for varlen attention
set_context(
    is_prefill=True,
    cu_seqlens_q=jnp.array([0, 10, 25]),  # cumulative sequence lengths
    cu_seqlens_k=jnp.array([0, 10, 25]),
    max_seqlen_q=15,
    max_seqlen_k=15,
    slot_mapping=jnp.arange(25),  # slot indices for KV cache
    context_lens=None,
    block_tables=None,
)

# Allocate external KV cache
num_blocks = 100
k_cache = jnp.zeros((num_blocks, 256, num_kv_heads, head_dim))
v_cache = jnp.zeros((num_blocks, 256, num_kv_heads, head_dim))

# Forward pass
output, k_cache, v_cache = attn(
    hidden_states,
    positions=positions,
    k_cache=k_cache,
    v_cache=v_cache
)
```

For model_runner.py integration, the model_runner handles:
- Context setup via prepare_prefill() and prepare_decode()
- External KV cache allocation and management
- Flattened tensor preparation
"""

from typing import Optional
import jax
from jax import numpy as jnp
from flax import nnx
from transformers import PretrainedConfig
import logging
from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.layers.embed_head import Embed, ParallelLMHead
from nanovllm_jax.layers.attention import Attention
from nanovllm_jax.layers.layernorm import RMSNorm
from nanovllm_jax.layers.linear import Linear
from nanovllm_jax.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def _build_rope_cos_sin(seq_len: int, head_dim: int, base: float, dtype=jnp.float32):
    """Build RoPE cos/sin tables with shape [seq_len, head_dim//2]."""
    assert head_dim % 2 == 0
    # Generate inverse frequencies: theta_i = base^(-2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    # Generate position indices
    t = jnp.arange(seq_len, dtype=dtype)
    # Compute freqs: [seq_len, head_dim/2]
    freqs = jnp.einsum("i,j->ij", t, inv_freq)
    # Compute cos and sin
    cos = jnp.cos(freqs)  # [seq_len, head_dim/2]
    sin = jnp.sin(freqs)  # [seq_len, head_dim/2]
    return cos, sin


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply RoPE to tensor x of shape [batch, seq, heads, head_dim].

    Args:
        x: [batch, seq, heads, head_dim]
        cos: [seq, head_dim//2]
        sin: [seq, head_dim//2]
    """
    # x: [batch, seq, heads, head_dim]
    # Split x into two halves along the last dimension
    x1, x2 = jnp.split(x, 2, axis=-1)  # Each: [batch, seq, heads, head_dim/2]

    # cos/sin: [seq, head_dim/2] -> broadcast to [1, seq, 1, head_dim/2]
    cos = cos[None, :, None, :]  # [1, seq, 1, head_dim/2]
    sin = sin[None, :, None, :]  # [1, seq, 1, head_dim/2]

    # Apply rotation
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    return jnp.concatenate([y1, y2], axis=-1)


class MLP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.dtype = dtype

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = Linear(
            hidden_size, intermediate_size, dtype=dtype, rngs=rngs, use_bias=False
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, dtype=dtype, rngs=rngs, use_bias=False
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, dtype=dtype, rngs=rngs, use_bias=False
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        act = jax.nn.silu(gate) * up
        out = self.down_proj(act)
        return out


class SelfAttentionVarlen(nnx.Module):
    """Self-attention module using varlen attention from attention.py.

    This version uses external KV cache and supports:
    - Variable-length attention for flattened sequences
    - Paged attention for decode phase
    - Block-based KV cache management

    Usage:
        # Initialize
        attn = SelfAttentionVarlen(config, dtype=jnp.bfloat16)

        # For model_runner integration:
        # 1. Pass external k_cache and v_cache from model_runner
        # 2. The attention layer will use context from get_context()
        # 3. Returns output and updated caches
    """

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        block_size: int = 256,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.dtype = dtype
        self.block_size = block_size

        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        # Always calculate head_dim from hidden_size and num_heads
        # Do NOT use config.head_dim as it may be incorrect for GQA models
        self.head_dim = getattr(config, "head_dim", hidden_size // self.num_heads)
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

        # RoPE configuration
        self.max_position = getattr(config, "max_position_embeddings", 4096)
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        # Projections
        self.q_proj = Linear(
            hidden_size,
            self.num_heads * self.head_dim,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.k_proj = Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.v_proj = Linear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim,
            hidden_size,
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )

        # Layer norms for Q and K (applied on head_dim after reshape)
        self.q_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

        # Varlen attention module
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            block_size=block_size,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: Optional[jax.Array] = None,
        k_cache: Optional[jnp.ndarray] = None,
        v_cache: Optional[jnp.ndarray] = None,
    ) -> tuple[jax.Array, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Forward pass using varlen attention.

        Args:
            hidden_states: [num_tokens, hidden_size] - flattened sequences
            positions: [num_tokens] - position indices (optional, for RoPE)
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim] (optional)
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim] (optional)

        Returns:
            output: [num_tokens, hidden_size]
            k_cache: updated key cache
            v_cache: updated value cache

        Note:
            Context should be set via set_context() before calling this.
            The context includes cu_seqlens_q, cu_seqlens_k, slot_mapping, etc.
        """
        # Project Q, K, V
        q = self.q_proj(hidden_states)  # [num_tokens, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [num_tokens, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [num_tokens, num_kv_heads * head_dim]

        # Reshape for normalization
        num_tokens = hidden_states.shape[0]
        q = q.reshape(num_tokens, self.num_heads, self.head_dim)
        k = k.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.reshape(num_tokens, self.num_kv_heads, self.head_dim)

        # Normalize Q and K
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if positions provided
        if positions is not None:
            # Build cos/sin for the maximum position we need
            max_pos = int(jnp.max(positions)) + 1
            cos, sin = _build_rope_cos_sin(
                max_pos, self.head_dim, self.rope_theta, dtype=self.dtype
            )
            # Index cos/sin by positions to get only the positions we need
            # This ensures cos/sin shape matches the number of input tokens
            cos = cos[positions]  # [num_tokens, head_dim//2]
            sin = sin[positions]  # [num_tokens, head_dim//2]

            # Expand for batch and heads dimensions
            q_expanded = q[None, :, :, :]  # [1, num_tokens, num_heads, head_dim]
            k_expanded = k[None, :, :, :]  # [1, num_tokens, num_kv_heads, head_dim]

            q_expanded = _apply_rope(q_expanded, cos, sin)
            k_expanded = _apply_rope(k_expanded, cos, sin)

            q = q_expanded[0]  # [num_tokens, num_heads, head_dim]
            k = k_expanded[0]  # [num_tokens, num_kv_heads, head_dim]

        # Flatten back for attention
        q = q.reshape(num_tokens, -1)  # [num_tokens, num_heads * head_dim]
        k = k.reshape(num_tokens, -1)  # [num_tokens, num_kv_heads * head_dim]
        v = v.reshape(num_tokens, -1)  # [num_tokens, num_kv_heads * head_dim]

        # Apply varlen attention
        # Context must be set via set_context() before this call
        attn_output, k_cache, v_cache = self.attn(q, k, v, k_cache, v_cache)

        # Output projection
        output = self.o_proj(attn_output)

        return output, k_cache, v_cache


class Qwen3DecoderLayerVarlen(nnx.Module):
    """Decoder layer using varlen attention with external KV cache.

    This version is designed for model_runner.py integration.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        block_size: int = 256,
        rngs: nnx.Rngs | None = None,
    ):
        self.config = config
        self.layer_id = layer_id
        self.dtype = dtype
        self.block_size = block_size

        eps = getattr(config, "rms_norm_eps", 1e-6)
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        self.self_attn = SelfAttentionVarlen(
            config, dtype=dtype, block_size=block_size, rngs=rngs
        )
        self.mlp = MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        positions: Optional[jax.Array] = None,
        k_cache: Optional[jnp.ndarray] = None,
        v_cache: Optional[jnp.ndarray] = None,
    ) -> tuple[jax.Array, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Forward pass with varlen attention.

        Args:
            hidden_states: [num_tokens, hidden_size]
            positions: [num_tokens] position indices
            k_cache: layer's key cache
            v_cache: layer's value cache

        Returns:
            output: [num_tokens, hidden_size]
            k_cache: updated key cache
            v_cache: updated value cache
        """
        # Self-attention block
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, k_cache, v_cache = self.self_attn(
            x, positions=positions, k_cache=k_cache, v_cache=v_cache
        )
        x = x + residual

        # MLP block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual

        return x, k_cache, v_cache


class Qwen3ModelVarlen(nnx.Module):
    """Qwen3 model with varlen attention and external KV cache.

    This version is designed for model_runner.py integration with:
    - Flattened tensor inputs
    - External block-based KV cache
    - Variable-length sequence support
    """

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        block_size: int = 256,
        rngs: nnx.Rngs | None = None,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.block_size = block_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            rngs=rngs,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.layers = nnx.List(
            [
                Qwen3DecoderLayerVarlen(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    block_size=block_size,
                    rngs=rngs,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=eps, dtype=dtype)

    def __call__(
        self,
        input_ids: jax.Array,
        positions: Optional[jax.Array] = None,
        kv_caches: Optional[list[tuple[jnp.ndarray, jnp.ndarray]]] = None,
    ) -> tuple[jax.Array, list[tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass with varlen attention.

        Args:
            input_ids: [num_tokens] or [batch, seq] token ids
            positions: [num_tokens] position indices (optional)
            kv_caches: List of (k_cache, v_cache) tuples for each layer

        Returns:
            hidden_states: [num_tokens, hidden_size]
            updated_kv_caches: List of updated (k_cache, v_cache) tuples
        """
        # Handle both flattened [num_tokens] and batched [batch, seq] formats
        if input_ids.ndim == 2:
            batch_size, seq_len = input_ids.shape
            input_ids = input_ids.reshape(-1)  # Flatten to [num_tokens]

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)  # [num_tokens, hidden_size]

        # Initialize kv_caches if not provided
        if kv_caches is None:
            kv_caches = [(None, None)] * len(self.layers)

        updated_kv_caches = []

        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[layer_idx]
            hidden_states, k_cache, v_cache = layer(
                hidden_states,
                positions=positions,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            updated_kv_caches.append((k_cache, v_cache))

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, updated_kv_caches


class Qwen3ForCausalLMVarlen(nnx.Module):
    """Qwen3 for causal LM with varlen attention support.

    This version is optimized for model_runner.py with:
    - Varlen attention for flattened sequences
    - External KV cache management
    - Compatible with prepare_prefill/prepare_decode

    Usage with model_runner.py:
        model = Qwen3ForCausalLMVarlen(config, dtype=jnp.bfloat16, block_size=256)

        # model_runner will:
        # 1. Call prepare_prefill() or prepare_decode() to set context
        # 2. Pass flattened input_ids and positions
        # 3. Manage external KV caches per layer
    """

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        block_size: int = 256,
        rngs: nnx.Rngs | None = None,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.block_size = block_size

        self.transformers = Qwen3ModelVarlen(
            config, dtype=self.dtype, block_size=block_size, rngs=rngs, mesh=mesh
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, dtype=self.dtype, rngs=rngs
        )

    def __call__(
        self,
        input_ids: jax.Array,
        positions: Optional[jax.Array] = None,
        kv_caches: Optional[list[tuple[jnp.ndarray, jnp.ndarray]]] = None,
    ) -> tuple[jax.Array, list[tuple[jnp.ndarray, jnp.ndarray]]]:
        """Forward pass with varlen attention.

        Args:
            input_ids: [num_tokens] or [batch, seq] token ids
            positions: [num_tokens] position indices
            kv_caches: List of (k_cache, v_cache) tuples for each layer

        Returns:
            logits: [num_tokens, vocab_size] or [batch, seq, vocab_size]
            updated_kv_caches: List of updated (k_cache, v_cache) tuples
        """
        hidden_states, updated_kv_caches = self.transformers(
            input_ids, positions=positions, kv_caches=kv_caches
        )
        logits = self.lm_head(hidden_states)
        return logits, updated_kv_caches

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Compute logits from hidden states (for compatibility)."""
        return self.lm_head(hidden_states)

    def load_weights(self, config):
        model_config = ModelConfig(model_path=config.model, trust_remote_code=True)

        weight_mappings = self._build_weight_mappings()
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(weight_mappings)

    def _build_weight_mappings(self):
        """Build full weight mappings for Qwen3-0.6B.

        Embedding / LM head use vocab-parallel sharding along the tensor axis:
        - PartitionSpec('tensor', None) on weight of shape [vocab, hidden]

        Decoder layers use column/row-parallel sharding:
        - Q/K/V/Gate/Up:  sharding=(None, 'tensor')
        - O/Down:         sharding=('tensor', None)
        """
        weight_mappings: dict[str, WeightMapping] = {
            # Vocab-parallel embedding and LM head
            "model.embed_tokens.weight": WeightMapping(
                target_path="transformers.embed_tokens.embedding",
                sharding=("tensor", None),
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.weight",
                sharding=("tensor", None),
            ),
        }

        # Decoder layer mappings
        num_layers = self.config.num_hidden_layers
        for layer_id in range(num_layers):
            hf_prefix = f"model.layers.{layer_id}"
            nnx_prefix = f"transformers.layers.{layer_id}"

            # LayerNorms (replicated)
            weight_mappings[f"{hf_prefix}.input_layernorm.weight"] = WeightMapping(
                target_path=f"{nnx_prefix}.input_layernorm.weight",
            )
            weight_mappings[f"{hf_prefix}.post_attention_layernorm.weight"] = (
                WeightMapping(
                    target_path=f"{nnx_prefix}.post_attention_layernorm.weight",
                )
            )

            # Self-attention projections
            for proj in ["q_proj", "k_proj", "v_proj"]:
                hf_key = f"{hf_prefix}.self_attn.{proj}.weight"
                target = f"{nnx_prefix}.self_attn.{proj}.weight"
                if proj in ["k_proj", "v_proj"]:
                    weight_mappings[hf_key] = WeightMapping(
                        target_path=target,
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    )
                else:
                    weight_mappings[hf_key] = WeightMapping(
                        target_path=target,
                        sharding=(None, "tensor"),
                        transpose=True,
                    )

            weight_mappings[f"{hf_prefix}.self_attn.q_norm.weight"] = WeightMapping(
                target_path=f"{nnx_prefix}.self_attn.q_norm.weight",
            )

            weight_mappings[f"{hf_prefix}.self_attn.k_norm.weight"] = WeightMapping(
                target_path=f"{nnx_prefix}.self_attn.k_norm.weight",
            )

            # Output projection (row-parallel)
            weight_mappings[f"{hf_prefix}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{nnx_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            )

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                hf_key = f"{hf_prefix}.mlp.{proj}.weight"
                target = f"{nnx_prefix}.mlp.{proj}.weight"
                weight_mappings[hf_key] = WeightMapping(
                    target_path=target,
                    sharding=(None, "tensor"),
                    transpose=True,
                )

            # weight_mappings[f"{hf_prefix}.mlp.down_proj.weight"] = WeightMapping(
            #     target_path=f"{nnx_prefix}.mlp.down_proj.weight",
            #     sharding=("tensor", None),
            # )

        # Final layer norm
        weight_mappings["model.norm.weight"] = WeightMapping(
            target_path="transformers.final_layernorm.weight",
        )

        return weight_mappings
