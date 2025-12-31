import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional
from nanovllm_jax.utils.context import get_context


def store_kvcache(
    key: jnp.ndarray,
    value: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    slot_mapping: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Store key-value pairs into cache at specified slots.

    Args:
        key: [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] - global slot indices

    Returns:
        Updated k_cache and v_cache
    """
    num_tokens = key.shape[0]
    num_kv_heads = key.shape[1]
    head_dim = key.shape[2]

    # Flatten cache for easier indexing: [num_blocks * block_size, num_kv_heads, head_dim]
    k_cache_flat = k_cache.reshape(-1, num_kv_heads, head_dim)
    v_cache_flat = v_cache.reshape(-1, num_kv_heads, head_dim)

    # Update cache at specified slots
    k_cache_flat = k_cache_flat.at[slot_mapping].set(key)
    v_cache_flat = v_cache_flat.at[slot_mapping].set(value)

    # Reshape back to block structure
    k_cache = k_cache_flat.reshape(k_cache.shape)
    v_cache = v_cache_flat.reshape(v_cache.shape)

    return k_cache, v_cache


def varlen_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    causal: bool = True,
    block_table: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Variable-length attention for flattened sequences.

    Args:
        q: [total_tokens_q, num_heads, head_dim]
        k: [total_tokens_k, num_kv_heads, head_dim] or from cache
        v: [total_tokens_k, num_kv_heads, head_dim] or from cache
        cu_seqlens_q: [num_seqs + 1] cumulative sequence lengths for queries
        cu_seqlens_k: [num_seqs + 1] cumulative sequence lengths for keys
        max_seqlen_q: maximum sequence length in batch for queries
        max_seqlen_k: maximum sequence length in batch for keys
        scale: attention scale factor
        causal: whether to apply causal masking
        block_table: [num_seqs, max_blocks] for blocked KV cache (optional)

    Returns:
        output: [total_tokens_q, num_heads, head_dim]
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    num_seqs = len(cu_seqlens_q) - 1

    # Handle grouped-query attention: repeat k, v to match num_heads
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        k = jnp.repeat(k, repeat_factor, axis=1)
        v = jnp.repeat(v, repeat_factor, axis=1)

    # Process each sequence separately and concatenate results
    outputs = []

    for i in range(num_seqs):
        # Get sequence boundaries
        q_start = int(cu_seqlens_q[i])
        q_end = int(cu_seqlens_q[i + 1])
        k_start = int(cu_seqlens_k[i])
        k_end = int(cu_seqlens_k[i + 1])

        # Extract sequence
        q_seq = q[q_start:q_end]  # [seqlen_q, num_heads, head_dim]
        k_seq = k[k_start:k_end]  # [seqlen_k, num_heads, head_dim]
        v_seq = v[k_start:k_end]  # [seqlen_k, num_heads, head_dim]

        # Transpose for attention computation
        # [num_heads, seqlen_q/k, head_dim]
        q_seq = jnp.transpose(q_seq, (1, 0, 2))
        k_seq = jnp.transpose(k_seq, (1, 0, 2))
        v_seq = jnp.transpose(v_seq, (1, 0, 2))

        # Compute attention scores: [num_heads, seqlen_q, seqlen_k]
        scores = jnp.einsum("hqd,hkd->hqk", q_seq, k_seq) * scale

        # Apply causal mask if needed
        if causal:
            seqlen_q = q_seq.shape[1]
            seqlen_k = k_seq.shape[1]
            # Create causal mask accounting for different q and k lengths (for prefix caching)
            # The mask should allow attending to all k positions up to the corresponding q position
            q_indices = jnp.arange(seqlen_q)[:, None]
            k_indices = jnp.arange(seqlen_k)[None, :]
            # For prefix caching: q positions are offset by (seqlen_k - seqlen_q)
            offset = seqlen_k - seqlen_q
            mask = (q_indices + offset) >= k_indices
            scores = jnp.where(mask, scores, -jnp.inf)

        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention: [num_heads, seqlen_q, head_dim]
        out_seq = jnp.einsum("hqk,hkd->hqd", attn_weights, v_seq)

        # Transpose back: [seqlen_q, num_heads, head_dim]
        out_seq = jnp.transpose(out_seq, (1, 0, 2))

        outputs.append(out_seq)

    # Concatenate all sequence outputs
    output = jnp.concatenate(outputs, axis=0)  # [total_tokens_q, num_heads, head_dim]
    return output


def paged_attention(
    q: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    context_lens: jnp.ndarray,
    block_table: jnp.ndarray,
    scale: float,
    block_size: int = 256,
) -> jnp.ndarray:
    """Paged attention for decode phase.

    Args:
        q: [batch_size, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        context_lens: [batch_size] - length of context for each sequence
        block_table: [batch_size, max_blocks] - block indices for each sequence
        scale: attention scale factor
        block_size: size of each block

    Returns:
        output: [batch_size, num_heads, head_dim]
    """
    batch_size = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]

    # Handle grouped-query attention
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
    else:
        repeat_factor = 1

    outputs = []

    for i in range(batch_size):
        context_len = int(context_lens[i])
        num_blocks = (context_len + block_size - 1) // block_size

        # Gather KV from blocks
        k_seq_list = []
        v_seq_list = []

        for block_idx in range(num_blocks):
            block_num = int(block_table[i, block_idx])
            if block_num < 0:
                break

            # Get block data
            k_block = k_cache[block_num]  # [block_size, num_kv_heads, head_dim]
            v_block = v_cache[block_num]  # [block_size, num_kv_heads, head_dim]

            # For last block, only take valid tokens
            if block_idx == num_blocks - 1:
                valid_len = context_len - block_idx * block_size
                k_block = k_block[:valid_len]
                v_block = v_block[:valid_len]

            k_seq_list.append(k_block)
            v_seq_list.append(v_block)

        # Concatenate blocks: [context_len, num_kv_heads, head_dim]
        k_seq = jnp.concatenate(k_seq_list, axis=0)
        v_seq = jnp.concatenate(v_seq_list, axis=0)

        # Repeat for grouped-query attention
        if repeat_factor > 1:
            k_seq = jnp.repeat(k_seq, repeat_factor, axis=1)
            v_seq = jnp.repeat(v_seq, repeat_factor, axis=1)

        # q: [num_heads, head_dim], k/v: [context_len, num_heads, head_dim]
        q_i = q[i]  # [num_heads, head_dim]
        k_seq = jnp.transpose(k_seq, (1, 0, 2))  # [num_heads, context_len, head_dim]
        v_seq = jnp.transpose(v_seq, (1, 0, 2))  # [num_heads, context_len, head_dim]

        # Compute attention: [num_heads, context_len]
        scores = jnp.einsum("hd,hkd->hk", q_i, k_seq) * scale
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention: [num_heads, head_dim]
        out_i = jnp.einsum("hk,hkd->hd", attn_weights, v_seq)

        outputs.append(out_i)

    # Stack batch: [batch_size, num_heads, head_dim]
    output = jnp.stack(outputs, axis=0)
    return output


class Attention(nnx.Module):
    """Multi-head attention with KV cache support and varlen attention."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        block_size: int = 256,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.dtype = dtype

        # These will be set externally by model_runner
        # Using nnx.Variable for mutable state
        self.k_cache = nnx.Variable(None)
        self.v_cache = nnx.Variable(None)

    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        k_cache: Optional[jnp.ndarray] = None,
        v_cache: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Forward pass of attention mechanism.

        Args:
            q: [num_tokens, num_heads * head_dim]
            k: [num_tokens, num_kv_heads * head_dim]
            v: [num_tokens, num_kv_heads * head_dim]
            k_cache: [num_blocks, block_size, num_kv_heads, head_dim] (optional)
            v_cache: [num_blocks, block_size, num_kv_heads, head_dim] (optional)

        Returns:
            output: [num_tokens, num_heads * head_dim]
            k_cache: updated key cache (or None)
            v_cache: updated value cache (or None)
        """
        # Reshape inputs
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_kv_heads, self.head_dim)
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)

        # Get context for varlen attention
        context = get_context()

        # Store KV in cache if available
        if (
            k_cache is not None
            and v_cache is not None
            and context.slot_mapping is not None
        ):
            k_cache, v_cache = store_kvcache(
                k, v, k_cache, v_cache, context.slot_mapping
            )

        # Choose attention implementation based on context
        if context.is_prefill:
            # Prefill: use varlen attention
            if context.block_tables is not None:
                # Prefix caching: use cached KV
                # Flatten cache: [num_blocks * block_size, num_kv_heads, head_dim]
                k = k_cache.reshape(-1, self.num_kv_heads, self.head_dim)
                v = v_cache.reshape(-1, self.num_kv_heads, self.head_dim)

            o = varlen_attention(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            # Decode: use paged attention with KV cache
            o = paged_attention(
                q,
                k_cache,
                v_cache,
                context_lens=context.context_lens,
                block_table=context.block_tables,
                scale=self.scale,
                block_size=self.block_size,
            )

        # Reshape output: [num_tokens, num_heads * head_dim]
        o = o.reshape(-1, self.num_heads * self.head_dim)

        # Return output and updated caches
        return o, k_cache, v_cache
