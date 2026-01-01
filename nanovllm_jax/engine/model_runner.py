import jax
import jax.numpy as jnp
from typing import List, Tuple
from nanovllm_jax.config import Config
from nanovllm_jax.engine.sequence import Sequence
from nanovllm_jax.layers.sampler import Sampler
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLMVarlen
from nanovllm_jax.utils.context import set_context, get_context, reset_context


import logging

logger = logging.getLogger(__name__)


class ModelRunnerVarlen:
    """JAX model runner with varlen attention support.

    This version uses Qwen3ForCausalLMVarlen with:
    - External KV cache management per layer
    - Varlen attention for flattened sequences
    - Paged attention for decode phase
    """

    def __init__(self, config: Config, rank: int = 0):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        # Initialize JAX devices
        self.devices = jax.devices()
        logger.info(f"Node has devices: {self.devices}")
        if len(self.devices) > 1:
            self.device = self.devices[rank % len(self.devices)]
        else:
            self.device = self.devices[0]

        # Create a simple 1D mesh over available devices
        self.mesh = jax.sharding.Mesh(self.devices[: self.world_size], ("tensor",))

        # Set default dtype
        self.default_dtype = jnp.bfloat16

        # Initialize model with varlen support, and load model weights
        # Need to activate mesh context for sharded parameter initialization
        with self.mesh:
            self.model = Qwen3ForCausalLMVarlen(
                config=hf_config,
                dtype=self.default_dtype,
                block_size=self.block_size,
                rngs=None,
                mesh=self.mesh,
            )
        # Load pretrained weights
        self.model.load_weights(self.config)
        logger.info(f"{self.config.model} Weights loaded finished.")
        # Initialize sampler
        self.sampler = Sampler()

        # Warmup and allocate KV cache
        self.warmup_model()
        self.allocate_kv_cache()

        # 暂时移除 JIT 编译，看看是否还有其他问题
        # self._create_jitted_functions()

    def warmup_model(self):
        """Warmup model to measure peak memory."""
        pass

    def _create_jitted_functions(self):
        """创建 JIT 编译的模型函数。

        在 Tensor Parallel 中，with_sharding_constraint 必须在 JIT 编译的函数中
        才能触发自动的集合通信（all-reduce 等）。
        """
        logger.info("Creating JIT compiled forward function...")

        # 在 mesh 上下文中使用 nnx.jit 编译模型
        with self.mesh:
            # nnx.jit 会自动处理模型的状态管理
            self.model = nnx.jit(self.model)

        logger.info("JIT compilation completed")

    def allocate_kv_cache(self):
        """Allocate external KV cache for each layer."""
        config = self.config
        hf_config = config.hf_config

        # 应该使用 ModelConfig 的方法来获取正确的 KV heads 数量
        from nanovllm_jax.configs.model_config import ModelConfig

        mc = ModelConfig(model_path=config.model, trust_remote_code=True)
        # 获取全局的 KV heads 数量，通过tp 切分
        num_kv_heads = mc.get_total_num_kv_heads()

        head_dim = (
            hf_config.head_dim
            if hasattr(hf_config, "head_dim")
            else hf_config.hidden_size // hf_config.num_attention_heads
        )

        # Get dtype from hf_config
        torch_dtype = getattr(hf_config, "torch_dtype", None)
        if torch_dtype is None:
            dtype = jnp.float32
        else:
            dt_str = str(torch_dtype)
            if "bfloat16" in dt_str:
                dtype = jnp.bfloat16
            elif "float16" in dt_str or "half" in dt_str:
                dtype = jnp.float16
            else:
                dtype = jnp.float32

        # Use conservative default if not specified
        if config.num_kvcache_blocks <= 0:
            config.num_kvcache_blocks = 100

        assert config.num_kvcache_blocks > 0

        # Allocate KV cache for each layer
        num_layers = hf_config.num_hidden_layers
        self.kv_caches = []
        # num_kv_heads 已经是按照tp 切分过的数据了

        from jax.sharding import PartitionSpec as P, NamedSharding

        for _ in range(num_layers):
            # KV Cache 需要在 KV head 维度上分片
            # shape: [num_blocks, block_size, num_kv_heads, head_dim]
            # sharding: (None, None, "tensor", None) - 在 num_kv_heads 维度分片
            k_cache = jnp.zeros(
                (
                    config.num_kvcache_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                ),
                dtype=dtype,
            )
            v_cache = jnp.zeros(
                (
                    config.num_kvcache_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                ),
                dtype=dtype,
            )

            # 将 KV cache 分片到各个设备
            # 因为 K/V 投影权重是按 num_kv_heads 维度切分的
            kv_sharding = NamedSharding(self.mesh, P(None, None, "tensor", None))
            k_cache = jax.device_put(k_cache, kv_sharding)
            v_cache = jax.device_put(v_cache, kv_sharding)

            logger.debug(f"KV cache sharding: {k_cache.sharding}, {v_cache.sharding}")

            self.kv_caches.append((k_cache, v_cache))

    def prepare_block_tables(self, seqs: List[Sequence]) -> jnp.ndarray | None:
        """Prepare block tables from sequences."""
        if not seqs:
            return jnp.array([], dtype=jnp.int32)
        max_len = (
            max(len(seq.block_table) for seq in seqs)
            if any(seq.block_table for seq in seqs)
            else 0
        )
        if max_len == 0:
            return None
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = jnp.array(block_tables, dtype=jnp.int32)
        return block_tables

    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for prefill phase."""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        # Check for prefix cache
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        input_ids = jnp.array(input_ids, dtype=jnp.int64)
        positions = jnp.array(positions, dtype=jnp.int64)
        cu_seqlens_q = jnp.array(cu_seqlens_q, dtype=jnp.int32)
        cu_seqlens_k = jnp.array(cu_seqlens_k, dtype=jnp.int32)
        slot_mapping = (
            jnp.array(slot_mapping, dtype=jnp.int32) if slot_mapping else None
        )

        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: List[Sequence]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prepare inputs for decode phase."""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        input_ids = jnp.array(input_ids, dtype=jnp.int64)
        positions = jnp.array(positions, dtype=jnp.int64)
        slot_mapping = jnp.array(slot_mapping, dtype=jnp.int32)
        context_lens = jnp.array(context_lens, dtype=jnp.int32)
        block_tables = self.prepare_block_tables(seqs)

        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: List[Sequence]) -> jnp.ndarray:
        """Prepare sampling parameters."""
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = jnp.array(temperatures, dtype=jnp.float32)
        return temperatures

    def run_model(
        self, input_ids: jnp.ndarray, positions: jnp.ndarray, is_prefill: bool
    ) -> jnp.ndarray:
        """Run the model forward pass with varlen attention."""
        # 在 mesh 上下文中执行
        # 即使没有 JIT，with_sharding_constraint 也会触发 all-reduce
        with self.mesh:
            from jax.sharding import PartitionSpec as P

            # 确保输入是 replicated 的
            input_ids = jax.lax.with_sharding_constraint(input_ids, P(None))
            positions = jax.lax.with_sharding_constraint(positions, P(None))

            logger.debug(
                f"Input IDs shape: {input_ids.shape} sharding {input_ids.sharding}, Positions shape: {positions.shape} sharding {positions.sharding}"
            )

            # 直接调用模型（不使用 JIT）
            logits, self.kv_caches = self.model(
                input_ids,
                positions=positions,
                kv_caches=self.kv_caches,
            )

        # Extract logits for sampling
        if is_prefill:
            context = get_context()
            if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                # Get the indices of the last token for each sequence
                last_indices = []
                for i in range(1, len(context.cu_seqlens_q)):
                    last_idx = int(context.cu_seqlens_q[i]) - 1
                    last_indices.append(last_idx)
                logits = logits[jnp.array(last_indices), :]
            else:
                # Single sequence, take last token
                logits = logits[-1:, :]
        # For decode, logits are already [batch_size, vocab_size]

        return logits

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int] | None:
        """Run inference on sequences."""
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        # Sample tokens
        if self.rank == 0:
            import time

            rng = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
            token_ids = self.sampler(logits, temperatures, rng).tolist()
        else:
            token_ids = None

        reset_context()
        return token_ids
