from collections import deque
import xxhash
import numpy as np

from nanovllm_jax.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        # block id
        self.block_id = block_id
        # 当有缓存时，该block 可能被其他所引用，ref_count 就是被引用的次数
        self.ref_count = 0
        # 比较两个block是否相等
        self.hash = -1
        # block 内的具体tokens值
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 调度器初始化的时候进行初始化，分配总共会有多少block，然后开辟出来，供后面分配使用
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        # 每个block 多大
        self.block_size = block_size
        # 总共多少个block
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hashid -> block_id
        self.hash_to_block_id: dict[int, int] = dict()
        # 有多少个block 空闲可以被分配
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 有多少个block 被使用了
        self.used_block_ids: set[int] = set()
        # 上面两个构造成双端对列

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        # 该block_id 对应的block 被分配了。
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # prefill的时候，得看一下Kvcache 还够不够给它分配
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    # 只会执行一次，也就是在初始化的时候执行一些(准备做prefill)
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                # 如果block 填充满了，则计算hash，不然设置-1,先填满
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:  # 该block_id 对应的block 已经存在，说明命中了缓存, 那么该sequence更新命中的cache数量
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
    # decode阶段执行
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    # decode阶段执行
    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
