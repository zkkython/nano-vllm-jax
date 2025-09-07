from collections import deque
from typing import Dict, List, Set
import xxhash
import numpy as np

from vllm_jax.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # at least one reference to itself
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        for token_id in token_ids:
            h.update(token_id.to_bytes(8, "little"))
        return h.intdigest()

    @classmethod
    def compute_hash_byarray(cls, token_ids: List[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block: Block = self.blocks[block_id]
        # make sure the block is not used
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> int:
        block: Block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash in self.hash_to_block_id:
            del self.hash_to_block_id[block.hash]
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        return block_id

    def can_allocate(self, seq: Sequence) -> bool:
        # whether consider the cached blocks
        # if consider:
        # needed_blocks = seq.num_blocks - seq.num_cached_blocks
        # return needed_blocks <= len(self.free_block_ids)
        # else:
        # just verify: seq.num_blocks <= len(self.free_block_ids)
        return seq.num_blocks <= len(self.free_block_ids)

    # allocate blocks for a complete sequence
    def allocate(self, seq: Sequence):
        # make sure the sequence is not allocated, equivalent to len(seq.block_table) == 0
        assert not seq.block_table
        h = -1
        for i in range(seq.num_blocks):
            # get the the i-th block
            token_ids = seq.block(i)
            # when the block is not full, we needn't compute the hash
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            if h in self.hash_to_block_id:

                seq.num_cached_tokens += self.block_size
                block_id = self.hash_to_block_id[h]
                block = self.blocks[block_id]
                assert block.hash == h
                block.ref_count += 1
                seq.block_table.append(block_id)

            else:

                if not self.free_block_ids:
                    # no free blocks
                    raise RuntimeError("No free blocks available")
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
                if h != -1:
                    block.update(h, token_ids)
                    self.hash_to_block_id[h] = block_id
                seq.block_table.append(block_id)

    # for the sequence that has been allocated, deallocate it
    def deallocate(self, seq: Sequence):

        # remove blocks in reverse order
        # because the previous block may be reused by the later block, so, we need to remove the later block first
        for i in reversed(seq.block_table):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table = []

    def can_append(self, seq: Sequence) -> bool:
        if seq.is_finished():
            return False
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence) -> bool:
        if seq.is_finished():
            return False
        block_table = seq.block_table
        # the last_block may be these conditions
        # 1. the last block is full, it has been in hash_to_block_id
        # 2. the last block is not full, it is not in hash_to_block_id
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block((seq.num_blocks - 1))
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash_byarray(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
