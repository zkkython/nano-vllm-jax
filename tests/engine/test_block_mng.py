import unittest

from vllm_jax.engine.block_mng import BlockManager
from vllm_jax.engine.sequence import Sequence
from vllm_jax.sampling_params import SamplingParams
from vllm_jax.utils.log import logger


class TestBlockManager(unittest.TestCase):

    def test_xxhash(self):
        hash = BlockManager.compute_hash([1, 2, 3, 4, 5])
        hash2 = BlockManager.compute_hash_byarray([1, 2, 3, 4, 5])
        logger.info(f"hash = {hash}, hash2 = {hash2}")
        self.assertEqual(hash, hash2)

    def test_add_block(self):
        bm = BlockManager(num_blocks=10, block_size=256)
        block = bm._allocate_block(0)
        block.update(hash=12345, token_ids=[1, 2, 3])
        self.assertEqual(block.hash, 12345)
        self.assertEqual(block.token_ids, [1, 2, 3])
        self.assertEqual(block.ref_count, 1)
        self.assertIn(0, bm.used_block_ids)
        self.assertNotIn(0, bm.free_block_ids)

    def test_add_block_for_sequence(self):
        bm = BlockManager(num_blocks=10, block_size=256)
        seq = Sequence(token_ids=list(range(600)), sampling_params=SamplingParams())
        bm.allocate(seq)
        logger.info(f"seq.block_table = {seq.block_table}")
        self.assertEqual(len(seq.block_table), 3)

        for hash, block in bm.hash_to_block_id.items():
            logger.info(f"hash = {hash}, block_id = {block}")

        self.assertEqual(len(bm.used_block_ids), 3)
        self.assertEqual(len(bm.free_block_ids), 7)

    def test_block_boundery(self):
        bm = BlockManager(num_blocks=4, block_size=256)
        with self.assertRaises(RuntimeError) as cm:
            seq = Sequence(token_ids=list(range(600)), sampling_params=SamplingParams())
            bm.allocate(seq)
            seq2 = Sequence(
                token_ids=list(range(800)), sampling_params=SamplingParams()
            )
            bm.allocate(seq2)
            logger.info(f"seq.block_table = {seq.block_table}")
            logger.info(f"seq2.block_table = {seq2.block_table}")
        self.assertIn("No free blocks available", str(cm.exception))

    def test_block_can_allocate(self):
        bm = BlockManager(num_blocks=4, block_size=256)
        seq = Sequence(token_ids=list(range(600)), sampling_params=SamplingParams())
        bm.allocate(seq)
        seq2 = Sequence(token_ids=list(range(800)), sampling_params=SamplingParams())
        self.assertFalse(bm.can_allocate(seq2))

    def test_seq_cache_tokens(self):
        bm = BlockManager(num_blocks=10, block_size=256)
        seq = Sequence(token_ids=list(range(600)), sampling_params=SamplingParams())
        bm.allocate(seq)
        seq2 = Sequence(token_ids=list(range(800)), sampling_params=SamplingParams())
        bm.allocate(seq2)
        self.assertTrue(seq2.num_cached_tokens, 512)
        self.assertEqual(seq2.num_cached_blocks, 2)


if __name__ == "__main__":
    unittest.main()
