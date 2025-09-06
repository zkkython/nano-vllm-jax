from itertools import count
from tkinter import SE
import unittest

from vllm_jax.engine.sequence import Sequence
import logging


logging.basicConfig(
    level=logging.DEBUG,
    filemode="w",
    format=f"%(levelname)s-%(asctime)s-%(filename)s-%(funcName)s: %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
)
log = logging.getLogger(__name__)


class TestSequence(unittest.TestCase):
    def test_sequence(self):
        # Test the sequence function
        s = Sequence(token_ids=[1, 2, 3])
        log.info(f"len = {len(s)}")
        self.assertEqual(len(s), 3)

        s2 = Sequence(token_ids=list(range((10))))
        log.info(f"seq_id = {s2.seq_id}")
        self.assertEqual(s2.seq_id, 1)
        c: count[int] = Sequence.counter
        log.info(c)


if __name__ == "__main__":
    unittest.main()
