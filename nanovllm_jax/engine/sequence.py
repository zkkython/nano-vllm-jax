from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm_jax.sampling_params import SamplingParams


# request->sequence 状态
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        # 每个请求的唯一id
        self.seq_id = next(Sequence.counter)
        # 请求的状态
        self.status = SequenceStatus.WAITING
        # 请求的 token 列表，初始列表，随着prefill和decode 列表会动态变更
        self.token_ids = copy(token_ids)
        # 请求的最后一个 token
        self.last_token = token_ids[-1]
        # 值会动态变化，因为prefill和decode会动态增加
        self.num_tokens = len(self.token_ids)
        # 请求的 prompt token 数量，不会动态变化
        self.num_prompt_tokens = len(token_ids)
        # prompt前面缓存了多少个token
        self.num_cached_tokens = 0
        # sequece 横跨了多少个block
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    # 推理是否结束
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # 完成的token数量
    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    # 命中的缓存token 占了几个block, 一个block 256个slot
    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # 获取第i个block 有哪些tokens
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    # 添加一个token
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
        ) = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
