import jax
import torch
from torch import nn
import torch.nn.functional as F
from flax import nnx
from jax import numpy as jnp


class TorchSiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class SiluAndMul(nnx.Module):
    @jax.jit
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_part, y_part = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(x_part) * y_part
