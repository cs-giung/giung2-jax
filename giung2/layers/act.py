import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Iterable, Optional, Tuple, Union


PRNGKey = Any
Shape = Iterable[int]
Array = Any
Dtype = Any


__all__ = [
    "Sigmoid",
    "ReLU",
    "SiLU",
]


class Sigmoid(nn.Module):
    @nn.compact
    def __call__(self, x, **kwargs):
        return jax.scipy.special.expit(x)


class ReLU(nn.Module):
    @nn.compact
    def __call__(self, x, **kwargs):
        return jnp.maximum(x, 0)


class SiLU(nn.Module):
    @nn.compact
    def __call__(self, x, **kwargs):
        return x * jax.scipy.special.expit(x)
