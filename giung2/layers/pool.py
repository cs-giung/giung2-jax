import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Iterable, Optional, Tuple, Union


PRNGKey = Any
Shape = Iterable[int]
Array = Any
Dtype = Any


__all__ = [
    "MaxPool2d",
    "AvgPool2d",
]


class MaxPool2d(nn.Module):
    kernel_size: int
    stride: int
    padding: Union[str, int] = 'VALID'

    @nn.compact
    def __call__(self, x, **kwargs):
        strides = (1, self.stride, self.stride, 1,)
        padding = self.padding
        if not isinstance(padding, str):
            padding = ((0, 0,), (self.padding, self.padding,), (self.padding, self.padding,), (0, 0,))
        kernel_sizes = (1, self.kernel_size, self.kernel_size, 1,)
        y = jax.lax.reduce_window(x, -jnp.inf, jax.lax.max, kernel_sizes, strides, padding)
        return y


class AvgPool2d(nn.Module):
    kernel_size: int
    stride: int
    padding: Union[str, int] = 'VALID'

    @nn.compact
    def __call__(self, x, **kwargs):
        strides = (1, self.stride, self.stride, 1,)
        padding = self.padding
        if not isinstance(padding, str):
            padding = ((0, 0,), (self.padding, self.padding,), (self.padding, self.padding,), (0, 0,))
        kernel_sizes = (1, self.kernel_size, self.kernel_size, 1,)
        y = jax.lax.reduce_window(x, 0., jax.lax.add, kernel_sizes, strides, padding)
        y = y / (self.kernel_size * self.kernel_size)
        return y
