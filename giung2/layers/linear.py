import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Iterable, Optional, Tuple, Union


PRNGKey = Any
Shape = Iterable[int]
Array = Any
Dtype = Any


__all__ = [
    "Linear",
    "Linear_BatchEnsemble",
]


class Linear(nn.Module):
    features: int
    use_bias: bool = True
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.kaiming_normal()
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Args:
            x (Array): An input array with shape [N, C1,].
        
        Returns:
            y (Array): An output array with shape [N, C2,].
        """
        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1], self.features,)), x.dtype)
        y = jnp.dot(x, w)
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.features,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, -1,)))
        return y


class Linear_BatchEnsemble(nn.Module):
    ensemble_size: int
    features: int
    use_bias: bool = True
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.kaiming_normal()
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    r_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones
    s_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Args:
            x (Array): An input array with shape [N, C1,].
        
        Returns:
            y (Array): An output array with shape [N, C2,].
        """
        r = jnp.asarray(self.param('r', self.r_init, (self.ensemble_size, x.shape[-1],)), x.dtype)
        x = jnp.reshape(x, (self.ensemble_size, x.shape[0] // self.ensemble_size, -1,))
        x = jnp.multiply(x, jnp.reshape(r, (self.ensemble_size, 1, -1,)))

        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1], self.features,)), x.dtype)
        y = jax.lax.dot_general(x, w, (((2,), (0,),), ((), (),)))
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.features,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, 1, -1,)))

        s = jnp.asarray(self.param('s', self.s_init, (self.ensemble_size, y.shape[-1],)), x.dtype)
        y = jnp.multiply(y, jnp.reshape(s, (self.ensemble_size, 1, -1,)))
        y = jnp.reshape(y, (y.shape[0] * y.shape[1], -1,))
        return y
