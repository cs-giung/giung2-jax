import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Iterable, Optional, Tuple, Union


PRNGKey = Any
Shape = Iterable[int]
Array = Any
Dtype = Any


__all__ = [
    "Identity",
    "BatchNorm2d",
    "FilterResponseNorm2d",
]


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x, **kwargs):
        return x


class BatchNorm2d(nn.Module):
    use_running_average: Optional[bool] = None
    momentum: float = 0.9
    epsilon: float = 1e-5
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, **kwargs):
        use_running_average = kwargs.pop("use_running_average", True)
        use_running_average = nn.merge_param('use_running_average', self.use_running_average, use_running_average)
        m = self.variable('batch_stats', 'm', jnp.zeros, (x.shape[-1],))
        v = self.variable('batch_stats', 'v', jnp.ones,  (x.shape[-1],))

        if use_running_average:
            mean, var = m.value, v.value
        else:
            mean = jnp.mean(x, (0, 1, 2,))
            sq_mean = jnp.mean(jax.lax.square(x), (0, 1, 2,))
            var = jnp.maximum(0., sq_mean - jax.lax.square(mean))
            if not self.is_mutable_collection('params'):
                m.value = self.momentum * m.value + (1 - self.momentum) * mean
                v.value = self.momentum * v.value + (1 - self.momentum) * var

        y = x - jnp.reshape(mean, (1, 1, 1, -1,))
        y = jnp.multiply(y, jnp.reshape(jax.lax.rsqrt(var + self.epsilon), (1, 1, 1, -1,)))

        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1],)), x.dtype)
        b = jnp.asarray(self.param('b', self.b_init, (x.shape[-1],)), x.dtype)
        y = jnp.multiply(y, jnp.reshape(w, (1, 1, 1, -1,)))
        y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,)))
        return y


class FilterResponseNorm2d(nn.Module):
    epsilon: float = 1e-6
    use_learnable_epsilon: bool = False
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    e_init: Callable[[PRNGKey, Shape, Dtype], Array] = lambda e1, e2: jnp.array([1e-4,]) # TODO: init const.
    t_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, **kwargs):
        if self.use_learnable_epsilon:
            e = jnp.asarray(self.param('e', self.e_init, (1,)), x.dtype)
            eps = self.epsilon + jnp.abs(e)
        else:
            eps = self.epsilon

        nu2 = jnp.mean(x**2, axis=(1, 2,), keepdims=True)
        y = x * jax.lax.rsqrt(nu2 + eps)

        w = jnp.asarray(self.param('w', self.w_init, (x.shape[-1],)), x.dtype)
        b = jnp.asarray(self.param('b', self.b_init, (x.shape[-1],)), x.dtype)
        y = jnp.multiply(y, jnp.reshape(w, (1, 1, 1, -1,)))
        y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,)))

        t = jnp.asarray(self.param('t', self.t_init, (x.shape[-1],)), x.dtype)
        z = jnp.maximum(y, t)
        return z
