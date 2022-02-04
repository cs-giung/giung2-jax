import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Iterable, Optional, Tuple, Union


PRNGKey = Any
Shape = Iterable[int]
Array = Any
Dtype = Any


__all__ = [
    "Conv2d",
]


class Conv2d(nn.Module):
    channels: int
    kernel_size: int
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    num_groups: int = 1
    use_bias: bool = True
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.kaiming_normal()
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, **kwargs):
        in_channels = x.shape[-1]
        w_shape = (self.kernel_size, self.kernel_size, in_channels // self.num_groups, self.channels,)
        padding = self.padding if isinstance(self.padding, str) else [
            (self.padding, self.padding,), (self.padding, self.padding,),
        ]
        w = jnp.asarray(self.param('w', self.w_init, w_shape), x.dtype)
        y = jax.lax.conv_general_dilated(
            x, w, (self.stride, self.stride,), padding,
            lhs_dilation=(1,1,), rhs_dilation=(1,1,),
            dimension_numbers   = jax.lax.ConvDimensionNumbers((0,3,1,2,), (3,2,0,1,), (0,3,1,2,)),
            feature_group_count = self.num_groups,
        )
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.channels,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,)))
        return y
