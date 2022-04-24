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
    "Conv2d_WeightStandardization",
    "Conv2d_SpatialDropout",
    "Conv2d_SpatialGaussianDropout",
    "Conv2d_BatchEnsemble",
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
        """
        Args:
            x (Array): An input array with shape [N, H1, W1, C1,].
        
        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
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


class Conv2d_WeightStandardization(nn.Module):
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
        """
        Args:
            x (Array): An input array with shape [N, H1, W1, C1,].
        
        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
        in_channels = x.shape[-1]
        w_shape = (self.kernel_size, self.kernel_size, in_channels // self.num_groups, self.channels,)
        padding = self.padding if isinstance(self.padding, str) else [
            (self.padding, self.padding,), (self.padding, self.padding,),
        ]
        w = jnp.asarray(self.param('w', self.w_init, w_shape), x.dtype)
        w = w - jnp.mean(w, axis=(0, 1, 2,))
        w = w / (jnp.std(w, axis=(0, 1, 2,)) + 1e-5)
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


class Conv2d_BatchEnsemble(nn.Module):
    ensemble_size: int
    channels: int
    kernel_size: int
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    num_groups: int = 1
    use_bias: bool = True
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.kaiming_normal()
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros
    r_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones
    s_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Args:
            x (Array): An input array with shape [N, H1, W1, C1,].
        
        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
        in_channels = x.shape[-1]
        w_shape = (self.kernel_size, self.kernel_size, in_channels // self.num_groups, self.channels,)
        padding = self.padding if isinstance(self.padding, str) else [
            (self.padding, self.padding,), (self.padding, self.padding,),
        ]

        r = jnp.asarray(self.param('r', self.r_init, (self.ensemble_size, x.shape[-1],)), x.dtype)
        x = jnp.reshape(x, (self.ensemble_size, x.shape[0] // self.ensemble_size, x.shape[1], x.shape[2], -1))
        x = jnp.multiply(x, jnp.reshape(r, (self.ensemble_size, 1, 1, 1, -1,)))

        w = jnp.asarray(self.param('w', self.w_init, w_shape), x.dtype)
        y = jax.lax.conv_general_dilated(
            jnp.reshape(
                x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], -1)
            ), w, (self.stride, self.stride,), padding,
            lhs_dilation=(1,1,), rhs_dilation=(1,1,),
            dimension_numbers   = jax.lax.ConvDimensionNumbers((0,3,1,2,), (3,2,0,1,), (0,3,1,2,)),
            feature_group_count = self.num_groups,
        )
        if self.use_bias:
            b = jnp.asarray(self.param('b', self.b_init, (self.channels,)), x.dtype)
            y = jnp.add(y, jnp.reshape(b, (1, 1, 1, -1,)))

        s = jnp.asarray(self.param('s', self.s_init, (self.ensemble_size, y.shape[-1],)), x.dtype)
        y = jnp.reshape(y, (self.ensemble_size, y.shape[0] // self.ensemble_size, y.shape[1], y.shape[2], -1))
        y = jnp.multiply(y, jnp.reshape(s, (self.ensemble_size, 1, 1, 1, -1,)))
        y = jnp.reshape(y, (y.shape[0] * y.shape[1], y.shape[2], y.shape[3], -1))
        return y


class Conv2d_SpatialDropout(nn.Module):
    channels: int
    kernel_size: int
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    num_groups: int = 1
    use_bias: bool = True
    drop_rate: float = 0.5
    deterministic: Optional[bool] = None
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.kaiming_normal()
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Args:
            x (Array): An input array with shape [N, H1, W1, C1,].
        
        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
        deterministic = kwargs.pop('deterministic', True)
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        if not deterministic:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.drop_rate
            mask = jnp.broadcast_to(
                jax.random.bernoulli(
                    rng, p=keep, shape=x[:, :1, :1, :].shape
                ), x.shape)
            x = jax.lax.select(mask, x / keep, jnp.zeros_like(x))

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


class Conv2d_SpatialGaussianDropout(nn.Module):
    channels: int
    kernel_size: int
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    num_groups: int = 1
    use_bias: bool = True
    drop_rate: float = 0.5
    deterministic: Optional[bool] = None
    w_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.kaiming_normal()
    b_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x, **kwargs):
        """
        Args:
            x (Array): An input array with shape [N, H1, W1, C1,].
        
        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
        deterministic = kwargs.pop('deterministic', True)
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        if not deterministic:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.drop_rate
            mask = jnp.ones_like(x[:, :1, :1, :]) + jax.random.normal(
                rng, shape=x[:, :1, :1, :].shape
            ) * jnp.sqrt(keep / (1 - keep))
            x = jnp.multiply(x, mask)

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
