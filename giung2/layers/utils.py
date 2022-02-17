import jax
import jax.numpy as jnp
import functools

from .act import *
from .conv import *
from .linear import *
from .norm import *


__all__ = [
    'get_norm2d_layers',
    'get_conv2d_layers',
    'get_linear_layers',
    'get_activation_layers',
]


def _constant(value=0., dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return value + jnp.zeros(shape, dtype)
    return init


def _normal(mean=0., stddev=1e-2, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return mean + jax.random.normal(key, shape, dtype) * stddev
    return init


def get_norm2d_layers(cfg, name: str):
    """
    Args:
        cfg (CfgNode): CfgNode instance that contains blueprint of layers.
        name (str): a name for norm2d layers.

    Returns:
        `nn.Module` instance.
    """
    if name == 'NONE':
        return Identity

    if name == 'BatchNorm2d':
        return functools.partial(
            BatchNorm2d,
            momentum = cfg.MODEL.BATCH_NORMALIZATION.MOMENTUM,
            epsilon  = cfg.MODEL.BATCH_NORMALIZATION.EPSILON,
        )

    if name == 'LayerNorm2d':
        return functools.partial(
            LayerNorm2d,
            epsilon  = cfg.MODEL.LAYER_NORMALIZATION.EPSILON,
        )

    if name == 'GroupNorm2d':
        return functools.partial(
            GroupNorm2d,
            num_groups = cfg.MODEL.GROUP_NORMALIZATION.NUM_GROUPS,
            epsilon    = cfg.MODEL.GROUP_NORMALIZATION.EPSILON,
        )

    if name == 'FilterResponseNorm2d':
        return functools.partial(
            FilterResponseNorm2d,
            epsilon               = cfg.MODEL.FILTER_RESPONSE_NORMALIZATION.EPSILON,
            use_learnable_epsilon = cfg.MODEL.FILTER_RESPONSE_NORMALIZATION.USE_LEARNABLE_EPSILON,
        )

    raise NotImplementedError(f'Unknown name for norm2d layers: {name}')


def get_conv2d_layers(cfg, name, use_bias=False):
    """
    Args:
        cfg (CfgNode): CfgNode instance that contains blueprint of layers.
        name (str): a name for conv2d layers.

    Returns:
        `nn.Module` instance.
    """
    if name == 'Conv2d':
        return functools.partial(
            Conv2d,
            use_bias = use_bias,
        )

    if name == 'Conv2d_WeightStandardization':
        return functools.partial(
            Conv2d_WeightStandardization,
            use_bias = use_bias,
        )

    if name == 'Conv2d_BatchEnsemble':

        init_fn_name = cfg.MODEL.BATCH_ENSEMBLE.INITIALIZER.NAME

        if init_fn_name == 'constant':
            value = cfg.MODEL.BATCH_ENSEMBLE.INITIALIZER.VALUES[0]
            init_fn = _constant(value)

        elif init_fn_name == 'normal':
            mean, std = cfg.MODEL.BATCH_ENSEMBLE.INITIALIZER.VALUES
            init_fn = _normal(mean, std)

        else:
            raise NotImplementedError(
                f'Unknown name for initializer of BatchEnsemble: {init_fn_name}'
            )

        return functools.partial(
            Conv2d_BatchEnsemble,
            ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE,
            use_bias      = use_bias,
            r_init        = init_fn,
            s_init        = init_fn,
        )

    raise NotImplementedError(f'Unknown name for conv2d layers: {name}')


def get_linear_layers(cfg, name, use_bias=False):
    """
    Args:
        cfg (CfgNode): CfgNode instance that contains blueprint of layers.
        name (str): a name for linear layers.

    Returns:
        `nn.Module` instance.
    """
    if name == 'Linear':
        return functools.partial(
            Linear,
            use_bias = use_bias,
        )

    if name == 'Linear_Dropout':
        return functools.partial(
            Linear_Dropout,
            use_bias  = use_bias,
            drop_rate = cfg.MODEL.DROPOUT.DROP_RATE,
        )

    if name == 'Linear_BatchEnsemble':

        init_fn_name = cfg.MODEL.BATCH_ENSEMBLE.INITIALIZER.NAME

        if init_fn_name == 'constant':
            value = cfg.MODEL.BATCH_ENSEMBLE.INITIALIZER.VALUES[0]
            init_fn = _constant(value)

        elif init_fn_name == 'normal':
            mean, std = cfg.MODEL.BATCH_ENSEMBLE.INITIALIZER.VALUES
            init_fn = _normal(mean, std)

        else:
            raise NotImplementedError(
                f'Unknown name for initializer of BatchEnsemble: {init_fn_name}'
            )

        return functools.partial(
            Linear_BatchEnsemble,
            ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE,
            use_bias      = use_bias,
            r_init        = init_fn,
            s_init        = init_fn,
        )

    raise NotImplementedError(f'Unknown name for linear layers: {name}')


def get_activation_layers(cfg, name):
    """
    Args:
        cfg (CfgNode): CfgNode instance that contains blueprint of layers.
        name (str): a name for activation layers.

    Returns:
        `nn.Module` instance.
    """
    if name == 'Sigmoid':
        return Sigmoid

    if name == 'ReLU':
        return ReLU

    if name == 'SiLU':
        return SiLU

    raise NotImplementedError(f'Unknown name for activation layers: {name}')
