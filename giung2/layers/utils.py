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

    if name == 'Conv2d_BatchEnsemble':
        return functools.partial(
            Conv2d_BatchEnsemble,
            ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE,
            use_bias      = use_bias,
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

    if name == 'Linear_BatchEnsemble':
        return functools.partial(
            Linear_BatchEnsemble,
            ensemble_size = cfg.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE,
            use_bias      = use_bias,
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
