import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Dict, List, Union

from giung2.config import CfgNode
from giung2.layers import *


_VGG_CONFIGS = {
    11: [1,    'M', 2,    'M', 2, 1,       'M', 2, 1,       'M', 1, 1,       'M'],
    13: [1, 1, 'M', 2, 1, 'M', 2, 1,       'M', 2, 1,       'M', 1, 1,       'M'],
    16: [1, 1, 'M', 2, 1, 'M', 2, 1, 1,    'M', 2, 1, 1,    'M', 1, 1, 1,    'M'],
    19: [1, 1, 'M', 2, 1, 'M', 2, 1, 1, 1, 'M', 2, 1, 1, 1, 'M', 1, 1, 1, 1, 'M'],
}


class VGGNet(nn.Module):
    depth: int
    in_planes: int
    mlp_hiddens: List[int]
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module
    linear: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        block_idx = 0

        widen_factor = _VGG_CONFIGS[self.depth][0]
        y = self.conv(channels    = self.in_planes * widen_factor,
                      kernel_size = 3,
                      stride      = 1,
                      padding     = 1,)(x, **kwargs)
        y = self.norm()(y, **kwargs)
        y = self.relu()(y, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        for widen_factor in _VGG_CONFIGS[self.depth][1:]:
            if widen_factor == 'M':
                y = MaxPool2d(kernel_size = 2,
                              stride      = 2,
                              padding     = 0,)(y, **kwargs)
            else:
                y = self.conv(channels    = y.shape[3] * widen_factor,
                              kernel_size = 3,
                              stride      = 1,
                              padding     = 1,)(y, **kwargs)
                y = self.norm()(y, **kwargs)
                y = self.relu()(y, **kwargs)
            self.sow('intermediates', f'features.block.{block_idx}', y)
            block_idx += 1

        y = jnp.reshape(y, (-1, y.shape[1] * y.shape[2] * y.shape[3]))
        for hidden_dim in self.mlp_hiddens:
            y = self.linear(features = hidden_dim)(y, **kwargs)
            y = self.relu()(y, **kwargs)
            self.sow('intermediates', f'features.block.{block_idx}', y)
            block_idx += 1

        return y


def build_vggnet_backbone(cfg: CfgNode):

    _conv_layers = cfg.MODEL.BACKBONE.VGGNET.CONV_LAYERS
    _norm_layers = cfg.MODEL.BACKBONE.VGGNET.NORM_LAYERS
    _activations = cfg.MODEL.BACKBONE.VGGNET.ACTIVATIONS
    _linear_layers = cfg.MODEL.BACKBONE.VGGNET.LINEAR_LAYERS

    # normalization layers
    if _norm_layers == 'NONE':
        norm = Identity
    elif _norm_layers == 'BatchNorm2d':
        norm = partial(
            BatchNorm2d,
            momentum = cfg.MODEL.BATCH_NORMALIZATION.MOMENTUM,
            epsilon  = cfg.MODEL.BATCH_NORMALIZATION.EPSILON,
        )
    elif _norm_layers == 'FilterResponseNorm2d':
        norm = partial(
            FilterResponseNorm2d,
            epsilon               = cfg.MODEL.FILTER_RESPONSE_NORMALIZATION.EPSILON,
            use_learnable_epsilon = cfg.MODEL.FILTER_RESPONSE_NORMALIZATION.USE_LEARNABLE_EPSILON,
        )
    else:
        raise NotImplementedError()

    # convolutional layers
    if _conv_layers == 'Conv2d':
        conv = partial(
            Conv2d,
            use_bias = False if not isinstance(norm, Identity) else True,
        )
    else:
        raise NotImplementedError()

    # activation functions
    if _activations == 'Sigmoid':
        relu = Sigmoid
    elif _activations == 'ReLU':
        relu = ReLU
    elif _activations == 'SiLU':
        relu = SiLU
    else:
        raise NotImplementedError()

    # linear layers
    if _linear_layers == 'Linear':
        linear = partial(
            Linear, use_bias=cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.USE_BIAS
        )
    else:
        raise NotImplementedError()

    return VGGNet(
        depth       = cfg.MODEL.BACKBONE.VGGNET.DEPTH,
        in_planes   = cfg.MODEL.BACKBONE.VGGNET.IN_PLANES,
        mlp_hiddens = cfg.MODEL.BACKBONE.VGGNET.MLP_HIDDENS,
        conv        = conv,
        norm        = norm,
        relu        = relu,
        linear      = linear,
    )
