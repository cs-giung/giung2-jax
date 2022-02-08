import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Dict, List, Union

from giung2.config import CfgNode
from giung2.layers import *


class LeNet5(nn.Module):
    conv: nn.Module
    relu: nn.Module
    linear: nn.Module
    
    @nn.compact
    def __call__(self, x, **kwargs):
        block_idx = 0

        y = self.conv(channels    = 6,
                      kernel_size = 5,
                      stride      = 1,
                      padding     = 2,)(x, **kwargs)
        y = self.relu()(y, **kwargs)
        y = AvgPool2d(kernel_size = 2,
                      stride      = 2,
                      padding     = 0,)(y, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        y = self.conv(channels    = 16,
                      kernel_size = 5,
                      stride      = 1,
                      padding     = 0,)(y, **kwargs)
        y = self.relu()(y, **kwargs)
        y = AvgPool2d(kernel_size = 2,
                      stride      = 2,
                      padding     = 0,)(y, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        y = jnp.reshape(y, (-1, y.shape[1] * y.shape[2] * y.shape[3]))

        y = self.linear(features = 120)(y, **kwargs)
        y = self.relu()(y, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        y = self.linear(features = 84)(y, **kwargs)
        y = self.relu()(y, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        return y


def build_lenet_backbone(cfg: CfgNode):

    _conv_layers = cfg.MODEL.BACKBONE.LENET.CONV_LAYERS
    _activations = cfg.MODEL.BACKBONE.LENET.ACTIVATIONS
    _linear_layers = cfg.MODEL.BACKBONE.LENET.LINEAR_LAYERS

    # convolutional layers
    if _conv_layers == 'Conv2d':
        conv = partial(
            Conv2d,
            use_bias = True,
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

    return LeNet5(
        conv   = conv,
        relu   = relu,
        linear = linear,
    )
