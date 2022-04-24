import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import List, Union

from giung2.config import CfgNode
from giung2.layers import *


class IdentityShortcut(nn.Module):
    channels: int
    stride: int
    expansion: int
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        pad_offset = self.expansion * self.channels - x.shape[-1]
        return jnp.pad(
            array           = x[:, ::self.stride, ::self.stride, :],
            pad_width       = ((0, 0), (0, 0), (0, 0), (0, pad_offset)),
            mode            = 'constant',
            constant_values = 0,
        )


class ProjectionShortcut(nn.Module):
    channels: int
    stride: int
    expansion: int
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        y = self.conv(channels    = self.channels * self.expansion,
                      kernel_size = 1,
                      stride      = self.stride,
                      padding     = 0,)(x, **kwargs)
        y = self.norm()(y, **kwargs)
        return y


class FirstBlock(nn.Module):
    channels: int
    conv_ksp: List[Union[int, str]]
    pool_ksp: List[Union[int, str]]
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module
    pool: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        y = self.conv(channels    = self.channels,
                      kernel_size = self.conv_ksp[0],
                      stride      = self.conv_ksp[1],
                      padding     = self.conv_ksp[2],)(x, **kwargs)
        if self.norm is not None:
            y = self.norm()(y, **kwargs)
        if self.relu is not None:
            y = self.relu()(y, **kwargs)
        if self.pool is not None:
            y = self.pool(kernel_size = self.pool_ksp[0],
                          stride      = self.pool_ksp[1],
                          padding     = self.pool_ksp[2],)(y, **kwargs)
        return y


class BottleneckBlock(nn.Module):
    channels: int
    stride: int
    groups: int
    width_per_group: int
    shortcut: nn.Module
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        width = int(self.channels * (self.width_per_group / 64.)) * self.groups
        
        y = self.conv(channels    = width,
                      kernel_size = 1,
                      stride      = 1,
                      padding     = 0,
                      num_groups  = 1,)(x, **kwargs)
        y = self.norm()(y, **kwargs)
        y = self.relu()(y, **kwargs)
        y = self.conv(channels    = width,
                      kernel_size = 3,
                      stride      = self.stride,
                      padding     = 1,
                      num_groups  = self.groups,)(y, **kwargs)
        y = self.norm()(y, **kwargs)
        y = self.relu()(y, **kwargs)
        y = self.conv(channels    = self.channels * 4,
                      kernel_size = 1,
                      stride      = 1,
                      padding     = 0,)(y, **kwargs)
        y = self.norm()(y, **kwargs)

        if self.stride != 1 or x.shape[-1] != self.channels * 4:
            y = y + self.shortcut(channels  = self.channels,
                                  stride    = self.stride,
                                  expansion = 4,
                                  conv      = self.conv,
                                  norm      = self.norm,
                                  relu      = self.relu,)(x, **kwargs)
        else:
            y = y + x

        y = self.relu()(y, **kwargs)
        return y


class LastBlock(nn.Module):
    channels: int
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        """for compatibility with PreResNet
        """
        return x


class ResNeXt(nn.Module):
    in_planes: int
    groups: int
    width_per_group: int
    first_block: nn.Module
    block: nn.Module
    block_expansion: int
    shortcut: nn.Module
    num_blocks: List[int]
    widen_factor: int
    conv: nn.Module
    norm: nn.Module
    relu: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        block_idx = 0

        y = self.first_block()(x, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        for group_idx, num_block in enumerate(self.num_blocks):
            _strides = (1,) if group_idx == 0 else (2,)
            _strides = _strides + (1,) * (num_block - 1)
            for _stride in _strides:
                y = self.block(channels        = self.in_planes * self.widen_factor * (2 ** group_idx),
                               stride          = _stride,
                               groups          = self.groups,
                               width_per_group = self.width_per_group,
                               shortcut        = self.shortcut,
                               conv            = self.conv,
                               norm            = self.norm,
                               relu            = self.relu,)(y, **kwargs)
                self.sow('intermediates', f'features.block.{block_idx}', y)
                block_idx += 1

        group_idx = len(self.num_blocks) - 1
        y = LastBlock(channels = self.in_planes * self.widen_factor * (2 ** group_idx) * self.block_expansion,
                      conv     = self.conv,
                      norm     = self.norm,
                      relu     = self.relu,)(y, **kwargs)
        self.sow('intermediates', f'features.block.{block_idx}', y)
        block_idx += 1

        y = jnp.mean(y, [1, 2])
        self.sow('intermediates', f'features', y)

        return y


def build_resnext_backbone(cfg: CfgNode):

    # define layers
    norm = get_norm2d_layers(
        cfg      = cfg,
        name     = cfg.MODEL.BACKBONE.RESNEXT.NORM_LAYERS,
    )
    conv = get_conv2d_layers(
        cfg      = cfg,
        name     = cfg.MODEL.BACKBONE.RESNEXT.CONV_LAYERS,
        use_bias = False if not isinstance(norm, Identity) else True,
    )
    relu = get_activation_layers(
        cfg      = cfg,
        name     = cfg.MODEL.BACKBONE.RESNEXT.ACTIVATIONS,
    )

    # define the first block
    first_block = partial(
        FirstBlock, channels = cfg.MODEL.BACKBONE.RESNEXT.IN_PLANES,
                    conv_ksp = cfg.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.CONV_KSP,
                    pool_ksp = cfg.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.POOL_KSP,
                    conv     = conv,
                    norm     = norm      if cfg.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.USE_NORM_LAYER else None,
                    relu     = relu      if cfg.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.USE_ACTIVATION else None,
                    pool     = MaxPool2d if cfg.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.USE_POOL_LAYER else None,
        )

    # define intermediate blocks
    _block = cfg.MODEL.BACKBONE.RESNEXT.BLOCK
    if _block == 'BottleneckBlock':
        block = BottleneckBlock
        block_expansion = 4

    # define shortcuts
    _shortcut = cfg.MODEL.BACKBONE.RESNEXT.SHORTCUT
    if _shortcut == 'IdentityShortcut':
        shortcut = IdentityShortcut
    elif _shortcut == 'ProjectionShortcut':
        shortcut = ProjectionShortcut

    return ResNeXt(
        in_planes       = cfg.MODEL.BACKBONE.RESNEXT.IN_PLANES,
        groups          = cfg.MODEL.BACKBONE.RESNEXT.GROUPS,
        width_per_group = cfg.MODEL.BACKBONE.RESNEXT.WIDTH_PER_GROUP,
        first_block     = first_block,
        block           = block,
        block_expansion = block_expansion,
        shortcut        = shortcut,
        num_blocks      = cfg.MODEL.BACKBONE.RESNEXT.NUM_BLOCKS,
        widen_factor    = cfg.MODEL.BACKBONE.RESNEXT.WIDEN_FACTOR,
        conv            = conv,
        norm            = norm,
        relu            = relu,
    )
