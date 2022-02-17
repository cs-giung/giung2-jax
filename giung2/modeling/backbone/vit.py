"""
Code adapted from
https://github.com/google-research/vision_transformer

Copyright 2021 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Any, Callable, Iterable, Optional, List, Union

from giung2.config import CfgNode
from giung2.layers import *


PRNGKey = Any
Shape = Iterable[int]
Array = Any
Dtype = Any


class EncoderBlock(nn.Module):
    mlp_dim: int
    num_heads: int
    dropout_rate: float
    attention_dropout_rate: float

    @nn.compact
    def __call__(self, x, **kwargs):
        deterministic = kwargs.pop('deterministic', True)

        # forward attention block
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads         = self.num_heads,
            broadcast_dropout = False,
            dropout_rate      = self.attention_dropout_rate,
            kernel_init       = jax.nn.initializers.xavier_uniform(),
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = y + x

        # forward MLP block
        mlp_out_dim = y.shape[-1]
        z = nn.LayerNorm()(y)
        z = nn.Dense(
            features    = self.mlp_dim,
            kernel_init = jax.nn.initializers.xavier_uniform(),
            bias_init   = jax.nn.initializers.normal(stddev=1e-6),
        )(z)
        z = nn.gelu(z)
        z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=deterministic)
        z = nn.Dense(
            features    = mlp_out_dim,
            kernel_init = jax.nn.initializers.xavier_uniform(),
            bias_init   = jax.nn.initializers.normal(stddev=1e-6),
        )(z)
        z = nn.Dropout(rate=self.dropout_rate)(z, deterministic=deterministic)

        return y + z


class Encoder(nn.Module):
    mlp_dim: int
    num_heads: int
    num_layers: int
    dropout_rate: float
    attention_dropout_rate: float
    pos_embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = jax.nn.initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, x, **kwargs):
        deterministic = kwargs.pop('deterministic', True)

        # add learnable 1D positional embeddings
        pe_shape = (1, x.shape[1], x.shape[2]) # [N, L, D1]
        pe = jnp.asarray(self.param('pe', self.pos_embedding_init, pe_shape))
        x = x + pe
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # forward blocks
        for _ in range(self.num_layers):
            x = EncoderBlock(
                mlp_dim                = self.mlp_dim,
                num_heads              = self.num_heads,
                dropout_rate           = self.dropout_rate,
                attention_dropout_rate = self.attention_dropout_rate,
            )(x, deterministic=deterministic)

        y = nn.LayerNorm()(x)
        return y


class VisionTransformer(nn.Module):
    patch_size: int
    hidden_size: int
    transformer: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):

        # embed inputs
        y = nn.Conv(
            features = self.hidden_size,
            kernel_size = (self.patch_size, self.patch_size),
            strides     = (self.patch_size, self.patch_size),
            padding     = 'VALID',
        )(x)

        N, H, W, C = y.shape
        y = jnp.reshape(y, [N, H * W, C])

        # add a class token
        cls = self.param('cls', jax.nn.initializers.zeros, (1, 1, C))
        cls = jnp.tile(cls, [N, 1, 1])
        y = jnp.concatenate([cls, y], axis=1)

        # forward transformer
        y = self.transformer()(y, **kwargs)[:, 0]

        return y


def build_vit_backbone(cfg: CfgNode):

    # define the transformer
    transformer = partial(
        Encoder,
        mlp_dim                = cfg.MODEL.BACKBONE.VIT.TRANSFORMER.MLP_DIM,
        num_heads              = cfg.MODEL.BACKBONE.VIT.TRANSFORMER.NUM_HEADS,
        num_layers             = cfg.MODEL.BACKBONE.VIT.TRANSFORMER.NUM_LAYERS,
        dropout_rate           = cfg.MODEL.BACKBONE.VIT.TRANSFORMER.DROPOUT_RATE,
        attention_dropout_rate = cfg.MODEL.BACKBONE.VIT.TRANSFORMER.ATTENTION_DROPOUT_RATE,
    )

    return VisionTransformer(
        patch_size  = cfg.MODEL.BACKBONE.VIT.PATCH_SIZE,
        hidden_size = cfg.MODEL.BACKBONE.VIT.HIDDEN_SIZE,
        transformer = transformer,
    )
