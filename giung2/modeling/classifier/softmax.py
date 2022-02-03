import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from giung2.config import CfgNode
from giung2.layers import Linear


class SoftmaxClassifier(nn.Module):
    num_classes: int
    num_heads: int
    linear: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):

        y = self.linear(features = self.num_classes)(x, **kwargs)
        self.sow('intermediates', f'logits', y)

        y = nn.log_softmax(y, axis=-1)
        self.sow('intermediates', f'log_confidences', y)

        return y


def build_softmax_classifier(cfg: CfgNode):

    _linear_layers = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.LINEAR_LAYERS
    if _linear_layers == 'Linear':
        linear = partial(
            Linear, use_bias=cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.USE_BIAS
        )

    return SoftmaxClassifier(
        num_classes = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_CLASSES,
        num_heads   = cfg.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_HEADS,
        linear      = linear,
    )

