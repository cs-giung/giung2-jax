import jax.numpy as jnp
import flax.linen as nn

from giung2.config import CfgNode
from giung2.modeling.architecture import *
from giung2.modeling.backbone import *
from giung2.modeling.classifier import *


def build_backbone(cfg: CfgNode) -> nn.Module:
    name = cfg.MODEL.BACKBONE.NAME

    if name == 'PreResNet':
        backbone = build_preresnet_backbone(cfg)

    return backbone


def build_classifier(cfg: CfgNode) -> nn.Module:
    name = cfg.MODEL.CLASSIFIER.NAME

    if name == 'SoftmaxClassifier':
        classifier = build_softmax_classifier(cfg)

    return classifier


def build_model(cfg: CfgNode) -> nn.Module:
    name = cfg.MODEL.META_ARCHITECTURE.NAME

    if name == 'ImageClassificationModelBase':
        model = ImageClassificationModelBase(
            backbone   = build_backbone(cfg),
            classifier = build_classifier(cfg),
            pixel_mean = jnp.asarray(cfg.MODEL.PIXEL_MEAN),
            pixel_std  = jnp.asarray(cfg.MODEL.PIXEL_STD),
        )

    return model
