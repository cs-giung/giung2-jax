import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Union, Tuple


Array = Any


class ImageClassificationModelBase(nn.Module):
    backbone: nn.Module
    classifier: nn.Module
    pixel_mean: Array
    pixel_std: Array

    @nn.compact
    def __call__(
            self,
            images: Array,
            labels: Array = None,
            **kwargs,
        ) -> Union[Array, Tuple]:
        """
        Args:
            images
            labels

        Returns:
            log_confidences 
        """
        # per-channel statistics
        m = self.variable('image_stats', 'm', lambda _ : self.pixel_mean, (images.shape[-1],))
        s = self.variable('image_stats', 's', lambda _ : self.pixel_std,  (images.shape[-1],))

        # preprocess images using per-channel statistics
        x = jnp.asarray(images)
        x = x - jnp.reshape(m.value, (1, 1, 1, -1,))
        x = x / jnp.reshape(s.value, (1, 1, 1, -1,))

        # forward passes
        y = self.backbone(x, **kwargs)
        y = self.classifier(y, **kwargs)

        return y
