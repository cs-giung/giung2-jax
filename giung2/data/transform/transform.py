import jax
import jax.numpy as jnp
from typing import List
from abc import ABCMeta, abstractmethod


__all__ = [
    "Transform",
    "TransformChain",
    "ToTensorTransform",
    "RandomUniformDequantizeTransform",
    "RandomHFlipTransform",
    "RandomCropTransform",
    "RandomResizedCropTransform",
]


class Transform(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, rng, image):
        """
        Apply the transform on an image.
        """


class TransformChain(Transform):

    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, rng, image):
        for _t in self.transforms:
            image = _t(rng, image)
        return image


class ToTensorTransform(Transform):

    def __init__(self):
        """
        Change the range of [0, 255] into [0., 1.].
        """
    
    def __call__(self, rng, image):
        return image / 255.0


class RandomUniformDequantizeTransform(Transform):

    def __init__(self):
        """
        Convert discrete [0, 255] to continuous [-0.5, 255.5].
        """

    def __call__(self, rng, image):
        return image + jax.random.uniform(rng, image.shape, minval=-0.5, maxval=0.5)


class RandomHFlipTransform(Transform):

    def __init__(self, prob=0.5):
        """
        Flip the image horizontally with the given probability.

        Args:
            prob: probability of the flip.
        """
        self.prob = prob

    def __call__(self, rng, image):
        return jnp.where(
            condition = jax.random.bernoulli(rng, self.prob),
            x         = jnp.flip(image, axis=1),
            y         = image,
        )


class RandomCropTransform(Transform):

    def __init__(self, size, padding):
        """
        Crop the image at a random location with given size and padding.

        Args:
            size (int): desired output size of the crop.
            padding (int): padding on each border of the image before cropping.
        """
        self.size = size
        self.padding = padding

    def __call__(self, rng, image):

        image = jnp.pad(
            array           = image,
            pad_width       = ((self.padding, self.padding),
                               (self.padding, self.padding),
                               (           0,            0),),
            mode            = 'constant',
            constant_values = 0,
        )

        rng1, rng2 = jax.random.split(rng, 2)
        h0 = jax.random.randint(rng1, shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        w0 = jax.random.randint(rng2, shape=(1,), minval=0, maxval=2*self.padding+1)[0]
        image = jax.lax.dynamic_slice(
            operand       = image,
            start_indices = (h0, w0, 0),
            slice_sizes   = (self.size, self.size, image.shape[2]),
        )

        return image


class RandomResizedCropTransform(Transform):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        """
        Crop the image at a random location and resize it to a given size.

        Args:
            size (int): desired output size of the crop.
            scale (tuple): it specifies lower and upper bounds for the random
                area of the crop, before resizing.
            ratio (tuple): it specifies lower and upper bounds for the random
                aspect ratio of the crop, before resizing.
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, rng, image):

        h, w = image.shape[1:3]
        area = h * w

        rng, sub_rng = jax.random.split(rng, 2)
        target_area = area * jax.random.uniform(
            sub_rng, shape=(1,),
            minval = self.scale[0],
            maxval = self.scale[1],
        )[0]

        rng, sub_rng = jax.random.split(rng, 2)
        aspect_ratio = jnp.exp(
            jax.random.uniform(
                sub_rng, shape=(1,),
                minval = jnp.log(self.ratio[0]),
                maxval = jnp.log(self.ratio[1]),
            )
        )[0]

        rng, sub_rng1, sub_rng2 = jax.random.split(rng, 3)
        new_h = jnp.round(jnp.sqrt(target_area / aspect_ratio)).astype(int)
        new_w = jnp.round(jnp.sqrt(target_area * aspect_ratio)).astype(int)
        h0    = jax.random.randint(sub_rng1, shape=(1,), minval=0, maxval=h - new_h + 1)[0]
        w0    = jax.random.randint(sub_rng2, shape=(1,), minval=0, maxval=w - new_w + 1)[0]

        def fn1(image, h0, w0, new_h, new_w):
            image = jax.lax.dynamic_slice(
                operand = image,
                start_indices = (h0, w0, 0),
                slice_sizes   = (new_h, new_w, image.shape[2]),
            )
            return jax.image.resize(
                image     = image,
                shape     = (self.size, self.size, image.shape[2]),
                method    = jax.image.ResizeMethod.LINEAR,
                antialias = True,
            )

        def fn2(image, h, w):
            image = jnp.where(
                condition = jnp.greater(h, w),
                x = jax.image.resize(
                    image     = image,
                    shape     = (h * int(self.size / w), self.size, image.shape[2]),
                    method    = jax.image.ResizeMethod.LINEAR,
                    antialias = True,
                ),
                y = jax.image.resize(
                    image     = image,
                    shape     = (self.size, w * int(self.size / h), image.shape[2]),
                    method    = jax.image.ResizeMethod.LINEAR,
                    antialias = True,
                ),
            )
            offset_h = int( ((h - self.size) + 1) / 2 )
            offset_w = int( ((w - self.size) + 1) / 2 )
            return jax.lax.dynamic_slice(
                operand       = image,
                start_indices = (offset_h, offset_w, 0),
                slice_sizes   = (self.size, self.size, image.shape[2])
            )

        return jnp.where(
            condition = jnp.logical_and(0 < new_h <= h, 0 < new_w <= w),
            x         = fn1(image, h0, w0, new_h, new_w),
            y         = fn2(image, h, w),
        )
