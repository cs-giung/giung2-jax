import os
import random
import warnings
import functools
import numpy as np
from typing import Dict, Union, Iterable

import jax
import jax.numpy as jnp

from giung2.config import CfgNode
from giung2.data.transform import *


DATA_AUGMENTATION = {

    'STANDARD': {
        'MNIST': TransformChain([
            ToTensorTransform(),
        ]),
        'KMNIST': TransformChain([
            ToTensorTransform(),
        ]),
        'FashionMNIST': TransformChain([
            ToTensorTransform(),
        ]),
        "CIFAR10": TransformChain([
            RandomCropTransform(size=32, padding=4),
            RandomHFlipTransform(prob=0.5),
            ToTensorTransform(),
        ]),
        "CIFAR100": TransformChain([
            RandomCropTransform(size=32, padding=4),
            RandomHFlipTransform(prob=0.5),
            ToTensorTransform(),
        ]),
        "TinyImageNet200": TransformChain([
            RandomCropTransform(size=64, padding=4),
            RandomHFlipTransform(prob=0.5),
            ToTensorTransform(),
        ]),
    },

    'DEQUANTIZED_STANDARD': {
        'MNIST': TransformChain([
            RandomUniformDequantizeTransform(),
            ToTensorTransform(),
        ]),
        'KMNIST': TransformChain([
            RandomUniformDequantizeTransform(),
            ToTensorTransform(),
        ]),
        'FashionMNIST': TransformChain([
            RandomUniformDequantizeTransform(),
            ToTensorTransform(),
        ]),
        "CIFAR10": TransformChain([
            RandomCropTransform(size=32, padding=4),
            RandomHFlipTransform(prob=0.5),
            RandomUniformDequantizeTransform(),
            ToTensorTransform(),
        ]),
        "CIFAR100": TransformChain([
            RandomCropTransform(size=32, padding=4),
            RandomHFlipTransform(prob=0.5),
            RandomUniformDequantizeTransform(),
            ToTensorTransform(),
        ]),
        "TinyImageNet200": TransformChain([
            RandomCropTransform(size=64, padding=4),
            RandomHFlipTransform(prob=0.5),
            RandomUniformDequantizeTransform(),
            ToTensorTransform(),
        ]),
    },

}


def build_dataloaders(
        cfg: CfgNode,
        batch_size: Union[int, Iterable[int]],
    ) -> Dict[str, Iterable]:
    name = cfg.DATASETS.NAME

    if name in ['MNIST', 'KMNIST', 'FashionMNIST',]:
        indices = list(range(60000))
        if cfg.DATASETS.MNIST.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)
        trn_indices = indices[cfg.DATASETS.MNIST.TRAIN_INDICES[0] : cfg.DATASETS.MNIST.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.MNIST.VALID_INDICES[0] : cfg.DATASETS.MNIST.VALID_INDICES[1]]

    elif name in ['CIFAR10', 'CIFAR100',]:
        indices = list(range(50000))
        if cfg.DATASETS.CIFAR.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)
        trn_indices = indices[cfg.DATASETS.CIFAR.TRAIN_INDICES[0] : cfg.DATASETS.CIFAR.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.CIFAR.VALID_INDICES[0] : cfg.DATASETS.CIFAR.VALID_INDICES[1]]

    elif name in ['TinyImageNet200',]:
        indices = list(range(100000))
        if cfg.DATASETS.TINY.SHUFFLE_INDICES:
            random.Random(cfg.DATASETS.SEED).shuffle(indices)
        trn_indices = indices[cfg.DATASETS.TINY.TRAIN_INDICES[0] : cfg.DATASETS.TINY.TRAIN_INDICES[1]]
        val_indices = indices[cfg.DATASETS.TINY.VALID_INDICES[0] : cfg.DATASETS.TINY.VALID_INDICES[1]]

    trn_images = np.load(os.path.join(cfg.DATASETS.ROOT, f'{name}/train_images.npy'))
    trn_labels = np.load(os.path.join(cfg.DATASETS.ROOT, f'{name}/train_labels.npy'))
    tst_images = np.load(os.path.join(cfg.DATASETS.ROOT, f'{name}/test_images.npy'))
    tst_labels = np.load(os.path.join(cfg.DATASETS.ROOT, f'{name}/test_labels.npy'))

    # validation split
    val_images, val_labels = trn_images[val_indices], trn_labels[val_indices]
    trn_images, trn_labels = trn_images[trn_indices], trn_labels[trn_indices]

    # specify mini-batch settings
    if isinstance(batch_size, int):
        batch_size = (batch_size, batch_size, batch_size,)
    trn_batch_size, val_batch_size, tst_batch_size = batch_size

    if len(val_images) % val_batch_size != 0:
        warnings.warn(f'val_batch_size={val_batch_size} cannot utilize all {len(val_images)} examples.')
    if len(tst_images) % tst_batch_size != 0:
        warnings.warn(f'tst_batch_size={tst_batch_size} cannot utilize all {len(tst_images)} examples.')

    trn_steps_per_epoch = len(trn_images) // trn_batch_size
    val_steps_per_epoch = len(val_images) // val_batch_size
    tst_steps_per_epoch = len(tst_images) // tst_batch_size

    # build dataloaders
    dataloaders = {
        'dataloader': functools.partial(
            _build_dataloader,
            images          = trn_images,
            labels          = trn_labels,
            batch_size      = trn_batch_size,
            steps_per_epoch = trn_steps_per_epoch,
            shuffle         = True,
            transform       = jax.jit(jax.vmap(DATA_AUGMENTATION[cfg.DATASETS.DATA_AUGMENTATION][name])),
        ),
        'trn_loader': functools.partial(
            _build_dataloader,
            images          = trn_images,
            labels          = trn_labels,
            batch_size      = trn_batch_size,
            steps_per_epoch = trn_steps_per_epoch,
            shuffle         = False,
            transform       = jax.jit(jax.vmap(ToTensorTransform())),
        ),
        'val_loader': functools.partial(
            _build_dataloader,
            images          = val_images,
            labels          = val_labels,
            batch_size      = val_batch_size,
            steps_per_epoch = val_steps_per_epoch,
            shuffle         = False,
            transform       = jax.jit(jax.vmap(ToTensorTransform())),
        ),
        'tst_loader': functools.partial(
            _build_dataloader,
            images          = tst_images,
            labels          = tst_labels,
            batch_size      = tst_batch_size,
            steps_per_epoch = tst_steps_per_epoch,
            shuffle         = False,
            transform       = jax.jit(jax.vmap(ToTensorTransform())),
        ),
        'trn_steps_per_epoch': trn_steps_per_epoch,
        'val_steps_per_epoch': val_steps_per_epoch,
        'tst_steps_per_epoch': tst_steps_per_epoch,
    }
    return dataloaders


def _build_dataloader(images, labels, batch_size, steps_per_epoch, rng=None, shuffle=False, transform=None):
    indices = jax.random.permutation(rng, len(images)) if shuffle else jnp.arange(len(images))
    indices = indices[:steps_per_epoch*batch_size]
    indices = indices.reshape((steps_per_epoch, batch_size,))
    for batch_idx in indices:
        batch = {'images': jnp.array(images[batch_idx]), 'labels': jnp.array(labels[batch_idx])}
        if transform is not None:
            if rng is not None:
                _, rng = jax.random.split(rng)
            sub_rng = None if rng is None else jax.random.split(rng, batch['images'].shape[0])
            batch['images'] = transform(sub_rng, batch['images'])
        batch = jax.tree_map(
            lambda x: x.reshape(
                (jax.local_device_count(), -1,) + x.shape[1:]
            ), batch,
        )
        yield batch
