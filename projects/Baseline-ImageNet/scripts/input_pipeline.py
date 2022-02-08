# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import tensorflow as tf
import tensorflow_datasets as tfds


IMAGE_SIZE = 224
CROP_PADDING = 32


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
        image_bytes: `Tensor` of binary image data.
        bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
            where each coordinate is [0, 1) and the coordinates are arranged
            as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding
            box supplied.
        aspect_ratio_range: An optional list of `float`s. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `float`s. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
    Returns:
        cropped image `Tensor`
    """
    shape = tf.io.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _resize(image, image_size):
    return tf.image.resize([image], [image_size, image_size],
                            method=tf.image.ResizeMethod.BICUBIC)[0]


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10)
    original_shape = tf.io.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: _resize(image, image_size))

    return image


def _decode_and_center_crop(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
        tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = _resize(image, image_size)

    return image


def preprocess_for_train(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for training.
    Args:
        image_bytes: `Tensor` representing an image binary of arbitrary size.
        dtype: data type of the image.
        image_size: image size.
    Returns:
        A preprocessed image `Tensor`.
    """
    image = _decode_and_random_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def preprocess_for_eval(image_bytes, dtype=tf.float32, image_size=IMAGE_SIZE):
    """Preprocesses the given image for evaluation.
    Args:
        image_bytes: `Tensor` representing an image binary of arbitrary size.
        dtype: data type of the image.
        image_size: image size.
    Returns:
        A preprocessed image `Tensor`.
    """
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image


def create_split(dataset_builder, batch_size, train, dtype=tf.float32,
                 image_size=IMAGE_SIZE, cache=False, seed=0):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.
    Args:
        dataset_builder: TFDS dataset builder for ImageNet.
        batch_size: the batch size returned by the data pipeline.
        train: Whether to load the train or evaluation split.
        dtype: data type of the image.
        image_size: The target size of the images.
        cache: Whether to cache the dataset.
    Returns:
        A `tf.data.Dataset`.
    """
    if train:
        train_examples = dataset_builder.info.splits['train'].num_examples
        split_size = train_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = 'train[{}:{}]'.format(start, start + split_size)
    else:
        validate_examples = dataset_builder.info.splits['validation'].num_examples
        split_size = validate_examples // jax.process_count()
        start = jax.process_index() * split_size
        split = 'validation[{}:{}]'.format(start, start + split_size)

    def decode_example(example):
        if train:
            image = preprocess_for_train(example['image'], dtype, image_size)
        else:
            image = preprocess_for_eval(example['image'], dtype, image_size)
        return {'images': image, 'labels': example['label']}

    ds = dataset_builder.as_dataset(split=split, decoders={
        'image': tfds.decode.SkipDecoding(),
    })
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds = ds.with_options(options)

    if cache:
        ds = ds.cache()

    if train:
        ds = ds.shuffle(split_size, seed=seed)

    ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.prefetch(10)

    return ds


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()
    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train, cache, seed):
    ds = create_split(
        dataset_builder, batch_size,
        image_size=image_size, dtype=dtype, train=train, cache=cache, seed=seed
    )
    it = map(prepare_tf_data, ds)
    return it


def build_imagenet_dataloader(batch_size):
    
    # specify mini-batch settings
    if isinstance(batch_size, int):
        batch_size = (batch_size, batch_size, batch_size,)
    trn_batch_size, val_batch_size, tst_batch_size = batch_size

    trn_local_batch_size = trn_batch_size // jax.process_count()
    val_local_batch_size = val_batch_size // jax.process_count()
    tst_local_batch_size = tst_batch_size // jax.process_count()

    # create iterators
    ds_builder = tfds.builder('imagenet2012')

    def _trn_iter(rng):
        return create_input_iter(
            dataset_builder = ds_builder,
            batch_size      = trn_local_batch_size,
            image_size      = 224,
            dtype           = tf.float32,
            train           = True,
            cache           = False,
            seed            = int(rng[1]),
        )

    def _val_iter(rng=None):
        return create_input_iter(
            dataset_builder = ds_builder,
            batch_size      = val_local_batch_size,
            image_size      = 224,
            dtype           = tf.float32,
            train           = False,
            cache           = False,
            seed            = None,
        )

    dataloaders = {
        'dataloader': _trn_iter,
        'val_loader': _val_iter,
        'trn_steps_per_epoch': ds_builder.info.splits['train'].num_examples // trn_batch_size,
        'val_steps_per_epoch': ds_builder.info.splits['validation'].num_examples // val_batch_size,
    }

    return dataloaders
