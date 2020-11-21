from typing import Callable, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

Generator = Callable[[], Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]]


def get_ordered_mnist_generator(minibatch_size: int = 8) -> Tuple[Generator, int]:
    """Get a generator returning ordered batches of MNIST data.

    "Ordered" here means a batch of `minibatch_size` examples from each class in order, so by
    default a batch of 8 images of 0s, 8 images of 1s, 8 images of 2s, etc - to be used with the
    TopologicallyDenseRegularization layer.
    Will download MNIST data the first time.

    Args:
        minibatch_size: the number of examples of each class in each batch - so the total batch
            size will be minibatch_size * 10
    Returns:
        A generator of batches to pass to model.fit, and an int representing the total steps per
        epoch to be passed to model.fit
    """
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(_normalize_img)
    ds_test = ds_test.map(_normalize_img)

    images = tf.stack([image for image, _ in ds_train])
    labels = tf.stack([label for _, label in ds_train])

    by_label = {i: images[labels == i] for i in np.unique(labels.numpy())}

    minimum_label_length = min(map(len, by_label.values()))
    steps_per_epoch = minimum_label_length // minibatch_size

    def _gen():
        while True:
            for i in range(steps_per_epoch):
                batch = {
                    label: images[i * minibatch_size : (i + 1) * minibatch_size]
                    for label, images in by_label.items()
                }
                batch_labels = tf.concat([[i] * minibatch_size for i in batch.keys()], axis=0)
                batch_images = tf.concat([images for images in batch.values()], axis=0)
                yield (batch_images, batch_labels), batch_labels

    return _gen(), steps_per_epoch


def _normalize_img(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label
