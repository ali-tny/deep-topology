"""Recreate the SPHERES dataset from the Topological Autoencoders paper.

The dataset consists of points sampled from 100 100-dimensional spheres embedded in 101 dimensional
space, enclosed by a larger 100-sphere. Half of the sampled points are from the enclosing sphere,
and half from the enclosed spheres.
"""
from typing import Tuple

import math

import tensorflow as tf


def _sample_d_sphere(
    num_points: int, dimension: int, radius: float, noise: float = None
) -> tf.Tensor:
    samples = tf.random.uniform(shape=(num_points, dimension + 1), minval=-1, maxval=1)

    # Rescale the samples to lie on the d-sphere with radius `radius`
    norm = tf.sqrt(tf.reduce_sum(samples ** 2, axis=-1))
    samples = samples / tf.expand_dims(norm, axis=-1) * radius

    # Add noise, if we have any
    if noise is not None:
        samples += tf.random.normal(shape=(num_points, dimension + 1), mean=noise)

    return samples


def spheres(
    points_per_sphere: int,
    inner_radius: float = 5,
    outer_radius: float = 25,
    noise: float = None,
    num_inner_spheres: int = 10,
    dimension=100,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if outer_radius <= inner_radius:
        raise ValueError("Enclosing sphere radius less than inner sphere radius")
    samples = []
    labels = []
    # Add smaller inner spheres translated from the origin
    translations = tf.random.normal(
        shape=(num_inner_spheres, dimension + 1),
        stddev=num_inner_spheres / math.sqrt(dimension + 1),
    )
    for i, translation in enumerate(translations):
        inner_sample = _sample_d_sphere(
            num_points=points_per_sphere, dimension=dimension, radius=inner_radius, noise=noise
        )
        inner_sample += translation
        samples.append(inner_sample)
        labels.append(tf.fill(dims=inner_sample.shape[:1], value=i))

    # And finally add the enclosing sphere
    enclosing_samples = _sample_d_sphere(
        num_points=points_per_sphere * num_inner_spheres,
        dimension=dimension,
        radius=outer_radius,
        noise=noise,
    )
    samples.append(enclosing_samples)
    labels.append(tf.fill(dims=enclosing_samples.shape[:1], value=len(translations)))

    return tf.concat(samples, axis=0), tf.concat(labels, axis=0)
