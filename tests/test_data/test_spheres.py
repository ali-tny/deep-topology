import numpy as np
import tensorflow as tf

from deep_topology.data import spheres


def test_spheres():
    samples, labels = spheres(1000, noise=None)

    labels = labels.numpy()

    for i in range(10):
        assert (labels[i * 1000 : (i + 1) * 1000] == i).all()
    assert (labels[10000:] == 10).all()

    assert np.isclose(tf.norm(samples[10000:], axis=1).numpy(), 25).all()
    assert (tf.norm(samples[:10000], axis=1).numpy() < 25).all()
