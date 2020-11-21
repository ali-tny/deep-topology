import tensorflow as tf
import numpy as np

from deep_topology import layers


def test_get_distance_matrix_1d():
    vertices = tf.convert_to_tensor(np.array([[0], [1], [2]], dtype=float))

    distances = layers._get_distance_matrix(vertices).numpy()
    assert distances.shape == (3, 3)
    assert (distances.T == distances).all()
    assert (np.diagonal(distances) == 0).all()

    assert (distances[0, 1:] == np.array([1, 2])).all()
    assert (distances[1, 2:] == np.array([1])).all()


def test_get_distance_matrix_3d():
    vertices = tf.convert_to_tensor(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    )

    distances = layers._get_distance_matrix(vertices).numpy()
    assert distances.shape == (5, 5)
    assert (distances.T == distances).all()
    assert (np.diagonal(distances) == 0).all()

    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    assert (distances[0, 1:] == np.array([1, 1, 1, sqrt2])).all()
    assert (distances[1, 2:] == np.array([sqrt2, sqrt2, 1])).all()
    assert (distances[2, 3:] == np.array([sqrt2, 1])).all()
    assert (distances[3, 4:] == np.array([sqrt3])).all()
