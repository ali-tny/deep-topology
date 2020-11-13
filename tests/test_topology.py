import pytest
import tensorflow as tf
import numpy as np

from topo_ae import topology_np, topology_tf


@pytest.mark.parametrize("num_vertices", [8, 16, 32])
def test_get_0d_homology_edges_numpy(num_vertices):
    vertices = tf.random.uniform((num_vertices,))
    distance_matrix = tf.abs(tf.expand_dims(vertices, axis=-1) - vertices)

    edges = topology_np.get_0d_homology_edges(distance_matrix.numpy())

    # Check it's a tree - it touches every vertex with exactly n-1 edges => no cycles
    assert edges.shape[0] == num_vertices - 1
    assert set(tf.reshape(edges, [-1]).numpy()) == set(np.arange(num_vertices))

    # Our vertices are in 1 dimension, so the minimal tree is just a straight line
    # with length as the difference between the smallest and largest vertex
    expected_length = tf.reduce_max(vertices) - tf.reduce_min(vertices)
    edge_weights = tf.gather_nd(distance_matrix, edges)
    assert tf.reduce_sum(edge_weights) == expected_length


@pytest.mark.parametrize("num_vertices", [8, 16, 32])
def test_get_0d_homology_edges_tf(num_vertices):
    vertices = tf.random.uniform((num_vertices,))
    distance_matrix = tf.abs(tf.expand_dims(vertices, axis=-1) - vertices)
    components = tf.Variable(tf.range(num_vertices), trainable=False)

    edges = topology_tf.get_0d_homology_edges(distance_matrix, components)

    # Check it's a tree - it touches every vertex with exactly n-1 edges => no cycles
    assert edges.shape[0] == num_vertices - 1
    assert set(tf.reshape(edges, [-1]).numpy()) == set(np.arange(num_vertices))

    # Our vertices are in 1 dimension, so the minimal tree is just a straight line
    # with length as the difference between the smallest and largest vertex
    expected_length = tf.reduce_max(vertices) - tf.reduce_min(vertices)
    edge_weights = tf.gather_nd(distance_matrix, edges)
    assert tf.reduce_sum(edge_weights) == expected_length
