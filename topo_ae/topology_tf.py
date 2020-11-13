"""Calculate persistent homology features from distance tensors in a tensorflow graph.

This is actually about 10 times slower than just using numpy - presumably because it's forced to be
sequential anyway, and eg getting the upper-triangular part of the tensor is a pain - might see if
it can be streamlined. Anyway, it was really just an exercise in expressing the recursive
calculation to get the homology edges in a tf.Graph.

Primarily we'll focus on 0-d homology, which just checks if an edge is in the minimum spanning
tree (MST), which is the spanning tree (a subgraph which connects all vertices together with no
cycles) which has the minimal sum of edge weights (which in our case will be the distances between
vertices in the point cloud).
"""
import tensorflow as tf


def find(components: tf.Variable, u: int) -> int:
    """Find the parent node representing the component of vertex u."""

    def _body(i, parents):
        parents = parents.write(i + 1, components[parents.read(i)])
        return i + 1, parents

    def _cond(i, parents):
        return parents.read(i) != components[parents.read(i)]

    parents = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    parents = parents.unstack([u])
    _, parents = tf.while_loop(_cond, _body, loop_vars=(0, parents))

    # Perform path compression
    final_value = parents.read(parents.size() - 1)
    for i in range(parents.size() - 1):
        components[parents.read(i)].assign(final_value)

    return final_value


def merge(components: tf.Variable, u: int, v: int):
    """Merge the two components of vertices u and v."""
    if u != v:
        components[find(components, u)].assign(find(components, v))


@tf.function
def get_0d_homology_edges(distance_matrix: tf.Tensor, components) -> tf.Tensor:
    """Get a tensor of edge indices indicating the edges that destroy 0d homology features.

    In 0d homology, the features is the number of connected components, so edges in the minimum
    spanning tree (MST) destroy these features. Returned tensor is of shape (n-1, 2) where n is the
    number of vertices, each row representing the start and end vertex indices of an edge in the
    MST. They will always be in the upper right triangle of matrix (ie edge[0] < edge[1]).
    """

    ones = tf.ones_like(distance_matrix)
    upper_triangle = tf.linalg.band_part(ones, 0, -1)
    diagonal = tf.linalg.band_part(ones, 0, 0)
    mask = tf.cast(upper_triangle - diagonal, dtype=tf.bool)

    edge_weights = tf.boolean_mask(distance_matrix, mask)
    edge_idxs = tf.argsort(edge_weights)
    num_vertices = distance_matrix.shape[0]
    num_edges = num_vertices * (num_vertices - 1) // 2
    edge_idxs = tf.ensure_shape(edge_idxs, shape=(num_edges,))

    start_vertices = tf.expand_dims(tf.range(distance_matrix.shape[0]), axis=-1)
    start_vertices = tf.broadcast_to(start_vertices, distance_matrix.shape)
    start_vertices = tf.boolean_mask(start_vertices, mask)

    end_vertices = tf.expand_dims(tf.range(distance_matrix.shape[1]), axis=0)
    end_vertices = tf.broadcast_to(end_vertices, distance_matrix.shape)
    end_vertices = tf.boolean_mask(end_vertices, mask)

    sorted_edges = tf.gather(tf.stack([start_vertices, end_vertices], axis=-1), edge_idxs)

    def _body(i, j, persistence_pairs):
        edge = sorted_edges[i]
        if _edge_is_in_mst(edge, components):
            persistence_pairs = persistence_pairs.write(j, _convert_to_upper_triangular(edge))
            j += 1
        return i + 1, j, persistence_pairs

    def _cond(i, j, persitence_pairs):
        return j < num_vertices - 1

    persistence_pairs = tf.TensorArray(tf.int32, size=num_vertices - 1)
    _, _, persistence_pairs = tf.while_loop(_cond, _body, loop_vars=(0, 0, persistence_pairs))

    persistence_pairs = persistence_pairs.stack()
    return persistence_pairs


def _edge_is_in_mst(edge: tf.Tensor, components: tf.Tensor) -> bool:
    """Check an edge is is in the minimum spanning tree, and update connected components."""
    younger_component = find(components, edge[0])
    older_component = find(components, edge[1])

    if younger_component == older_component:
        # They're already in the same component - so it isn't an edge of the MST
        return False
    elif younger_component > older_component:
        merge(components, edge[1], edge[0])
    else:
        merge(components, edge[0], edge[1])
    return True


def _convert_to_upper_triangular(edge: tf.Tensor) -> tf.Tensor:
    """Convert edge indices to point to upper triangular entries of the distance matrix."""
    if edge[0] < edge[1]:
        return edge
    else:
        return tf.reverse(edge, axis=[0])
