"""Calculate persistent homology features from distance tensors using numpy.

Primarily we'll focus on 0-d homology, which just checks if an edge is in the minimum spanning tree
(MST), which is the spanning tree (a subgraph which connects all vertices together with no cycles)
which has the minimal sum of edge weights (which in our case will be the distances between
vertices in the point cloud).
"""
import numpy as np


def find(components: np.ndarray, u: int) -> int:
    """Find the parent node representing the component of vertex u."""
    if components[u] == u:
        return u
    else:
        components[u] = find(components, components[u])
        return components[u]


def merge(components: np.ndarray, u: int, v: int):
    """Merge the two components of vertices u and v."""
    if u != v:
        components[find(components, u)] = find(components, v)


def get_0d_homology_edges(distance_matrix: np.ndarray) -> np.ndarray:
    """Get a tensor of edge indices indicating the edges that destroy 0d homology features.

    In 0d homology, the features is the number of connected components, so edges in the minimum
    spanning tree (MST) destroy these features. Returned tensor is of shape (n-1, 2) where n is the
    number of vertices, each row representing the start and end vertex indices of an edge in the
    MST. They will always be in the upper right triangle of matrix (ie edge[0] < edge[1]).
    """
    n_vertices = distance_matrix.shape[0]

    triu_indices = np.triu_indices_from(distance_matrix, k=1)
    edge_weights = distance_matrix[triu_indices]
    edge_idxs = np.argsort(edge_weights, kind="stable")
    sorted_edges = np.array(list(zip(triu_indices[0][edge_idxs], triu_indices[1][edge_idxs])))

    components = np.arange(n_vertices, dtype=np.int32)
    mst_edges_mask = [_edge_is_in_mst(edge, components) for edge in sorted_edges]

    persistence_pairs = sorted_edges[mst_edges_mask]
    persistence_pairs = np.apply_along_axis(
        _convert_to_upper_triangular, axis=1, arr=persistence_pairs
    )
    return persistence_pairs.astype(np.int32)


def _edge_is_in_mst(edge: np.ndarray, components: np.ndarray) -> bool:
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


def _convert_to_upper_triangular(edge: np.ndarray) -> np.ndarray:
    """Convert edge indices to point to upper triangular entries of the distance matrix."""
    if edge[0] < edge[1]:
        return edge
    else:
        return edge[::-1]
