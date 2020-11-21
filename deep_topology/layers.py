import tensorflow as tf

from . import topology_np, topology_tf


class BaseTopologicalLayer(tf.keras.layers.Layer):
    """A base for layers that intend to calculate 0d persistent homology.

    Primarily exposes the `get_edges` method, which takes a distance matrix and returns the indices
    of edges that are in the minimum spanning tree (and so correspond to the death times of
    0d persistent homology features).

    Provided that distances between points are unique, we don't need to calculate gradients of the
    edge-selection calculation, since we can rewrite anything that sums over those specific edges
    as a function that sums over _all_ edges, with an indicator function that is 1 if that
    edge-length is in the minimum spanning tree and 0 otherwise. By uniqueness of distances between
    points, this indicator function is locally constant, and so has 0 gradient. So then, any
    function f (for example, a loss function) that is summed over the selected edges e_i has
    gradient that's just the sum over the selected edges of df(e_i)/dz_i.
    """

    def __init__(self, numpy=True):
        super().__init__()
        self.numpy = numpy

    def get_edges(self, distances):
        if self.numpy:
            return tf.numpy_function(topology_np.get_0d_homology_edges, [distances], Tout=tf.int32)
        else:
            # Reset the components before calculating edges
            self.components.assign(tf.range(self.components.shape[0]))
            return topology_tf.get_0d_homology_edges(distances, self.components)


class TopologicalAutoencoderLoss(BaseTopologicalLayer):
    def __init__(self, reg_lambda: float = 0.5, numpy=True):
        super().__init__(numpy=numpy)
        self.reg_lambda = reg_lambda
        self.latent_norm = tf.Variable(1.0)

    def build(self, input_shape):
        if not self.numpy:
            if input_shape[0] is None:
                raise ValueError("Batch size should be specified when running with numpy=False")
            batch_size = input_shape[0]
            self.components = tf.Variable(tf.range(batch_size), trainable=False, name="components")

    def call(self, y_true, latent, y_pred):
        ae_loss = tf.losses.mean_squared_error(y_true, y_pred)
        topo_loss = self._topo_loss(y_true, latent)
        return ae_loss + self.reg_lambda * topo_loss

    def _topo_loss(self, data, latent):
        data_distances = _get_distance_matrix(data)
        data_distances /= tf.reduce_max(data_distances)
        latent_distances = _get_distance_matrix(latent)
        latent_distances /= self.latent_norm

        edges = self.get_edges(tf.stop_gradient(data_distances))
        data_loss = tf.losses.mean_squared_error(
            tf.gather_nd(data_distances, edges), tf.gather_nd(latent_distances, edges)
        )

        edges = self.get_edges(tf.stop_gradient(latent_distances))
        latent_loss = tf.losses.mean_squared_error(
            tf.gather_nd(data_distances, edges), tf.gather_nd(latent_distances, edges)
        )
        return data_loss + latent_loss


class TopologicallyDenseRegularization(BaseTopologicalLayer):
    def __init__(self, beta, inner_batch_size, num_classes, numpy=True):
        super().__init__(numpy=numpy)
        self.beta = beta
        self.inner_batch_size = inner_batch_size
        self.num_classes = num_classes

    def build(self, input_shape):
        if not self.numpy:
            if input_shape[0] is None:
                raise ValueError("Batch size should be specified when running with numpy=False")
            assert input_shape[0] / self.num_classes == self.inner_batch_size
            self.components = tf.Variable(
                tf.range(self.inner_batch_size), trainable=False, name="components"
            )

    def call(self, features):
        loss = 0
        for i in range(self.num_classes):
            distances = _get_distance_matrix(
                features[i * self.inner_batch_size : (i + 1) * self.inner_batch_size]
            )
            edges = self.get_edges(tf.stop_gradient(distances))
            loss += tf.losses.mean_absolute_error(tf.gather_nd(features, edges), self.beta)
        return loss


def _get_distance_matrix(vertices: tf.Tensor) -> tf.Tensor:
    return tf.norm(tf.expand_dims(vertices, axis=0) - tf.expand_dims(vertices, axis=1), axis=-1)


def identity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(y_pred)
