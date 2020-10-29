import tensorflow as tf

from . import topology_np, topology_tf


class TopoLoss(tf.keras.layers.Layer):
    def __init__(self, reg_lambda: float = 0.5, numpy=True):
        super().__init__()
        self.reg_lambda = reg_lambda
        self.latent_norm = tf.Variable(1.0)
        self.numpy = numpy

    def call(self, y_true, latent, y_pred):
        ae_loss = tf.losses.mean_squared_error(y_true, y_pred)
        topo_loss = self._topo_loss(y_true, latent)
        return ae_loss + self.reg_lambda * topo_loss

    def build(self, input_shape):
        if not self.numpy:
            batch_size = input_shape[0]
            self.components = tf.Variable(tf.range(batch_size), trainable=False, name="components")

    def _get_edges(self, distances):
        if self.numpy:
            return tf.numpy_function(topology_np.get_0d_homology_edges, [distances], Tout=tf.int32)
        else:
            # Reset the components before calculating edges
            self.components.assign(tf.range(self.components.shape[0]))
            return topology_tf.get_0d_homology_edges(distances, self.components)

    def _topo_loss(self, data, latent):
        data_distances = _get_distance_matrix(data)
        data_distances /= tf.reduce_max(data_distances)
        latent_distances = _get_distance_matrix(latent)
        latent_distances /= self.latent_norm

        edges = self._get_edges(tf.stop_gradient(data_distances))
        data_loss = tf.losses.mean_squared_error(
            tf.gather_nd(data_distances, edges),
            tf.gather_nd(latent_distances, edges),
        )

        edges = self._get_edges(tf.stop_gradient(latent_distances))
        latent_loss = tf.losses.mean_squared_error(
            tf.gather_nd(data_distances, edges),
            tf.gather_nd(latent_distances, edges),
        )
        return data_loss + latent_loss


def _get_distance_matrix(vertices: tf.Tensor) -> tf.Tensor:
    return tf.norm(tf.expand_dims(vertices, axis=0) - tf.expand_dims(vertices, axis=1), axis=-1)


def identity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(y_pred)
