import tensorflow as tf


def mlp_model(input_t, z_size, hidden_dim=128):
    flattened = tf.layers.flatten(input_t)

    # encoder
    with tf.variable_scope('encoder'):
        encoder_hidden = tf.contrib.layers.fully_connected(
            flattened,
            hidden_dim
        )
        z_t = tf.contrib.layers.fully_connected(
            encoder_hidden,
            z_size,
            activation_fn=None
        )

    with tf.variable_scope('decoder'):
        decoder_hidden = tf.contrib.layers.fully_connected(
            z_t,
            hidden_dim
        )
        reconstructed_flattened = tf.contrib.layers.fully_connected(
            decoder_hidden,
            flattened.shape[1].value,
            activation_fn=tf.sigmoid
        )

    return tf.reshape(
        reconstructed_flattened,
        shape=input_t.shape
    ), z_t
