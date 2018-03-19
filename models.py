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


def two_layer_mlp(input_t, z_size, hidden_dim=128, activation_fn=tf.sigmoid):
    flattened = tf.layers.flatten(input_t)
    encoder_hidden = tf.contrib.layers.fully_connected(
        flattened,
        hidden_dim
    )
    z_t = tf.contrib.layers.fully_connected(
        encoder_hidden,
        z_size,
        activation_fn=activation_fn
    )
    return z_t


def nulti_layer_mlp(input_t, z_size, hidden_dims, activation_fn=tf.sigmoid):
    flattened = tf.layers.flatten(input_t)
    hidden_output = flattened
    for hidden_dim in hidden_dims:
        hidden_output = tf.contrib.layers.fully_connected(
            hidden_output,
            hidden_dim
        )

    z_t = tf.contrib.layers.fully_connected(
        hidden_output,
        z_size,
        activation_fn=activation_fn
    )
    return z_t
