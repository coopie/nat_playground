import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import scipy


def cost_matrix(predicted, targets, loss_func, scope=None):
    """
    Compute the cost for each predicted value to every target.

    Each row contains the cost of one predicted point to each target.
    """
    assert predicted.shape.is_compatible_with(targets.shape), 'predicted and targets must be of the same shape.'
    set_size = predicted.shape[0].value
    with ops.name_scope(scope, 'cost_matrix', [predicted, targets]):
        return tf.stack(
            [
                loss_func(repeat(p, set_size), targets)
                for p in tf.unstack(predicted)
            ]
        )


def repeat(x, n, scope=None):
    """
    Repeat a 1D vector N times
    """
    with ops.name_scope(scope, 'repeat', [x]):
        return tf.transpose(tf.tile(tf.expand_dims(x, -1), [1, n]))


def dot_product(x, y):
    coordinate_muls = tf.multiply(x, y)  # b x w
    return tf.reduce_sum(coordinate_muls, -1)


def cosine_sim(x, y):
    """
    Gets batches of left and right vectors b x w (w is the same on both sides)
        and returns the cosine similarity (b x 1)

    Also works for single vectors
    """
    with tf.name_scope('cosine_sim', values=[x, y]):
        x_norm = tf.nn.l2_normalize(x, dim=-1)
        y_norm = tf.nn.l2_normalize(y, dim=-1)
        cos_sim = dot_product(x_norm, y_norm)
    return cos_sim


def hungarian_method(cost_matrix):
    return tf.py_func(
        scipy.optimize.linear_sum_assignment,
        [cost_matrix],
        stateful=False,
        Tout=[tf.int32],
        name='hungarian_method'
    )
