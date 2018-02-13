import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
import numpy as np


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


def euclidean_distance(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    squared_distance = ((x - y) ** 2)
    return tf.sqrt(tf.reduce_sum(squared_distance, axis=1))


def repeat(x, n, scope=None):
    """
    Repeat a 1D vector N times
    """
    with ops.name_scope(scope, 'repeat', [x]):
        return tf.transpose(tf.tile(tf.expand_dims(x, -1), [1, n]))


def hungarian_method(cost_matrix):

    def optimal_matching(*args, **kwargs):
        indices, _ = scipy.optimize.linear_sum_assignment(*args, **kwargs)
        return indices.astype(np.int32)

    return tf.py_func(
        optimal_matching,
        [cost_matrix],
        stateful=False,
        Tout=[tf.int32],
        name='hungarian_method'
    )
