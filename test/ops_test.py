import numpy as np
import tensorflow as tf

import ops


class OpsTest(tf.test.TestCase):
    def test_euclidean_distance(self):
        with self.test_session():
            x = tf.constant([
                [0, 1],
                [0, 1],
                [1, 0]
            ])
            y = tf.constant([
                [0, 1],
                [0, 2],
                [-1, 0]
            ])
            dists_t = ops.euclidean_distance(x, y)
            dists = dists_t.eval()
            np.testing.assert_array_almost_equal(
                dists,
                np.array([0, 1, 2])
            )

    def test_hungarian_method(self):
        with self.test_session():
            distances = tf.constant([
                [
                    [0, 1],
                    [1, 0]
                ],
                [
                    [1, 0],
                    [0, 1]
                ]
            ])
            optimal_indices = ops.hungarian_method(distances).eval()
            np.testing.assert_array_almost_equal(
                optimal_indices,
                [
                    [0, 1],
                    [1, 0]
                ]

            )

            distances = tf.constant([[
                [0, 1, 2],
                [1, 1, 1],
                [4, 1, 9]
            ]])
            optimal_indices = ops.hungarian_method(distances).eval()
            np.testing.assert_array_almost_equal(
                optimal_indices,
                [
                    [0, 2, 1],
                ]

            )
