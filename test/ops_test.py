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
