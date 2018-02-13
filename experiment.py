import logging
import os

import numpy as np
import tensorflow as tf

import noise_as_targets
import datasets
import models
import ops


def run_experiment(
    targets_fn,
    run_name
):
    dataset_fn = datasets.mnist_digits
    model_fn = models.mlp_model
    batch_size = 100
    nat_loss_coefficient = 1e-2
    optimizer = tf.train.AdamOptimizer()

    (train_x, _), (validation_x, _), _ = dataset_fn()
    targets = targets_fn(len(train_x))

    input_t = tf.placeholder(
        dtype=tf.float32,
        name='input_t',
        shape=(batch_size, *train_x.shape[1:])
    )
    target_t = tf.placeholder(
        dtype=tf.float32,
        name='input_t',
        shape=(batch_size, targets.shape[1])
    )

    reconstructed_t, z_t = model_fn(input_t, targets.shape[1])

    mean_reconstruction_loss_t = tf.nn.l2_loss(
        reconstructed_t - input_t
    ) / batch_size

    cost_matrix_t = ops.cost_matrix(
        z_t, target_t,
        loss_func=ops.euclidean_distance
    )
    new_assignment_indices = ops.hungarian_method(cost_matrix_t)
    new_targets_t = tf.gather(target_t, new_assignment_indices)

    nat_loss_t = ops.euclidean_distance(new_targets_t, z_t)
    mean_nat_loss_t = tf.reduce_mean(nat_loss_t)

    total_loss = mean_nat_loss_t * nat_loss_coefficient + mean_reconstruction_loss_t

    global_step_t = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(
        total_loss,
        global_step_t
    )

    tf.summary.scalar('mean_nat_loss', mean_nat_loss_t)
    tf.summary.scalar('mean_reconstruction_loss', mean_reconstruction_loss_t)

    logdir = os.path.join('model_logs', run_name)
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=200,
        save_summaries_steps=200
    )

    logging.info('Training')
    while True:
        batch_indices = np.random.choice(len(train_x), size=batch_size)
        batch_images = train_x[batch_indices]
        batch_target = targets[batch_indices]

        current_step, new_targets, _ = sess.run(
            [global_step_t, new_targets_t, train_op],
            feed_dict={
                input_t: batch_images,
                target_t: batch_target
            }
        )
        targets[batch_indices] = new_targets














if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = {
        "targets_fn": lambda num_targets: noise_as_targets.gaussian_noise(num_targets, 2)
    }
    run_name = 'testing'
    run_experiment(**config, run_name=run_name)
