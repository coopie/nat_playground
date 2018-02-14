import logging
import os
import sys

import numpy as np
import tensorflow as tf

import noise_as_targets
import datasets
import models
import ops
import metric_logging


def run_experiment(
    targets_fn,
    run_name
):
    dataset_fn = datasets.mnist_digits
    model_fn = models.mlp_model
    batch_size = 100
    nat_loss_coefficient = 20.0
    optimizer = tf.train.AdamOptimizer()
    eval_steps = 200

    (train_x, _), (validation_x, _), _ = dataset_fn()
    targets = targets_fn(len(train_x))

    input_t = tf.placeholder(
        dtype=tf.float32,
        name='input_t',
        shape=(batch_size, *train_x.shape[1:])
    )
    target_t = tf.placeholder_with_default(
        input=np.zeros((batch_size, targets.shape[1])),
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

    total_loss = (mean_nat_loss_t * nat_loss_coefficient) + mean_reconstruction_loss_t

    global_step_t = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(
        total_loss,
        global_step_t
    )

    tf.summary.scalar('mean_nat_loss', mean_nat_loss_t)
    tf.summary.scalar('mean_reconstruction_loss/train', mean_reconstruction_loss_t)

    logdir = os.path.join('model_logs', run_name)
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=200,
        save_summaries_steps=200
    )
    metric_logger = metric_logging.TFEventsLogger(log_dir=logdir)

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

        if current_step % eval_steps == 2:
            validation_session_results = [
                sess.run(
                    [mean_reconstruction_loss_t, reconstructed_t],
                    feed_dict={
                        input_t: validation_x[i:i + batch_size],
                    }
                ) for i in range(0, len(validation_x), batch_size)
            ]

            validation_reconstruction_loss = np.mean([
                v[0] for v in validation_session_results
            ])
            # just get the first 100 of validation images
            validation_images = validation_session_results[0][1]

            metric_logger.log_scalar(
                'mean_reconstruction_loss/validation',
                validation_reconstruction_loss,
                current_step
            )
            metric_logger.log_images(
                'validation_reconstructed',
                # workaround needed to save monochrome images
                validation_images[:10][:, :, :, 0],
                current_step
            )



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = {
        "targets_fn": lambda num_targets: noise_as_targets.gaussian_noise(num_targets, 2)
    }
    args = sys.argv[1:]
    assert len(args) == 1
    run_name = 'testing'
    run_experiment(**config, run_name=args[0])
