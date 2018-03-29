import shutil
import logging
import os
import sys

import numpy as np
import tensorflow as tf

import ops
import utils
import metric_logging


def run_experiment(
    targets_fn,
    dataset_fn,
    model_fn,
    batch_size,
    run_name,
    config_path=None
):
    optimizer = tf.train.AdamOptimizer(1e-3)
    eval_steps = 2000

    train_x = dataset_fn()
    targets = targets_fn(len(train_x))

    assert len(train_x) % batch_size == 0, \
        'batch_size must be a multiple for validation size due to laziness'

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

    z_t = model_fn(input_t, targets.shape[1])

    cost_matrix_t = ops.cost_matrix(
        z_t, target_t,
        loss_func=ops.euclidean_distance
    )
    new_assignment_indices_t = ops.hungarian_method(
        tf.expand_dims(cost_matrix_t, 0)
    )[0]
    new_targets_t = tf.gather(target_t, new_assignment_indices_t)

    nat_loss_t = ops.euclidean_distance(new_targets_t, z_t)
    mean_nat_loss_t = tf.reduce_mean(nat_loss_t)

    global_step_t = tf.train.get_or_create_global_step()
    train_op = optimizer.minimize(
        mean_nat_loss_t,
        global_step_t
    )

    tf.summary.scalar('nat_loss/train', mean_nat_loss_t)

    logdir = os.path.join('model_logs', run_name)
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=200,
        save_summaries_steps=eval_steps
    )
    metric_logger = metric_logging.TensorboardLogger(
        writer=tf.summary.FileWriterCache.get(logdir)
    )
    shutil.copyfile(config_path, logdir + '/config.py')

    train_noise_image, *_ = np.histogram2d(targets[:, 0], targets[:, 1], bins=(256, 256))
    metric_logger.log_images(
        'train_noise_image',
        [train_noise_image.T / train_noise_image.max()],
        0
    )

    logging.info('Training')
    epoch_indices = np.random.permutation(np.arange(len(train_x)))
    while True:
        if len(epoch_indices) < batch_size:
            logging.info('Completed Epoch')
            epoch_indices = np.random.permutation(np.arange(len(train_x)))
        batch_indices = epoch_indices[:batch_size]
        epoch_indices = epoch_indices[batch_size:]

        batch_input_noise = train_x[batch_indices]
        batch_target_noise = targets[batch_indices]

        current_step, new_assignment_indices, _ = sess.run(
            [global_step_t, new_assignment_indices_t, train_op],
            feed_dict={
                input_t: batch_input_noise,
                target_t: batch_target_noise
            }
        )

        train_x[batch_indices] = batch_input_noise[new_assignment_indices]

        if current_step % eval_steps == 1:
            # log change in targets
            number_changed = (new_assignment_indices != np.arange(len(new_assignment_indices))).sum()
            metric_logger.log_scalar('fraction_of_targets_changing', number_changed / batch_size, current_step)

            validation_results = [
                sess.run(
                    [z_t],
                    feed_dict={
                        input_t: train_x[i:i + batch_size],
                    }
                ) for i in range(0, len(train_x), batch_size)
            ]
            validation_z = np.concatenate([x[0] for x in validation_results])

            validation_noise_image, *_ = np.histogram2d(validation_z[:, 0], validation_z[:, 1], bins=(256, 256))
            metric_logger.log_images(
                'validation_noise_image',
                [validation_noise_image.T / validation_noise_image.max()],
                current_step
            )


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) == 2, 'usage: <path_to_config> <run_name>'
    config_path, run_name = args

    logging.basicConfig(level=logging.INFO)
    user_config = utils.import_module(config_path).config

    default_config = {
        'targets_fn': None,
        'dataset_fn': None,
        'model_fn': None,
        'batch_size': 100,
    }

    config = {}
    for k in default_config:
        config[k] = default_config[k]
        # overwrite with our config
        if k in user_config:
            config[k] = user_config[k]

    run_experiment(**config, run_name=run_name, config_path=config_path)
