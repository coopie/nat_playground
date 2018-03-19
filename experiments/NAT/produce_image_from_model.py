import sys

import numpy as np
import tensorflow as tf

import metric_logging
from utils import import_module
from scipy.misc import imsave


def produce_image_from_model(
    dataset_fn,
    model_fn,
    logdir,
    **unused
):
    batch_size = None
    number_of_points = 10_000_000
    z_dim = 2  # we are genrating a greyscale image, so if z != 2, then something is wrong

    (train_x, _), (validation_x, _), _ = dataset_fn()
    # HACK we are assuming a gaussian noise input here, might eb worth refactoring to be a bit more general
    noise_dims = train_x.shape[1]
    new_noise = np.random.normal(size=(number_of_points, noise_dims))

    input_t = tf.placeholder(
        dtype=tf.float32,
        name='input_t',
        shape=(batch_size, *train_x.shape[1:])
    )
    z_t = model_fn(input_t, z_dim)
    tf.train.get_or_create_global_step()

    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=logdir,
        save_checkpoint_secs=None,
        save_summaries_steps=None
    )
    metric_logger = metric_logging.TensorboardLogger(
        writer=tf.summary.FileWriterCache.get(logdir)
    )

    print('generating points....')
    points = sess.run(z_t, feed_dict={input_t: new_noise})

    reconstructed, *_ = np.histogram2d(points[:, 0], points[:, 1], bins=(256, 256))
    reconstructed_image = reconstructed.T / reconstructed.max()
    metric_logger.log_images(
        f'reconstructed_{number_of_points}_points',
        [reconstructed_image],
        0
    )
    imsave(f'{logdir}/reconstructed.png', reconstructed_image)


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) == 1, 'usage: <path_to_logdir>'

    config = import_module(f'{args[0]}/config.py').config

    produce_image_from_model(**config, logdir=args[0], config=config)
