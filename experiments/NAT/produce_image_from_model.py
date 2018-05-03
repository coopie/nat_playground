import sys
import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import metric_logging
from utils import import_module
from scipy.misc import imsave


def produce_image_from_model(
    model_fn,
    logdir,
    input_noise_fn,
    **unused
):
    batch_size = None
    number_of_points = 10_000_000

    inputs = input_noise_fn(10_000_000)

    input_t = tf.placeholder(
        dtype=tf.float32,
        name='input_t',
        shape=(batch_size, *inputs.shape[1:])
    )
    z_t = model_fn(input_t, 2)
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
    batch_size = 512
    points = np.concatenate(
        [
            sess.run(z_t, feed_dict={input_t: inputs[i: i + batch_size]})
            for i in tqdm(range(0, len(inputs), batch_size))
        ]
    )

    reconstructed, *_ = np.histogram2d(points[:, 0], points[:, 1], bins=(256, 256))
    reconstructed_image = reconstructed.T / reconstructed.max()
    metric_logger.log_images(
        f'reconstructed_{number_of_points}_points',
        [reconstructed_image],
        0
    )
    imsave(f'{logdir}/reconstructed.png', reconstructed_image)


if __name__ == '__main__':
    tf_log = logging.getLogger('tensorflow')
    tf_log.setLevel(logging.ERROR)
    args = sys.argv[1:]
    assert len(args) == 1, 'usage: <path_to_logdir>'

    config = import_module(f'{args[0]}/config.py').config

    produce_image_from_model(**config, logdir=args[0], config=config)
