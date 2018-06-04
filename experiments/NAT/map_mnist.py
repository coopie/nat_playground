import sys
import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from utils import import_module, save_embeddings


def produce_image_from_model(
    model_fn,
    logdir,
    **unused
):

    dataset = input_data.read_data_sets("data/MNIST/", one_hot=False, reshape=False)
    inputs = np.concatenate(
        [x.images for x in [dataset.train, dataset.validation, dataset.test]]
    )
    labels = np.concatenate(
        [x.labels for x in [dataset.train, dataset.validation, dataset.test]]
    )
    images = vector_to_matrix_mnist(inputs)
    sprite_array = 1 - create_sprite_image(images)
    imsave(f'{logdir}/sprites.png', sprite_array)

    batch_size = None
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

    print('generating points....')
    batch_size = 1000
    points = np.concatenate(
        [
            sess.run(z_t, feed_dict={input_t: inputs[i: i + batch_size]})
            for i in tqdm(range(0, len(inputs), batch_size))
        ]
    )
    # add another zeroed dimension to get 3 dimensions for tensorflow projector
    points = np.stack((points[:, 0], points[:, 1], np.zeros(shape=len(points))), axis=1)
    save_embeddings(outputs=points, labels=labels, logdir=logdir)


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[
                    i * img_h:(i + 1) * img_h,
                    j * img_w:(j + 1) * img_w
                ] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))



if __name__ == '__main__':
    tf_log = logging.getLogger('tensorflow')
    tf_log.setLevel(logging.ERROR)
    args = sys.argv[1:]
    assert len(args) == 1, 'usage: <path_to_logdir>'

    config = import_module(f'{args[0]}/config.py').config

    produce_image_from_model(**config, logdir=args[0], config=config)
