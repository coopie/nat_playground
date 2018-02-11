import os
import logging
from subprocess import call

from tensorflow.examples.tutorials.mnist import input_data


BASEDIR = os.path.dirname(os.path.realpath(__file__))


def mnist_digits():
    dataset = input_data.read_data_sets(f'{BASEDIR}/data/MNIST/', one_hot=False, reshape=False)
    return [(x.images, x.labels) for x in [dataset.train, dataset.validation, dataset.test]]


def mnist_fashion():
    if not os.path.isdir(os.path.join(BASEDIR, 'data', 'fashion-mnist')):
        logging.info('downloading fashion-mnist..')
        call(f'bash {BASEDIR}/data/download_fashion_mnist.sh', shell=True)
        logging.info('DONE')

    dataset = input_data.read_data_sets(f'{BASEDIR}/data/fashion-mnist/', one_hot=False, reshape=False)
    return [(x.images, x.labels) for x in [dataset.train, dataset.validation, dataset.test]]
