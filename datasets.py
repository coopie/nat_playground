import os
from subprocess import call
import logging

from tensorflow.examples.tutorials.mnist import input_data


BASEDIR = os.path.dirname(os.path.realpath(__file__))


def mnist_digits():
    path = os.path.join(BASEDIR, 'data', 'mnist_digits')
    dataset = input_data.read_data_sets(path, one_hot=False, reshape=False)
    return [(x.images, x.labels) for x in [dataset.train, dataset.validation, dataset.test]]


def mnist_fashion():
    path = os.path.join(BASEDIR, 'data', 'fashion-mnist')
    if not os.path.isdir(path):
        logging.info('downloading fashion-mnist..')
        call(f'bash {BASEDIR}/data/download_fashion_mnist.sh', shell=True)
        logging.info('DONE')

    dataset = input_data.read_data_sets(path, one_hot=False, reshape=False)
    return [(x.images, x.labels) for x in [dataset.train, dataset.validation, dataset.test]]
