from scipy.misc import imread
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import utils
import noise_as_targets
import models
import batching_functions

im = imread('data/images/donald_duck.jpeg', mode='I')
# im = imread('data/images/spiral.png', mode='I')
# im = imread('data/images/mandelbrot.jpeg', mode='I')
# im = 255 - imread('data/images/me.png', mode='I')
# im = 255 - imread('data/images/all_that_must_be.jpg', mode='I')
# im = 255 - imread('data/images/smart_ali_crop.gif', mode='I')
# im = 255 - imread('data/images/ali-cropped.jpg', mode='I')
heatmap = utils.image_to_square_greyscale_array(im)

seed = 1337
np.random.seed(seed)

dataset = input_data.read_data_sets("data/MNIST/", one_hot=False, reshape=False)
data_points = np.concatenate(
    [x.images for x in [dataset.train, dataset.validation, dataset.test]]
)
batch_size = 128
data_points = data_points.reshape((len(data_points), -1))
np.random.shuffle(data_points)

targets = noise_as_targets.sample_from_heatmap(
    heatmap, len(data_points), sampling_method='even',
)

batching_function = batching_functions.random_batching(targets)
# batching_function = batching_functions.progressive_local_search(targets)

config = {
    'dataset_fn': lambda: (data_points, targets),
    'model_fn': lambda input_t, output_size: models.multi_layer_mlp(
        input_t, output_size, hidden_dims=[128, 128], activation_fn=tf.sigmoid
    ),
    'batch_size': batch_size,
    'batching_fn': batching_function,
    'eval_steps': 50,
    'input_noise_fn': lambda x: exit('input is not noise')
}
