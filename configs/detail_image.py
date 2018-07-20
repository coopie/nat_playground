from scipy.misc import imread
import numpy as np
import tensorflow as tf

import utils
import noise_as_targets
import models
import batching_functions

# im = imread('data/images/donald_duck.jpeg', mode='I')
# im = imread('data/images/spiral.png', mode='I')
# im = imread('data/images/mandelbrot.jpeg', mode='I')
# im = 255 - imread('data/images/me2.png', mode='I')
# im = 255 - imread('data/images/all_that_must_be.gif', mode='I')
# im = 255 - imread('data/images/yanny.png', mode='I')
im = imread('data/images/me3.png', mode='I')
# im = 255 - imread('data/images/smart_ali_crop.gif', mode='I')
# im = 255 - imread('data/images/ali-cropped.jpg', mode='I')
heatmap = utils.image_to_square_greyscale_array(im)

seed = 1337
np.random.seed(seed)
# train_size = 64_000 * 8
train_size = 64_000 * 8 * 4

# data_points = np.random.normal(size=(train_size, 3))
# # l2 normalize the points
# data_points /= np.linalg.norm(data_points, axis=1, ord=2).reshape((-1, 1))
# data_points = np.random.uniform(size=(train_size, 100))
input_noise_fn = lambda size: np.random.uniform(size=(size, 256))  # NOQA
data_points = input_noise_fn(train_size)

targets = noise_as_targets.sample_from_heatmap(
    heatmap, train_size, sampling_method='even',
)

# batching_function = batching_functions.progressive_local_search(targets)
batching_function = batching_functions.random_batching(targets)

config = {
    'dataset_fn': lambda: (data_points, targets),
    'model_fn': lambda input_t, output_size: models.multi_layer_mlp(
        input_t, output_size, hidden_dims=[512, 512], activation_fn=tf.tanh
    ),
    'batch_size': 128,
    'batching_fn': batching_function,
    # 'eval_steps': 500
    'eval_steps': 2000,
    'input_noise_fn': input_noise_fn,
    'train_steps': 20_000_000,
    # 'image_dimensions': (512, 512)
    'image_dimensions': (700, 700),
    'optimizer': tf.train.AdamOptimizer(1e-4)
}
