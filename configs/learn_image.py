from scipy.misc import imread
import utils
import numpy as np

import noise_as_targets
import models
import batching_functions

# im = imread('data/images/donald_duck.jpeg', mode='I')
im = imread('data/images/spiral.png', mode='I')
# im = 255 - imread('data/images/me.png', mode='I')
# im = 255 - imread('data/images/all_that_must_be.jpg', mode='I')
# im = 255 - imread('data/images/smart_ali_crop.gif', mode='I')
# im = 255 - imread('data/images/ali-cropped.jpg', mode='I')
heatmap = utils.image_to_square_greyscale_array(im)

seed = 1337
train_size = 20_000
data_points = np.random.normal(size=(train_size, 8))
# l2 normalize the points
data_points /= np.linalg.norm(data_points, axis=1, ord=2).reshape((-1, 1))

targets = noise_as_targets.sample_from_heatmap(
    heatmap, train_size, sampling_method='even',
)

batching_function = batching_functions.random_batching(targets)
# batching_function = batching_functions.progressive_local_search(targets)

config = {
    'dataset_fn': lambda: (data_points, targets),
    'model_fn': lambda input_t, output_size: models.multi_layer_mlp(input_t, output_size, hidden_dims=[32, 32]),
    'batch_size': 32,
    'batching_fn': batching_function
}
