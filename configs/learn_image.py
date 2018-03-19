from scipy.misc import imread
import utils
import numpy as np

import noise_as_targets
import models

# im = imread('data/images/donald_duck.jpeg', mode='I')
im = imread('data/images/spiral.png', mode='I')
# im = 255 - imread('data/images/me.png', mode='I')
# im = 255 - imread('data/images/smart_ali_crop.gif', mode='I')
# im = 255 - imread('data/images/ali-cropped.jpg', mode='I')
heatmap = utils.image_to_square_greyscale_array(im)

data_points = np.random.normal(size=(100_000, 2))


config = {
    'dataset_fn': lambda: [(data_points, None), (data_points, None), (None, None)],
    'targets_fn': lambda num_targets: noise_as_targets.sample_uniformly_from_heatmap(heatmap, num_targets),
    'model_fn': lambda input_t, output_size: models.nulti_layer_mlp(input_t, output_size, hidden_dims=[256, 256])
}
