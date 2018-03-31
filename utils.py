import importlib.util

import numpy as np


def import_module(path):
    spec = importlib.util.spec_from_file_location('', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def image_to_square_greyscale_array(image):
    intensity_map = (1 - (image / 255))

    if not intensity_map.shape[1] == intensity_map.shape[0]:
        dimension_to_pad = np.argmin(intensity_map.shape)
        dominant_dimension = np.argmax(intensity_map.shape)
        amount_to_pad = intensity_map.shape[dominant_dimension] - intensity_map.shape[dimension_to_pad]

        pad_amounts = [[0, 0], [0, 0]]
        pad_amounts[dimension_to_pad] = [
            amount_to_pad // 2, amount_to_pad // 2 + amount_to_pad % 2
        ]

        intensity_map = np.pad(
            intensity_map,
            pad_width=pad_amounts,
            mode='constant',
            constant_values=0
        )
    return intensity_map


def fast_random_choice(x, size, num_tries=2):
    """
    Try a really fast random sample (with replacement) as it's much faster than without.
    if it doesn't work, try without replacement.
    """
    for _ in range(num_tries):
        sample = np.random.choice(x, size, replace=True)
        if len(sample) == len(np.unique(sample)):
            return sample

    # otherwise do the slow sampling
    return np.random.choice(x, size, replace=False)
