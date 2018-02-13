import numpy as np


def gaussian_noise(num_targets, dims):
    return np.random.normal(size=(num_targets, dims))


def sample_uniformly_from_heatmap(heatmap, num_targets):
    """
    TODO: what this does

        for each pixel, sample uniformly within that pixel
    """
    assert heatmap.shape[0] == heatmap.shape[1]
    width = heatmap.shape[0]

    probability_buckets = heatmap.flatten() / heatmap.sum()
    coordinates_of_buckets = np.array([
        [i % width, i // width] for i in range(len(probability_buckets))
    ])
    assert coordinates_of_buckets.max() == width - 1

    sampled_flattened_pixel_indices = np.random.choice(
        len(probability_buckets),
        num_targets,
        replace=False,
        p=probability_buckets
    )

    sample_pixel_coordinates = [
        coordinates_of_buckets[sampled_index]
        for sampled_index in sampled_flattened_pixel_indices
    ]

    unnormalized_points = np.array([
        [np.random.uniform(x, x + 1), np.random.uniform(y, y + 1)]
        for x, y in sample_pixel_coordinates
    ])

    return unnormalized_points / width
