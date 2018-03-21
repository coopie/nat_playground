import numpy as np


def evenly_distribute(bucket_weights, number_of_points):
    normalized = bucket_weights / bucket_weights.sum()

    distributed = np.around(normalized * number_of_points).astype(np.int32)
    return distributed


def sample_from_heatmap(heatmap, num_targets, sampling_method='random'):
    """
    TODO: what this does

        for each pixel, sample uniformly within that pixel
    """
    allowed_methods = {'random', 'even'}
    assert sampling_method in allowed_methods, f'Incorrect sampling method {sampling_method}. Allowed {allowed_methods}'
    assert heatmap.shape[0] == heatmap.shape[1]
    width = heatmap.shape[0]

    probability_buckets = heatmap.flatten() / heatmap.sum()
    coordinates_of_buckets = np.array([
        [i % width, i // width] for i in range(len(probability_buckets))
    ])
    assert coordinates_of_buckets.max() == width - 1

    if sampling_method == 'random':
        sampled_flattened_pixel_indices = np.random.choice(
            len(probability_buckets),
            num_targets,
            replace=True,
            p=probability_buckets
        )
    elif sampling_method == 'even':
        bucket_counts = evenly_distribute(probability_buckets, num_targets)
        sampled_flattened_pixel_indices = np.concatenate([
            np.ones(bucket_count, dtype=np.int64) * i for i, bucket_count in enumerate(bucket_counts)
        ])
        # there is always some missing indices due to rounding down, or too many from rounding up
        sampled_flattened_pixel_indices = np.concatenate(
            (
                sampled_flattened_pixel_indices,
                np.random.choice(
                    len(probability_buckets),
                    max(num_targets - len(sampled_flattened_pixel_indices), 0),
                    replace=True,
                    p=probability_buckets
                )
            )
        )
        sampled_flattened_pixel_indices = sampled_flattened_pixel_indices[:num_targets]

    sample_pixel_coordinates = [
        coordinates_of_buckets[sampled_index]
        for sampled_index in sampled_flattened_pixel_indices
    ]

    unnormalized_points = np.array([
        [np.random.uniform(x, x + 1), np.random.uniform(y, y + 1)]
        for x, y in sample_pixel_coordinates
    ])

    return unnormalized_points / width
