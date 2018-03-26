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


def bucket_into_sub_regions(targets, bounds=((0, 1), (0, 1)), buckets=(20, 20)):
    """
    Produce a 2D array where the (i,j)th entry is a list of all indices in targets which exust in that bucketed area
    """
    assert len(targets.shape) == 2 and targets.shape[1] == 2
    x_buckets, y_buckets = buckets
    (x_min, x_max), (y_min, y_max) = bounds
    x_step = (x_max - x_min) / x_buckets
    y_step = (y_max - y_min) / y_buckets

    bucketed = [
        [[] for _ in range(y_buckets)] for _ in range(x_buckets)
    ]
    x_bucket_indices = ((targets[:, 0] - x_min) / x_step).astype(np.int32)
    y_bucket_indices = ((targets[:, 1] - y_min) / y_step).astype(np.int32)

    for i, (x, y) in enumerate(zip(x_bucket_indices, y_bucket_indices)):
        bucketed[x][y] += [i]

    for i in range(x_buckets):
        for j in range(y_buckets):
            bucketed[i][j] = np.array(bucketed[i][j])

    return bucketed
