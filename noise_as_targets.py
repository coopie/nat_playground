import numpy as np
import itertools


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


def bucket_into_sub_regions(points, bounds=((0, 1), (0, 1)), buckets=(20, 20)):
    """
    Produce a ND array where the (i,j...)th entry is a list of all indices in points which exist in that bucketed area
    """
    # TODO: maybe allow non-linear bucketing?
    dimensions = points.shape[1]
    assert len(buckets) == dimensions, \
        f'buckets must have the same dimensions as the noise, expected {dimensions}, got {len(buckets)}'
    assert len(bounds) == dimensions, \
        f'bounds must have the same dimensions as the noise, expected {dimensions}, got {len(bounds)}'
    assert all(len(bound) == 2 for bound in bounds), 'all dimension bounds must be (low, high)'

    # calculate the bucket step size for each dimension
    step_sizes = np.array([
        (max_bound - min_bound) / num_buckets
        for (min_bound, max_bound), num_buckets in zip(bounds, buckets)
    ])

    def bucket_lookup(point):
        """
        point can be single point or multiple points
        """
        nonlocal dimensions
        nonlocal step_sizes
        nonlocal bounds
        # for each dimension, find the index for each point, then stack them
        window_fracton = np.stack([
            # point - min_bound / step_size
            (point[:, i] - bounds[i][0]) / step_sizes[i]
            for i in range(dimensions)
        ], axis=1)

        # truncate to integers to find the indices
        return window_fracton.astype(np.int32)

    # create a n-dimensional bucket array (A bit hacky, but everything else was much too fiddly)
    bucketed = np.zeros(buckets).tolist()

    for index in itertools.product(*[range(num_buckets) for num_buckets in buckets]):
        # drill into the list
        bucket_view = bucketed
        for coord in index[:-1]:
            bucket_view = bucket_view[coord]

        bucket_view[index[-1]] = []

    # add the index of each target to the corresponding buckets
    for i, point_index in enumerate(bucket_lookup(points)):
        bucket_view = bucketed
        for index in point_index:
            bucket_view = bucket_view[index]

        bucket_view.append(i)

    return np.array(bucketed), bucket_lookup
