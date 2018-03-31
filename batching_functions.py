import itertools
import logging

import numpy as np

import noise_as_targets
import utils


def progressive_local_search(targets):
    bounds = ((0, 1), (0, 1))
    buckets = (256, 256)

    bucket_matrix, index_fn = noise_as_targets.bucket_into_sub_regions(
        targets,
        bounds=bounds,
        buckets=buckets
    )

    def batching_function(batch_size, targets, context):
        nonlocal index_fn
        nonlocal bucket_matrix
        random_target_sample = targets[np.random.choice(len(targets))]

        # HACK for the time being, only look at the current step?
        current_step = context['current_step']
        index_x, index_y = index_fn(random_target_sample)
        search_radius = max(256 - (current_step // 1000), 1)
        # END

        point_indices = np.concatenate([
            bucket_matrix[i, j]
            for i, j in itertools.product(
                range(max(index_x - search_radius, 0), min(index_x + search_radius, 256)),
                range(max(index_y - search_radius, 0), min(index_y + search_radius, 256))
            )
        ])
        if len(point_indices) < batch_size:
            logging.warning(
                f'Batching function of search search_radius {search_radius} with batch_size ' +
                f'{batch_size} could not find enough points at {(index_x, index_y)}'
            )
            return np.concatenate(
                (point_indices, utils.fast_random_choice(targets, size=batch_size - len(point_indices)))
            )

        return utils.fast_random_choice(point_indices, size=batch_size)

    return batching_function


def random_batching(targets):
    epoch_indices = np.random.permutation(np.arange(len(targets)))

    def batching_function(batch_size, targets, context):
        nonlocal epoch_indices
        if len(epoch_indices) < batch_size:
            epoch_indices = np.random.permutation(np.arange(len(targets)))

        batch_indices = epoch_indices[:batch_size]
        epoch_indices = epoch_indices[batch_size:]
        return batch_indices

    return batching_function
