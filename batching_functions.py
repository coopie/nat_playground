import itertools
import logging

import numpy as np

import noise_as_targets
import utils


def progressive_local_search(targets):
    bounds = ((0, 1), (0, 1))
    num_buckets = 256
    buckets = (num_buckets, num_buckets)

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
        if current_step > 2 * (len(targets) / batch_size):  # after an epoch
            average_loss = context['average_l2_loss']  # average distance from points to targets
            search_radius = int(average_loss * num_buckets) + 1
        else:
            search_radius = 256
        index_x, index_y = index_fn(random_target_sample)
        # END

        # original idea - a bit cleaner
        # point_indices = np.concatenate([
        #     bucket_matrix[i, j]
        #     for i, j in itertools.product(
        #         range(max(index_x - search_radius, 0), min(index_x + search_radius, num_buckets)),
        #         range(max(index_y - search_radius, 0), min(index_y + search_radius, num_buckets))
        #     )
        # ])
        # if len(point_indices) < batch_size:
        #     logging.warning(
        #         f'Batching function of search search_radius {search_radius} with batch_size ' +
        #         f'{batch_size} could not find enough points at {(index_x, index_y)}'
        #     )
        #     return np.concatenate(
        #         (point_indices, utils.fast_random_choice(len(targets), size=batch_size - len(point_indices)))
        #     )
        #
        # return utils.fast_random_choice(point_indices, size=batch_size)

        # sorry - this is much faster than something more sane
        buckets = [
            bucket_matrix[i, j] for i, j in itertools.product(
                range(max(index_x - search_radius, 0), min(index_x + search_radius, num_buckets)),
                range(max(index_y - search_radius, 0), min(index_y + search_radius, num_buckets))
            )
        ]
        probability_buckets = np.array([len(bucket) for bucket in buckets], dtype=np.float32)
        probability_buckets /= probability_buckets.sum()

        sampled_bucket_indices = np.random.choice(len(buckets), batch_size, replace=True, p=probability_buckets)
        bucket_indices, counts = np.unique(sampled_bucket_indices, return_counts=True)

        indices_for_batch = []
        for bucket_index, count in zip(bucket_indices, counts):
            bucket = np.array(buckets[bucket_index])

            if count > len(bucket):  # Impossible to sample enough points from this bucket
                indices_for_batch += [bucket]
                continue

            indices_from_bucket = utils.fast_random_choice(
                bucket,
                size=count
            )
            indices_for_batch += [indices_from_bucket]

        indices_for_batch = [] if len(indices_for_batch) == 0 else np.concatenate(indices_for_batch)
        if len(indices_for_batch) == batch_size:  # then we didn't break in the loop
            return indices_for_batch

        logging.warning(f'Slow batching used for {batch_size - len(indices_for_batch)} examples in batch')
        # this did'nt work, so just return a random batch along with what's gathered
        return np.concatenate((
            indices_for_batch,
            np.random.choice(len(targets), size=batch_size - len(indices_for_batch), replace=False)
        ))

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
