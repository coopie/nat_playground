import numpy as np

import batching_functions


def test_cyclical_batching():
    points = np.random.uniform(size=(10, 2))
    batching_function = batching_functions.cyclical_batching(points)

    a, b = batching_function(5, points, {}), batching_function(5, points, {})
    # check that this is the whole dataset
    epoch_indices = np.concatenate((a, b)).tolist()
    assert len(epoch_indices) == 10
    assert len(epoch_indices) == len(set(epoch_indices))
    assert set(epoch_indices) == set(range(10))

    batches = [batching_function(2, points, {}) for _ in range(5)]
    assert all(len(batch) == 2 for batch in batches)
    epoch_indices2 = np.concatenate(batches).tolist()

    assert set(epoch_indices2) == set(epoch_indices)


def test_random_batching():
    points = np.random.uniform(size=(10, 2))
    batching_function = batching_functions.cyclical_batching(points)
    for i in range(20):
        indices = batching_function(5, points, {})
        assert len(indices) == 5
        assert len(set(indices)) == len(indices)


def test_progressive_local_search():
    # TODO: proper testing other than covering the code
    points = np.random.uniform(size=(200, 2))
    batching_function = batching_functions.progressive_local_search(points)

    for i in range(10):
        a = batching_function(batch_size=10, context={'current_step': i}, targets=points)
        assert a.ndim == 1
        assert len(a) == 10
        assert len(set(a.tolist())) == len(a)
