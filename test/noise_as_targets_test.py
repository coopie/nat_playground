import numpy as np

import noise_as_targets


def test_bucket_into_sub_regions_simple():
    points = np.random.uniform(size=(20, 2))
    bucketed, _ = noise_as_targets.bucket_into_sub_regions(
        points,
        bounds=((0, 1), (0, 1)),
        buckets=(1, 1)
    )
    assert len(bucketed) == 1, 'expected only one bucket'
    assert len(bucketed[0]) == 1, 'expected only one bucket'
    assert len(bucketed[0][0]) == 20, 'bucket expected to contain all 20 points'

    assert all(i in bucketed[0][0] for i in range(20)), 'expected one buckets with all points in it'


def test_bucket_into_sub_regions_points_are_in_correct_regions_bucketed_correct():
    points = np.array([
        [0.1, 0.1],
        [0.1, 0.1],
        [0.1, 0.1],
        [0.49, 0.49],
        [0.51, 0.49],
        [0.49, 0.51],
        [0.51, 0.51]
    ]) + 1
    # first bucket in the x direction only
    bucketed, _ = noise_as_targets.bucket_into_sub_regions(
        points,
        bounds=((1, 2), (1, 2)),
        buckets=(2, 1)
    )

    # so the bucketed point indices should be shape [2, 1]
    assert len(bucketed) == 2, 'should be 2 buckets in the x direction'
    assert len(bucketed[0]) == 1, 'should be one bucket in the y direction'

    np.testing.assert_array_almost_equal(
        bucketed[0][0],
        [0, 1, 2, 3, 5]
    )
    np.testing.assert_array_almost_equal(
        bucketed[1][0],
        [4, 6]
    )

    bucketed, _ = noise_as_targets.bucket_into_sub_regions(
        points,
        bounds=((1, 2), (1, 2)),
        buckets=(2, 2)
    )
    np.testing.assert_array_almost_equal(
        bucketed[0][0],
        [0, 1, 2, 3]
    )
    np.testing.assert_array_almost_equal(
        bucketed[1][0],
        [4, ]
    )
    np.testing.assert_array_almost_equal(
        bucketed[0][1],
        [5]
    )
    np.testing.assert_array_almost_equal(
        bucketed[1][1],
        [6]
    )


def test_bucket_into_sub_regions_points_are_in_correct_regions_index_fn_correct():
    points = np.random.uniform(low=0, high=1.0, size=(500, 2))

    bucketed, index_fn = noise_as_targets.bucket_into_sub_regions(
        points,
        bounds=((0, 1), (0, 1)),
        buckets=(20, 20)
    )

    # test reverse lookup is correct for a few points
    for i, point in enumerate(points[:10]):
        assert i in bucketed[index_fn(point)], \
            f'Index lookup of {point} ({index_fn(point)}) did not return a bucket containing index {i}'

    # np.testing.assert_array_equal(
    #     points[50:60],
    #     bucketed[index]
    # )
