import utils


def test_fast_random_choice():
    things = list(range(50))

    for i in range(50):
        sample = utils.fast_random_choice(things, 49, num_tries=5)
        assert len(sample) == len(set(sample)), 'fast_random_choice should always return unique numbers'
