import numpy as np

from ferroscope.contrast import min_max_from_counts, percentile_from_counts


def test_min_max_basic():
    counts = np.array([0, 3, 0, 2])  # pixel values: 1, 1, 1, 3, 3
    assert min_max_from_counts(counts) == (1, 3)


def test_percentile_median():
    counts = np.array([0, 3, 0, 2])  # pixels: 1, 1, 1, 3, 3
    assert percentile_from_counts(counts, 50) == 1


def test_percentile_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4096, size=10_000)
    counts = np.bincount(data)
    for p in [0, 0.1, 1, 25, 50, 75, 99, 99.9, 100]:
        assert percentile_from_counts(counts, p) == np.percentile(
            data, p, method="inverted_cdf"
        )
