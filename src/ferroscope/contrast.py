import math

import dask.array as da
import numpy as np

from ferroscope.io import Image


def min_max_from_counts(counts) -> tuple[int, int]:
    """Lowest and highest pixel value present in a histogram.

    counts[i] = number of pixels with value i (as from np.bincount).
    """
    array = np.array(counts)
    if array.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {array.shape}")

    nonzero = np.flatnonzero(array)

    if nonzero.size == 0:
        raise ValueError("no pixels counted")

    first_index = int(nonzero[0])
    last_index = int(nonzero[-1])

    return (first_index, last_index)


def percentile_from_counts(counts, p: float) -> int:
    """Pixel value at percentile p (0–100) of a histogram.

    Matches np.percentile(data, p, method="inverted_cdf").
    """
    array = np.array(counts)
    if array.ndim != 1:
        raise ValueError(f"counts must be 1-D, got shape {array.shape}")

    if not 0 <= p <= 100:
        raise ValueError(f"p must be in 0..100, got {p}")

    cdf = np.cumsum(array)

    n = cdf[-1]

    if n == 0:
        raise ValueError("no pixels")

    k = max(math.ceil(n * (p / 100)), 1)

    val = np.searchsorted(cdf, k, side="left")

    return int(val)


def channel_histogram(img: Image, c: int) -> np.ndarray:
    """Pixel-value counts of channel c over all T, Z, Y, X. Lazy, chunk by chunk."""

    n_channels = img.dask_data.shape[1]

    if not 0 <= c < n_channels:
        raise ValueError(f"channel {c} out of range, image has {n_channels}")

    data = img.dask_data[:, c]

    if not np.issubdtype(data.dtype, np.unsignedinteger):
        raise ValueError(f"need unsigned integer pixels, got {data.dtype}")

    nbins = np.iinfo(data.dtype).max + 1
    counts = da.bincount(data.ravel(), minlength=nbins).compute()

    return counts


def shared_contrast(
    images: list[Image], low: float = 0.1, high: float = 99.9
) -> list[tuple[int, int]]:
    """One (lo, hi) per channel, from the pooled histogram of all images."""
    # 1. images empty → ValueError
    if not images:
        raise ValueError("images is empty")

    if low >= high:
        raise ValueError(f"low must be < high, got low={low}, high={high}")

    ref = images[0].dask_data
    n_channels = ref.shape[1]

    for i, image in enumerate(images):
        data = image.dask_data
        if data.shape[1] != n_channels:
            raise ValueError(
                f"image {i} has {data.shape[1]} channels, image 0 has {n_channels}"
            )
        if data.dtype != ref.dtype:
            raise ValueError(f"image {i} is {data.dtype}, image 0 is {ref.dtype}")

    limits = []

    for c in range(n_channels):
        total = sum(channel_histogram(img, c) for img in images)
        lo = percentile_from_counts(total, low)
        hi = percentile_from_counts(total, high)
        limits.append((lo, hi))

    return limits
