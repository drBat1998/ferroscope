import dask.array as da
import pytest

from ferroscope.io import Image

DIMS = ("T", "C", "Z", "Y", "X")


def test_create(arr):
    img = Image(data=da.from_array(arr), dims=DIMS)
    assert img.data.shape == (2, 3, 1, 8, 8)
    assert img.pixel_size == (None, None, None)
    assert img.channel_names == []


def test_defaults_not_shared(arr):
    a = Image(data=da.from_array(arr), dims=DIMS)
    b = Image(data=da.from_array(arr), dims=DIMS)
    a.channel_names.append("DRAQ7")
    assert b.channel_names == []


def test_wrong_dims_rejected(arr):
    with pytest.raises(ValueError):
        Image(data=da.from_array(arr), dims=("Y", "X"))
