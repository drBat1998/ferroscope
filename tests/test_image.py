import math

import dask.array as da
import pytest

from ferroscope.io import Image


def make(shape=(2, 3, 4, 5, 6), dims=tuple("TCZYX"), channel_names=("a", "b", "c"), **kw):
    return Image(data=da.zeros(shape), dims=dims, channel_names=list(channel_names), **kw)


def test_valid_tczyx():
    assert make().channel_axis == 1


def test_rgb_six_dims_accepted():
    img = make(shape=(1, 1, 1, 4, 5, 3), dims=tuple("TCZYXS"), channel_names=["rgb"])
    assert img.dims[-1] == "S"


def test_dims_length_mismatch():
    with pytest.raises(ValueError, match="has 4 axes"):
        make(dims=tuple("CZYX"))


def test_duplicate_dims():
    with pytest.raises(ValueError, match="duplicate"):
        make(dims=tuple("TCCYX"))


def test_unknown_dim():
    with pytest.raises(ValueError, match="unknown dims"):
        make(dims=tuple("TCQYX"))


def test_channel_count_mismatch():
    with pytest.raises(ValueError, match="2 channel names"):
        make(channel_names=["a", "b"])


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_bad_pixel_size(bad):
    with pytest.raises(ValueError, match="pixel size Y"):
        make(pixel_size_um=(None, bad, 0.3))


@pytest.mark.parametrize("bad", [0.0, -60.0, math.nan])
def test_bad_time_interval(bad):
    with pytest.raises(ValueError, match="time_interval_s"):
        make(time_interval_s=bad)


def test_axis_scale_known_and_unknown():
    img = make(pixel_size_um=(None, 0.3, 0.3), time_interval_s=600.0)
    assert img.axis_scale == {"T": 600.0, "C": 1.0, "Z": 1.0, "Y": 0.3, "X": 0.3}


def test_axis_scale_all_unknown():
    assert set(make().axis_scale.values()) == {1.0}
