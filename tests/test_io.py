import dask.array as da
import re
from ferroscope.io import Image
import pytest

def make_image(**overrides) -> Image:
    """A valid Image; each test overrides only the field it checks."""
    fields = {
        "data": da.zeros((2, 3, 4, 16, 24), dtype="uint16"),  # T C Z Y X
        "dims": ("T", "C", "Z", "Y", "X"),
        "channel_names": ("AO", "DRAQ7", "BF"),
        "pixel_size_um": (2.0, 0.6, 0.5),  # z, y, x
        "time_interval_s": 300.0,
        "source": None,
        "scene": "P1-A1",
    }
    fields.update(overrides)
    return Image(**fields)


def test_accepts_tczyx():
    img = make_image()
    assert img.dims == ("T", "C", "Z", "Y", "X")

def test_rejects_wrong_order():
    bad = ("C", "T", "Z", "Y", "X")
    with pytest.raises(ValueError, match=re.escape(repr(bad))):
        make_image(dims=bad)


def test_rejects_fewer_dims():
    bad = ("Y", "X")
    with pytest.raises(ValueError, match=re.escape(repr(bad))):
        make_image(data=da.zeros((16, 24), dtype="uint16"), dims=bad)


def test_rejects_dims_data_mismatch():
    data = da.zeros((3, 4, 16, 24), dtype="uint16")  # 4-D
    with pytest.raises(ValueError, match="4"):
        make_image(data=data)


#data	
def test_rejects_numpy_array():
    bad = np.zeros((2, 3, 4, 16, 24), dtype="uint16")
    with pytest.raises(TypeError, match="ndarray"):
        make_image(data=bad)

#channels	
@pytest.mark.parametrize(
    "names",
    [("AO", "DRAQ7"), ("AO", "DRAQ7", "BF", "PI")],
    ids=["too_few", "too_many"],
)
def test_rejects_wrong_channel_count(names):
    with pytest.raises(ValueError, match="3 channels"):
        make_image(channel_names=names)


def test_rejects_list_channel_names():


#pixel size	test_accepts_unknown_pixel_sizes	all None is fine
def test_rejects_non_positive_pixel_size	parametrize: each axis × (0, −1)
def test_rejects_pixel_size_wrong_length	2 or 4 values

#time	test_accepts_unknown_time_interval	None is fine
test_rejects_non_positive_time_interval	parametrize: 0, −1
#immutable	test_cannot_reassign_field	which exception? check the dataclasses docs
axis_scale	test_axis_scale_known_values	your earlier prediction: T in s, Z/Y/X in µm, C = 1.0
test_axis_scale_unknown_is_one	None → 1.0
#lazy	test_building_image_does_not_compute	huge da.zeros + Callback counter = 0
