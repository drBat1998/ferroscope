import dask.array as da
from dask.callbacks import Callback
import re
from ferroscope.io import Image, _open , _um, _seconds, load_scene, describe_image
import pytest
import numpy as np

import dataclasses
import bioio_ome_tiff 
import tifffile 
from bioio import BioImage 

from datetime import timedelta

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
    with pytest.raises(TypeError, match="list"):
        make_image(channel_names=["AO", "DRAQ7", "BF"])


def test_accepts_unknown_pixel_sizes():
    img = make_image(pixel_size_um=(None, None, None))
    assert img.pixel_size_um == (None, None, None)


@pytest.mark.parametrize("value", [0, -1.0], ids=["zero", "negative"])
@pytest.mark.parametrize("i, axis", [(0, "Z"), (1, "Y"), (2, "X")])
def test_rejects_non_positive_pixel_size(i, axis, value):
    sizes = [2.0, 0.6, 0.5]
    sizes[i] = value
    with pytest.raises(ValueError, match=f"pixel size {axis}"):
        make_image(pixel_size_um=tuple(sizes))


@pytest.mark.parametrize(
    "sizes", [(0.6, 0.5), (2.0, 0.6, 0.5, 1.0)], ids=["two", "four"]
)
def test_rejects_pixel_size_wrong_length(sizes):
    with pytest.raises(ValueError, match="pixel_size_um"):
        make_image(pixel_size_um=sizes)


# --- time ---------------------------------------------------------------------

def test_accepts_unknown_time_interval():
    img = make_image(time_interval_s=None)
    assert img.time_interval_s is None


@pytest.mark.parametrize("value", [0, -1.0], ids=["zero", "negative"])
def test_rejects_non_positive_time_interval(value):
    with pytest.raises(ValueError, match="time_interval_s"):
        make_image(time_interval_s=value)


# --- immutable ----------------------------------------------------------------

def test_cannot_reassign_field():
    img = make_image()
    with pytest.raises(dataclasses.FrozenInstanceError):
        img.scene = "P2-B2"


# --- axis_scale ---------------------------------------------------------------

def test_axis_scale_known_values():
    assert make_image().axis_scale == {
        "T": 300.0, "C": 1.0, "Z": 2.0, "Y": 0.6, "X": 0.5,
    }


def test_axis_scale_unknown_is_one():
    img = make_image(pixel_size_um=(None, None, None), time_interval_s=None)
    assert img.axis_scale == {"T": 1.0, "C": 1.0, "Z": 1.0, "Y": 1.0, "X": 1.0}


# --- lazy ---------------------------------------------------------------------

class ComputeCounter(Callback):
    """Counts dask computes started inside a `with` block."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def _start(self, dsk):
        self.calls += 1


def test_counter_detects_compute():
    with ComputeCounter() as counter:
        da.zeros(4).sum().compute()
    assert counter.calls == 1


def test_building_image_does_not_compute():
    huge = da.zeros((1000, 3, 100, 2048, 2048), dtype="uint16")  # ~2.5 TB, never allocated
    with ComputeCounter() as counter:
        make_image(data=huge)
    assert counter.calls == 0




@pytest.mark.parametrize("value, expected", [(0.5, 0.5), (2, 2.0)], ids=["float", "int"])
def test_um_positive_returns_float(value, expected):
    result = _um(value)
    assert result == expected
    assert isinstance(result, float)


@pytest.mark.parametrize("value", [None, 0, 0.0], ids=["none", "int_zero", "float_zero"])
def test_um_unknown_is_none(value):
    assert _um(value) is None


def test_um_negative_passes_through():
    assert _um(-0.5) == -0.5


@pytest.fixture(scope="session")
def ome_tiff(tmp_path_factory):
    """Small OME-TIFF, written once per test session. Y != X on purpose."""
    path = tmp_path_factory.mktemp("data") / "cells.ome.tif"
    tifffile.imwrite(
        path,
        np.zeros((2, 3, 4, 16, 24), dtype="uint16"),
        ome=True,
        metadata={
            "axes": "TCZYX",
            "PhysicalSizeZ": 2.0,
            "PhysicalSizeY": 0.6,
            "PhysicalSizeX": 0.5,
            "Channel": {"Name": ["AO", "DRAQ7", "BF"]},
        },
    )
    return path


def test_open_returns_path_format_bioimage(ome_tiff):
    path, fmt, img = _open(ome_tiff)
    assert path == ome_tiff
    assert fmt == "ome-tiff"
    assert isinstance(img, BioImage)


def test_open_uses_reader_for_format(ome_tiff):
    _, _, img = _open(ome_tiff)
    assert isinstance(img.reader, bioio_ome_tiff.Reader)


def test_open_missing_file(tmp_path):
    missing = tmp_path / "missing.ome.tif"
    with pytest.raises(FileNotFoundError, match="missing.ome.tif"):
        _open(missing)


@pytest.mark.parametrize("name", ["notes.png", "cells.tif", "cells.tiff"])
def test_open_unsupported_extension(tmp_path, name):
    with pytest.raises(ValueError, match="unsupported format"):
        _open(tmp_path / name)


def test_open_expands_tilde(ome_tiff, monkeypatch):
    monkeypatch.setenv("HOME", str(ome_tiff.parent))
    path, _, _ = _open(f"~/{ome_tiff.name}")
    assert path == ome_tiff


# --- _seconds -----------------------------------------------------------------

def test_seconds_converts_timedelta():
    result = _seconds(timedelta(minutes=5))
    assert result == 300.0
    assert isinstance(result, float)


@pytest.mark.parametrize("value", [None, timedelta(0)], ids=["none", "zero"])
def test_seconds_unknown_is_none(value):
    assert _seconds(value) is None


# --- load_scene ---------------------------------------------------------------

def test_load_scene_metadata_matches_file(ome_tiff):
    img = load_scene(ome_tiff)
    assert img.data.shape == (2, 3, 4, 16, 24)
    assert img.channel_names == ("AO", "DRAQ7", "BF")
    assert img.pixel_size_um == (2.0, 0.6, 0.5)
    assert img.source == ome_tiff


def test_load_scene_invalid_index(ome_tiff):
    with pytest.raises(ValueError, match="valid"):
        load_scene(ome_tiff, scene=99)


def test_load_scene_does_not_compute(ome_tiff):
    with ComputeCounter() as counter:
        load_scene(ome_tiff)
    assert counter.calls == 0


# --- describe_image -----------------------------------------------------------

def test_describe_matches_load(ome_tiff):
    info = describe_image(ome_tiff)
    for i, scene in enumerate(info["scenes"]):
        img = load_scene(ome_tiff, scene=i)
        assert scene["id"] == img.scene
        assert scene["shape"] == img.data.shape
        assert scene["channel_names"] == img.channel_names
        assert scene["pixel_size_um"] == img.pixel_size_um
        assert scene["time_interval_s"] == img.time_interval_s


def test_describe_does_not_compute(ome_tiff):
    with ComputeCounter() as counter:
        describe_image(ome_tiff)
    assert counter.calls == 0
