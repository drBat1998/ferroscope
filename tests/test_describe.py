import json

import pytest

from ferroscope.io import describe_image


def test_file_level(info, ome_tiff):
    assert info["format"] == "ome-tiff"
    assert info["n_scenes"] == 1
    assert info["path"] == str(ome_tiff)


def test_dims_and_sizes(info):
    scene = info["scenes"][0]
    assert scene["dims"] == "TCZYX"
    assert scene["T"] == 2
    assert scene["Z"] == 4
    assert tuple(scene["shape"]) == (2, 3, 4, 16, 16)
    assert scene["data_type"] == "uint16"


def test_channels_are_plain_strings(info):
    channels = info["scenes"][0]["channels"]
    assert channels == ["AO", "DRAQ7", "BF"]
    assert all(type(c) is str for c in channels)


def test_pixel_size(info):
    assert info["scenes"][0]["pixel_size"] == pytest.approx((2.0, 0.65, 0.65))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe_image(tmp_path / "nope.czi")


def test_unsupported_format_checked_before_bioio(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    with pytest.raises(ValueError, match="unsupported format"):
        describe_image(bad)


def test_summary_is_json_ready(info):
    json.dumps(info)
