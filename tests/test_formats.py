from pathlib import Path

import pytest

from ferroscope.io import detect_format


@pytest.mark.parametrize(
    "name, expected",
    [
        # CZI
        ("plate.czi", "czi"),
        ("PLATE.CZI", "czi"),  # upper case
        ("2026.10.28_plate.czi", "czi"),  # dots in a date
        ("data.v2/plate.czi", "czi"),  # dot in a folder name
        ("~/data/OGD_24h.czi", "czi"),  # home-dir path, not expanded
        # TIFF
        ("cells.tif", "tiff"),
        ("cells.tiff", "tiff"),
        ("CELLS.TIF", "tiff"),
        # OME-TIFF: must win over plain .tif
        ("cells.ome.tif", "ome-tiff"),
        ("cells.ome.tiff", "ome-tiff"),
        ("Cells.OME.TIF", "ome-tiff"),
        # Zarr: a folder, often with a trailing slash
        ("plate.zarr", "zarr"),
        ("plate.ome.zarr", "zarr"),
        ("plate.zarr/", "zarr"),
        ("data/plate.zarr/", "zarr"),
    ],
)
def test_detect_format(name, expected):
    assert detect_format(name) == expected


def test_accepts_path_objects():
    assert detect_format(Path("plate.czi")) == "czi"


@pytest.mark.parametrize(
    "name",
    [
        "notes.txt",
        "scan.nd2",
        "README",
        "plate.czi.bak",
        "cells.tif.zip",
        "czi",
        "plate_czi",
        "data.zarr/plate.png",
    ],
)
def test_unknown_format_raises(name):
    with pytest.raises(ValueError, match="unsupported format"):
        detect_format(name)
