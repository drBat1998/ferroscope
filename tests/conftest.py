import numpy as np
import pytest


@pytest.fixture
def arr():
    return np.arange(2 * 3 * 1 * 8 * 8, dtype=np.uint16).reshape(2, 3, 1, 8, 8)


import tifffile

from ferroscope.io import describe_image


@pytest.fixture
def ome_tiff(tmp_path):
    path = tmp_path / "cells.ome.tif"
    arr = np.zeros((2, 3, 4, 16, 16), dtype=np.uint16)  # T C Z Y X
    tifffile.imwrite(
        path,
        arr,
        ome=True,
        metadata={
            "axes": "TCZYX",
            "PhysicalSizeX": 0.65,
            "PhysicalSizeY": 0.65,
            "PhysicalSizeZ": 2.0,
            "Channel": {"Name": ["AO", "DRAQ7", "BF"]},
        },
    )
    return path


@pytest.fixture
def info(ome_tiff):
    return describe_image(ome_tiff)
