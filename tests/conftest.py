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
    arr = make_array((2, 3, 4, 16, 16))  # T C Z Y X
    tifffile.imwrite(
        path,
        arr,
        ome=True,
        metadata={
            "axes": "TCZYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.5,
            "PhysicalSizeZ": 2.0,
            "Channel": {"Name": ["AO", "DRAQ7", "BF"]},
        },
    )
    return path


@pytest.fixture
def info(ome_tiff):
    return describe_image(ome_tiff)


def make_array(shape=(2, 3, 4, 16, 24)):  
    t_idx, c_idx, z_idx, y_idx, x_idx = shape
    T = np.arange(t_idx).reshape(t_idx,1,1,1,1) 
    C = np.arange(c_idx).reshape(1,c_idx,1,1,1) 
    Z = np.arange(z_idx).reshape(1,1,z_idx,1,1) 
    code = 1000*T + 100*C + 10*Z
    made_array = np.broadcast_to(code, shape).astype(np.uint16)
    return made_array

if __name__ == "__main__":
    a = make_array()[1,2,3]
    print(a)
    print(type(a))
    print(a.flags.writeable)
