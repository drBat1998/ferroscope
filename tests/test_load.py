import dask.array as da
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback

from ferroscope.io import load_scene


class CountTasks(Callback):
    """Counts how many dask tasks actually run (= how much real work happened)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def _posttask(self, key, result, dsk, state, id):
        self.n += 1


def test_load_basic(ome_tiff):
    img = load_scene(ome_tiff)
    assert isinstance(img.data, da.Array)
    assert img.dims == ("T", "C", "Z", "Y", "X")
    assert img.data.shape == (2, 3, 4, 16, 16)
    assert img.channel_names == ["AO", "DRAQ7", "BF"]
    assert img.pixel_size == pytest.approx((2.0, 0.65, 0.65))
    assert img.scene == "Image:0"


def test_load_reads_correct_pixels(tmp_path):
    path = tmp_path / "values.ome.tif"
    arr = np.arange(2 * 3 * 1 * 8 * 8, dtype=np.uint16).reshape(2, 3, 1, 8, 8)
    tifffile.imwrite(path, arr, ome=True, metadata={"axes": "TCZYX"})
    img = load_scene(path)
    np.testing.assert_array_equal(img.data[1, 2, 0].compute(), arr[1, 2, 0])


def test_load_is_lazy(ome_tiff):
    with CountTasks() as counter:
        img = load_scene(ome_tiff)
        plane = img.data[1, 2, 3]          # selecting must not read anything
    assert counter.n == 0

    with CountTasks() as counter:
        plane.compute()                    # only now pixels are read
    assert counter.n > 0


def test_bad_scene_raises(ome_tiff):
    with pytest.raises(IndexError):
        load_scene(ome_tiff, scene=5)
