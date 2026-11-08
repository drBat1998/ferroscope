"""Manual check of ferroscope.io on a real file.

Usage:
    uv run python scripts/try_io.py path/to/live_cell.czi
"""

import sys
import time
from pprint import pprint

from ferroscope.io import _open, describe_image, load_scene


def main(path: str) -> None:
    t0 = time.perf_counter()
    info = describe_image(path)
    print(f"describe_image: {time.perf_counter() - t0:.2f} s")
    print(f"format: {info['format']}, scenes: {info['n_scenes']}")
    pprint(info["scenes"][0])

    t0 = time.perf_counter()
    img = load_scene(path, scene=0)
    print(f"\nload_scene: {time.perf_counter() - t0:.3f} s")
    print(f"scene {img.scene}, shape {img.data.shape}, dtype {img.data.dtype}")
    print(f"channels: {img.channel_names}")
    print(f"pixel size (z, y, x) µm: {img.pixel_size_um}")
    print(f"time interval s: {img.time_interval_s}")
    print(f"axis_scale: {img.axis_scale}")

    # Raw bioio values, to compare with ZEN and with what io converted.
    _, _, bio = _open(path)
    print(f"\nbioio time_interval: {bio.time_interval!r}")
    

    # One plane only: the first real read from disk.
    t0 = time.perf_counter()
    plane = img.data[0, 0, 0].compute()
    print(f"\none plane {plane.shape} read in {time.perf_counter() - t0:.3f} s, "
          f"min {plane.min()}, max {plane.max()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
