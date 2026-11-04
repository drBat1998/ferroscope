from itertools import product
from pathlib import Path

import numpy as np
from bioio_base.types import PhysicalPixelSizes
from bioio_ome_tiff.writers import OmeTiffWriter
from PIL import Image as PILImage

from ferroscope.io import Image, describe_image, load_scene
from ferroscope.render import apply_lut, to_uint8


def _check_color(name: str, color) -> list[int]:
    """(r, g, b) with 0..255 → plain list of ints; else ValueError naming the channel."""
    rgb = [int(v) for v in color]
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(
            f"color for {name!r} must be 3 values 0..255, got {list(color)}"
        )
    return rgb


def _check_t_labels(t_labels, n_t: int) -> list[int]:
    if t_labels is None:
        return list(range(n_t))
    if len(t_labels) != n_t:
        raise ValueError(
            f"t_labels has {len(t_labels)} values, image has {n_t} time points"
        )
    if len(set(t_labels)) != len(t_labels):
        dup = next(t for t in t_labels if t_labels.count(t) > 1)
        raise ValueError(f"time point {dup} appears twice in t_labels")
    return list(t_labels)


def export_tiff_planes(
    img: Image,
    out_dir: str | Path,
    t_labels: list[int] | None = None,
    channel_colors: dict | None = None,
) -> list[Path]:
    """Every T, C, Z plane → one OME-TIFF, raw pixels, µm and channel colour kept.

    t_labels: the real time-point numbers for the file names (e.g. [0, 3] for a crop).
    channel_colors: {channel name: (r, g, b)}; channels without one are stored white.
    """
    n_t, n_c, n_z = img.data.shape[:3]

    # --- check everything BEFORE writing anything ---
    t_labels = _check_t_labels(t_labels, n_t)

    channel_colors = channel_colors or {}
    colors = {
        name: _check_color(name, color)
        for name, color in channel_colors.items()
        if name in img.channel_names
    }

    # --- now write ---
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(img.source).stem}_{img.scene}"

    written = []
    for t, c, z in product(range(n_t), range(n_c), range(n_z)):
        channel = img.channel_names[c]
        path = out_dir / f"{stem}_T{t_labels[t]:03d}_{channel}_Z{z:03d}.ome.tif"
        extra = {"channel_colors": [colors[channel]]} if channel in colors else {}
        OmeTiffWriter.save(
            img.data[t, c, z],
            path,
            dim_order="YX",
            channel_names=[channel],
            physical_pixel_sizes=PhysicalPixelSizes(*img.pixel_size),
            **extra,
        )
        written.append(path)
    return written


def export_plate(path, out_dir=None) -> list[Path]:
    """Every scene of a file → all planes, all into ONE folder."""
    path = Path(path).expanduser()
    if out_dir is None:
        out_dir = path.parent / f"{path.stem}_tiff"
    written = []
    n_scenes = describe_image(path)["n_scenes"]
    for i in range(n_scenes):
        written += export_tiff_planes(load_scene(path, i), out_dir)
    return written


def export_display(
    img: Image,
    out_dir: str | Path,
    limits: dict,
    luts: dict,
    gamma: dict | None = None,
    z: int = 0,
    t_labels: list[int] | None = None,
) -> list[Path]:
    """One RGB PNG per T x C of plane z, coloured with each channel's LUT.

    limits: {channel: (lo, hi)}, luts: {channel: (256, 3) uint8},
    gamma: {channel: float}, missing channels use 1.0.
    """
    n_t, n_c, n_z = img.data.shape[:3]

    # --- check everything BEFORE writing anything ---
    if not 0 <= z < n_z:
        raise ValueError(f"z {z} out of range (0..{n_z - 1})")
    t_labels = _check_t_labels(t_labels, n_t)
    gamma = gamma or {}

    for name in img.channel_names:
        if name not in limits:
            raise ValueError(f"no contrast limits for channel {name!r}")
        if name not in luts:
            raise ValueError(f"no LUT for channel {name!r}")
        # dry run on one pixel: reuses the checks in to_uint8 and apply_lut
        lo, hi = limits[name]
        g = gamma.get(name, 1.0)
        apply_lut(to_uint8(np.zeros(1, dtype=np.uint16), lo, hi, g), luts[name])

    # --- now write ---
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(img.source).stem}_{img.scene}"

    written = []
    for t, c in product(range(n_t), range(n_c)):
        name = img.channel_names[c]
        lo, hi = limits[name]
        plane = np.asarray(img.data[t, c, z])  # reads just this one plane
        rgb = apply_lut(to_uint8(plane, lo, hi, gamma.get(name, 1.0)), luts[name])
        path = out_dir / f"{stem}_T{t_labels[t]:03d}_{name}_Z{z:03d}.png"
        PILImage.fromarray(rgb).save(path)
        written.append(path)
    return written
