from itertools import product
from pathlib import Path

from bioio_base.types import PhysicalPixelSizes
from bioio_ome_tiff.writers import OmeTiffWriter

from ferroscope.io import Image, describe_image, load_scene


def _check_color(name: str, color) -> list[int]:
    """(r, g, b) with 0..255 → plain list of ints; else ValueError naming the channel."""
    rgb = [int(v) for v in color]
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(
            f"color for {name!r} must be 3 values 0..255, got {list(color)}"
        )
    return rgb


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
    if t_labels is None:
        t_labels = list(range(n_t))
    if len(t_labels) != n_t:
        raise ValueError(
            f"t_labels has {len(t_labels)} values, image has {n_t} time points"
        )
    if len(set(t_labels)) != len(t_labels):
        dup = next(t for t in t_labels if t_labels.count(t) > 1)
        raise ValueError(f"time point {dup} appears twice in t_labels")

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
