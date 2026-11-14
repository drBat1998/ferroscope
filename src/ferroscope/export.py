from itertools import product
from pathlib import Path

import numpy as np
import tifffile
from bioio_base.types import PhysicalPixelSizes
from bioio_ome_tiff.writers import OmeTiffWriter
from PIL import Image as PILImage

from ferroscope.io import Image, describe_image, load_scene
from ferroscope.render import apply_lut, composite, to_uint8

_ALLOWED = {("png", "8bit"), ("tiff", "8bit"), ("tiff", "raw"), ("ome-tiff", "raw")}
_BAD_CHARS = set('/\\:')
_DEFAULT_FMT = {"raw": "ome-tiff", "8bit": "png"}

WORKERS = 2      # threads per file for compression; 1 = quietest
ZLIB_LEVEL = 1   # 1 fastest … 9 smallest; 1 ≈ same size as 6 on camera data
PNG_LEVEL = 1    # PIL compress_level; 6 = PIL default, smaller but ~4× slower


def _tiff_opts() -> dict:
    """Compression settings shared by every TIFF writer (read at call time)."""
    return {
        "compression": "zlib",
        "compressionargs": {"level": ZLIB_LEVEL},
        "predictor": True,
        "maxworkers": WORKERS,
    }

def check_format(fmt: str, depth: str) -> None:
    """Raise ValueError unless (fmt, depth) is a pair the exporters accept."""
    depths = sorted({d for _, d in _ALLOWED})
    if depth not in depths:
        raise ValueError(f"depth {depth!r} unknown; choose one of {depths}")
    if (fmt, depth) not in _ALLOWED:
        ok = sorted(f for f, d in _ALLOWED if d == depth)
        raise ValueError(f"{depth} export can't be {fmt!r}; choose one of {ok}")

def _check_t_labels(t_labels, n_t: int) -> list[int]:
    """None → 0..n_t-1; else n_t unique, whole, non-negative numbers → ints."""
    if t_labels is None:
        return list(range(n_t))
    if len(t_labels) != n_t:
        raise ValueError(
            f"t_labels has {len(t_labels)} values, image has {n_t} time points"
        )
    labels = []
    for t in t_labels:
        if int(t) != t or t < 0:  # 3.0 ok, 3.5 / "3" / -1 rejected
            raise ValueError(f"time point {t!r} is not a whole number >= 0")
        labels.append(int(t))
    if len(set(labels)) != len(labels):
        dup = next(t for t in labels if labels.count(t) > 1)
        raise ValueError(f"time point {dup} appears twice in t_labels")
    return labels

def _check_channel_names(names) -> None:
    """Names go into file names: no path characters, no empties, no duplicates."""
    for name in names:
        if not name or _BAD_CHARS & set(name):
            raise ValueError(f"channel name {name!r} can't be used in a file name")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate channel names: {list(names)}")


def _plane_name(
    stem: str, t_label: int, channel: str, z: int, depth: str, ext: str
) -> str:
    """File name for one plane; ext has no leading dot.

    plate01_B2_T0003_DAPI_Z000_raw.ome.tif
    """ 
    return f"{stem}_T{t_label:04d}_{channel}_Z{z:03d}_{depth}.{ext}"

def _write_ome_tiff(
    plane: np.ndarray,
    path: Path,
    channel: str,
    pixel_size_um: tuple[float | None, float | None, float | None],
) -> None:
    """One YX plane → one grey OME-TIFF, raw pixels kept.

    pixel_size_um is (z, y, x) in µm; None means unknown.
    """
    OmeTiffWriter.save(
        plane,
        path,
        dim_order="YX",
        channel_names=[channel],
        physical_pixel_sizes=PhysicalPixelSizes(*pixel_size_um),
        channel_colors=[[255, 255, 255]],  # raw is always grey
        tifffile_kwargs=_tiff_opts(),
    )

def _write_tiff(
    plane: np.ndarray,
    path: Path,
    channel: str,
    pixel_size_um: tuple[float | None, float | None, float | None],
) -> None:
    """One YX plane → one grey ImageJ TIFF that opens in plain ImageJ.

    Deflate + horizontal predictor (ImageJ reads it; zstd it can't).
    pixel_size_um is (z, y, x) in µm; µm calibration is written only
    when both y and x are known. channel is unused: plain TIFF has no
    place for it, it is here only so all raw writers share one signature.
    """
    _, py, px = pixel_size_um

    extra = {}
    if py is not None and px is not None:
        extra = {
            "resolution": (1 / px, 1 / py),  # pixels per µm, (X, Y) order
            "metadata": {"unit": "um"},
        }

    tifffile.imwrite(path, plane, kind="imagej", **_tiff_opts(), **extra)

_RAW_WRITERS = {"ome-tiff": _write_ome_tiff, "tiff": _write_tiff}
_RAW_EXT = {"ome-tiff": "ome.tif", "tiff": "tif"}



def export_raw_planes(
    img: Image,
    out_dir: str | Path,
    *,
    fmt: str = "ome-tiff",
    t_labels: list[int] | None = None,
) -> list[Path]:
    """Every T, C, Z plane → one grey file, raw pixels and µm kept.

    fmt: "ome-tiff" (bioio, napari, Fiji) or "tiff" (opens in plain ImageJ).
    t_labels: the real time-point numbers for the file names (e.g. [0, 3] for a crop).
    """
    n_t, n_c, n_z = img.data.shape[:3]

    # --- check everything BEFORE writing anything ---
    check_format(fmt, "raw")
    t_labels = _check_t_labels(t_labels, n_t)
    _check_channel_names(img.channel_names)
    write = _RAW_WRITERS[fmt]
    ext = _RAW_EXT[fmt]

    # --- now write ---
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(img.source).stem}_{img.scene}"

    written = []
    for t, c, z in product(range(n_t), range(n_c), range(n_z)):
        channel = img.channel_names[c]
        path = out_dir / _plane_name(stem, t_labels[t], channel, z, "raw", ext)
        write(img.data[t, c, z], path, channel, img.pixel_size_um)
        written.append(path)
    return written

def _write_png(rgb: np.ndarray, path: Path, pixel_size_um) -> None:
    """RGB → PNG. pixel_size_um unused: PNG can't store µm."""
    PILImage.fromarray(rgb).save(path, compress_level=PNG_LEVEL)

def _write_rgb_tiff(rgb: np.ndarray, path: Path, pixel_size_um) -> None:
    """RGB → ImageJ TIFF with µm (for journals; opens in plain ImageJ)."""
    _, py, px = pixel_size_um
    extra = {}
    if py is not None and px is not None:
        extra = {"resolution": (1 / px, 1 / py), "metadata": {"unit": "um"}}
    tifffile.imwrite(
        path, rgb, kind="imagej", photometric="rgb", **_tiff_opts(), **extra
    )

_DISPLAY_WRITERS = {"png": _write_png, "tiff": _write_rgb_tiff}
_DISPLAY_EXT = {"png": "png", "tiff": "tif"}

def export_display(
    img: Image,
    out_dir: str | Path,
    limits: dict,
    luts: dict,
    *,
    fmt: str = "png",
    gamma: dict | None = None,
    z: int = 0,
    t_labels: list[int] | None = None,
    singles: bool = True,
    merge: bool = False,
) -> list[Path]:
    """8-bit RGB PNG or TIFF of plane z: one per T x C (singles) and/or one
    mix of all C per T (merge).

    limits: {channel: (lo, hi)}, luts: {channel: (256, 3) uint8},
    gamma: {channel: float}, missing channels use 1.0.
    """
    n_t, n_c, n_z = img.data.shape[:3]

    # --- check everything BEFORE writing anything ---
    check_format(fmt, "8bit")
    if not singles and not merge:
        raise ValueError("nothing to write: choose singles, merge or both")
    if not 0 <= z < n_z:
        raise ValueError(f"z {z} out of range (0..{n_z - 1})")
    t_labels = _check_t_labels(t_labels, n_t)
    _check_channel_names(img.channel_names)
    if merge and "merge" in img.channel_names:
        raise ValueError("a channel is named 'merge'; its file would clash with the merge")
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

    write = _DISPLAY_WRITERS[fmt]
    ext = _DISPLAY_EXT[fmt]

    # --- now write ---
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(img.source).stem}_{img.scene}"

    written = []
    for t in range(n_t):
        rgbs = []
        for c in range(n_c):
            name = img.channel_names[c]
            lo, hi = limits[name]
            plane = np.asarray(img.data[t, c, z])  # reads just this one plane
            rgb = apply_lut(to_uint8(plane, lo, hi, gamma.get(name, 1.0)), luts[name])
            rgbs.append(rgb)
            if singles:
                path = out_dir / _plane_name(stem, t_labels[t], name, z, "8bit", ext)
                write(rgb, path, img.pixel_size_um)
                written.append(path)
        if merge:
            path = out_dir / _plane_name(stem, t_labels[t], "merge", z, "8bit", ext)
            write(composite(rgbs), path, img.pixel_size_um)
            written.append(path)
    return written


def export_plate(
    path: str | Path,
    out_dir: str | Path | None = None,
    *,
    scenes: list[int] | None = None,
    channels: list[str] | None = None,
    depth: str = "raw",
    fmt: str | None = None,
    limits: dict | None = None,
    luts: dict | None = None,
    gamma: dict | None = None,
    z: int = 0,
    singles: bool = True,
    merge: bool = False,
) -> list[Path]:
    """Every scene of a file → one folder, raw planes or 8-bit images.

    scenes: which scene indices; None = all.
    channels: which channel names; None = all.
    depth "raw":  fmt "ome-tiff" (default) or "tiff".
    depth "8bit": fmt "png" (default) or "tiff"; needs limits and luts,
                  the SAME for every scene, so wells stay comparable.
    Default folder: {stem}_{depth} next to the file.
    """
    # --- checks that don't need the file ---
    if fmt is None:
        fmt = _DEFAULT_FMT.get(depth)  # unknown depth → None → check_format says so
    check_format(fmt, depth)
    if depth == "8bit" and (limits is None or luts is None):
        raise ValueError("8bit export needs limits and luts")

    path = Path(path).expanduser()
    if out_dir is None:
        out_dir = path.parent / f"{path.stem}_{depth}"

    # --- scene checks ---
    n_scenes = describe_image(path)["n_scenes"]
    if scenes is None:
        scenes = list(range(n_scenes))
    if not scenes:
        raise ValueError("choose at least one scene")
    bad = [i for i in scenes if not 0 <= i < n_scenes]
    if bad:
        raise ValueError(f"scenes {bad} out of range (0..{n_scenes - 1})")
    if len(set(scenes)) != len(scenes):
        dup = next(i for i in scenes if scenes.count(i) > 1)
        raise ValueError(f"scene {dup} appears twice in scenes")

    # --- now write ---
    written = []
    for i in scenes:
        img = load_scene(path, i)
        if channels is not None:
            img = img.select_channels(channels)

        if depth == "raw":
            written += export_raw_planes(img, out_dir, fmt=fmt)
        else:
            written += export_display(
                img, out_dir, limits, luts, fmt=fmt,
                gamma=gamma, z=z, singles=singles, merge=merge,
            )
    return written
