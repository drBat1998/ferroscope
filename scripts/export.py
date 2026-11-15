"""Try export settings on a few scenes; each preset gets its own folder.

Channel names come from the file. Contrast limits are computed once from ALL
scenes (shared_contrast) and cached in {stem}_limits.json next to the CZI.
Edit the block below, then:  python scripts/export.py
"""

import json
import time
from pathlib import Path
import dask
import numpy as np
from napari.utils.colormaps import ensure_colormap

from ferroscope.colors import colormap_list
from ferroscope.contrast import shared_contrast
from ferroscope.export import export_plate
from ferroscope.io import describe_image, load_scene

# --- edit here -------------------------------------------------------------
DATA = Path("~/data/Pl1_IHC_LC3_v1.czi").expanduser()  # your CZI
SCENES = None  # scenes to export; the limits still use ALL scenes
LOW_P, HIGH_P = 0.1, 99.9  # percentiles for the shared contrast
COLOR_OVERRIDES = {}  # {channel name: colormap}; others: magenta, cyan, yellow…
RUN = ["8bit_png"] #["raw_ome", "raw_imagej", "8bit_png"]  # presets to try (see make_presets)
CHANNELS = ["AF647","AF488"]  # e.g. ["AF488"]; None = all; names as printed below
# ---------------------------------------------------------------------------

OUT = DATA.parent / f"{DATA.stem}_trials"
LIMITS_FILE = DATA.parent / f"{DATA.stem}_limits.json"


def make_presets(limits: dict, luts: dict) -> dict:
    """Named export settings. New idea → new name, so folders stay truthful."""
    grey = {name: lut_from_colormap("gray") for name in luts}
    return {
        "raw_ome": dict(depth="raw"),
        "raw_imagej": dict(depth="raw", fmt="tiff"),
        "8bit_png": dict(depth="8bit", limits=limits, luts=luts, merge=True),
        "8bit_tiff": dict(depth="8bit", fmt="tiff", limits=limits, luts=luts),
        "8bit_grey": dict(depth="8bit", limits=limits, luts=grey),
        "8bit_gamma07": dict(
            depth="8bit",
            limits=limits,
            luts=luts,
            gamma=dict.fromkeys(limits, 0.7),
            merge=True,
        ),
    }


def lut_from_colormap(name: str) -> np.ndarray:
    """napari colormap name → (256, 3) uint8 LUT, the same colours as in napari."""
    rgba = ensure_colormap(name).map(np.linspace(0, 1, 256))
    return np.round(rgba[:, :3] * 255).astype(np.uint8)


def load_all_scenes(path: Path) -> tuple[list, list[str]]:
    """Every scene (lazy) and the channel names, checked to be the same in all."""
    images = [load_scene(path, i) for i in range(describe_image(path)["n_scenes"])]
    names = list(images[0].channel_names)
    for i, img in enumerate(images):
        if list(img.channel_names) != names:
            raise ValueError(
                f"scene {i} has channels {list(img.channel_names)}, scene 0 {names}"
            )
    return images, names


def plate_limits(images: list, names: list[str]) -> dict:
    """One (lo, hi) per channel from all scenes; reuse the cache when it matches."""
    wanted = {"source": DATA.name, "percentiles": [LOW_P, HIGH_P], "channels": names}
    if LIMITS_FILE.exists():
        saved = json.loads(LIMITS_FILE.read_text())
        if {k: saved.get(k) for k in wanted} == wanted:
            return {n: tuple(v) for n, v in saved["limits"].items()}
        print(f"{LIMITS_FILE.name} is for other settings; computing again")

    print(f"computing shared contrast over {len(images)} scenes…")
    limits = dict(zip(names, shared_contrast(images, LOW_P, HIGH_P)))
    LIMITS_FILE.write_text(json.dumps({**wanted, "limits": limits}, indent=2))
    return limits


def main() -> None:
    dask.config.set(num_workers=8)  # shared_contrast: 2 threads, not all cores
    images, names = load_all_scenes(DATA)
    if CHANNELS is not None:
        unknown = [c for c in CHANNELS if c not in names]
        if unknown:
            raise SystemExit(f"unknown channels {unknown}; file has {names}")
    limits = plate_limits(images, names)
    colors = colormap_list(names, COLOR_OVERRIDES)
    luts = {name: lut_from_colormap(c) for name, c in zip(names, colors)}

    for name, color in zip(names, colors):
        print(f"  {name:12s} {color:8s} limits {limits[name]}")

    presets = make_presets(limits, luts)
    unknown = [p for p in RUN if p not in presets]
    if unknown:
        raise SystemExit(f"unknown presets {unknown}; have {list(presets)}")

    for preset in RUN:
        out = OUT / preset
        if out.exists() and any(out.iterdir()):
            print(f"{preset:14s} skipped: {out} already has files")
            continue
        start = time.perf_counter()
        paths = export_plate(
            DATA, out, scenes=SCENES, channels=CHANNELS, **presets[preset]
        )
        seconds = time.perf_counter() - start
        mb = sum(p.stat().st_size for p in paths) / 1e6
        print(f"{preset:14s} {len(paths):4d} files  {mb:8.1f} MB  {seconds:6.1f} s")


if __name__ == "__main__":
    main()
