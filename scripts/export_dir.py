"""Export every CZI in a folder with the same presets; each file × preset gets
its own folder, and one bad file doesn't stop the others.

Channel names come from the files. Contrast limits come from shared_contrast:
  LIMITS_SCOPE = "file":   one set per file, cached as {stem}_limits.json
  LIMITS_SCOPE = "folder": one set for ALL files (they must have the same
                           channels), cached as _folder_limits.json
Edit the block below, then:  nice -n 19 uv run scripts/export_folder.py
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
DATA_DIR = Path("~/data/NaPr_110626/080726_data").expanduser()  # folder of CZIs
PATTERN = "*.czi"  # which files; "pL1_*.czi" for a subset
SCENES = None  # None = all scenes of every file
CHANNELS = ["AF555", "AF488"]  # None = all; names as printed below
LIMITS_SCOPE = "file"  # "file" or "folder" (see docstring)
LOW_P, HIGH_P = 0.1, 99.9  # percentiles for the shared contrast
COLOR_OVERRIDES = {"Oblique": "gray"}  # {channel name: colormap}
RUN = ["raw_imagej", "8bit_png"]  # presets to run (see make_presets)
# ---------------------------------------------------------------------------

OUT_ROOT = DATA_DIR / "exports"  # → exports/{file stem}/{preset}/


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
                f"{path.name}: scene {i} has channels {list(img.channel_names)}, "
                f"scene 0 {names}"
            )
    return images, names


def cached_limits(cache: Path, sources: list[str], images, names) -> dict:
    """shared_contrast over images; reuse the cache file when it matches."""
    wanted = {"sources": sources, "percentiles": [LOW_P, HIGH_P], "channels": names}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if {k: saved.get(k) for k in wanted} == wanted:
            return {n: tuple(v) for n, v in saved["limits"].items()}
        print(f"  {cache.name} is for other settings; computing again")

    print(f"  computing shared contrast over {len(images)} scenes…")
    limits = dict(zip(names, shared_contrast(images, LOW_P, HIGH_P)))
    cache.write_text(json.dumps({**wanted, "limits": limits}, indent=2))
    return limits


def export_file(path: Path, names: list[str], limits: dict) -> None:
    """All presets in RUN for one file; prints one line per preset."""
    colors = colormap_list(names, COLOR_OVERRIDES)
    luts = {name: lut_from_colormap(c) for name, c in zip(names, colors)}
    for name, color in zip(names, colors):
        print(f"  {name:12s} {color:8s} limits {limits[name]}")

    presets = make_presets(limits, luts)
    for preset in RUN:
        out = OUT_ROOT / path.stem / preset
        if out.exists() and any(out.iterdir()):
            print(f"  {preset:14s} skipped: {out} already has files")
            continue
        start = time.perf_counter()
        paths = export_plate(
            path, out, scenes=SCENES, channels=CHANNELS, **presets[preset]
        )
        seconds = time.perf_counter() - start
        mb = sum(p.stat().st_size for p in paths) / 1e6
        print(f"  {preset:14s} {len(paths):4d} files  {mb:8.1f} MB  {seconds:6.1f} s")


def main() -> None:
    dask.config.set(num_workers=2)  # shared_contrast: 2 threads, not all cores

    # --- checks before any work ---
    if LIMITS_SCOPE not in ("file", "folder"):
        raise SystemExit(f"LIMITS_SCOPE must be 'file' or 'folder': {LIMITS_SCOPE!r}")
    files = sorted(DATA_DIR.glob(PATTERN))
    if not files:
        raise SystemExit(f"no files match {PATTERN!r} in {DATA_DIR}")
    unknown = [p for p in RUN if p not in make_presets({}, {})]
    if unknown:
        raise SystemExit(f"unknown presets {unknown}")
    print(f"{len(files)} files in {DATA_DIR}")

    # --- open every file (lazy) and check channels ---
    opened, failed = {}, {}
    for path in files:
        try:
            images, names = load_all_scenes(path)
            missing = [c for c in CHANNELS or [] if c not in names]
            if missing:
                raise ValueError(f"no channels {missing}; file has {names}")
            opened[path] = (images, names)
        except Exception as e:  # one bad file must not stop the batch
            failed[path] = e

    # --- limits ---
    folder_limits = None
    if LIMITS_SCOPE == "folder" and opened:
        all_names = {tuple(names) for _, names in opened.values()}
        if len(all_names) > 1:
            raise SystemExit(f"folder limits need the same channels; found {all_names}")
        names = list(all_names.pop())
        images = [img for imgs, _ in opened.values() for img in imgs]
        print("folder-wide contrast:")
        folder_limits = cached_limits(
            DATA_DIR / "_folder_limits.json",
            [p.name for p in opened],
            images,
            names,
        )

    # --- export file by file ---
    for path, (images, names) in opened.items():
        print(f"\n{path.name}")
        try:
            limits = folder_limits or cached_limits(
                path.parent / f"{path.stem}_limits.json", [path.name], images, names
            )
            export_file(path, names, limits)
        except Exception as e:
            failed[path] = e
            print(f"  FAILED: {type(e).__name__}: {e}")

    # --- summary ---
    ok = [p for p in opened if p not in failed]
    print(f"\ndone: {len(ok)} ok, {len(failed)} failed")
    for path, e in failed.items():
        print(f"  {path.name}: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
