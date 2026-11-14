import json
from dataclasses import dataclass

from ferroscope.crop import crop_image


def _check_color(name: str, color) -> list[int]:
    """(r, g, b) with 0..255 → plain list of ints; else ValueError naming the channel."""
    rgb = [int(v) for v in color]
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(
            f"color for {name!r} must be 3 values 0..255, got {list(color)}"
        )
    return rgb


def crop_record(
    crop: Image,
    box: tuple[slice, slice],
    t_labels,
    z: int = 0,
    limits: dict | None = None,
    luts: dict | None = None,
    gamma: dict | None = None,
) -> dict:
    """Everything needed to find the crop again and rebuild the figure."""
    n_t, _, _, n_y, n_x = crop.data.shape

    # 1. times: plain ints, then the usual checks
    if t_labels is not None:
        t_labels = [int(t) for t in t_labels]
    t_labels = _check_t_labels(t_labels, n_t)

    # 2. box → origin and size as plain ints; must match the crop
    sy, sx = box
    y0, x0 = int(sy.start), int(sx.start)
    h, w = int(sy.stop) - y0, int(sx.stop) - x0
    if (h, w) != (n_y, n_x):
        raise ValueError(f"box is {h} x {w} px, crop is {n_y} x {n_x} px")

    # 3. the same box in µm, only when the pixel size is known
    pz, py, px = crop.pixel_size_um
    box_um = None
    if py is not None and px is not None:
        box_um = {
            "y": round(y0 * py, 6),
            "x": round(x0 * px, 6),
            "height": round(h * py, 6),
            "width": round(w * px, 6),
        }

    # 4. per channel, in file order
    gamma = gamma or {}
    channels = {}
    for name in crop.channel_names:
        if limits is not None and name not in limits:
            raise ValueError(f"no contrast limits for channel {name!r}")
        if luts is not None and name not in luts:
            raise ValueError(f"no LUT for channel {name!r}")
        channels[name] = {
            "limits": None if limits is None else [float(v) for v in limits[name]],
            "gamma": float(gamma.get(name, 1.0)),
            "rgb": None if luts is None else [int(v) for v in luts[name][-1]],
        }

    return {
        "source": str(crop.source),
        "scene": crop.scene,
        "pixel_size_um": {"z": pz, "y": py, "x": px},
        "box_px": {"y": y0, "x": x0, "height": h, "width": w},
        "box_um": box_um,
        "z": int(z),
        "times": t_labels,
        "channels": channels,
    }


def write_crop_json(record: dict, path: str | Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def read_crop_json(path: str | Path) -> dict:
    return json.loads(Path(path).expanduser().read_text())


@dataclass(frozen=True)
class CropExport:
    """All settings of one crop export, checked when created."""

    box: tuple[slice, slice]
    channels: tuple[str, ...]
    times: tuple[int, ...]
    z: int = 0
    raw: bool = True
    display: bool = True
    limits: dict | None = None  # {channel: (lo, hi)}, needed for display
    luts: dict | None = None  # {channel: (256, 3) uint8}, needed for display
    gamma: dict | None = None  # {channel: float}, missing → 1.0

    merge: bool = False  # one PNG per T mixing all chosen channels

    def __post_init__(self):
        if not (self.raw or self.display or self.merge):
            raise ValueError("nothing to export: choose raw, display, merge")
        if (self.display or self.merge) and self.limits is None:
            raise ValueError("display/merge export needs limits")
        if (self.display or self.merge) and self.luts is None:
            raise ValueError("display/merge export needs luts")


def crop_folder_name(img: Image, box: tuple[slice, slice]) -> str:
    """{stem}_{scene}_y{y0}_x{x0}_{h}x{w}, unique per region."""
    sy, sx = box
    h, w = sy.stop - sy.start, sx.stop - sx.start
    return f"{Path(img.source).stem}_{img.scene}_y{sy.start}_x{sx.start}_{h}x{w}"


def export_crop(img: Image, settings: CropExport, folder: str | Path) -> Path:
    """Crop img and write raw/, display/ and crop.json into folder."""
    s = settings
    folder = Path(folder).expanduser()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"{folder} already has files; choose another folder")

    # checks channels, times and box — still nothing written
    crop = crop_image(img, s.box, list(s.channels), list(s.times))
    t_labels = list(s.times)
    record = crop_record(
        crop,
        s.box,
        t_labels,
        z=s.z,
        limits=s.limits if (s.display or s.merge) else None,
        luts=s.luts,
        gamma=s.gamma,
    )

    # display FIRST: it checks limits/gamma/LUTs before writing anything
    if s.display or s.merge:
        export_display(
            crop, folder / "display", s.limits, s.luts,
            gamma=s.gamma, z=s.z, t_labels=t_labels,
            singles=s.display, merge=s.merge,
        )
    if s.raw:
        colors = None if s.luts is None else {n: lut[-1] for n, lut in s.luts.items()}
        export_tiff_planes(crop, folder / "raw", t_labels=t_labels, channel_colors=colors)

    write_crop_json(record, folder / "crop.json")
    return folder
