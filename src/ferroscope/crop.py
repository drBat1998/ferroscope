import math
import re

from ferroscope.io import Image


def um_to_px(size_um: float, pixel_um: float | None) -> int:

    if pixel_um is None:
        raise ValueError("pixel size is not define")

    if pixel_um <= 0:
        raise ValueError(f"pixel size must be > 0, got {pixel_um!r}")

    if size_um <= 0:
        raise ValueError("size can not be equal or smaller than zero")

    px = math.floor(size_um / pixel_um + 0.5)

    if px <= 0:
        raise ValueError(f"{size_um} µm is smaller than one pixel ({pixel_um} µm/px)")
    return px


def px_to_um(size_px: int, pixel_um: float | None) -> float:

    if pixel_um is None:
        raise ValueError("pixel size is not define")

    if pixel_um <= 0:
        raise ValueError("pixel size can not be equal or smaller than zero")

    if size_px <= 0:
        raise ValueError(f"size must be > 0, got {size_px!r}")

    return size_px * pixel_um


def _axis_slice(c, h, n):
    if h <= 0:
        raise ValueError(f"size must be > 0, got {h!r}")
    if h > n:
        raise ValueError(f"size {h} px is larger than the image ({n} px)")

    start = math.floor(c - (h - 1) / 2 + 0.5)
    start = min(max(start, 0), n - h)
    return slice(start, start + h)


def crop_box(
    center_yx: tuple[float, float], size_yx: tuple[int, int], shape_yx: tuple[int, int]
) -> tuple[slice, slice]:

    center_y, center_x = center_yx
    size_y, size_x = size_yx
    shape_y, shape_x = shape_yx

    crop_y = _axis_slice(center_y, size_y, shape_y)
    crop_x = _axis_slice(center_x, size_x, shape_x)

    return (crop_y, crop_x)


_TOKEN = re.compile(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", re.ASCII)


def parse_indices(text: str, n: int) -> list[int]:
    # 1. n <= 0            → ValueError with n
    if n <= 0:
        raise ValueError("n can not be equal or smaller to zero")
    # 2. text.strip() == "" → ValueError mentioning "empty"
    if text.strip() == "":
        raise ValueError("input is empty")
    # 3. "all"             → return list(range(n))
    if text.strip().lower() == "all":
        return list(range(n))

    selected = set()
    for piece in text.split(","):
        match = _TOKEN.fullmatch(piece)
        # 4. match is None  → ValueError with piece
        if match is None:
            raise ValueError(f"cannot read {piece!r} (use e.g. 0,5,10-20)")

        start = int(match[1])
        end = int(match[2]) if match[2] is not None else start
        # 5. end < start    → ValueError with piece
        if end < start:
            raise ValueError(f"range {piece.strip()!r} goes backwards")
        # 6. end >= n       → ValueError with end and n
        if end >= n:
            raise ValueError(f"index {end} out of range: {n} time points (0..{n - 1})")
        selected.update(range(start, end + 1))

    return sorted(selected)


def crop_image(
    img: Image, box: tuple[slice, slice], channels: list[str], times: list[int]
) -> Image:

    if img.dims != ("T", "C", "Z", "Y", "X"):
        raise ValueError(f"crop_image needs TCZYX, got {img.dims}")
    n_t, _, _, n_y, n_x = img.data.shape

    if not channels:
        raise ValueError("no channel selected")

    for name in channels:
        if name not in img.channel_names:
            raise ValueError(
                f"unknown channel {name!r}, available: {img.channel_names}"
            )

    if len(set(channels)) != len(channels):
        dup = next(name for name in channels if channels.count(name) > 1)
        raise ValueError(f"channel {dup!r} selected twice")

    if not times:
        raise ValueError("no time point selected")

    for t in times:
        if not 0 <= t < n_t:
            raise ValueError(f"time point {t} out of range (0..{n_t - 1})")

    sy, sx = box

    for s, n in ((sy, n_y), (sx, n_x)):
        if s.start < 0 or s.stop > n or s.stop <= s.start:
            raise ValueError(f"box {box} outside or empty for image {n_y} x {n_x}")

    c_idx = [i for i, name in enumerate(img.channel_names) if name in channels]

    data = img.data[list(times)]
    data = data[:, c_idx]
    data = data[..., sy, sx]

    return Image(
        data=data,
        dims=img.dims,
        channel_names=[img.channel_names[i] for i in c_idx],
        pixel_size=img.pixel_size,
        source=img.source,
        scene=img.scene,
    )

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
    pz, py, px = crop.pixel_size
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
