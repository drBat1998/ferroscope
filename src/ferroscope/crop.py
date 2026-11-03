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


import math
import re

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



def crop_image(img: Image, box: tuple[slice, slice],
               channels: list[str], times: list[int]) -> Image:
    raise NotImplementedError


def crop_image(img: Image, box: tuple[slice, slice],
               channels: list[str], times: list[int]) -> Image:
    # 1. dims must be ("T", "C", "Z", "Y", "X")  → ValueError mentioning "TCZYX"
    n_t, _, _, n_y, n_x = img.data.shape
    
    # 2. channels: empty → "channel"; unknown name → name + img.channel_names;
    #    the same name twice → that name

    # 3. times: empty → "time"; any t outside 0..n_t-1 → that t

    # 4. box: for (sy, n_y) and (sx, n_x): start >= 0, stop <= n, stop > start
    #    → otherwise ValueError mentioning "box"
    sy, sx = box

    # 5. channel indices in FILE order
    c_idx = [i for i, name in enumerate(img.channel_names) if name in channels]

    # 6. index one step at a time
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
