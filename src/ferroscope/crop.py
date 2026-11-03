import math


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
