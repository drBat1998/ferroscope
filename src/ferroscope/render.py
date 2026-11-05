import numpy as np


def to_uint8(plane, lo, hi, gamma=1.0) -> np.ndarray:
    if lo >= hi:
        raise ValueError(f"lo must be < hi, got lo={lo}, hi={hi}")    
    if gamma <= 0:
        raise ValueError("Gamma is lower than 0")
    float_plane = plane.astype(np.float64)
    norm = np.clip((float_plane - lo) / (hi - lo), 0, 1) ** gamma
    return np.floor(norm * 255 + 0.5).astype(np.uint8)


def apply_lut(u8, lut) -> np.ndarray:
    if lut.shape != (256, 3):
        raise ValueError("Lut shape must be (256,3)")
    if u8.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    return lut[u8]

def composite(rgbs) -> np.ndarray:
    """Additive mix of RGB images, like napari's additive blending: sum, cap at 255."""
    rgbs = [np.asarray(rgb) for rgb in rgbs]
    if not rgbs:
        raise ValueError("nothing to mix: empty list of images")
    shape = rgbs[0].shape
    for i, rgb in enumerate(rgbs):
        if rgb.dtype != np.uint8:
            raise ValueError(f"image {i} must be uint8, got {rgb.dtype}")
        if rgb.shape != shape:
            raise ValueError(f"image {i} has shape {rgb.shape}, image 0 has {shape}")
        if rgb.shape[-1] != 3:
            raise ValueError(f"images must end in 3 colours (RGB), got shape {rgb.shape}")
    total = np.zeros(shape, dtype=np.uint16)  # room for sums above 255
    for rgb in rgbs:
        total += rgb
    return np.minimum(total, 255).astype(np.uint8)
