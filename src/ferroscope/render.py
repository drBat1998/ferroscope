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
