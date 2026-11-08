"""Lazy loading of microscopy images and their metadata.

Supported files: .czi, .tif/.tiff, .ome.tif/.ome.tiff, .zarr (OME-Zarr only).
DECIDE: plain .zarr without OME metadata -> rejected with ValueError, or left to bioio?

Public functions:
    detect_format(path)         -> format name, decided by file extension.
    describe_image(path)        -> metadata for every scene; no pixels read.
    load_scene(path, scene=0)   -> one scene as an Image; scene by index or id.

Every Image guarantees:
    - data is a lazy dask array; pixels are read from disk only on compute.
    - data is always 5-D, axes in TCZYX order.
    - one channel name per channel on the C axis.
    - pixel sizes are in µm, time intervals in seconds.
    - unknown pixel sizes and time intervals are None, never 0.
    - Image is immutable.
DECIDE: plain TIFF reporting a pixel size of 1.0 -> treated as unknown (None), or kept?

Errors:
    FileNotFoundError   the path does not exist.
    ValueError          unsupported file extension.
    ValueError          scene index or id not in the file; message lists valid ids.
    ValueError          file is read as RGB; RGB is not supported.

This module only reads files. Display, export and analysis live in other
modules that import io, never the other way round.
"""

from dataclasses import dataclass
from pathlib import Path
import dask.array as da
from bioio import BioImage


VALID_DIMS = frozenset("TCZYX")


def _check_positive_or_none(name: str, value: float | None) -> None:
    if value is not None and not value > 0:
        raise ValueError(f"{name} must be > 0 or None, got {value!r}")

@dataclass
class Image:
    """One scene of a microscopy image, loaded lazily.

    Units: physical sizes in µm, times in seconds.
    Unknown sizes or intervals are None, never 0.
    """

    data: da.Array
    dims: tuple[str, ...]
    channel_names: list[str]
    pixel_size_um: tuple[float | None, float | None, float | None] = (None, None, None)
    time_interval_s: float | None = None
    source: Path | None = None
    scene: str | None = None

    def __post_init__(self) -> None:
        self._check_dims()
        self._check_channels()
        if len(self.pixel_size_um) != 3:
            raise ValueError(f"pixel_size_um must be (z, y, x), got {self.pixel_size_um!r}")
        for axis, value in zip("ZYX", self.pixel_size_um):
            _check_positive_or_none(f"pixel size {axis}", value)
        _check_positive_or_n:one("time_interval_s", self.time_interval_s)

    

    def _check_dims(self) -> None:
        if len(self.dims) != self.data.ndim:
            raise ValueError(
                f"dims {self.dims!r} has {len(self.dims)} axes, "
                f"but data has {self.data.ndim} (shape {self.data.shape})"
            )

        unknown = set(self.dims) - VALID_DIMS

        if unknown:
            raise ValueError(f"unknown dims {sorted(unknown)} in {self.dims!r}; allowed: T C Z Y X")

        if len(set(self.dims)) != len(self.dims):
            raise ValueError(f"duplicate dims in {self.dims!r}")

    
    def _check_channels(self) -> None:
        if self.channel_axis is None:
            return

        n = self.data.shape[self.channel_axis]
        
        if len(self.channel_names) != n:
            raise ValueError(
                f"{len(self.channel_names)} channel names {self.channel_names!r} "
                f"for {n} channels on the C axis"
            )

    @property
    def channel_axis(self) -> int | None:
        return self.dims.index("C") if "C" in self.dims else None

    @property
    def axis_scale(self) -> dict[str, float]:
        """Step per axis: µm for Z/Y/X, seconds for T; 1.0 where unknown, and for C."""
        z, y, x = self.pixel_size_um
        lookup = {"T": self.time_interval_s, "Z": z, "Y": y, "X": x}
        return {d: lookup.get(d) or 1.0 for d in self.dims}




FORMATS = [(".ome.tiff", "ome-tiff"),
        (".ome.tif", "ome-tiff"),
        (".tiff", "tiff"),
        (".tif", "tiff"),
        (".czi", "czi"),
        (".zarr", "zarr"),]

def detect_format(path: str | Path) -> str:
    """
    'czi' | 'ome-tiff' | 'tiff' | 'zarr'.
    Raises ValueError for anything else.
    """
    file_path = Path(path)
   
    for ending, name in FORMATS:
        if file_path.name.lower().endswith(ending):
            return name
    raise ValueError(
        f"unsupported format: {file_path.name!r} (supported: {', '.join([ending for ending, name in FORMATS])})")


def _um(value):
    """bioio gives 0.0 ..."""
    return float(value) if value else None


def _open(path: str | Path) -> tuple[Path, str, BioImage]:
    path_file = Path(path).expanduser()

    if not path_file.exists():
        raise FileNotFoundError(f" no such file:{path_file}")
    fmt = detect_format(path_file)
    return path_file, fmt, BioImage(path_file)


def describe_image(path: str | Path) -> dict:

    path_file, fmt, img = _open(path)

    scenes = []
    for i, scene_id in enumerate(img.scenes):
        img.set_scene(i)
        pps = img.physical_pixel_sizes
        scenes.append(
            {
                "id": scene_id,
                "dims": img.dims.order,
                "T": img.dims.T,
                "Z": img.dims.Z,
                "channels": [str(c) for c in img.channel_names],
                "shape": img.shape,
                "data_type": str(img.dtype),
                "pixel_size": (_um(pps.Z), _um(pps.Y), _um(pps.X)),
            }
        )

    return {
        "path": str(path_file),
        "format": fmt,
        "n_scenes": len(scenes),
        "scenes": scenes,
    }


def load_scene(path: str | Path, scene: int | str = 0) -> Image:
    """Open one scene lazily. ``scene``: index (0, 1, ...) or id ("P3-B3")."""
    path_file, _, img = _open(path)
    
    img.set_scene(scene)
    
    pps = img.physical_pixel_sizes

    return Image(
        data = img.dask_data,
        dims = tuple(img.dims.order),
        channel_names = [str(c) for c in img.channel_names],
        pixel_size_um = (_um(pps.Z), _um(pps.Y), _um(pps.X)),
        time_interval_s = None,
        source = path_file,
        scene = str(img.current_scene),
    )

if __name__ == "__main__":

