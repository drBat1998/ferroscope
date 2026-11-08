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
import bioio_czi
import bioio_ome_tiff
import bioio_ome_zarr
from datetime import timedelta

DIMS = ("T", "C", "Z", "Y", "X")


def _check_positive_or_none(name: str, value: float | None) -> None:
    if value is not None and not value > 0:
        raise ValueError(f"{name} must be > 0 or None, got {value!r}")

@dataclass(frozen=True)
class Image:
    """One scene of a microscopy image, loaded lazily.

    Units: physical sizes in µm, times in seconds.
    Unknown sizes or intervals are None, never 0.
    """

    data: da.Array
    dims: tuple[str, ...]
    channel_names: tuple[str, ...]    
    pixel_size_um: tuple[float | None, float | None, float | None] = (None, None, None)
    time_interval_s: float | None = None
    source: Path | None = None
    scene: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, da.Array):
            raise TypeError(
                f"data have to be in da.Array must be got {type(self.data).__name__}"
            )        
        self._check_dims()
        self._check_channels()
        if len(self.pixel_size_um) != 3:
            raise ValueError(f"pixel_size_um must be (z, y, x), got {self.pixel_size_um!r}")
        for axis, value in zip("ZYX", self.pixel_size_um):
            _check_positive_or_none(f"pixel size {axis}", value)
        _check_positive_or_none("time_interval_s", self.time_interval_s)

    

    def _check_dims(self) -> None:
        if self.dims != DIMS:
            raise ValueError(f"dims must be {DIMS!r}, got {self.dims!r}")
        if self.data.ndim != len(DIMS):
            raise ValueError(
                f"data must be {len(DIMS)}-D, got {self.data.ndim}-D "
                f"(shape {self.data.shape})"
            )

    
    def _check_channels(self) -> None:
        if not isinstance(self.channel_names, tuple):
            raise TypeError(
                f"channel_names must be a tuple, got {type(self.channel_names).__name__}"
            )

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


def _um(value: float | None) -> float | None:
    """Pixel size from bioio as a float in µm, or None when unknown.

    bioio reports a missing size as None or 0.0; both become None.
    Negative values pass through unchanged, so Image rejects them loudly.
    """
    return float(value) if value else None


FORMATS = (
    (".ome.tiff", "ome-tiff"),
    (".ome.tif", "ome-tiff"),
    (".czi", "czi"),
    (".zarr", "zarr"),
)

READERS = {
    "czi": bioio_czi.Reader,
    "ome-tiff": bioio_ome_tiff.Reader,
    "zarr": bioio_ome_zarr.Reader,
}


def detect_format(path: str | Path) -> str:
    """Format name from the file name: 'czi' | 'ome-tiff' | 'zarr'.

    Raises ValueError for anything else, including plain .tif/.tiff.
    """
    name = Path(path).name.lower()
    for ending, fmt in FORMATS:
        if name.endswith(ending):
            return fmt
    supported = ", ".join(ending for ending, _ in FORMATS)
    raise ValueError(f"unsupported format: {name!r} (supported: {supported})")


def _open(path: str | Path) -> tuple[Path, str, BioImage]:
    """Expand ~, check format then existence, open with that format's reader."""
    path_file = Path(path).expanduser()
    fmt = detect_format(path_file)
    if not path_file.exists():
        raise FileNotFoundError(f"no such file: {path_file}")
    return path_file, fmt, BioImage(path_file, reader=READERS[fmt])


def _seconds(value: timedelta | None) -> float | None:
    """Time interval from bioio as float seconds, or None when unknown.

    bioio gives a timedelta or None; None and a zero interval become None.
    Negative intervals pass through unchanged, so Image rejects them loudly.
    """
    return value.total_seconds() if value else None


def _set_scene(bio: BioImage, scene: int | str, path: Path) -> None:
    """Select a scene, or raise ValueError listing the valid scene ids."""
    try:
        bio.set_scene(scene)
    except (IndexError, ValueError) as err:
        raise ValueError(
            f"scene {scene!r} not in {path.name}; valid: {list(bio.scenes)}"
        ) from err


def _image_from(bio: BioImage, path: Path) -> Image:
    """The current scene of an open BioImage as an Image. No pixels are read.

    Time interval comes from bio.time_interval, not standard_metadata:
    standard_metadata builds full OME metadata on every call (~190 ms per
    scene on CD7 files, plus a warning per scene).
    """
    order = bio.dims.order
    if "S" in order:
        raise ValueError(
            f"scene {bio.current_scene!r} in {path.name} is read as RGB "
            f"(dims {order}); RGB is not supported"
        )
    pps = bio.physical_pixel_sizes
    return Image(
        data=bio.dask_data,
        dims=tuple(order),
        channel_names=tuple(str(c) for c in bio.channel_names),
        pixel_size_um=(_um(pps.Z), _um(pps.Y), _um(pps.X)),
        time_interval_s=_seconds(bio.time_interval),
        source=path,
        scene=str(bio.current_scene),
    )


def describe_image(path: str | Path) -> dict:
    """Metadata for every scene; no pixels read. Fails if any scene is invalid."""
    path_file, fmt, bio = _open(path)
    scenes = []
    for i in range(len(bio.scenes)):
        bio.set_scene(i)
        image = _image_from(bio, path_file)
        scenes.append(
            {
                "id": image.scene,
                "shape": image.data.shape,
                "dtype": str(image.data.dtype),
                "channel_names": image.channel_names,
                "pixel_size_um": image.pixel_size_um,
                "time_interval_s": image.time_interval_s,
            }
        )
    return {"path": str(path_file), "format": fmt, "n_scenes": len(scenes), "scenes": scenes}


def load_scene(path: str | Path, scene: int | str = 0) -> Image:
    """Open one scene lazily. ``scene``: index (0, 1, ...) or id ("P3-B3")."""
    path_file, _, bio = _open(path)
    _set_scene(bio, scene, path_file)
    return _image_from(bio, path_file)
