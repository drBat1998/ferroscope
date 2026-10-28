"""
The program have to import image mainly tiff, czi, and zarr.

First, the file type should be detected. Then the metadata should be displayed, including type of data, channels, time information, z stack information. Scenes information. Then the image scene by sence lazy import to mamory for further use.

Should I have several
"""

from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
from bioio import BioImage


@dataclass
class Image:
    data: da.Array
    dims: tuple[str, str, str, str, str]
    channel_names: list = field(default_factory=list)
    pixel_size: tuple[float | None, float | None, float | None] = (None, None, None)
    source: str | Path | None = None
    scene: str | None = None

    def __post_init__(self):
        if len(self.dims) != self.data.ndim:
            raise ValueError("Dims is not good")

    @property
    def channel_axis(self) -> int | None:
        return self.dims.index("C") if "C" in self.dims else None

    @property
    def axis_scale(self) -> dict[str, float]:
        """µm per pixel for every axis; 1.0 where unknown or non-spatial (T, C)."""
        z, y, x = self.pixel_size
        lookup = {"Z": z, "Y": y, "X": x}
        return {d: (lookup.get(d) or 1.0) for d in self.dims}


def detect_format(path: str | Path) -> str:
    """
    'czi' | 'ome-tiff' | 'tiff' | 'zarr'.
    Raises ValueError for anything else.
    """
    file_path = Path(path)

    FORMATS = [
        (".ome.tiff", "ome-tiff"),
        (".ome.tif", "ome-tiff"),
        (".tiff", "tiff"),
        (".tif", "tiff"),
        (".czi", "czi"),
        (".zarr", "zarr"),
    ]
    for ending, name in FORMATS:
        if file_path.name.lower().endswith(ending):
            return name
    raise ValueError(
        f"unsupported format: {file_path.name!r} (supported: .czi, .tif, .tiff, .ome.tif, .zarr)"
    )


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
    path_file, fmt, img = _open(path)
    img.set_scene(scene)
    pps = img.physical_pixel_sizes

    return Image(
        data=img.dask_data,
        dims=tuple(img.dims.order),
        channel_names=[str(c) for c in img.channel_names],
        pixel_size=(_um(pps.Z), _um(pps.Y), _um(pps.X)),
        source=path_file,
        scene=str(img.current_scene),
    )
