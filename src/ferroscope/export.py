from itertools import product
from pathlib import Path

from bioio_base.types import PhysicalPixelSizes
from bioio_ome_tiff.writers import OmeTiffWriter

from ferroscope.io import Image, describe_image, load_scene


def export_tiff_planes(img: Image, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(img.source).stem}_{img.scene}"  
    n_t, n_c, n_z = img.data.shape[:3]  

    written = []
    for t, c, z in product(range(n_t), range(n_c), range(n_z)):
        channel = img.channel_names[c]
        path = out_dir / f"{stem}_T{t:03d}_{channel}_Z{z:03d}.ome.tif"
        OmeTiffWriter.save(
            img.data[t, c, z],                                   
            path,
            dim_order="YX",
            channel_names=[channel],
            physical_pixel_sizes=PhysicalPixelSizes(*img.pixel_size),
        )
        written.append(path)
    return written


def export_plate(path, out_dir=None) -> list[Path]:
    """Every scene of a file → all planes, all into ONE folder."""
    path = Path(path).expanduser()
    if out_dir is None:
        out_dir = path.parent / f"{path.stem}_tiff"    
    written = []
    n_scenes = describe_image(path)["n_scenes"]
    for i in range(n_scenes): 
        written += export_tiff_planes(load_scene(path, i), out_dir)
    return written
