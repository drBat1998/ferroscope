"""
CZI -> OME-Zarr export for ferroscope.

"""

from ome_zarr import OMEZarrImage, OMEZarrMultiscale
import numpy as np
import zarr

from ome_zarr.writer import write_image, write_plate_metadata, write_well_metadata

path = "test_ngff_plate.zarr"
row_names = ["A", "B"]
col_names = ["1", "2", "3"]
well_paths = ["A/2", "B/3"]
field_paths = ["0", "1", "2"]

# generate data
mean_val = 10
num_wells = len(well_paths)
num_fields = len(field_paths)
size_xy = 128
size_z = 10
rng = np.random.default_rng(0)
data = rng.poisson(
    mean_val, size=(num_wells, num_fields, size_z, size_xy, size_xy)
).astype(np.uint8)

# write the plate of images and corresponding metadata
# Use zarr_format=2 to write v0.4 format (zarr v2)
root = zarr.open_group(path, mode="w")
write_plate_metadata(root, row_names, col_names, well_paths)

for wi, wp in enumerate(well_paths):
    row, col = wp.split("/")
    row_group = root.require_group(row)
    well_group = row_group.require_group(col)
    write_well_metadata(well_group, field_paths)
    for fi, field in enumerate(field_paths):
        image_group = well_group.require_group(str(field))
        write_image(
            image=data[wi, fi],
            group=image_group,
            axes="zyx",
            storage_options=dict(chunks=(1, size_xy, size_xy)),
        )


image = OMEZarrImage(
    data = data, 
    axes = ["t", "c", "z", "y", "x"],
    scale = {"t": 1.0, "c": 1.0, "z": 0.5, "y": 0.1, "x": 0.1},
    axes_units = {"t": "seconds", "z": "micrometer", "y": "micrometer", "x": "micrometer"},
    axes_types = {"t": "time", "c": "channel", "z": "space", "x": "space", "y": "space"},
    name = "name_of_the_file"
)

multiscales = OMEZarrMultiscale(
    image=image,
    scale_factors=(2, 4, 8),
    method="resize",
    channel_names=["DAPI", "GFP"],
    contrast_limits=[(0, 255), (0, 255)]  
    )

multiscales.to_ome_zarr("test_ngff.ome.zarr", version="0.6")
