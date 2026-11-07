from ferroscope.io import load_scene
from ome_zarr.writer import write_image
import time

img = load_scene("~/data/Tox/78_tox.czi", scene = 0)
print(img.data.shape)

t0 = time.perf_counter()
write_image(
image = img.data, 
group = "78_tox_s0.ome.zarr",
axes = "tczyx",
scale = {"y": 0.5, "x": 0.5,"z":0.4 }
)
print(time.perf_counter()-t0)

