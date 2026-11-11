import numpy as np
import napari
from bioio import BioImage

OFFSET = 100.0  # replace with your dark-frame median

img = BioImage("~/data/OGD_but/test_export/PL1_live_P1-A1_T000_Coherence_Z000.ome.tif")
plane = img.get_image_data("YX", T=0, C=0, Z=0).astype(np.float32) - OFFSET

def sum_bin(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f   # trim to a multiple of f
    return a[:h, :w].reshape(h // f, f, w // f, f).sum(axis=(1, 3))

viewer = napari.Viewer()
viewer.add_image(plane, name="bin1x1")
for f in (2, 3):
    viewer.add_image(sum_bin(plane, f), name=f"bin{f}x{f}", scale=(f, f))
napari.run()
