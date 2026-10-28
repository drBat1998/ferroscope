import napari
from ferroscope.io import load_scene
from ferroscope.viewer import show

img = load_scene("~/data/OGD_but/PL1_live.czi", 0)
show(img)
napari.run()
