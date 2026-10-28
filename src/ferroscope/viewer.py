
from ferroscope.io import Image


def layer_kwargs(img: Image, contrast_limits = None) -> dict:

    kw = {
        "data":img.data,
        "channel_axis": img.channel_axis,
        "name": img.channel_names if img.channel_axis is not None else (img.scene or "image"),
        "scale": tuple(img.axis_scale[d] for d in img.dims if d != "C"),
    }

    if contrast_limits is not None:
        kw["contrast_limits"] = contrast_limits
  
    return kw

def show(img: Image, viewer=None, contrast_limits=None):
    import napari

    if viewer is None:
        viewer = napari.Viewer()
    layers = viewer.add_image(**layer_kwargs(img, contrast_limits))
    viewer.dims.axis_labels = tuple(d for d in img.dims if d != "C")
    return layers
