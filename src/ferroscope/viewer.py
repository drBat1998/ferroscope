from pathlib import Path

import dask.array as da

from ferroscope.colors import colormap_list
from ferroscope.io import Image, describe_image, load_scene


def layer_kwargs(img: Image, contrast_limits=None, colormap=None) -> dict:

    kw = {
        "data": img.data,
        "channel_axis": img.channel_axis,
        "name": img.channel_names
        if img.channel_axis is not None
        else (img.scene or "image"),
        "scale": tuple(img.axis_scale[d] for d in img.dims if d != "C"),
    }

    if contrast_limits is not None:
        kw["contrast_limits"] = contrast_limits

    if colormap is not None:
        kw["colormap"] = colormap

    return kw


def show(img: Image, viewer=None, contrast_limits=None, colormap=None):
    import napari

    if viewer is None:
        viewer = napari.Viewer()

    layers = viewer.add_image(**layer_kwargs(img, contrast_limits, colormap))
    viewer.dims.axis_labels = tuple(d for d in img.dims if d != "C")

    return layers


class SceneBrowser:
    def __init__(self, viewer, path, colors=None):
        self.viewer = viewer
        self.path = path
        self.colors = colors or {}
        self.scenes = [s["id"] for s in describe_image(path)["scenes"]]
        self.index = 0

    def go(self, index: int) -> None:
        index = index % len(self.scenes)
        img = load_scene(self.path, index)

        if not self._swap_in_place(img):
            colormap = colormap_list(img.channel_names, self.colors)
            self.viewer.layers.clear()
            self.viewer.add_image(**layer_kwargs(img, colormap=colormap))
            self.viewer.dims.axis_labels = tuple(d for d in img.dims if d != "C")

        self.index = index
        self.viewer.title = (
            f"{Path(self.path).name} | {img.scene} ({index + 1}/{len(self.scenes)})"
        )
    def go_to(self, scene_id: str) -> None:
        self.go(self.scenes.index(scene_id))

    def _swap_in_place(self, img) -> bool:
        names = [layer.name for layer in self.viewer.layers]
        if names != img.channel_names:
            return False
        for i, layer in enumerate(self.viewer.layers):
            layer.data = da.take(img.data, i, axis=img.channel_axis)
        return True

    def next(self) -> None:
        self.go(self.index + 1)

    def prev(self) -> None:
        self.go(self.index - 1)
