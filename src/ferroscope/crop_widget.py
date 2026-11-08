"""Crop & export dock widget for napari.

Pure helpers first (tested without a window), the widget last. The widget only
reads the UI, calls the helpers and export_crop, and shows errors in its status
line instead of crashing napari.
"""

from pathlib import Path

import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    LineEdit,
    PushButton,
    Select,
    SpinBox,
)

from ferroscope.contrast import shared_contrast
from ferroscope.crop import crop_box, crop_image, parse_indices, px_to_um, um_to_px
from ferroscope.export import CropExport, crop_folder_name, export_crop
from ferroscope.io import Image

MODES = ("crop", "field", "viewer")


# ---------------------------------------------------------------- pure helpers


def rectangle_from_box(box: tuple[slice, slice]) -> np.ndarray:
    """Four (y, x) corners on the pixel EDGES around the box (napari convention)."""
    sy, sx = box
    y0, y1 = sy.start - 0.5, sy.stop - 0.5
    x0, x1 = sx.start - 0.5, sx.stop - 0.5
    return np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float)


def centre_of_rectangle(corners) -> tuple[float, float]:
    """Centre (y, x) of the corners' bounding box — the inverse of rectangle_from_box."""
    c = np.asarray(corners, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"corners must be (n, 2) y/x points, got shape {c.shape}")
    lo, hi = c.min(axis=0), c.max(axis=0)
    return float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)


def _image_centre(n_y: int, n_x: int) -> tuple[float, float]:
    """Pixels run 0..n-1, so the middle is (n - 1) / 2, not n / 2."""
    return (n_y - 1) / 2, (n_x - 1) / 2


def lut_from_layer(layer) -> np.ndarray:
    """The layer's colormap as a (256, 3) uint8 table."""
    rgba = np.asarray(layer.colormap.map(np.linspace(0, 1, 256)))
    return np.floor(rgba[:, :3] * 255 + 0.5).astype(np.uint8)


def display_from_layers(layers, channels: list[str]) -> tuple[dict, dict, dict]:
    """LUT, gamma and contrast limits of each channel's layer, by channel name."""
    by_name = {layer.name: layer for layer in layers}
    luts, gamma, limits = {}, {}, {}
    for name in channels:
        if name not in by_name:
            raise ValueError(f"no layer named {name!r} in the viewer")
        layer = by_name[name]
        luts[name] = lut_from_layer(layer)
        gamma[name] = float(layer.gamma)
        limits[name] = tuple(float(v) for v in layer.contrast_limits)
    return luts, gamma, limits


def limits_for_mode(
    img: Image,
    box: tuple[slice, slice],
    channels: list[str],
    times: list[int],
    mode: str,
    low: float = 0.1,
    high: float = 99.9,
    viewer_limits: dict | None = None,
) -> dict:
    """{channel: (lo, hi)} — from the crop, the whole field, or the viewer."""
    if mode == "viewer":
        if viewer_limits is None:
            raise ValueError("limits mode 'viewer' needs the viewer's limits")
        for name in channels:
            if name not in viewer_limits:
                raise ValueError(f"no viewer limits for channel {name!r}")
        return {name: viewer_limits[name] for name in channels}

    if mode == "crop":
        region = box
    elif mode == "field":
        n_y, n_x = img.data.shape[-2:]
        region = (slice(0, n_y), slice(0, n_x))
    else:
        raise ValueError(f"unknown limits mode {mode!r}, use one of {MODES}")

    part = crop_image(img, region, list(channels), list(times))
    return dict(zip(part.channel_names, shared_contrast([part], low, high)))


# ---------------------------------------------------------------- widget


class CropWidget(Container):
    """Dock widget: square crop sized in µm or px, chosen channels and T, export."""

    def __init__(self, viewer, get_image, out_dir="~/ferroscope_crops", size_px=200):
        self.viewer = viewer
        self.get_image = get_image  # callable → the current Image (lazy)

        img = get_image()
        n_y, n_x = img.data.shape[-2:]
        self._pixel_um = img.pixel_size[1]
        size_px = min(size_px, n_y, n_x)

        self.size_px = SpinBox(
            value=size_px, min=1, max=min(n_y, n_x), label="Size (px)"
        )
        self.size_um = FloatSpinBox(
            value=0.0, min=0.0, max=1e6, step=1.0, label="Size (µm)"
        )
        self.size_info = Label(value="")
        self.channels = Select(
            choices=img.channel_names, value=img.channel_names, label="Channels"
        )
        self.times = LineEdit(value="all", label="Time points")
        self.raw = CheckBox(value=True, text="Raw OME-TIFF")
        self.display = CheckBox(value=True, text="Display PNG (one per channel)")
        self.merge = CheckBox(value=False, text="Merge PNG (channels mixed)")
        self.mode = ComboBox(choices=MODES, value="crop", label="Limits from")
        self.low = FloatSpinBox(value=0.1, min=0.0, max=100.0, step=0.1, label="Low %")
        self.high = FloatSpinBox(
            value=99.9, min=0.0, max=100.0, step=0.1, label="High %"
        )
        self.folder = FileEdit(
            mode="d", value=Path(out_dir).expanduser(), label="Save to"
        )
        self.export_button = PushButton(text="Export crop")
        self.status = Label(value="")

        super().__init__(
            widgets=[
                self.size_px,
                self.size_um,
                self.size_info,
                self.channels,
                self.times,
                self.raw,
                self.display,
                self.merge,
                self.mode,
                self.low,
                self.high,
                self.folder,
                self.export_button,
                self.status,
            ]
        )

        # the rectangle lives in pixel coordinates; scale puts it on top in µm
        scale = (img.axis_scale["Y"], img.axis_scale["X"])
        box = crop_box(_image_centre(n_y, n_x), (size_px, size_px), (n_y, n_x))
        self.layer = viewer.add_shapes(
            [rectangle_from_box(box)],
            shape_type="rectangle",
            name="crop",
            scale=scale,
            edge_color="yellow",
            face_color="transparent",
            edge_width=max(2, min(n_y, n_x) // 300),
        )
        # keep the channel list short so the options below it stay visible
        self.channels.max_height = 22 * len(img.channel_names) + 10
        if self._pixel_um is None:
            self.size_um.enabled = False  # µm unknown → px only
        self._sync_um_from_px()

        self.size_px.changed.connect(self._on_px)
        self.merge.changed.connect(self._update_enabled)
        self.size_um.changed.connect(self._on_um)
        self.raw.changed.connect(self._update_enabled)
        self.display.changed.connect(self._update_enabled)
        self.mode.changed.connect(self._update_enabled)
        self.export_button.changed.connect(self.export)
        self._update_enabled()

    # ---- size fields, linked both ways without calling each other forever

    def _sync_um_from_px(self):
        px = self.size_px.value
        if self._pixel_um is None:
            self.size_info.value = f"{px} px (pixel size unknown)"
            return
        um = px_to_um(px, self._pixel_um)
        with self.size_um.changed.blocked():
            self.size_um.value = round(um, 2)
        self.size_info.value = f"{px} px = {um:.2f} µm"

    def _on_px(self, _value=None):
        self._sync_um_from_px()
        self._redraw()

    def _on_um(self, value):
        try:
            px = min(um_to_px(value, self._pixel_um), self.size_px.max)
        except ValueError as e:
            self.status.value = f"✗ {e}"
            return
        with self.size_px.changed.blocked():
            self.size_px.value = px
        self.size_info.value = f"{px} px = {px_to_um(px, self._pixel_um):.2f} µm"
        self._redraw()

    # ---- rectangle

    def current_box(self, img: Image | None = None) -> tuple[slice, slice]:
        img = img or self.get_image()
        n_y, n_x = img.data.shape[-2:]
        if self.layer in self.viewer.layers and len(self.layer.data):
            centre = centre_of_rectangle(self.layer.data[-1][:, -2:])
        else:
            centre = _image_centre(n_y, n_x)
        size = self.size_px.value
        return crop_box(centre, (size, size), (n_y, n_x))

    def _redraw(self):
        if self.layer not in self.viewer.layers:
            self.viewer.layers.append(self.layer)  # brought back if deleted
        self.layer.data = [rectangle_from_box(self.current_box())]

    def current_z(self) -> int:
        labels = list(self.viewer.dims.axis_labels)
        if "Z" not in labels:
            return 0
        return int(self.viewer.dims.current_step[labels.index("Z")])

    # ---- enabling

    def _update_enabled(self, _value=None):
        coloured = self.display.value or self.merge.value  # both need limits + LUTs
        self.mode.enabled = coloured
        self.low.enabled = self.high.enabled = coloured and self.mode.value != "viewer"
        self.export_button.enabled = self.raw.value or coloured
    # ---- export

    def export(self, _value=None) -> Path | None:
        try:
            img = self.get_image()
            box = self.current_box(img)
            chosen = set(self.channels.value)
            channels = [n for n in img.channel_names if n in chosen]  # file order
            times = parse_indices(self.times.value, img.data.shape[0])

            # colours and gamma always come from the layers (raw files get colours too)
            luts, gamma, viewer_limits = display_from_layers(
                self.viewer.layers, channels
            )
            limits = None
            if self.display.value or self.merge.value:
                limits = limits_for_mode(
                    img,
                    box,
                    channels,
                    times,
                    self.mode.value,
                    self.low.value,
                    self.high.value,
                    viewer_limits,
                )

            settings = CropExport(
                box=box,
                channels=tuple(channels),
                times=tuple(times),
                z=self.current_z(),
                raw=self.raw.value,
                display=self.display.value,
                merge=self.merge.value,
                limits=limits,
                luts=luts,
                gamma=gamma,
            )
            folder = Path(self.folder.value).expanduser() / crop_folder_name(img, box)
            export_crop(img, settings, folder)
        except (ValueError, FileExistsError) as e:
            self.status.value = f"✗ {e}"
            return None

        self.status.value = f"✓ saved {folder.name}"
        return folder
