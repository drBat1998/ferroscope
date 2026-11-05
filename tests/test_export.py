"""Tests for ferroscope.export.export_tiff_planes — raw OME-TIFF per plane.

New in the crop step:
- t_labels: the real time-point numbers for the file names. A crop of T [0, 3]
  must write ..._T000_... and ..._T003_..., not T000 and T001.
  None -> 0, 1, 2, ... (the old behaviour). Wrong length or duplicates -> ValueError.
- channel_colors: {channel name: (r, g, b)} with 0..255 ints, stored as the OME
  Channel Color so Fiji and napari open each channel in its colour.
  A channel without an entry gets the OME default (white); extra names are ignored.
- Everything is validated BEFORE the first file is written (no half exports).
- Pixels are written unchanged (raw = raw).

export_display(img, out_dir, limits, luts, gamma=None, z=0, t_labels=None):
- One RGB PNG per T x C for plane z: {stem}_T{label:03d}_{channel}_Z{z:03d}.png
- Pixels = apply_lut(to_uint8(plane, lo, hi, gamma), lut), exactly what render does.
- limits and luts are dicts keyed by channel NAME; a channel without one -> ValueError
  naming it. gamma is optional per channel (default 1.0). Extra names are ignored.
- z out of range -> ValueError mentioning "z"; t_labels follow the same rules as above.
- Bad limits, gamma or LUT are found BEFORE the first PNG is written.

crop_record(crop, box, t_labels, z=0, limits=None, luts=None, gamma=None) -> dict:
- Everything needed to find the crop again and rebuild the figure:
  source, scene, pixel size, box in px (and in µm when known), z, real time
  points, and per channel: limits, gamma, colour (lut[-1]).
- Only plain Python types (numpy ints/floats would break json.dumps).
- box must match the crop's Y, X size; t_labels follow the usual rules.
write_crop_json(record, path) / read_crop_json(path): round trip, nothing lost.

CropExport(box, channels, times, z=0, raw=True, display=True, limits, luts, gamma):
- All settings of one export in one frozen object, checked when it is created:
  at least one of raw/display; display needs limits and luts.
crop_folder_name(img, box) -> "{stem}_{scene}_y{y0}_x{x0}_{h}x{w}"
export_crop(img, settings, folder) -> folder with raw/, display/, crop.json:
- raw uses the real time points and the LUT colours; display uses limits/luts/gamma.
- An existing non-empty folder -> FileExistsError (never mix two exports).
- Bad display settings -> nothing written at all.
"""

import json

import dask.array as da
import numpy as np
import pytest
import tifffile
from ome_types import from_xml
from PIL import Image as PILImage

from ferroscope.crop import crop_image
from ferroscope.export import (
    CropExport,
    crop_folder_name,
    crop_record,
    export_crop,
    export_display,
    export_tiff_planes,
    read_crop_json,
    write_crop_json,
)
from ferroscope.io import Image
from ferroscope.render import apply_lut, to_uint8

NAMES = ["AO", "DRAQ7"]


@pytest.fixture
def stack():
    # T=2, C=2, Z=1, Y=6, X=8
    return np.arange(2 * 2 * 1 * 6 * 8, dtype=np.uint16).reshape(2, 2, 1, 6, 8)


@pytest.fixture
def img(stack):
    return Image(
        data=da.from_array(stack),
        dims=("T", "C", "Z", "Y", "X"),
        channel_names=list(NAMES),
        pixel_size=(None, 0.4536, 0.4536),
        source="PL1_live.czi",
        scene="P3-B3",
    )


def ome_channel(path):
    return from_xml(tifffile.TiffFile(path).ome_metadata).images[0].pixels.channels[0]


# ---------------------------------------------------------------- names


def test_default_names_unchanged(img, tmp_path):
    names = sorted(p.name for p in export_tiff_planes(img, tmp_path))
    assert names == [
        "PL1_live_P3-B3_T000_AO_Z000.ome.tif",
        "PL1_live_P3-B3_T000_DRAQ7_Z000.ome.tif",
        "PL1_live_P3-B3_T001_AO_Z000.ome.tif",
        "PL1_live_P3-B3_T001_DRAQ7_Z000.ome.tif",
    ]


def test_t_labels_name_files_by_real_time_point(img, tmp_path):
    names = {p.name for p in export_tiff_planes(img, tmp_path, t_labels=[0, 3])}
    assert "PL1_live_P3-B3_T003_DRAQ7_Z000.ome.tif" in names
    assert not any("T001" in n for n in names)


def test_t_labels_file_holds_the_right_plane(img, stack, tmp_path):
    export_tiff_planes(img, tmp_path, t_labels=[0, 3])
    plane = tifffile.imread(tmp_path / "PL1_live_P3-B3_T003_DRAQ7_Z000.ome.tif")
    np.testing.assert_array_equal(plane, stack[1, 1, 0])  # 2nd plane of the crop


@pytest.mark.parametrize("labels", [[0], [0, 1, 2]])
def test_t_labels_wrong_length_raises(img, tmp_path, labels):
    with pytest.raises(ValueError, match="t_labels"):
        export_tiff_planes(img, tmp_path, t_labels=labels)


def test_t_labels_duplicates_raise(img, tmp_path):
    # [3, 3] would silently overwrite the first file with the second
    with pytest.raises(ValueError, match="3"):
        export_tiff_planes(img, tmp_path, t_labels=[3, 3])


# ---------------------------------------------------------------- raw pixels


def test_pixels_written_unchanged(img, stack, tmp_path):
    export_tiff_planes(img, tmp_path)
    plane = tifffile.imread(tmp_path / "PL1_live_P3-B3_T001_AO_Z000.ome.tif")
    assert plane.dtype == np.uint16
    np.testing.assert_array_equal(plane, stack[1, 0, 0])


# ---------------------------------------------------------------- colours


def test_channel_colors_written_to_ome(img, tmp_path):
    colors = {"AO": (0, 255, 0), "DRAQ7": (255, 0, 255)}
    export_tiff_planes(img, tmp_path, channel_colors=colors)
    ao = ome_channel(tmp_path / "PL1_live_P3-B3_T000_AO_Z000.ome.tif")
    draq7 = ome_channel(tmp_path / "PL1_live_P3-B3_T000_DRAQ7_Z000.ome.tif")
    assert ao.name == "AO"
    assert ao.color.as_rgb_tuple() == (0, 255, 0)
    assert draq7.color.as_rgb_tuple() == (255, 0, 255)


def test_channel_without_color_gets_ome_default(img, tmp_path):
    export_tiff_planes(img, tmp_path, channel_colors={"DRAQ7": (255, 0, 255)})
    ao = ome_channel(tmp_path / "PL1_live_P3-B3_T000_AO_Z000.ome.tif")
    assert ao.color.as_rgb_tuple() == (255, 255, 255)


def test_extra_color_names_are_ignored(img, tmp_path):
    colors = {"AO": (0, 255, 0), "DRAQ7": (255, 0, 255), "PI": (255, 0, 0)}
    assert len(export_tiff_planes(img, tmp_path, channel_colors=colors)) == 4


def test_numpy_color_values_accepted(img, tmp_path):
    # a LUT row, e.g. lut[-1], is a numpy array of uint8
    color = np.array([255, 0, 255], dtype=np.uint8)
    export_tiff_planes(img, tmp_path, channel_colors={"DRAQ7": color})
    draq7 = ome_channel(tmp_path / "PL1_live_P3-B3_T000_DRAQ7_Z000.ome.tif")
    assert draq7.color.as_rgb_tuple() == (255, 0, 255)


@pytest.mark.parametrize(
    "color",
    [(256, 0, 0), (-1, 0, 0), (255, 0), (255, 0, 255, 255)],
)
def test_bad_color_raises(img, tmp_path, color):
    with pytest.raises(ValueError, match="DRAQ7"):
        export_tiff_planes(img, tmp_path, channel_colors={"DRAQ7": color})


def test_nothing_written_when_input_is_bad(img, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        export_tiff_planes(img, out, channel_colors={"DRAQ7": (300, 0, 0)})
    assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------- export_display

GRAY = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
GREEN = GRAY * np.array([0, 1, 0], dtype=np.uint8)
MAGENTA = GRAY * np.array([1, 0, 1], dtype=np.uint8)
LIMITS = {"AO": (0, 191), "DRAQ7": (20, 150)}
LUTS = {"AO": GREEN, "DRAQ7": MAGENTA}


def read_png(path):
    with PILImage.open(path) as im:
        return im.mode, np.asarray(im)


def test_display_one_png_per_time_and_channel(img, tmp_path):
    written = export_display(img, tmp_path, LIMITS, LUTS)
    assert sorted(p.name for p in written) == [
        "PL1_live_P3-B3_T000_AO_Z000.png",
        "PL1_live_P3-B3_T000_DRAQ7_Z000.png",
        "PL1_live_P3-B3_T001_AO_Z000.png",
        "PL1_live_P3-B3_T001_DRAQ7_Z000.png",
    ]
    assert all(p.exists() for p in written)


def test_display_png_is_rgb_with_image_shape(img, tmp_path):
    export_display(img, tmp_path, LIMITS, LUTS)
    mode, rgb = read_png(tmp_path / "PL1_live_P3-B3_T001_DRAQ7_Z000.png")
    assert mode == "RGB"
    assert rgb.shape == (6, 8, 3)
    assert rgb.dtype == np.uint8


def test_display_pixels_match_render(img, stack, tmp_path):
    export_display(img, tmp_path, LIMITS, LUTS)
    _, rgb = read_png(tmp_path / "PL1_live_P3-B3_T001_DRAQ7_Z000.png")
    expected = apply_lut(to_uint8(stack[1, 1, 0], 20, 150), MAGENTA)
    np.testing.assert_array_equal(rgb, expected)


def test_display_uses_each_channels_own_colour(img, tmp_path):
    export_display(img, tmp_path, LIMITS, LUTS)
    _, ao = read_png(tmp_path / "PL1_live_P3-B3_T000_AO_Z000.png")
    _, draq7 = read_png(tmp_path / "PL1_live_P3-B3_T000_DRAQ7_Z000.png")
    assert not ao[..., 0].any() and not ao[..., 2].any()  # green only
    assert not draq7[..., 1].any()  # magenta: no green


def test_display_gamma_per_channel(img, stack, tmp_path):
    export_display(img, tmp_path, LIMITS, LUTS, gamma={"DRAQ7": 2.0})
    _, draq7 = read_png(tmp_path / "PL1_live_P3-B3_T000_DRAQ7_Z000.png")
    _, ao = read_png(tmp_path / "PL1_live_P3-B3_T000_AO_Z000.png")
    np.testing.assert_array_equal(
        draq7, apply_lut(to_uint8(stack[0, 1, 0], 20, 150, gamma=2.0), MAGENTA)
    )
    # AO has no gamma entry -> 1.0
    np.testing.assert_array_equal(
        ao, apply_lut(to_uint8(stack[0, 0, 0], 0, 191), GREEN)
    )


def test_display_float_limits_from_napari(img, tmp_path):
    limits = {"AO": (0.0, 191.0), "DRAQ7": (20.5, 150.25)}
    assert len(export_display(img, tmp_path, limits, LUTS)) == 4


def test_display_t_labels(img, tmp_path):
    names = {
        p.name for p in export_display(img, tmp_path, LIMITS, LUTS, t_labels=[0, 3])
    }
    assert "PL1_live_P3-B3_T003_AO_Z000.png" in names
    assert not any("T001" in n for n in names)


def test_display_t_labels_wrong_length_raises(img, tmp_path):
    with pytest.raises(ValueError, match="t_labels"):
        export_display(img, tmp_path, LIMITS, LUTS, t_labels=[0])


def test_display_extra_names_ignored(img, tmp_path):
    limits = {**LIMITS, "PI": (0, 10)}
    luts = {**LUTS, "PI": GRAY}
    assert len(export_display(img, tmp_path, limits, luts)) == 4


def test_display_picks_the_requested_z(tmp_path):
    arr = np.zeros((1, 1, 3, 4, 4), dtype=np.uint16)
    arr[0, 0, 2] = 100  # only Z = 2 is bright
    img3 = Image(
        data=da.from_array(arr),
        dims=("T", "C", "Z", "Y", "X"),
        channel_names=["AO"],
        source="s.czi",
        scene="A1",
    )
    (path,) = export_display(img3, tmp_path, {"AO": (0, 100)}, {"AO": GRAY}, z=2)
    assert path.name == "s_A1_T000_AO_Z002.png"
    assert read_png(path)[1].min() == 255


@pytest.mark.parametrize("z", [1, -1])
def test_display_z_out_of_range_raises(img, tmp_path, z):
    with pytest.raises(ValueError, match="z"):
        export_display(img, tmp_path, LIMITS, LUTS, z=z)


def test_display_missing_limits_raises(img, tmp_path):
    with pytest.raises(ValueError, match="DRAQ7"):
        export_display(img, tmp_path, {"AO": (0, 191)}, LUTS)


def test_display_missing_lut_raises(img, tmp_path):
    with pytest.raises(ValueError, match="DRAQ7"):
        export_display(img, tmp_path, LIMITS, {"AO": GREEN})


@pytest.mark.parametrize(
    "limits, luts, gamma",
    [
        ({"AO": (0, 191), "DRAQ7": (150, 150)}, LUTS, None),  # lo == hi on the last
        (LIMITS, {"AO": GREEN, "DRAQ7": GRAY[:255]}, None),  # bad LUT on the last
        (LIMITS, LUTS, {"DRAQ7": 0}),  # bad gamma on the last
    ],
)
def test_display_nothing_written_when_input_is_bad(img, tmp_path, limits, luts, gamma):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        export_display(img, out, limits, luts, gamma=gamma)
    assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------- crop_record

CROP_BOX = (slice(1, 5), slice(2, 8))  # 4 x 6 px at y=1, x=2


@pytest.fixture
def crop(img):
    return crop_image(img, CROP_BOX, list(NAMES), [0, 1])


def test_record_source_scene_and_pixel_size(crop):
    rec = crop_record(crop, CROP_BOX, t_labels=[0, 3])
    assert rec["source"] == "PL1_live.czi"
    assert rec["scene"] == "P3-B3"
    assert rec["pixel_size_um"] == {"z": None, "y": 0.4536, "x": 0.4536}


def test_record_box_in_pixels(crop):
    rec = crop_record(crop, CROP_BOX, t_labels=[0, 3])
    assert rec["box_px"] == {"y": 1, "x": 2, "height": 4, "width": 6}


def test_record_box_in_um(crop):
    box = crop_record(crop, CROP_BOX, t_labels=[0, 3])["box_um"]
    assert box["y"] == pytest.approx(0.4536)
    assert box["x"] == pytest.approx(0.9072)
    assert box["height"] == pytest.approx(1.8144)
    assert box["width"] == pytest.approx(2.7216)


def test_record_box_in_um_is_none_without_pixel_size(crop):
    crop.pixel_size = (None, None, None)
    assert crop_record(crop, CROP_BOX, t_labels=[0, 3])["box_um"] is None


def test_record_times_and_z(crop):
    rec = crop_record(crop, CROP_BOX, t_labels=[0, 3], z=0)
    assert rec["times"] == [0, 3]
    assert rec["z"] == 0


def test_record_default_times(crop):
    assert crop_record(crop, CROP_BOX, t_labels=None)["times"] == [0, 1]


def test_record_channels_with_display_settings(crop):
    rec = crop_record(
        crop,
        CROP_BOX,
        t_labels=[0, 3],
        limits=LIMITS,
        luts=LUTS,
        gamma={"DRAQ7": 2.0},
    )
    assert list(rec["channels"]) == ["AO", "DRAQ7"]  # file order
    assert rec["channels"]["AO"] == {
        "limits": [0, 191],
        "gamma": 1.0,
        "rgb": [0, 255, 0],
    }
    assert rec["channels"]["DRAQ7"] == {
        "limits": [20, 150],
        "gamma": 2.0,
        "rgb": [255, 0, 255],
    }


def test_record_raw_only_has_no_display_settings(crop):
    rec = crop_record(crop, CROP_BOX, t_labels=[0, 3])
    assert rec["channels"]["AO"] == {"limits": None, "gamma": 1.0, "rgb": None}


def test_record_missing_channel_in_limits_raises(crop):
    with pytest.raises(ValueError, match="DRAQ7"):
        crop_record(crop, CROP_BOX, t_labels=[0, 3], limits={"AO": (0, 191)}, luts=LUTS)


def test_record_numpy_values_become_plain_python(crop):
    # what really comes in: numpy box ends, float32 napari limits, uint8 LUT rows
    box = (slice(np.int64(1), np.int64(5)), slice(np.int64(2), np.int64(8)))
    limits = {
        "AO": np.array([0.0, 191.5], dtype=np.float32),
        "DRAQ7": (np.float32(20), np.float32(150)),
    }
    rec = crop_record(crop, box, t_labels=np.array([0, 3]), limits=limits, luts=LUTS)
    json.dumps(rec)  # raises TypeError on any numpy value
    assert type(rec["box_px"]["y"]) is int
    assert type(rec["channels"]["AO"]["limits"][1]) is float
    assert type(rec["channels"]["AO"]["rgb"][1]) is int
    assert type(rec["times"][1]) is int


def test_record_path_source_becomes_string(crop):
    from pathlib import Path

    crop.source = Path("/data/OGD_but/PL1_live.czi")
    rec = crop_record(crop, CROP_BOX, t_labels=[0, 3])
    assert rec["source"] == "/data/OGD_but/PL1_live.czi"


def test_record_box_not_matching_crop_raises(crop):
    with pytest.raises(ValueError, match="box"):
        crop_record(
            crop, (slice(1, 6), slice(2, 8)), t_labels=[0, 3]
        )  # 5 tall, crop is 4


def test_record_t_labels_wrong_length_raises(crop):
    with pytest.raises(ValueError, match="t_labels"):
        crop_record(crop, CROP_BOX, t_labels=[0])


# ---------------------------------------------------------------- crop.json


def test_json_round_trip(crop, tmp_path):
    rec = crop_record(crop, CROP_BOX, t_labels=[0, 3], limits=LIMITS, luts=LUTS)
    path = tmp_path / "crop.json"
    write_crop_json(rec, path)
    assert read_crop_json(path) == rec


def test_json_is_readable_text(crop, tmp_path):
    path = tmp_path / "crop.json"
    write_crop_json(crop_record(crop, CROP_BOX, t_labels=[0, 3]), path)
    text = path.read_text()
    assert text.count("\n") > 5  # indented, one key per line
    assert '"source": "PL1_live.czi"' in text


def test_json_creates_missing_folder(crop, tmp_path):
    path = tmp_path / "a" / "b" / "crop.json"
    write_crop_json(crop_record(crop, CROP_BOX, t_labels=[0, 3]), path)
    assert path.exists()


# ---------------------------------------------------------------- CropExport


def settings(**changes):
    base = dict(
        box=CROP_BOX,
        channels=("AO", "DRAQ7"),
        times=(1,),
        z=0,
        raw=True,
        display=True,
        limits=LIMITS,
        luts=LUTS,
    )
    base.update(changes)
    return CropExport(**base)


def test_settings_are_frozen():
    s = settings()
    with pytest.raises(AttributeError):
        s.z = 1


def test_settings_need_raw_or_display():
    with pytest.raises(ValueError, match="raw"):
        settings(raw=False, display=False)


@pytest.mark.parametrize("missing", ["limits", "luts"])
def test_settings_display_needs_limits_and_luts(missing):
    with pytest.raises(ValueError, match=missing):
        settings(**{missing: None})


def test_settings_raw_only_needs_no_display_settings():
    s = settings(display=False, limits=None, luts=None)
    assert s.raw and not s.display


def test_settings_gamma_defaults_to_none():
    assert settings().gamma is None


# ---------------------------------------------------------------- crop_folder_name


def test_crop_folder_name(img):
    assert crop_folder_name(img, CROP_BOX) == "PL1_live_P3-B3_y1_x2_4x6"


def test_crop_folder_name_differs_per_region(img):
    other = (slice(0, 4), slice(2, 8))
    assert crop_folder_name(img, CROP_BOX) != crop_folder_name(img, other)


# ---------------------------------------------------------------- export_crop


def test_export_crop_writes_raw_display_and_json(img, tmp_path):
    folder = export_crop(img, settings(), tmp_path / "c")
    assert folder == tmp_path / "c"
    assert sorted(p.name for p in (folder / "raw").iterdir()) == [
        "PL1_live_P3-B3_T001_AO_Z000.ome.tif",
        "PL1_live_P3-B3_T001_DRAQ7_Z000.ome.tif",
    ]
    assert sorted(p.name for p in (folder / "display").iterdir()) == [
        "PL1_live_P3-B3_T001_AO_Z000.png",
        "PL1_live_P3-B3_T001_DRAQ7_Z000.png",
    ]
    assert (folder / "crop.json").exists()


def test_export_crop_raw_pixels_are_the_crop(img, stack, tmp_path):
    folder = export_crop(img, settings(), tmp_path / "c")
    plane = tifffile.imread(folder / "raw" / "PL1_live_P3-B3_T001_DRAQ7_Z000.ome.tif")
    np.testing.assert_array_equal(plane, stack[1, 1, 0, 1:5, 2:8])


def test_export_crop_raw_has_lut_colours(img, tmp_path):
    folder = export_crop(img, settings(), tmp_path / "c")
    ch = ome_channel(folder / "raw" / "PL1_live_P3-B3_T001_DRAQ7_Z000.ome.tif")
    assert ch.color.as_rgb_tuple() == (255, 0, 255)


def test_export_crop_display_matches_render(img, stack, tmp_path):
    folder = export_crop(img, settings(gamma={"AO": 2.0}), tmp_path / "c")
    _, rgb = read_png(folder / "display" / "PL1_live_P3-B3_T001_AO_Z000.png")
    plane = stack[1, 0, 0, 1:5, 2:8]
    np.testing.assert_array_equal(rgb, apply_lut(to_uint8(plane, 0, 191, 2.0), GREEN))


def test_export_crop_json_describes_the_export(img, tmp_path):
    folder = export_crop(img, settings(gamma={"AO": 2.0}), tmp_path / "c")
    rec = read_crop_json(folder / "crop.json")
    assert rec["box_px"] == {"y": 1, "x": 2, "height": 4, "width": 6}
    assert rec["times"] == [1]
    assert rec["channels"]["AO"] == {
        "limits": [0, 191],
        "gamma": 2.0,
        "rgb": [0, 255, 0],
    }


def test_export_crop_only_selected_channels(img, tmp_path):
    folder = export_crop(img, settings(channels=("DRAQ7",)), tmp_path / "c")
    assert all("DRAQ7" in p.name for p in (folder / "raw").iterdir())
    assert list(read_crop_json(folder / "crop.json")["channels"]) == ["DRAQ7"]


def test_export_crop_raw_only(img, tmp_path):
    s = settings(display=False, limits=None, luts=None)
    folder = export_crop(img, s, tmp_path / "c")
    assert (folder / "raw").exists()
    assert not (folder / "display").exists()
    assert read_crop_json(folder / "crop.json")["channels"]["AO"]["limits"] is None


def test_export_crop_display_only(img, tmp_path):
    folder = export_crop(img, settings(raw=False), tmp_path / "c")
    assert not (folder / "raw").exists()
    assert (folder / "display").exists()


def test_export_crop_existing_folder_raises(img, tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    (folder / "old.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="c"):
        export_crop(img, settings(), folder)
    assert [p.name for p in folder.iterdir()] == ["old.txt"]


def test_export_crop_empty_existing_folder_is_fine(img, tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    export_crop(img, settings(), folder)
    assert (folder / "crop.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"limits": {"AO": (0, 191), "DRAQ7": (150, 150)}},  # lo == hi
        {"limits": {"AO": (0, 191)}},  # DRAQ7 missing
        {"gamma": {"DRAQ7": -1.0}},
        {"channels": ("AO", "PI")},  # unknown channel
        {"times": (7,)},  # out of range
    ],
)
def test_export_crop_nothing_written_when_input_is_bad(img, tmp_path, changes):
    folder = tmp_path / "c"
    with pytest.raises(ValueError):
        export_crop(img, settings(**changes), folder)
    assert not folder.exists() or not any(folder.rglob("*"))
