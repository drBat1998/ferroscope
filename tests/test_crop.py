"""Tests for ferroscope.crop.

Step 1 — µm <-> pixel conversion:
- um_to_px rounds to the nearest pixel, exact halves round UP (not Python's
  banker's rounding).
- Unknown (None), zero or negative pixel size -> ValueError mentioning "pixel size".
- Zero or negative crop size -> ValueError mentioning "size".
- A size that rounds to 0 px -> ValueError mentioning "one pixel".
- Converting µm -> px -> µm lands within half a pixel of the start value.

Step 2 — crop_box(center_yx, size_yx, shape_yx) -> (slice_y, slice_x):
- napari convention: pixel i has its centre at coordinate i.
- start = floor(c - (h - 1) / 2 + 0.5), then shifted inside [0, shape - h].
- The box always keeps its size (shifted at edges, never clipped).
- Size larger than the image -> ValueError mentioning "larger".
- Zero or negative size -> ValueError mentioning "size".
- box -> rectangle centre -> box gives back the same box (no drift on redraw).

Step 3 — parse_indices(text, n) -> sorted list of unique ints:
- 0-based, like the napari slider (ZEN counts from 1, napari from 0).
- "0,5,9" single indices, "2-5" inclusive ranges, mixed freely.
- Spaces allowed around numbers and the dash, never between two digits ("1 2").
- "all" (any case) -> every index; empty text -> ValueError mentioning "empty".
- Out of range, backwards ranges and garbled pieces -> ValueError with the bad value.

Step 4 — crop_image(img, box, channels, times) -> new lazy Image:
- Only TCZYX images; anything else -> ValueError mentioning "TCZYX".
- Channels by name, kept in file order; unknown / duplicate / none -> ValueError.
- Times by index; out of range / none -> ValueError with the bad value.
- Box must lie inside Y and X and not be empty (dask would silently clip).
- All Z kept; data stays lazy; pixel_size, source, scene carried over.
- The original Image is never changed.
"""

import dask.array as da
import numpy as np
import pytest

from ferroscope.crop import crop_box, crop_image, parse_indices, px_to_um, um_to_px
from ferroscope.io import Image

# ---------------------------------------------------------------- um_to_px


@pytest.mark.parametrize(
    "size_um, pixel_um, expected",
    [
        (100.0, 0.5, 200),  # exact
        (100.0, 0.3, 333),  # 333.33 -> down
        (100.0, 0.6, 167),  # 166.67 -> up
        (50.0, 0.4536, 110),  # realistic pixel size: 110.23 -> down
    ],
)
def test_um_to_px_rounds_to_nearest(size_um, pixel_um, expected):
    assert um_to_px(size_um, pixel_um) == expected


@pytest.mark.parametrize(
    "size_um, expected",
    [
        (0.5, 1),  # banker's rounding would give 0
        (2.5, 3),  # banker's rounding would give 2
        (4.5, 5),  # banker's rounding would give 4
    ],
)
def test_um_to_px_half_rounds_up(size_um, expected):
    assert um_to_px(size_um, 1.0) == expected


def test_um_to_px_returns_int():
    result = um_to_px(100.0, 0.5)
    assert isinstance(result, int)


def test_um_to_px_unknown_pixel_size_raises():
    with pytest.raises(ValueError, match="pixel size"):
        um_to_px(100.0, None)


@pytest.mark.parametrize("pixel_um", [0.0, -0.4])
def test_um_to_px_zero_or_negative_pixel_raises(pixel_um):
    with pytest.raises(ValueError, match="pixel size"):
        um_to_px(100.0, pixel_um)


@pytest.mark.parametrize("size_um", [0.0, -10.0])
def test_um_to_px_zero_or_negative_size_raises(size_um):
    with pytest.raises(ValueError, match="size"):
        um_to_px(size_um, 0.5)


def test_um_to_px_below_one_pixel_raises():
    # 0.1 µm / 0.5 µm/px = 0.2 px -> rounds to 0
    with pytest.raises(ValueError, match="one pixel"):
        um_to_px(0.1, 0.5)


def test_um_to_px_error_message_contains_bad_value():
    with pytest.raises(ValueError, match="-0.4"):
        um_to_px(100.0, -0.4)


# ---------------------------------------------------------------- px_to_um


@pytest.mark.parametrize(
    "size_px, pixel_um, expected",
    [
        (200, 0.5, 100.0),
        (220, 0.4536, 99.792),
        (1, 0.4536, 0.4536),
    ],
)
def test_px_to_um_basic(size_px, pixel_um, expected):
    assert px_to_um(size_px, pixel_um) == pytest.approx(expected)


def test_px_to_um_unknown_pixel_size_raises():
    with pytest.raises(ValueError, match="pixel size"):
        px_to_um(200, None)


@pytest.mark.parametrize("pixel_um", [0.0, -0.4])
def test_px_to_um_zero_or_negative_pixel_raises(pixel_um):
    with pytest.raises(ValueError, match="pixel size"):
        px_to_um(200, pixel_um)


@pytest.mark.parametrize("size_px", [0, -5])
def test_px_to_um_zero_or_negative_size_raises(size_px):
    with pytest.raises(ValueError, match="size"):
        px_to_um(size_px, 0.5)


# ---------------------------------------------------------------- round trip


@pytest.mark.parametrize("size_um", [1.0, 13.7, 50.0, 100.0, 333.3, 1000.0])
@pytest.mark.parametrize("pixel_um", [0.1135, 0.2268, 0.4536, 0.9072, 1.0])
def test_round_trip_within_half_pixel(size_um, pixel_um):
    back = px_to_um(um_to_px(size_um, pixel_um), pixel_um)
    # tiny tolerance for floating-point error at exact halves
    assert abs(back - size_um) <= pixel_um / 2 + 1e-9


def test_px_to_um_error_message_contains_bad_value():
    with pytest.raises(ValueError, match="-5"):
        px_to_um(-5, 0.5)


# ---------------------------------------------------------------- crop_box

SHAPE = (100, 100)


def test_crop_box_odd_size_is_exactly_centred():
    # pixel 50 is the middle pixel of 45..55
    assert crop_box((50, 50), (11, 11), SHAPE) == (slice(45, 56), slice(45, 56))


def test_crop_box_even_size_is_half_a_pixel_off():
    # 10 px can't be centred on a pixel; pinned: 46..55 (one more pixel after 50)
    assert crop_box((50, 50), (10, 10), SHAPE) == (slice(46, 56), slice(46, 56))


def test_crop_box_float_centre():
    # y: 50.4 - 5 + 0.5 = 45.9 -> 45 ; x: 50.6 - 5 + 0.5 = 46.1 -> 46
    assert crop_box((50.4, 50.6), (11, 11), SHAPE) == (slice(45, 56), slice(46, 57))


def test_crop_box_one_pixel_follows_napari_pixel_centres():
    # 10.2 lies in pixel 10, 10.7 lies in pixel 11 (pixel i spans i-0.5 .. i+0.5)
    assert crop_box((10.2, 10.7), (1, 1), SHAPE) == (slice(10, 11), slice(11, 12))


def test_crop_box_shifted_inside_at_low_edge():
    assert crop_box((2, 50), (10, 10), SHAPE) == (slice(0, 10), slice(46, 56))


def test_crop_box_shifted_inside_at_high_edge():
    assert crop_box((98, 50), (10, 10), SHAPE) == (slice(90, 100), slice(46, 56))


def test_crop_box_non_square_shape_and_size():
    # y: 100 - 24.5 + 0.5 = 76 ; x: 95 - 9.5 + 0.5 = 86 -> shifted to 80
    box = crop_box((100, 95), (50, 20), (200, 100))
    assert box == (slice(76, 126), slice(80, 100))


@pytest.mark.parametrize(
    "center",
    [
        (0, 0),
        (0, 99),
        (99, 0),
        (99, 99),
        (50, 0),
        (-20, 50),  # centre outside the image
        (150, 150),  # far outside
        (50, 50),
    ],
)
def test_crop_box_always_keeps_size_and_stays_inside(center):
    size = (10, 20)
    sy, sx = crop_box(center, size, SHAPE)
    assert (sy.stop - sy.start, sx.stop - sx.start) == size
    assert sy.start >= 0 and sx.start >= 0
    assert sy.stop <= SHAPE[0] and sx.stop <= SHAPE[1]


def test_crop_box_size_equal_to_shape_gives_whole_image():
    assert crop_box((10, 80), SHAPE, SHAPE) == (slice(0, 100), slice(0, 100))


@pytest.mark.parametrize("size", [(101, 10), (10, 101)])
def test_crop_box_larger_than_image_raises(size):
    with pytest.raises(ValueError, match="larger"):
        crop_box((50, 50), size, SHAPE)


def test_crop_box_larger_message_contains_bad_value():
    with pytest.raises(ValueError, match="101"):
        crop_box((50, 50), (101, 10), SHAPE)


@pytest.mark.parametrize("size", [(0, 10), (10, 0), (-5, 10)])
def test_crop_box_zero_or_negative_size_raises(size):
    with pytest.raises(ValueError, match="size"):
        crop_box((50, 50), size, SHAPE)


def test_crop_box_returns_plain_ints():
    sy, sx = crop_box((50.4, 50.6), (11, 11), SHAPE)
    for value in (sy.start, sy.stop, sx.start, sx.stop):
        assert type(value) is int


def test_crop_box_slices_a_real_array_to_the_right_shape():
    image = np.zeros((100, 120))
    box = crop_box((30.3, 117.0), (25, 40), image.shape)
    assert image[box].shape == (25, 40)


@pytest.mark.parametrize("start", [0, 7, 45, 150])
@pytest.mark.parametrize("h", [1, 10, 11, 25])
def test_crop_box_is_stable_under_redraw(start, h):
    # the widget draws the rectangle from start-0.5 to start+h-0.5,
    # so its centre is start + (h - 1) / 2; reading it back must give the same box
    centre = start + (h - 1) / 2
    sy, sx = crop_box((centre, centre), (h, h), (200, 200))
    assert sy == slice(start, start + h)
    assert sx == slice(start, start + h)


# ---------------------------------------------------------------- parse_indices

N = 10


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3", [3]),
        ("0", [0]),  # 0-based: first time point
        ("9", [9]),  # last time point when n = 10
        ("0,5,9", [0, 5, 9]),
        ("2-5", [2, 3, 4, 5]),  # both ends included
        ("4-4", [4]),  # range of one
        ("0,5,7-9", [0, 5, 7, 8, 9]),
    ],
)
def test_parse_indices_basic(text, expected):
    assert parse_indices(text, N) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5,1,3", [1, 3, 5]),  # sorted
        ("2,2,2", [2]),  # duplicates removed
        ("1-4,3-6", [1, 2, 3, 4, 5, 6]),  # overlapping ranges merged
        ("7-9,0", [0, 7, 8, 9]),
    ],
)
def test_parse_indices_sorted_and_unique(text, expected):
    assert parse_indices(text, N) == expected


@pytest.mark.parametrize(
    "text",
    [" 0 , 5 - 7 ", "0,5-7", "0 ,5 -7", "\t0,5-7\n"],
)
def test_parse_indices_allows_spaces_around_numbers(text):
    assert parse_indices(text, N) == [0, 5, 6, 7]


@pytest.mark.parametrize("text", ["all", "ALL", " All "])
def test_parse_indices_all(text):
    assert parse_indices(text, N) == list(range(N))


@pytest.mark.parametrize("text", ["", "   "])
def test_parse_indices_empty_raises(text):
    with pytest.raises(ValueError, match="empty"):
        parse_indices(text, N)


@pytest.mark.parametrize("text, bad", [("10", "10"), ("8-12", "12"), ("0,99", "99")])
def test_parse_indices_out_of_range_raises(text, bad):
    with pytest.raises(ValueError, match=bad):
        parse_indices(text, N)


def test_parse_indices_backwards_range_raises():
    with pytest.raises(ValueError, match="5-2"):
        parse_indices("5-2", N)


@pytest.mark.parametrize(
    "text",
    [
        "a",
        "1 2",  # forgotten comma must NOT become 12
        "1,,2",
        "1,",
        ",1",
        "1-",
        "-1",  # negative
        "1--3",
        "1-2-3",
        "1.5",
        "+3",
        "²",  # Unicode digit
        "all,3",
    ],
)
def test_parse_indices_garbled_raises(text):
    with pytest.raises(ValueError):
        parse_indices(text, N)


def test_parse_indices_garbled_message_contains_piece():
    with pytest.raises(ValueError, match="abc"):
        parse_indices("0,abc,5", N)


@pytest.mark.parametrize("n", [0, -3])
def test_parse_indices_bad_n_raises(n):
    with pytest.raises(ValueError):
        parse_indices("0", n)


def test_parse_indices_returns_plain_int_list():
    result = parse_indices("0,3-5", N)
    assert isinstance(result, list)
    assert all(type(i) is int for i in result)


# ---------------------------------------------------------------- crop_image

NAMES = ["AO_red", "AO_green", "DRAQ7"]
BOX = (slice(4, 14), slice(10, 25))  # 10 x 15 px


@pytest.fixture
def plane_stack():
    # T=5, C=3, Z=2, Y=20, X=30; every pixel has a unique value
    return np.arange(5 * 3 * 2 * 20 * 30, dtype=np.uint16).reshape(5, 3, 2, 20, 30)


@pytest.fixture
def img(plane_stack):
    return Image(
        data=da.from_array(plane_stack, chunks=(1, 1, 1, 20, 30)),
        dims=("T", "C", "Z", "Y", "X"),
        channel_names=list(NAMES),
        pixel_size=(2.0, 0.4536, 0.4536),
        source="PL1_live.czi",
        scene="P3-B3",
    )


def test_crop_image_returns_image_with_right_shape(img):
    out = crop_image(img, BOX, ["AO_red", "DRAQ7"], [0, 3, 4])
    assert isinstance(out, Image)
    assert out.data.shape == (3, 2, 2, 10, 15)  # T, C, all Z, Y, X
    assert out.dims == ("T", "C", "Z", "Y", "X")


def test_crop_image_values_match_numpy(img, plane_stack):
    out = crop_image(img, BOX, ["AO_green", "DRAQ7"], [1, 4])
    expected = plane_stack[[1, 4]][:, [1, 2]][..., 4:14, 10:25]
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_crop_image_stays_lazy(img):
    out = crop_image(img, BOX, ["DRAQ7"], [0])
    assert isinstance(out.data, da.Array)


def test_crop_image_never_reads_a_huge_lazy_image():
    # ~160 GB if it were ever computed; cropping must only build the graph
    huge = Image(
        data=da.zeros((100, 4, 1, 20000, 20000), dtype=np.uint16, chunks=(1, 1, 1, 2048, 2048)),
        dims=("T", "C", "Z", "Y", "X"),
        channel_names=["a", "b", "c", "d"],
    )
    out = crop_image(huge, (slice(0, 100), slice(0, 100)), ["b"], [0, 99])
    assert out.data.shape == (2, 1, 1, 100, 100)


def test_crop_image_channels_kept_in_file_order(img, plane_stack):
    out = crop_image(img, BOX, ["DRAQ7", "AO_red"], [0])  # asked in reverse
    assert out.channel_names == ["AO_red", "DRAQ7"]
    expected = plane_stack[[0]][:, [0, 2]][..., 4:14, 10:25]
    np.testing.assert_array_equal(out.data.compute(), expected)


def test_crop_image_carries_metadata_over(img):
    out = crop_image(img, BOX, ["DRAQ7"], [2])
    assert out.pixel_size == (2.0, 0.4536, 0.4536)
    assert out.source == "PL1_live.czi"
    assert out.scene == "P3-B3"


def test_crop_image_does_not_change_the_original(img, plane_stack):
    crop_image(img, BOX, ["DRAQ7"], [2])
    assert img.data.shape == (5, 3, 2, 20, 30)
    assert img.channel_names == NAMES


def test_crop_image_channel_names_are_a_new_list(img):
    out = crop_image(img, BOX, list(NAMES), [0])
    out.channel_names.append("extra")
    assert img.channel_names == NAMES


def test_crop_image_all_channels_and_times_only_crops_yx(img, plane_stack):
    out = crop_image(img, BOX, list(NAMES), list(range(5)))
    np.testing.assert_array_equal(out.data.compute(), plane_stack[..., 4:14, 10:25])


def test_crop_image_unknown_channel_raises(img):
    with pytest.raises(ValueError, match="PI"):
        crop_image(img, BOX, ["PI"], [0])


def test_crop_image_unknown_channel_message_lists_available(img):
    with pytest.raises(ValueError, match="DRAQ7"):
        crop_image(img, BOX, ["PI"], [0])


def test_crop_image_duplicate_channel_raises(img):
    with pytest.raises(ValueError, match="AO_red"):
        crop_image(img, BOX, ["AO_red", "AO_red"], [0])


def test_crop_image_no_channels_raises(img):
    with pytest.raises(ValueError, match="channel"):
        crop_image(img, BOX, [], [0])


@pytest.mark.parametrize("times, bad", [([5], "5"), ([0, 7], "7"), ([-1], "-1")])
def test_crop_image_time_out_of_range_raises(img, times, bad):
    with pytest.raises(ValueError, match=bad):
        crop_image(img, BOX, ["DRAQ7"], times)


def test_crop_image_no_times_raises(img):
    with pytest.raises(ValueError, match="time"):
        crop_image(img, BOX, ["DRAQ7"], [])


@pytest.mark.parametrize(
    "box",
    [
        (slice(15, 25), slice(0, 10)),  # past the bottom (Y = 20)
        (slice(0, 10), slice(25, 35)),  # past the right (X = 30)
        (slice(-2, 8), slice(0, 10)),  # negative start
        (slice(5, 5), slice(0, 10)),  # empty
    ],
)
def test_crop_image_box_outside_or_empty_raises(img, box):
    with pytest.raises(ValueError, match="box"):
        crop_image(img, box, ["DRAQ7"], [0])


def test_crop_image_accepts_a_box_from_crop_box(img):
    box = crop_box((19.0, 29.0), (10, 15), (20, 30))  # bottom-right corner
    out = crop_image(img, box, ["DRAQ7"], [0])
    assert out.data.shape[-2:] == (10, 15)


def test_crop_image_wrong_dims_raises():
    other = Image(data=da.zeros((3, 20, 30), dtype=np.uint16), dims=("C", "Y", "X"))
    with pytest.raises(ValueError, match="TCZYX"):
        crop_image(other, BOX, ["a"], [0])


def test_crop_image_exports_with_existing_export_tiff_planes(img, tmp_path):
    from ferroscope.export import export_tiff_planes

    out = crop_image(img, BOX, ["AO_red", "DRAQ7"], [0, 3])
    written = export_tiff_planes(out, tmp_path)
    assert len(written) == 2 * 2 * 2  # T x C x Z
    assert all(p.exists() for p in written)
