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
"""

import numpy as np
import pytest

from ferroscope.crop import crop_box, px_to_um, um_to_px

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
