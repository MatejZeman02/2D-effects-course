"""Krita's bytes to float RGBA and back, in plain Python: python -m pytest krita/tests"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pga_filter"))
import pixels  # noqa: E402

RED = np.array([[[1.0, 0.0, 0.0, 1.0]]], dtype=np.float32)


def test_8_bit_red_is_stored_as_bgra():
    assert pixels.to_bytes(RED, "U8") == bytes([0, 0, 255, 255])


def test_float_red_is_stored_as_rgba():
    data = pixels.to_bytes(RED, "F32")
    assert np.frombuffer(data, dtype=np.float32).tolist() == [1.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("depth, tolerance", [("U8", 0.5 / 255), ("U16", 0.5 / 65535),
                                              ("F16", 1e-3), ("F32", 0.0)])
def test_every_depth_survives_a_round_trip(depth, tolerance):
    rng = np.random.default_rng(1)
    img = rng.random((5, 7, 4), dtype=np.float32)
    back = pixels.to_float(pixels.to_bytes(img, depth), 7, 5, depth)
    assert back.shape == (5, 7, 4) and back.dtype == np.float32
    assert np.max(np.abs(back - img)) <= tolerance


def test_integer_depths_clip_and_float_depths_do_not():
    bright = np.full((1, 1, 4), 2.0, dtype=np.float32)
    assert pixels.to_float(pixels.to_bytes(bright, "U8"), 1, 1, "U8").max() == 1.0
    assert pixels.to_float(pixels.to_bytes(bright, "F32"), 1, 1, "F32").max() == 2.0


def test_blend_follows_the_selection():
    original = np.zeros((1, 3, 4), dtype=np.float32)
    result = np.ones((1, 3, 4), dtype=np.float32)
    mask = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    assert pixels.blend(original, result, mask)[0, :, 0].tolist() == [0.0, 0.5, 1.0]
    assert pixels.blend(original, result, None) is result


def test_check_result_names_the_mistake():
    shape = (2, 2, 4)
    with pytest.raises(pixels.PixelError, match="NumPy array"):
        pixels.check_result(None, shape)
    with pytest.raises(pixels.PixelError, match="shape"):
        pixels.check_result(np.zeros((2, 2, 3)), shape)
    with pytest.raises(pixels.PixelError, match="NaN"):
        pixels.check_result(np.full(shape, np.nan), shape)
    assert pixels.check_result(np.zeros(shape, dtype=np.float64), shape).dtype == np.float32


def test_downscale_averages_blocks_under_the_limit():
    img = np.zeros((100, 60, 4), dtype=np.float32)
    img[:50] = 1.0
    small = pixels.downscale(img, 25)
    assert max(small.shape[:2]) <= 25
    assert small[0, 0, 0] == 1.0 and small[-1, 0, 0] == 0.0


def test_centre_crop_and_display():
    img = np.zeros((10, 20, 4), dtype=np.float32)
    assert pixels.centre_crop(img, 4).shape == (4, 4, 4)
    shown = pixels.to_display(img)
    assert shown.shape == (10, 20, 3) and shown.dtype == np.uint8
    assert set(np.unique(shown)) == {153, 204}    # the checkerboard shows through
