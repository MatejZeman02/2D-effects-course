"""Runs lesson 1's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_01.py

The kernels need Sara and are not run here. What runs is the NumPy half: the
helpers cell, the given cells unfinished and the solutions, on the shipped
test picture, with PARAMS read by the Krita template's own parser.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "krita" / "pga_filter"))

import lesson_01_notebook as nb  # noqa: E402
import params  # noqa: E402


def namespace(*sources):
    space = {"np": np}
    for source in sources:
        exec(source, space)
    return space


@pytest.fixture(scope="module")
def balls():
    rgba = plt.imread(ROOT / "imgs" / "1" / "balls.png").astype(np.float32)
    return rgba


@pytest.fixture(scope="module")
def defaults():
    return params.defaults(params.parse(namespace(nb.HELPERS_NP)["PARAMS"]))


def test_the_template_reads_params(defaults):
    assert set(defaults) == {"dark", "mid", "light", "mid_at", "steps", "dither"}
    assert isinstance(defaults["steps"], int) and defaults["dither"] == "Bayer"


def test_the_goal_colours_are_the_params():
    """Each colour picker of the goal holds its PARAMS colour to four decimals, and mid_at the same slider."""
    listed = namespace(nb.HELPERS_NP)["PARAMS"]
    for name in ("dark", "mid", "light"):
        rgb = [int(listed[name][i:i + 2], 16) / 255 for i in (1, 3, 5)]
        assert f"uniform vec3 {name}: source_color = vec3({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})" in nb.GOAL
    default, low, high = listed["mid_at"]
    assert f"uniform float mid_at: hint_range({low}, {high}) = {default}" in nb.GOAL


def test_the_gradient_passes_through_its_three_colours(defaults):
    space = namespace(nb.HELPERS_NP)
    for mid_at in (0.3, 0.5, 0.8):
        t = np.array([[0.0, mid_at, 1.0]], dtype=np.float32)
        colours = space["gradient"](t, defaults["dark"], defaults["mid"], defaults["light"], mid_at)[0]
        for got, name in zip(colours, ("dark", "mid", "light")):
            assert np.allclose(got, defaults[name], atol=1e-3), (mid_at, name)


def test_the_dither_labels_agree():
    """The goal's hint_enum and PARAMS list the same labels, and both open on Bayer."""
    labels = [label.strip(' "') for label in nb.DITHERS.split(",")]
    listed = namespace(nb.HELPERS_NP)["PARAMS"]["dither"]
    assert sorted(listed) == sorted(labels) and listed[0] == "Bayer"
    default = labels.index("Bayer")
    assert f"hint_enum({nb.DITHERS}) = {default}" in nb.GOAL
    for i, constant in enumerate(["OFF", "WHITE", "BAYER", "IGN"]):
        assert f"const int {constant} = {i}" in nb.GOAL


def test_unfinished_cells_are_harmless(balls, defaults):
    space = namespace(nb.HELPERS_NP, nb.APPLY_NP_GIVEN)
    assert np.array_equal(space["apply"](balls, **defaults), balls)


@pytest.mark.parametrize("dither", ["Bayer", "White noise", "IGN", "Off"])
def test_the_solution_runs(balls, defaults, dither, tmp_path_factory):
    space = namespace(nb.HELPERS_NP, nb.SOLUTION_APPLY_NP)
    out = space["apply"](balls, **{**defaults, "dither": dither})
    assert out.shape == balls.shape and np.isfinite(out).all()
    assert np.array_equal(out[..., 3], balls[..., 3])
    # Five levels give at most five colours.
    colours = np.unique(out[..., :3].reshape(-1, 3).round(4), axis=0)
    assert len(colours) <= 5
    # Black lands on dark, white on light.
    ramp = np.zeros((1, 2, 4), np.float32)
    ramp[0, 1, :3] = 1.0
    ramp[..., 3] = 1.0
    ends = space["apply"](ramp, **{**defaults, "dither": "Off"})[0, :, :3]
    assert np.allclose(ends[0], defaults["dark"], atol=1e-3)
    assert np.allclose(ends[1], defaults["light"], atol=1e-3)
    plt.imsave(tmp_path_factory.mktemp("lesson_01") / f"goal_{dither}.png", np.clip(out, 0, 1))


def test_the_blue_noise_texture_is_even_and_has_no_clumps():
    """Every threshold is about as common, and the texture holds less of the low frequencies, the
    clumps, than white noise does."""
    blue = plt.imread(ROOT / "imgs" / "1" / "blue_noise.png")[..., 0]
    assert blue.shape == (64, 64)
    counts = np.histogram(blue, bins=8, range=(0, 1))[0] / blue.size
    assert abs(counts - 0.125).max() < 0.01, counts

    def low(d):
        power = np.abs(np.fft.fftshift(np.fft.fft2(d - d.mean()))) ** 2
        return power[28:37, 28:37].sum() / power.sum()

    white = np.random.default_rng(0).random((64, 64))
    assert low(blue) < low(white) / 10


def test_hash21_is_the_kernel_hash():
    """The NumPy hash21 follows sa_hash21 line by line, checked on single pixels in float32."""
    f = np.float32
    hash21 = namespace(nb.HELPERS_NP)["hash21"]

    def fract(x):
        return x - np.floor(x)

    for x, y in [(0, 0), (1, 0), (0, 1), (17, 3), (1919, 1079), (640, 360)]:
        q = fract(np.array([x, y, x], dtype=f) * f(0.1031))
        q = q + np.dot(q, q[[1, 2, 0]] + f(33.33))
        expected = fract((q[0] + q[1]) * q[2])
        assert abs(hash21(np.array(x), np.array(y)) - expected) < 1e-6, (x, y)


@pytest.mark.parametrize("name", ["White noise", "IGN"])
def test_the_noises_are_even_thresholds(name):
    """Both noises stay in 0 to 1 and every tenth of the range is about as common."""
    space = namespace(nb.HELPERS_NP)
    yy, xx = np.mgrid[0:256, 0:256]
    if name == "White noise":
        d = space["hash21"](xx, yy)
    else:
        d = space["fract"](52.9829189 * space["fract"](0.06711056 * xx + 0.00583715 * yy))
    assert d.min() >= 0 and d.max() < 1
    counts = np.histogram(d, bins=10, range=(0, 1))[0] / d.size
    assert abs(counts - 0.1).max() < 0.01, counts


def test_dither_keeps_the_mean(defaults):
    """A dithered ramp averages to the ramp, a rounded one does not."""
    space = namespace(nb.HELPERS_NP)
    ramp = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (64, 1))
    yy, xx = np.mgrid[0:64, 0:256]
    d = (space["BAYER"][yy % 4, xx % 4] + 0.5) / 16
    dithered = np.floor(ramp + d)
    rounded = np.floor(ramp + 0.5)

    def blocks(a):  # the mean of every 4 x 4 tile, where the matrix holds all sixteen thresholds
        return a.reshape(16, 4, 64, 4).mean(axis=(1, 3))

    assert abs(blocks(dithered) - blocks(ramp)).max() < 0.07
    assert abs(blocks(rounded) - blocks(ramp)).max() > 0.45
