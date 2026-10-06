"""Runs lesson 2's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_02.py

The kernels need Sara and are not run here. What runs is the NumPy half: the
helpers cell, the given cells unfinished and the solutions, on a white canvas,
with PARAMS read by the Krita template's own parser. The kernels' constants and
sliders are compared with the NumPy ones as text.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "krita" / "pga_filter"))

import lesson_02_notebook as nb  # noqa: E402
import params  # noqa: E402

M32 = 0xFFFFFFFF


def namespace(*sources):
    space = {"np": np}
    for source in sources:
        exec(source, space)
    return space


@pytest.fixture(scope="module")
def canvas():
    return np.ones((1080, 1920, 4), np.float32)              # File > New in Sara


@pytest.fixture(scope="module")
def defaults():
    return params.defaults(params.parse(namespace(nb.HELPERS_NP)["PARAMS"]))


@pytest.fixture(scope="module")
def solved():
    return namespace(nb.HELPERS_NP, nb.SOLUTION_VALUE_NP, nb.FBM_NP, nb.SOLUTION_APPLY_NP)


def test_the_template_reads_params(defaults):
    assert set(defaults) == {"scale", "octaves", "sea", "relief", "sun"}
    assert isinstance(defaults["octaves"], int)


def test_the_sliders_agree():
    """Every uniform of the terrain kernel is a slider of PARAMS with the same range and default."""
    sliders = namespace(nb.HELPERS_NP)["PARAMS"]
    found = re.findall(r"uniform (float|int) (\w+): hint_range\(([\d.]+), ([\d.]+)\) = ([\d.]+)", nb.GOAL)
    assert [name for _, name, *_ in found] == list(sliders)
    for kind, name, low, high, default in found:
        if kind == "int":
            assert sliders[name] == (int(float(default)), int(float(low)), int(float(high))), name
        else:
            assert sliders[name] == (float(default), float(low), float(high)), name


def test_the_check_cell_uses_the_defaults(defaults):
    check = [c for c in nb.cells if c["cell_type"] == "code" and "sara.check(" in "".join(c["source"])][0]
    space = {}
    exec("".join(check["source"]).split("sara.check")[0], space)
    assert space["defaults"].keys() == defaults.keys()
    for name, value in defaults.items():
        assert np.allclose(space["defaults"][name], value), name


def test_the_stops_agree():
    """The kernel's STOP_AT and STOP_LAB are the NumPy stops to the printed four decimals."""
    space = namespace(nb.HELPERS_NP)
    at = re.search(r"float\[8\]\(([^)]*)\)", nb.STOPS_GMACS).group(1)
    assert np.allclose([float(v) for v in at.split(",")], space["STOP_AT"])
    lab = re.findall(r"vec3\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", nb.STOPS_GMACS)
    assert np.allclose(np.array(lab, dtype=np.float64), space["STOP_LAB"], atol=6e-5)
    assert np.all(np.diff(space["STOP_AT"]) > 0)


def test_the_hash_constants_agree():
    """The kernel's cell_hash and the NumPy one multiply and shift by the same numbers."""
    numbers = re.findall(r"\d{3,}|>> ?\d+", nb.HASH_GMACS.split("def random1")[0].replace("u", ""))
    helpers = nb.HELPERS_NP[nb.HELPERS_NP.index("def cell_hash"):nb.HELPERS_NP.index("def random1")]
    assert re.findall(r"\d{3,}|>> ?\d+", helpers) == numbers


def test_cell_hash_is_uint32_on_the_gpu():
    """The NumPy cell_hash wraps like GLSL's uint, negative cells included."""
    cell_hash = namespace(nb.HELPERS_NP)["cell_hash"]

    def reference(x, y):
        h = ((x & M32) * 1597334677 + (y & M32) * 3812015801) & M32
        h = ((h ^ (h >> 16)) * 2146121005) & M32
        h = ((h ^ (h >> 15)) * 2221713035) & M32
        return h ^ (h >> 16)

    cells = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (17, 3), (1919, 1079), (-31, 17), (40000, -40000)]
    got = cell_hash(np.array([x for x, _ in cells]), np.array([y for _, y in cells]))
    assert got.dtype == np.uint32
    assert [int(v) for v in got] == [reference(x, y) for x, y in cells]


def test_random1_is_even():
    """random1 stays in 0 to 1 and every tenth of the range is about as common."""
    random1 = namespace(nb.HELPERS_NP)["random1"]
    cy, cx = np.mgrid[0:256, 0:256]
    r = random1(cx, cy)
    assert r.min() >= 0 and r.max() < 1
    counts = np.histogram(r, bins=10, range=(0, 1))[0] / r.size
    assert abs(counts - 0.1).max() < 0.01, counts


def test_value_noise_is_smooth(solved):
    """The solved value noise meets random1 at the corners and changes without a jump between them."""
    value_noise, random1 = solved["value_noise"], solved["random1"]
    corners = np.arange(-2, 6)
    assert np.allclose(value_noise(corners.astype(np.float64), np.full(8, 3.0)), random1(corners, np.full(8, 3)))
    x = np.linspace(-2, 6, 8001)
    line = value_noise(x, np.full_like(x, 2.37))
    assert line.min() >= 0 and line.max() <= 1
    assert np.abs(np.diff(line)).max() < 0.003
    slope = np.diff(line) / np.diff(x)
    assert np.abs(np.diff(slope)).max() < 0.05          # the smooth curve leaves no kink at a corner


def test_fbm_stays_in_range(solved):
    y, x = np.mgrid[0:128, 0:128] / 40.0
    for octaves in (1, 4, 8):
        h = solved["fbm"](x, y, octaves)
        assert h.min() >= 0 and h.max() <= 1
    assert np.allclose(solved["fbm"](x, y, 1), solved["value_noise"](x, y))


def test_unfinished_cells_are_harmless(canvas, defaults):
    space = namespace(nb.HELPERS_NP, nb.VALUE_NP_GIVEN, nb.FBM_NP, nb.APPLY_NP_GIVEN)
    assert np.array_equal(space["apply"](canvas, **defaults), canvas)
    x = np.linspace(0.05, 3.95, 40)
    squares = space["value_noise"](x, np.full_like(x, 0.5))
    assert len(np.unique(squares)) == 4                  # one value a cell
    assert space["fbm"](x, x[:, None], 3).shape == (40, 40)


def test_the_solution_runs(canvas, defaults, solved, tmp_path_factory):
    out = solved["apply"](canvas, **defaults)
    assert out.shape == canvas.shape and out.dtype == np.float32 and np.isfinite(out).all()
    assert np.all(out[..., 3] == 1.0)
    assert out[..., :3].min() > -0.01 and out[..., :3].max() < 1.01
    # The defaults give both sea and land on the canvas.
    h, w = canvas.shape[:2]
    height = solved["fbm"]((np.arange(w)[None, :] + 0.5) / defaults["scale"],
                           (np.arange(h)[:, None] + 0.5) / defaults["scale"], defaults["octaves"])
    land = np.mean(height > defaults["sea"])
    assert 0.1 < land < 0.9, land
    plt.imsave(tmp_path_factory.mktemp("lesson_02") / "goal.png", np.clip(out, 0, 1))


def test_the_sun_lights_the_slope_facing_it(defaults):
    """With the sun at 135 degrees, from the upper left, a slope rising to the right is the brightest."""

    def brightness(rise_x, rise_y, sun):
        space = namespace(nb.HELPERS_NP, nb.SOLUTION_VALUE_NP, nb.FBM_NP, nb.SOLUTION_APPLY_NP)
        sea = defaults["sea"]
        # a plane through the middle pixel, 0.0002 higher each pixel to the right and down
        space["fbm"] = lambda x, y, octaves: sea + 0.05 + 2e-4 * (rise_x * (x * 400 - 32) + rise_y * (y * 400 - 32))
        out = space["apply"](np.ones((64, 64, 4), np.float32), **{**defaults, "scale": 400.0, "sun": sun})
        return float(out[32, 32, :3].mean())

    flat = brightness(0, 0, 135.0)
    assert brightness(1, 0, 135.0) > flat > brightness(-1, 0, 135.0)
    assert brightness(0, 1, 135.0) > flat > brightness(0, -1, 135.0)     # rising downwards faces up
    assert brightness(-1, 0, 315.0) > flat > brightness(1, 0, 315.0)


def test_the_water_is_flat(defaults):
    """Under the sea every pixel gets the light of flat ground, whatever the bottom does."""
    space = namespace(nb.HELPERS_NP, nb.SOLUTION_VALUE_NP, nb.FBM_NP, nb.SOLUTION_APPLY_NP)
    sea = defaults["sea"]
    space["fbm"] = lambda x, y, octaves: sea - 0.1 - 0.05 * np.sin(x * 40) * np.cos(y * 30)
    out = space["apply"](np.ones((48, 48, 4), np.float32), **defaults)
    h, w = out.shape[:2]
    depth = space["fbm"]((np.arange(w)[None, :] + 0.5) / defaults["scale"],
                         (np.arange(h)[:, None] + 0.5) / defaults["scale"], 1) - sea
    lab = np.stack([np.interp(depth, space["STOP_AT"], space["STOP_LAB"][:, k]) for k in range(3)], axis=-1)
    flat_light = 0.25 + 0.75 / np.sqrt(2.0)                 # n = (0, 0, 1) and l 45 degrees up
    expected = space["linear_to_srgb"](space["lab_to_linear"](lab) * flat_light)
    assert np.allclose(out[..., :3], expected, atol=1e-5)


def test_the_pictures_exist():
    """Every picture the notebook shows is in imgs/2."""
    shown = set()
    for cell in nb.cells:
        shown |= set(re.findall(r'src="(imgs/2/[^"]+)"', "".join(cell["source"])))
    assert shown, "no pictures found"
    for path in shown:
        assert (ROOT / path).is_file(), path
