"""Runs lesson 3's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_03.py

The kernels need Sara and are not run here. What runs is the NumPy half: the
helpers cell, the given cells unfinished and the solutions, on the lesson's
photo, with PARAMS read by the Krita template's own parser. The kernels'
arithmetic is redone in NumPy where a test needs it: the 2D Gaussian with its
edges clamped, the edge rules' index formulas and Sobel's weights.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "krita" / "pga_filter"))

import lesson_03_diagrams as diagrams  # noqa: E402
import lesson_03_notebook as nb  # noqa: E402
import params  # noqa: E402

W, H = nb.WIDTH, nb.HEIGHT


def namespace(*sources):
    space = {"np": np}
    for source in sources:
        exec(source, space)
    return space


def code_cells():
    return ["".join(c["source"]) for c in nb.cells if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def photo():
    return diagrams.load("coffee.png")


@pytest.fixture(scope="module")
def defaults():
    return params.defaults(params.parse(namespace(nb.PARAMS_NP)["PARAMS"]))


@pytest.fixture(scope="module")
def solved():
    return namespace(nb.HELPERS_NP, nb.SOLUTION_BLUR_NP, nb.PARAMS_NP, nb.SOLUTION_APPLY)


def test_the_template_reads_params(defaults):
    assert defaults == {"mode": "Blur", "sigma": nb.SIGMA, "amount": 1.0}


def test_the_check_cell_uses_the_defaults(defaults):
    check = [c for c in code_cells() if 'sara.check("Goal"' in c][0]
    space = {}
    exec(check.split("sara.check")[0], space)
    assert space["defaults"] == defaults


def test_the_photo_is_the_canvas_size():
    assert Image.open(ROOT / nb.IMG / "coffee.png").size == (W, H)
    assert f"sara.init(width={W}, height={H})" in code_cells()[0]
    assert f"const ivec2 SIZE = ivec2({W}, {H})" in nb.EDGES


def test_the_matrix_starts_as_the_sharpen_the_check_uses():
    numbers = re.search(r"weights\[9\] = \{([^}]*)\}", nb.CONV)[1]
    assert [float(n) for n in numbers.split(",")] == nb.SHARPEN


def test_every_layer_is_made_before_it_is_read():
    """Run top to bottom, no cell reads a layer that no cell above it made."""
    made = set()
    for cell in code_cells():
        first = cell.splitlines()[0]
        reads = re.findall(r'^%%gmacs "([^"]+)"', first)
        reads += re.findall(r'(?:doc\.layer|sara\.check|sara\.bench)\("([^"]+)"', cell)
        reads += re.findall(r'source="([^"]+)"', cell)
        for name in reads:
            assert name in made, f"{name} is read before any cell makes it"
        made |= set(re.findall(r'import_image\("imgs/3/([^"]+)\.png"\)', cell))
        made |= set(re.findall(r'(?:new_layer\(|-> |target=)"([^"]+)"', cell))


def test_the_blur_is_the_2d_kernel(photo, solved):
    """blur, rows then columns, is the 2D kernel's weighted square of 3 sigma with the edges clamped."""
    sigma = 1.5
    r = int(np.ceil(3 * sigma))
    lin = solved["srgb_to_linear"](photo[:120, :160, :3].astype(np.float64))
    p = np.pad(lin, ((r, r), (r, r), (0, 0)), mode="edge")
    total, weight = np.zeros_like(lin), 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            total += k * p[r + dy:r + dy + 120, r + dx:r + dx + 160]
            weight += k
    assert np.abs(solved["blur"](lin, sigma) - total / weight).max() < 1e-9


def test_the_edge_rules_are_numpys_pads():
    """The solution's index formulas for Mirror and Wrap, done on integers, are np.pad's symmetric and wrap.

    Mirror reflects once, which covers every shift the sliders allow, up to a canvas beyond each edge."""
    size = 7
    p = np.arange(-size, 2 * size)
    mirror = np.minimum(np.maximum(p, -1 - p), 2 * size - 1 - p)
    wrap = p - size * np.floor(p / size).astype(int)
    row = np.arange(size)
    far = np.arange(-20, 27)
    assert np.array_equal(row[far - size * np.floor(far / size).astype(int)], np.pad(row, (20, 20), mode="wrap"))
    assert np.array_equal(row[mirror], np.pad(row, (size, size), mode="symmetric"))
    assert np.array_equal(row[wrap], np.pad(row, (size, size), mode="wrap"))
    assert "min(max(p, -1 - p), 2 * SIZE - 1 - p)" in nb.SOLUTION_FETCH
    assert "p - SIZE * ivec2(floor(vec2(p) / vec2(SIZE)))" in nb.SOLUTION_FETCH


def test_the_shift_check_is_a_roll(photo):
    """The kernel reads at - (dx, dy), so with Wrap the layer is the photo rolled by (dy, dx)."""
    dx, dy = nb.SHIFT
    rolled = np.roll(photo, (dy, dx), axis=(0, 1))
    assert np.array_equal(rolled[dy + 5, dx + 7], photo[5, 7])
    assert f"np.roll(photo.read(), ({dy}, {dx}), axis=(0, 1))" in "".join(code_cells())


def test_sobels_weights_are_the_matrices():
    gx = np.array([[i * (2 - abs(j)) for i in (-1, 0, 1)] for j in (-1, 0, 1)])
    gy = np.array([[j * (2 - abs(i)) for i in (-1, 0, 1)] for j in (-1, 0, 1)])
    assert np.array_equal(gx, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    assert np.array_equal(gy, gx.T)
    assert "vec2(i * (2 - abs(j)), j * (2 - abs(i)))" in nb.SOLUTION_SOBEL


def test_the_shared_row_fits_the_largest_sigma():
    assert "R_MAX = 24" in nb.SHARED and int(np.ceil(3 * 8.0)) == 24
    assert 'halo=32 sigma=3' in nb.SHARED.splitlines()[0]


def test_unfinished_cells_are_harmless(photo, defaults):
    space = namespace(nb.HELPERS_NP, nb.BLUR_NP_GIVEN, nb.PARAMS_NP, nb.APPLY_GIVEN)
    a = photo[..., :3]
    assert np.array_equal(space["blur"](a, 3.0), a)
    assert np.array_equal(space["apply"](photo, **defaults), photo)


def test_the_solution_runs(photo, defaults, solved):
    img = photo[:200, :300]
    for mode in ("Blur", "Sharpen"):
        out = solved["apply"](img, **{**defaults, "mode": mode})
        assert out.shape == img.shape and out.dtype == np.float32, mode
        assert np.isfinite(out).all() and np.array_equal(out[..., 3], img[..., 3]), mode
    same = solved["apply"](img, **{**defaults, "mode": "Sharpen", "amount": 0.0})
    assert np.abs(same - img).max() < 1e-5
    soft = solved["apply"](img, **defaults)
    sharp = solved["apply"](img, **{**defaults, "mode": "Sharpen"})
    detail = np.abs(np.diff(img[..., 1], axis=1)).mean()
    assert np.abs(np.diff(soft[..., 1], axis=1)).mean() < detail < np.abs(np.diff(sharp[..., 1], axis=1)).mean()


def test_the_goal_is_the_solution(photo, solved):
    """The hidden Goal cell runs the same filter, PARAMS and source as section 6."""
    called = {}

    class Sara:
        @staticmethod
        def live(apply, params, source, target):
            called.update(apply=apply, params=params, source=source, target=target)

    exec(nb.GOAL, {"np": np, "sara": Sara})
    assert called["source"] == "coffee" and called["target"] == "Goal"
    assert called["params"] == solved["PARAMS"]
    img = photo[:100, :140]
    for mode in ("Blur", "Sharpen"):
        values = {"mode": mode, "sigma": 2.5, "amount": 1.5}
        assert np.array_equal(called["apply"](img, **values), solved["apply"](img, **values)), mode


def test_the_pictures_exist():
    """Every picture the notebook shows or opens is in imgs/3, and the copied photo is credited."""
    shown = set()
    for cell in nb.cells:
        text = "".join(cell["source"])
        shown |= set(re.findall(r'src="(imgs/3/[^"]+)"', text))
        shown |= set(re.findall(r'"(imgs/3/[^"]+\.png)"', text))
    assert len(shown) >= 8, shown
    for path in shown:
        assert (ROOT / path).is_file(), path
    assert "`coffee.png`" in (ROOT / nb.IMG / "SOURCES.md").read_text()
