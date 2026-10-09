"""Runs lesson 6's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_06.py

The kernels need Sara and are not run here. What runs is the NumPy half: the
helpers cell, the given cells unfinished and the solutions, on the lesson's
HDR picture, with PARAMS read by the Krita template's own parser. The kernels'
arithmetic is redone in NumPy where a test needs it.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "krita" / "pga_filter"))

import lesson_06_diagrams as diagrams  # noqa: E402
import lesson_06_notebook as nb  # noqa: E402
import params  # noqa: E402

W, H = nb.WIDTH, nb.HEIGHT
CURVES = ["ACES", "Reinhard", "Clip"]


def namespace(*sources):
    space = {"np": np}
    for source in sources:
        exec(source, space)
    return space


def code_cells():
    return ["".join(c["source"]) for c in nb.cells if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def hdr():
    return diagrams.load_hdr()


@pytest.fixture(scope="module")
def img(hdr):
    return np.dstack([hdr, np.ones(hdr.shape[:2], np.float32)])


@pytest.fixture(scope="module")
def defaults():
    return params.defaults(params.parse(namespace(nb.PARAMS_NP)["PARAMS"]))


@pytest.fixture(scope="module")
def solved():
    return namespace(nb.HELPERS_NP, nb.CURVES_NP, nb.SOLUTION_AUTO_NP, nb.PARAMS_NP, nb.SOLUTION_APPLY)


def test_the_template_reads_params(defaults):
    assert defaults == {"auto": True, "key": nb.KEY, "stops": 0.0, "threshold": nb.THRESHOLD,
                        "sigma": nb.SIGMA, "strength": nb.STRENGTH, "curve": "ACES", "encode": True}


def test_the_check_cell_uses_the_defaults(defaults):
    check = [c for c in code_cells() if 'sara.check("Goal"' in c][0]
    space = {}
    exec(check.split("sara.check")[0], space)
    assert space["defaults"] == defaults


def test_the_picture_is_hdr_and_the_canvas_size(hdr):
    assert hdr.shape == (H, W, 3)
    assert f"sara.init(width={W}, height={H})" in code_cells()[0]
    assert hdr.min() >= 0 and hdr.max() > 50
    assert (hdr > 1).any(axis=-1).mean() < 0.01          # only the lamps are brighter than white


def test_every_layer_is_made_before_it_is_read():
    """Run top to bottom, no cell reads a layer that no cell above it made."""
    made = set()
    for cell in code_cells():
        first = cell.splitlines()[0]
        reads = re.findall(r'^%%gmacs "([^"]+)"', first) + re.findall(r' \w+=([A-Z]\w*)', first)
        reads += re.findall(r'(?:doc\.layer|sara\.check|sara\.bench|reduce)\("([^"]+)"', cell)
        reads += re.findall(r'source="([^"]+)"', cell)
        for name in reads:
            assert name in made, f"{name} is read before any cell makes it"
        made |= set(re.findall(r'(?:new_layer\(|-> |target=)"([^"]+)"', cell))


def test_every_variable_a_kernel_line_expands_is_set_above():
    """A {name} on a %%gmacs line is a Python variable an earlier cell sets."""
    seen = ""
    for cell in code_cells():
        first = cell.splitlines()[0]
        for name in re.findall(r"\{(\w+)\}", first) if first.startswith("%%gmacs") else ():
            assert re.search(rf"^{name} = ", seen, re.M), name
        seen += cell + "\n"


def test_the_curves_are_the_kernels(solved):
    x = np.logspace(-3, 3, 400)
    assert np.allclose(solved["reinhard"](x), x / (1 + x))
    aces = solved["aces"](x)
    assert aces.min() >= 0 and aces.max() == 1 and np.all(np.diff(aces) >= -1e-12)
    assert "x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)" in nb.SOLUTION_TONE
    assert "x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)" in nb.ACES_GLSL


def test_auto_exposure_brings_the_geometric_mean_to_the_key(hdr, solved):
    e = solved["auto_exposure"](hdr, 0.2)
    y = solved["luminance"](hdr * e)
    assert np.isclose(np.exp(np.log(1e-4 * e + y).mean()), 0.2, rtol=1e-4)
    assert np.isclose(solved["auto_exposure"](hdr * 4), solved["auto_exposure"](hdr) / 4, rtol=0.01)


def test_the_gamut_fit_keeps_hue_and_lands_inside(solved):
    rng = np.random.default_rng(6)
    c = rng.random((500, 3)) * 3
    fitted = diagrams.fit_numpy(c)
    assert fitted.min() >= 0 and fitted.max() <= 1
    inside = np.all(c <= 1, axis=-1)
    assert np.array_equal(fitted[inside], c[inside])
    lab, out = solved["oklab"](c[~inside]), solved["oklab"](fitted[~inside])
    coloured = np.hypot(out[:, 1], out[:, 2]) > 0.01
    hue = np.arctan2(lab[coloured, 2], lab[coloured, 1])
    hue_out = np.arctan2(out[coloured, 2], out[coloured, 1])
    assert np.abs(np.angle(np.exp(1j * (hue - hue_out)))).max() < 0.02


def test_unfinished_cells_are_harmless(img, defaults):
    space = namespace(nb.HELPERS_NP, nb.CURVES_NP, nb.AUTO_NP_GIVEN, nb.PARAMS_NP, nb.APPLY_GIVEN)
    assert space["auto_exposure"](img[..., :3]) == 1.0
    assert np.array_equal(space["apply"](img, **defaults), img)


def test_the_solution_runs(img, defaults, solved):
    crop = img[200:, :300]
    for curve in CURVES:
        for encode in (True, False):
            out = solved["apply"](crop, **{**defaults, "curve": curve, "encode": encode})
            assert out.shape == crop.shape and out.dtype == np.float32, curve
            assert np.isfinite(out).all() and np.array_equal(out[..., 3], crop[..., 3]), curve
            assert out[..., :3].min() >= 0 and out[..., :3].max() <= 1 + 1e-6, curve
    plain = solved["apply"](crop, **{**defaults, "auto": False, "strength": 0.0, "curve": "Clip"})
    assert np.allclose(plain[..., :3], solved["linear_to_srgb"](np.clip(crop[..., :3], 0, 1)), atol=1e-6)


def test_the_goal_is_the_solution(img, solved):
    """The hidden Goal cell runs the same tonemapper, PARAMS and source as section 5."""
    called = {}

    class Sara:
        @staticmethod
        def live(apply, params, source, target):
            called.update(apply=apply, params=params, source=source, target=target)

    exec(nb.GOAL, {"np": np, "sara": Sara})
    assert called["source"] == "HDR" and called["target"] == "Goal"
    assert called["params"] == solved["PARAMS"]
    crop = img[250:350, 100:240]
    values = {"auto": True, "key": 0.2, "stops": -1.0, "threshold": 1.0, "sigma": 3.0, "strength": 0.3,
              "curve": "Reinhard", "encode": True}
    assert np.array_equal(called["apply"](crop, **values), solved["apply"](crop, **values))


def test_the_pictures_exist():
    """Every picture the notebook shows or opens is in imgs/6, and the copied photo is credited."""
    shown = set()
    for cell in nb.cells:
        text = "".join(cell["source"])
        shown |= set(re.findall(r'src="(imgs/6/[^"]+)"', text))
        shown |= set(re.findall(r'"(imgs/6/[^"]+\.(?:png|npz))"', text))
    assert len(shown) >= 7, shown
    for path in shown:
        assert (ROOT / path).is_file(), path
    assert "`rocket_hdr.npz`" in (ROOT / nb.IMG / "SOURCES.md").read_text()


@pytest.fixture(scope="module")
def local():
    return namespace(nb.HELPERS_NP, nb.PYRAMID_NP, nb.SOLUTION_REMAP, nb.LOCAL_NP)


def test_the_pyramid_collapses_back_exactly(hdr, local):
    L = np.log2(1e-4 + local["luminance"](hdr))
    pyramid = local["laplacian_pyramid"](L, 7)
    assert [level.shape for level in pyramid][-1] == (7, 10)
    assert np.abs(local["collapse"](pyramid) - L).max() < 1e-4


def test_remap_is_continuous_and_the_check_expects_the_solution(local):
    remap = local["remap"]
    for alpha, beta in ((1.0, 1.0), (0.5, 0.4), (2.0, 0.1)):
        assert np.allclose(remap(np.array([1 - 1e-9, 1 + 1e-9, -1 - 1e-9]), 0.0, 1.0, alpha, beta), [1, 1, -1])
    x = np.linspace(-5, 5, 101)
    assert np.allclose(remap(x, 0.3, 0.7, 1.0, 1.0), x)
    check = [c for c in code_cells() if 'print("remap: ok"' in c][0]
    space = dict(local)
    exec(check.split("def remap")[0] + check.split("return collapse(out)")[1], space)
    assert np.allclose(space["got"], space["want"], atol=1e-3)


def test_the_unfinished_remap_leaves_the_picture(hdr, local):
    space = namespace(nb.HELPERS_NP, nb.PYRAMID_NP, nb.REMAP_GIVEN, nb.LOCAL_NP)
    L = np.log2(1e-4 + space["luminance"](hdr[::4, ::4]))
    assert np.abs(space["local_laplacian"](L, nb.SIGMA_R, 0.5, 0.4) - L).max() < 1e-4


def test_the_local_filter_compresses_the_range_and_keeps_neutral(hdr, local):
    L = np.log2(1e-4 + local["luminance"](hdr))
    same = local["local_laplacian"](L, nb.SIGMA_R, 1.0, 1.0)
    assert np.abs(same - L).max() < 1e-4
    out = local["local_laplacian"](L, nb.SIGMA_R, nb.ALPHA, nb.BETA)
    assert out.shape == L.shape and np.isfinite(out).all()
    assert np.ptp(out) < 0.7 * np.ptp(L)
