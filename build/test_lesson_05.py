"""Runs lesson 5's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_05.py

The kernels need Sara and are not run here. What runs is the NumPy half, which
section 9 hands over finished: the helpers, the styles and `apply`, on the
lesson's photo, with PARAMS read by the Krita template's own parser. The kernels'
arithmetic is redone in NumPy where a test needs it.
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

import lesson_05_diagrams as diagrams  # noqa: E402
import lesson_05_notebook as nb  # noqa: E402
import params  # noqa: E402

W, H = nb.WIDTH, nb.HEIGHT
STYLES = ["Toon", "Ink", "Hatching", "Halftone"]


def namespace(*sources):
    space = {"np": np}
    for source in sources:
        exec(source, space)
    return space


def code_cells():
    return ["".join(c["source"]) for c in nb.cells if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def photo():
    return diagrams.load("chelsea.png")


@pytest.fixture(scope="module")
def defaults():
    return params.defaults(params.parse(namespace(nb.PARAMS_NP)["PARAMS"]))


@pytest.fixture(scope="module")
def solved():
    return namespace(nb.CONSTANTS_NP, nb.HELPERS_NP, nb.STYLES_NP, nb.PARAMS_NP, nb.APPLY_NP)


def test_the_template_reads_params(defaults):
    assert defaults == {"style": "Toon", "sigma": nb.SIGMA, "p": nb.P, "epsilon": nb.EPSILON,
                        "phi": nb.PHI, "levels": nb.LEVELS, "spacing": nb.SPACING, "angle": nb.ANGLE,
                        "on_photo": False, "cell": nb.CELL, "inks": "CMYK"}


def test_the_photo_is_the_canvas_size():
    assert Image.open(ROOT / nb.IMG / "chelsea.png").size == (W, H)
    assert f"sara.init(width={W}, height={H})" in code_cells()[0]


def test_every_layer_is_made_before_it_is_read():
    """Run top to bottom, no cell reads a layer that no cell above it made."""
    made = set()
    for cell in code_cells():
        first = cell.splitlines()[0]
        reads = re.findall(r'^%%gmacs "([^"]+)"', first) + (re.findall(r' \w+=([A-Z]\w*)', first) if first.startswith("%%gmacs") else [])
        reads += re.findall(r'(?:doc\.layer|sara\.bench)\("([^"]+)"', cell)
        reads += re.findall(r'source="([^"]+)"', cell)
        for name in reads:
            assert name in made, f"{name} is read before any cell makes it"
        made |= set(re.findall(r'import_image\("imgs/5/([^"]+)\.png"\)', cell))
        made |= set(re.findall(r'(?:new_layer\(|-> |target=)"([^"]+)"', cell))


def test_the_kernels_defaults_are_the_filters():
    """The sliders open where PARAMS and the check cells expect them."""
    assert f"sigma: hint_range(0.3, 4) = {nb.SIGMA}" in nb.BLURS and f"= {nb.K}" in nb.BLURS
    for name, value in (("p", nb.P), ("epsilon", nb.EPSILON), ("phi", nb.PHI)):
        assert f"uniform float {name}: hint_range" in nb.XDOG and f"= {value}" in nb.XDOG
    assert f"levels={nb.LEVELS}" in nb.TOON.splitlines()[0]
    assert f"= {nb.RADIUS}" in nb.KUWAHARA
    for name, value in (("spacing", nb.SPACING), ("angle", nb.ANGLE), ("thick", nb.THICK), ("lightest", nb.LIGHTEST)):
        assert f"uniform float {name}: hint_range" in nb.HATCH and f"= {value}\n" in nb.HATCH
    assert f"uniform float cell: hint_range(4, 24) = {nb.CELL}" in nb.HALFTONE


def test_the_kernels_and_numpy_share_their_constants(solved):
    """The hatching's turns and band and the halftone's screen angles are the same in the kernel and in NumPy."""
    assert "float[4](" + ", ".join(f"{a:.1f}" for a in solved["TURN"]) + ")" in nb.HATCH
    assert "float[4](" + ", ".join(f"{a:.1f}" for a in solved["SCREEN"]) + ")" in nb.HALFTONE
    assert f"BAND = {solved['BAND']}" in nb.HATCH and solved["LUMA"] == 4 and "const int LUMA = 4" in nb.HALFTONE
    assert solved["THICK"] == nb.THICK and solved["LIGHTEST"] == nb.LIGHTEST


def test_the_blurs_kernel_stops_each_gaussian_at_its_own_three_sigma(photo, solved):
    """The kernel's square sums, redone on a crop, are blur's rows and columns, the narrow one cut at its r1."""
    sigma, k = nb.SIGMA, nb.K
    L = solved["lightness"](photo[:60, :80]).astype(np.float64)
    r1, r2 = int(np.ceil(3 * sigma)), int(np.ceil(3 * k * sigma))
    p = np.pad(L, r2, mode="edge")
    totals, weights = [np.zeros_like(L), np.zeros_like(L)], [0.0, 0.0]
    for dy in range(-r2, r2 + 1):
        for dx in range(-r2, r2 + 1):
            shifted = p[r2 + dy:r2 + dy + 60, r2 + dx:r2 + dx + 80]
            for i, (s, r) in enumerate(((sigma, r1), (k * sigma, r2))):
                if max(abs(dx), abs(dy)) <= r:
                    w = np.exp(-(dx * dx + dy * dy) / (2 * s * s))
                    totals[i] += w * shifted
                    weights[i] += w
    for i, s in enumerate((sigma, k * sigma)):
        assert np.abs(totals[i] / weights[i] - solved["blur"](L[..., None], s)[..., 0]).max() < 1e-9


def test_oklab_goes_both_ways(photo, solved):
    lin = solved["srgb_to_linear"](photo[..., :3].astype(np.float64))
    assert np.abs(solved["oklab_to_linear"](solved["oklab"](lin)) - lin).max() < 1e-5
    white = solved["oklab"](np.ones(3))
    assert np.allclose(white, [1, 0, 0], atol=1e-4)


def test_xdog_is_white_paper_with_ink(photo, defaults, solved):
    L = solved["lightness"](photo)
    lines = solved["xdog"](L, nb.SIGMA, nb.P, nb.EPSILON, nb.PHI)
    assert lines.shape == L.shape and lines.min() >= 0 and lines.max() == 1
    paper = (lines == 1).mean()
    assert 0.5 < paper < 0.95, paper
    # with p = 0 and a sharp step it is a threshold of the blurred lightness
    flat = solved["xdog"](L, nb.SIGMA, 0.0, 0.5, 1e6)
    blurred = solved["blur"](L[..., None], nb.SIGMA)[..., 0]
    assert np.array_equal(flat == 1, blurred >= 0.5)


def test_toon_with_no_ink_and_many_levels_is_the_photo(photo, solved):
    toon = solved["toon"](photo, np.ones(photo.shape[:2]), 1000)
    assert np.abs(toon - photo[..., :3]).max() < 0.01
    black = solved["toon"](photo, np.zeros(photo.shape[:2]), nb.LEVELS)
    assert np.abs(black).max() < 1e-6
    levels = solved["toon"](photo, np.ones(photo.shape[:2]), 3)
    L = solved["lightness"](np.dstack([levels, np.ones(photo.shape[:2])]))
    assert set(np.unique(np.round(L, 1))) <= {0.0, 0.5, 1.0}     # three levels, a little off where clipped to the gamut


def test_kuwahara_keeps_a_clean_edge():
    """On a sharp edge every pixel finds a square on its own side, so the edge stays where it was."""
    img = np.zeros((20, 30, 4), np.float32)
    img[..., 3] = 1
    img[:, 15:, :3] = 1
    out = diagrams.kuwahara_numpy(img, 4)
    assert np.allclose(out, img[..., :3], atol=1e-6)


def test_four_kernel_tasks_have_stubs_and_solutions():
    """DoG, XDoG, Kuwahara's quadrant and the screen dot: each student cell holds its stub, the solved copy its solution."""
    tasks = [c for c in nb.cells if c.get("solution")]
    assert len(tasks) == 4
    for cell in tasks:
        assert "TODO" in "".join(cell["source"]) and "TODO" not in "".join(cell["solution"])


def test_the_krita_port_is_given_finished(photo, defaults):
    """Section 9 gives its two code cells finished: nothing in them is left to write."""
    given = [c for c in code_cells() if "def apply" in c or "def hatch" in c or "def srgb_to_linear" in c]
    assert len(given) == 3 and not any("TODO" in c for c in given)
    space = namespace(nb.CONSTANTS_NP, nb.HELPERS_NP, nb.STYLES_NP, nb.PARAMS_NP, nb.APPLY_NP)
    assert not np.array_equal(space["apply"](photo[:60, :80], **defaults), photo[:60, :80])


def test_hatching_gets_denser_where_it_is_darker(solved):
    """No ink on white, more with each step darker, and the four layers are all at work on black."""
    shade = lambda v: solved["hatch"](np.full((120, 160), v, np.float32), 6.0, 45.0)
    assert shade(1.0).min() == 1.0
    ink = [1 - shade(v).mean() for v in (0.95, 0.8, 0.6, 0.4, 0.2, 0.0)]
    assert ink == sorted(ink) and ink[0] < 0.02 and ink[-1] > 0.5, ink


def test_a_stripe_covers_its_width_and_a_zero_width_stripe_nothing(solved):
    x, y = solved["pixel_grid"](np.zeros((200, 200)))
    assert solved["stripes"](x, y, 8.0, 30.0, 0.0).max() == 0
    assert abs(solved["stripes"](x, y, 8.0, 30.0, 2.0).mean() - 2 / 8) < 0.02


def test_the_screen_dot_covers_the_share_it_is_asked_for(solved):
    cell = 20
    x, y = solved["pixel_grid"](np.zeros((cell, cell)))
    ox, oy = x / cell - 0.5, y / cell - 0.5
    assert solved["screen_dot"](ox, oy, 0.0, cell).max() == 0
    for c in (0.1, 0.3, 0.6):
        assert abs(solved["screen_dot"](ox, oy, c, cell).mean() - c) < 0.03, c


def test_cmyk_put_back_together_is_the_colour(solved):
    """(1 - C, 1 - M, 1 - Y) times (1 - K) is the linear colour the coverages came from."""
    lin = np.random.default_rng(1).uniform(0.02, 1, (30, 40, 3)).astype(np.float32)
    cmy = np.dstack([solved["coverage_of"](lin, c) for c in range(3)])
    k = solved["coverage_of"](lin, 3)
    assert np.abs((1 - cmy) * (1 - k)[..., None] - lin).max() < 1e-5


def test_halftone_keeps_the_mean_tone(solved):
    """Grey, white and black photos print as the same grey on average, white as paper."""
    def printed(v, inks):
        img = np.full((90, 120, 4), v, np.float32)
        return solved["halftone"](img, 9.0, inks)
    assert printed(1.0, "CMYK").min() > 0.999
    for inks in ("CMYK", "Black only"):
        grey = solved["srgb_to_linear"](np.float32(0.6))
        out = solved["srgb_to_linear"](printed(0.6, inks)).mean()
        assert abs(out - grey) < 0.06, (inks, out, grey)
    assert printed(0.0, "CMYK").mean() < 0.2


def test_the_solution_runs(photo, defaults, solved):
    img = photo[:200, :300]
    for style in STYLES:
        out = solved["apply"](img, **{**defaults, "style": style})
        assert out.shape == img.shape and out.dtype == np.float32, style
        assert np.isfinite(out).all() and np.array_equal(out[..., 3], img[..., 3]), style
        assert out[..., :3].min() >= 0 and out[..., :3].max() <= 1 + 1e-6, style
    lines = solved["apply"](img, **{**defaults, "style": "Ink"})
    assert np.array_equal(lines[..., 0], lines[..., 2])


def test_the_goal_is_the_solution(photo, solved):
    """The hidden Goal cell runs the same filter, PARAMS and source as section 9."""
    called = {}

    class Sara:
        @staticmethod
        def live(apply, params, source, target):
            called.update(apply=apply, params=params, source=source, target=target)

    exec(nb.GOAL, {"np": np, "sara": Sara})
    assert called["source"] == "chelsea" and called["target"] == "Goal"
    assert called["params"] == solved["PARAMS"]
    img = photo[:100, :140]
    for style in STYLES:
        values = {"style": style, "sigma": 2.0, "p": 30.0, "epsilon": 0.4, "phi": 20.0, "levels": 4,
                  "spacing": 5.0, "angle": 20.0, "on_photo": style == "Hatching", "cell": 7.0, "inks": "Black only"}
        assert np.array_equal(called["apply"](img, **values), solved["apply"](img, **values)), style


def test_the_pictures_exist():
    """Every picture the notebook shows or opens is in imgs/5, and the copied photo is credited."""
    shown = set()
    for cell in nb.cells:
        text = "".join(cell["source"])
        shown |= set(re.findall(r'src="(imgs/5/[^"]+)"', text))
        shown |= set(re.findall(r'"(imgs/5/[^"]+\.png)"', text))
    assert len(shown) >= 13, shown
    for path in shown:
        assert (ROOT / path).is_file(), path
    assert "`chelsea.png`" in (ROOT / nb.IMG / "SOURCES.md").read_text()


def test_the_prose_has_no_semicolons_or_long_dashes():
    for cell in nb.cells:
        text = "".join(cell["source"])
        assert "\u2014" not in text and "\u2013" not in text, text[:60]
        if cell["cell_type"] == "markdown":
            assert ";" not in text, text[:60]


def test_every_details_is_closed_and_has_no_heading():
    for cell in nb.cells:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        assert text.count("<details") == text.count("</details>"), text[:60]
