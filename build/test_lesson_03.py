"""Runs lesson 3's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_03.py

The kernels need Sara and are not run here. What runs is the NumPy half: the
helpers cell, the given cells unfinished and the solutions, on the lesson's
photo, with PARAMS read by the Krita template's own parser. The kernels'
arithmetic is redone in NumPy where a test needs it: the 2D Gaussian with its
edges clamped, the edge rules' index formulas and Sobel's weights.
Sobel is a finished kernel, the lesson's tasks are the convolution, the first Gaussian
pass and the Krita copy, and the bilateral filter is a bonus.
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


def test_the_photo_is_the_canvas_size():
    assert Image.open(ROOT / nb.IMG / "coffee.png").size == (W, H)
    assert f"sara.init(width={W}, height={H})" in code_cells()[0]
    assert f"const ivec2 SIZE = ivec2({W}, {H})" in nb.EDGES


def test_the_matrix_starts_as_sharpen():
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


def test_sobels_weights_are_the_matrices():
    gx = np.array([[i * (2 - abs(j)) for i in (-1, 0, 1)] for j in (-1, 0, 1)])
    gy = np.array([[j * (2 - abs(i)) for i in (-1, 0, 1)] for j in (-1, 0, 1)])
    assert np.array_equal(gx, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    assert np.array_equal(gy, gx.T)
    assert "vec2(i * (2 - abs(j)), j * (2 - abs(i)))" in nb.SOBEL
    assert "TODO" not in nb.SOBEL


def test_the_kernels_reach_as_far_as_they_read():
    """A square of the grid at the far edge reads up to twice the largest radius past it, and the
    bilateral filter reads out to 3 sigma at its largest sigma."""
    radius = int(re.search(r"radius: hint_range\(0, (\d+)\)", nb.MEAN)[1])
    assert f"halo={2 * radius}" in nb.MEAN.splitlines()[0]
    sigma = float(re.search(r"sigma: hint_range\(0.5, (\d+)\)", nb.BILATERAL)[1])
    assert int(np.ceil(3 * sigma)) <= int(re.search(r"halo=(\d+)", nb.BILATERAL)[1])
    radius = int(re.search(r"radius: hint_range\(1, (\d+)\)", nb.SPACES)[1])
    assert f"halo={radius}" in nb.SPACES.splitlines()[0]


def test_the_enums_are_matched_by_name():
    for kernel, names in ((nb.MEAN, ["BLOCKS", "AROUND"]), (nb.EDGES, ["ZERO", "CLAMP", "MIRROR", "WRAP"])):
        for i, name in enumerate(names):
            assert f"const int {name} = {i}" in kernel
        assert "match " in kernel and " == 1" not in kernel
    for name in ("BLOCKS", "AROUND"):                   # every choice has its own case
        assert f"case {name}:" in nb.MEAN, name
    for name in ("BLUR", "SHARPEN", "SRGB", "LINEAR", "OKLAB"):
        assert f"case {name}:" in nb.SPACES, name
    for name in ("ZERO", "CLAMP", "MIRROR", "WRAP"):
        assert f"case {name}:" in nb.EDGES, name


def test_the_shared_row_fits_the_largest_sigma():
    assert "R_MAX = 24" in nb.SHARED and int(np.ceil(3 * 8.0)) == 24
    assert 'halo=32 sigma=3' in nb.SHARED.splitlines()[0]


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
    """The hidden Goal cell runs the same filter, PARAMS and source as section 5."""
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
    assert len(shown) >= 16, shown
    for path in shown:
        assert (ROOT / path).is_file(), path
    assert "`coffee.png`" in (ROOT / nb.IMG / "SOURCES.md").read_text()


def markdown_text():
    return "\n".join("".join(c["source"]) for c in nb.cells if c["cell_type"] == "markdown")


def test_the_krita_port_is_given_and_one_task_copies_it():
    """The NumPy cells run as they are, with no TODO, and the only task of section 5 is the copy."""
    cells = code_cells()
    assert not [c for c in cells if "TODO" in c and "def apply" in c]
    text = markdown_text()
    assert text.count("### 🎯 Úkol") == 3
    assert text.count("### 🎯 Bonus: bilaterální filtr") == 1
    assert "## 5. Doma: do Krity" in text and "Úkol 3: filtr v Kritě" in text
    assert "TITLE = \"Blur and sharpen\"" in text
    assert "def apply" in nb.SOLUTION_APPLY and "def blur" in nb.SOLUTION_BLUR_NP
    assert any("sara.live(apply, PARAMS" in c for c in cells)


def test_every_task_is_preceded_by_its_formula():
    """A $$ formula stands in its section above the task's details (the Krita copy needs none)."""
    seen = ""
    for cell in nb.cells:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        if text.startswith("## "):
            seen = ""
        if text.startswith("### 🎯") and "Úkol 3" not in text:
            assert "$$" in seen + text.split("<details>")[0], text.splitlines()[0]
        seen += text


def test_the_tasks_come_in_order_and_sobel_is_not_one():
    text = markdown_text()
    heads = re.findall(r"^### 🎯 (.*?):", text, flags=re.M)
    assert heads == ["Úkol 1", "Úkol 2", "Úkol 3", "Bonus"]
    sections = re.findall(r"^## (.*)$", text, flags=re.M)
    assert sections[:7] == ["Cíl lekce", "Jak byste to udělali?", "1. Rozmazání", "2. Konvoluce",
                            "3. Gauss ve dvou průchodech", "4. Hrany", "5. Doma: do Krity"]
    assert sections[7].startswith("6. ") and "(nepovinné)" in sections[7] and "7. " in sections[8]
    assert "Úkol 4" not in text and "Úkol 5" not in text and not hasattr(nb, "SOLUTION_SOBEL")
    assert "lekce 4" in text.split("## Shrnutí")[1]


def test_the_lesson_opens_with_a_brainstorm_after_the_goal():
    text = markdown_text()
    assert text.index("## Cíl lekce") < text.index("## Jak byste to udělali?") < text.index("## 1. Rozmazání")
    assert "A jak ve fotce najít hrany?" in text and "A jak rozmazat, aby hrany zůstaly?" in text
    assert text.index("❓ Další otázky: hrany a rozmazání") < text.index("A jak ve fotce najít hrany?")
    assert text.count("<details><summary>💭 Nápady</summary>") == 3
    assert "```" not in markdown_text().split("## 1. Rozmazání")[0]


def test_no_heading_hides_in_a_details_and_every_details_closes():
    for cell in nb.cells:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        assert text.count("<details") == text.count("</details>"), text[:60]
    open_ = 0
    for cell in nb.cells:
        if cell["cell_type"] != "markdown":
            continue
        fenced = False
        for line in "".join(cell["source"]).splitlines():
            fenced ^= line.startswith("```")
            if line.startswith("#") and not fenced:
                assert open_ == 0, line
            open_ += line.count("<details") - line.count("</details>")
    assert open_ == 0


def test_the_prose_has_no_semicolon_or_long_dash():
    for cell in nb.cells:
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            text = re.sub(r"\$[^$]*\$", "", text)
        assert "\u2014" not in text and "\u2013" not in text, text[:60]
        assert ";" not in text.replace("&nbsp;", ""), text[:60]


def test_the_goal_holds_no_unused_helper():
    assert "def srgb_to_linear" in nb.GOAL and "M2_INV" in nb.GOAL
    assert "def gaussian_1d" in nb.GOAL and "TODO" not in nb.GOAL
