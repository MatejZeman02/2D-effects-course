"""Runs lesson 4's NumPy solutions as the notebook prints them.

    python -m pytest build/test_lesson_04.py

The kernels need Sara and are not run here. What runs is the NumPy half: the
helpers cell, the given cells unfinished and the solutions, on the lesson's
photos, with PARAMS read by the Krita template's own parser. The kernels'
arithmetic is redone in NumPy where a test needs it, the DFT's whole turns
dropped as integers and the blur's neighbours wrapped around the canvas.
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

import lesson_04_diagrams as diagrams  # noqa: E402
import lesson_04_notebook as nb  # noqa: E402
import params  # noqa: E402

N = nb.SIDE
MODES = ["Deblur", "Low pass", "High pass"]


def namespace(*sources):
    space = {"np": np}
    for source in sources:
        exec(source, space)
    return space


def code_cells():
    return ["".join(c["source"]) for c in nb.cells if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def photo():
    return diagrams.load("astronaut.png")


@pytest.fixture(scope="module")
def blurry():
    return diagrams.load("astronaut_blur.png")


@pytest.fixture(scope="module")
def defaults():
    return params.defaults(params.parse(namespace(nb.PARAMS_NP)["PARAMS"]))


@pytest.fixture(scope="module")
def solved():
    return namespace(nb.HELPERS_NP, nb.SOLUTION_SPECTRUM, nb.SOLUTION_GAUSSIAN, nb.SOLUTION_BLUR,
                     nb.SOLUTION_WIENER, nb.PARAMS_NP, nb.SOLUTION_APPLY)


def test_the_template_reads_params(defaults):
    assert defaults == {"mode": "Deblur", "sigma": nb.BLUR_SIGMA, "noise": 0.01}


def test_the_check_cell_uses_the_defaults(defaults):
    check = [c for c in code_cells() if 'sara.check("Goal"' in c][0]
    space = {}
    exec(check.split("sara.check")[0], space)
    assert space["defaults"] == defaults


def test_the_photos_are_the_canvas_size():
    """The canvas the setup cell asks for, the photos and every N in the kernels are one size."""
    for name in ("astronaut.png", "astronaut_blur.png", "brick.png"):
        assert Image.open(ROOT / nb.IMG / name).size == (N, N), name
    assert f"sara.init(width={N}, height={N})" in code_cells()[0]
    kernels = [c for c in code_cells() if c.startswith("%%gmacs")]
    assert len(kernels) == 4
    for kernel in kernels:
        assert set(re.findall(r"const int N = (\d+)", kernel)) <= {str(N)}
    assert f"/ {N}.0" in nb.WAVE
    assert f"* {N} ** 2" in code_cells()[[i for i, c in enumerate(code_cells()) if "F_gpu =" in c][0]]


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
        made |= set(re.findall(r'import_image\("imgs/4/([^"]+)\.png"\)', cell))
        made |= set(re.findall(r'(?:new_layer\(|-> |target=)"([^"]+)"', cell))


def test_the_spectrum_of_a_wave_is_three_dots(solved):
    """The wave kernel's picture, redone in NumPy, has its dots at the middle and at plus and minus (u, v)."""
    y, x = np.mgrid[0:N, 0:N]
    for u, v, phase in ((8, 4, 0.0), (8, -4, 90.0), (-3, 17, 200.0)):
        wave = 0.5 + 0.5 * np.cos(2 * np.pi * (u * x + v * y) / N + np.radians(phase))
        picture = solved["spectrum"](np.fft.fft2(wave))
        assert picture.shape == (N, N, 4) and np.all(picture[..., 3] == 1)
        grey = picture[..., 0]
        assert grey.min() >= 0 and grey.max() == 1 and grey[N // 2, N // 2] == 1
        bright = {(int(r) - N // 2, int(c) - N // 2) for r, c in zip(*np.nonzero(grey > 0.5))}
        assert bright == {(0, 0), (v, u), (-v, -u)}


def test_two_passes_with_whole_turns_dropped_are_the_dft(photo, solved):
    """The rows' and the columns' sums of the kernels, angles from (u * x) % N, give np.fft.fft2 / N^2."""
    L = solved["lightness"](photo).astype(np.float64)
    k = np.arange(N)
    W = np.exp(1j * (-2 * np.pi * ((k[:, None] * k[None, :]) % N) / N))
    rows = L @ W.T / N                       # DFT rows: (y, u)
    both = W @ rows / N                      # DFT: (v, u)
    assert np.abs(rows - np.fft.fft(L, axis=1) / N).max() < 1e-9
    assert np.abs(both - np.fft.fft2(L) / N ** 2).max() < 1e-9


def test_gaussian_filter_is_the_spectrum_of_the_kernels_weights(solved):
    """H is 1 at zero frequency and the DFT of the weights the blur kernel sums, to a thousandth."""
    H = solved["gaussian_filter"](N, N, 3.0)
    assert H.shape == (N, N) and H[0, 0] == 1 and H.min() > 0
    d = np.minimum(np.arange(N), N - np.arange(N))         # distance to pixel 0, around the edge
    weights = np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * 3.0 ** 2))
    weights[np.maximum(d[:, None], d[None, :]) > 12] = 0    # the kernel's square of 4 sigma
    assert np.abs(np.fft.fft2(weights / weights.sum()).real - H).max() < 1e-3
    # a picture that is not square: rows and columns each get their own frequencies, minus and plus alike
    tall = solved["gaussian_filter"](64, 48, 2.0)
    assert tall.shape == (64, 48)
    assert np.isclose(tall[1, 0], np.exp(-2 * np.pi ** 2 * 4.0 / 64 ** 2))
    assert np.isclose(tall[0, 1], np.exp(-2 * np.pi ** 2 * 4.0 / 48 ** 2))
    assert np.allclose(tall, tall[(-np.arange(64)) % 64][:, (-np.arange(48)) % 48])


def test_blur_fft_matches_the_blur_kernel(photo, solved):
    """blur_fft in linear light is the kernel's wrapped, weighted sum of 4 sigma around each pixel."""
    sigma = 1.5
    r = int(np.ceil(4 * sigma))
    lin = solved["srgb_to_linear"](photo[..., :3].astype(np.float64))
    total, weight = np.zeros_like(lin), 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            k = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            total += k * np.roll(lin, (-dy, -dx), axis=(0, 1))
            weight += k
    gpu = solved["linear_to_srgb"](total / weight)
    blurred = np.dstack([solved["blur_fft"](lin[..., c], sigma) for c in range(3)])
    assert np.abs(gpu - solved["linear_to_srgb"](blurred)).max() < 0.003
    assert np.allclose(blurred.mean(axis=(0, 1)), lin.mean(axis=(0, 1)))     # H is 1 at zero frequency


def test_wiener_undoes_a_clean_blur(solved):
    """Without noise and with K near zero the filter gives back the picture the blur was made from."""
    rng = np.random.default_rng(4)
    picture = rng.random((64, 48))
    blurred = solved["blur_fft"](picture, 1.0)
    assert np.abs(blurred - picture).max() > 0.1
    assert np.abs(solved["wiener"](blurred, 1.0, 1e-14) - picture).max() < 1e-6


def test_wiener_sharpens_the_lesson_photo(photo, blurry, solved):
    """On the eight-bit photo dividing by H explodes, and the filter at the lesson's K gets closer to the sharp one."""
    lin = solved["srgb_to_linear"](blurry[..., :3].astype(np.float64))
    H = solved["gaussian_filter"](N, N, nb.BLUR_SIGMA)
    divided = np.fft.ifft2(np.fft.fft2(lin[..., 0]) / H).real
    assert np.abs(divided).max() > 100
    fixed = np.dstack([solved["wiener"](lin[..., c], nb.BLUR_SIGMA, 0.01 ** 2) for c in range(3)])
    before = np.abs(blurry[..., :3] - photo[..., :3]).mean()
    after = np.abs(np.clip(solved["linear_to_srgb"](fixed), 0, 1) - photo[..., :3]).mean()
    assert after < 0.8 * before, (before, after)


def test_the_blurred_photo_is_the_lessons_blur(photo):
    """astronaut_blur.png is what the diagram script makes of astronaut.png, to the last bit."""
    shipped = np.asarray(Image.open(ROOT / nb.IMG / "astronaut_blur.png").convert("RGB"))
    assert np.array_equal(diagrams.blurred_photo(photo), shipped)


def test_unfinished_cells_are_harmless(photo, defaults):
    space = namespace(nb.HELPERS_NP, nb.SPECTRUM_GIVEN, nb.GAUSSIAN_GIVEN, nb.BLUR_GIVEN, nb.WIENER_GIVEN,
                      nb.PARAMS_NP, nb.APPLY_GIVEN)
    black = space["spectrum"](np.fft.fft2(photo[..., 0]))
    assert black.shape == (N, N, 4) and np.all(black[..., :3] == 0) and np.all(black[..., 3] == 1)
    assert np.all(space["gaussian_filter"](N, N, 3.0) == 1)
    channel = photo[..., 1]
    assert np.array_equal(space["blur_fft"](channel, 3.0), channel)
    assert np.array_equal(space["wiener"](channel, 3.0, 1e-4), channel)
    assert np.array_equal(space["apply"](photo, **defaults), photo)
    # the given inverse filter with the unfinished H shows the photo it was given
    lin = space["srgb_to_linear"](photo[..., :3])
    H = space["gaussian_filter"](N, N, nb.BLUR_SIGMA)
    assert np.allclose(np.fft.ifft2(np.fft.fft2(lin[..., 0]) / H).real, lin[..., 0], atol=1e-6)


def test_the_solution_runs(blurry, defaults, solved):
    out = {}
    for mode in MODES:
        out[mode] = solved["apply"](blurry, **{**defaults, "mode": mode})
        assert out[mode].shape == blurry.shape and out[mode].dtype == np.float32, mode
        assert np.isfinite(out[mode]).all() and np.array_equal(out[mode][..., 3], blurry[..., 3]), mode
    assert np.abs(out["High pass"][..., :3] - 0.5).mean() < 0.05            # detail around grey
    # Grain Merge, bottom + top - 0.5, of the two passes is the picture again
    merged = out["Low pass"][..., :3] + out["High pass"][..., :3] - 0.5
    assert np.abs(merged - blurry[..., :3]).max() < 1e-5


def test_the_goal_is_the_solution(blurry, solved):
    """The hidden Goal cell runs the same filter, PARAMS and source as section 5."""
    called = {}

    class Sara:
        @staticmethod
        def live(apply, params, source, target):
            called.update(apply=apply, params=params, source=source, target=target)

    namespace_ = {"np": np, "sara": Sara}
    exec(nb.GOAL, namespace_)
    assert called["source"] == "astronaut_blur" and called["target"] == "Goal"
    assert called["params"] == solved["PARAMS"]
    for mode in MODES:
        values = {"mode": mode, "sigma": 2.5, "noise": 0.02}
        assert np.array_equal(called["apply"](blurry, **values), solved["apply"](blurry, **values)), mode


def test_the_pictures_exist():
    """Every picture the notebook shows or opens is in imgs/4, and the copied photos are credited."""
    shown = set()
    for cell in nb.cells:
        text = "".join(cell["source"])
        shown |= set(re.findall(r'src="(imgs/4/[^"]+)"', text))
        shown |= set(re.findall(r'"(imgs/4/[^"]+\.png)"', text))
    assert len(shown) >= 7, shown
    for path in shown:
        assert (ROOT / path).is_file(), path
    sources = (ROOT / nb.IMG / "SOURCES.md").read_text()
    assert "`astronaut.png`" in sources and "`brick.png`" in sources
