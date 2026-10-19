"""Runs the NumPy half of lesson 4's optional notebook, the complex image, as the notebook prints it.

    python -m pytest build/test_lesson_04b.py

The kernel needs Sara and is not run here. Its arithmetic is redone in NumPy
on vec2 components, as the solution writes it, and compared with NumPy's
complex numbers. The interactive pictures need ipywidgets and are not run.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

import lesson_04b_notebook as nb  # noqa: E402


def namespace(*sources, **values):
    space = {"np": np, **values}
    for source in sources:
        exec(source, space)
    return space


@pytest.fixture(scope="module")
def photo():
    picture = Image.open(ROOT / "imgs/4/astronaut.png").convert("RGB").resize((256, 256), Image.LANCZOS)
    return np.asarray(picture, dtype=np.float64) / 255


def test_oklab_goes_there_and_back(photo):
    space = namespace(nb.HELPERS_NP)
    L, z = space["to_oklab"](photo)
    assert np.iscomplexobj(z) and not np.iscomplexobj(L)
    assert np.abs(space["from_oklab"](L, z) - photo).max() < 1e-6


def test_grey_has_no_arrow():
    space = namespace(nb.HELPERS_NP)
    L, z = space["to_oklab"](np.full((2, 2, 3), 0.5))
    assert np.abs(z).max() < 1e-6


def test_the_kernels_product_is_the_complex_one(photo):
    space = namespace(nb.HELPERS_NP)
    _, z = space["to_oklab"](photo)
    theta, s = np.radians(nb.THETA), nb.STRETCH
    w = (s * np.cos(theta), s * np.sin(theta))
    product = (z.real * w[0] - z.imag * w[1]) + 1j * (z.real * w[1] + z.imag * w[0])
    assert np.abs(product - s * np.exp(1j * theta) * z).max() < 1e-12
    assert f"= {nb.THETA}" in nb.HUE and f"= {nb.STRETCH}" in nb.HUE


def test_white_balance_moves_the_mean_to_grey(photo):
    space = namespace(nb.HELPERS_NP)
    _, z = space["to_oklab"](photo)
    warm = 0.04 * np.exp(1j * np.radians(70))
    given = namespace(nb.BALANCE_GIVEN)["white_balance"](z + warm)
    solved = namespace(nb.SOLUTION_BALANCE)["white_balance"](z + warm)
    assert abs(given.mean()) > 0.03
    assert abs(solved.mean()) < 1e-12
    assert np.allclose(solved - z, -z.mean())


def test_one_transform_splits_into_two(photo):
    space = namespace(nb.HELPERS_NP)
    space = namespace(nb.SPLIT, **space, PICTURES={"fotka": photo})
    assert np.abs(space["Fa"] - np.fft.fft2(space["z"].real)).max() < 1e-9
    assert np.abs(space["Fb"] - np.fft.fft2(space["z"].imag)).max() < 1e-9
