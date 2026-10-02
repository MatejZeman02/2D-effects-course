"""The PARAMS reader, in plain Python: python -m pytest krita/tests"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pga_filter"))
import params as P  # noqa: E402


def one(value):
    return P.parse({"x": value})[0]


def test_lesson_0_slider_reads_as_decimals():
    p = one((1.0, 0.0, 1.0))
    assert (p.kind, p.default, p.minimum, p.maximum) == (P.SLIDER, 1.0, 0.0, 1.0)
    assert p.step == pytest.approx(0.01)


def test_three_ints_are_whole_numbers_in_steps_of_one():
    p = one((4, 1, 8))
    assert (p.kind, p.default, p.step) == (P.WHOLE, 4, 1)


def test_a_fourth_number_is_the_step():
    p = one((16.0, 2.0, 64.0, 0.5))
    assert (p.kind, p.step) == (P.SLIDER, 0.5)


def test_one_float_among_ints_makes_decimals():
    assert one((4, 1, 8.0)).kind == P.SLIDER


def test_colour_without_and_with_alpha():
    p = one("#ff8000")
    assert p.kind == P.COLOUR and not p.alpha
    assert p.default == pytest.approx((1.0, 128 / 255, 0.0))
    q = one("#ff800080")
    assert q.alpha and len(q.default) == 4


def test_choice_defaults_to_its_first_label():
    p = one(["Multiply", "Screen"])
    assert (p.kind, p.default, p.labels) == (P.CHOICE, "Multiply", ["Multiply", "Screen"])


def test_bool_is_a_toggle_not_a_whole_number():
    assert one(False).kind == P.TOGGLE
    assert one(True).default is True


def test_square_matrix_becomes_a_float32_array():
    p = one([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    assert (p.kind, p.size) == (P.MATRIX, 3)
    assert p.default.dtype == np.float32 and p.default[1, 1] == -4


def test_order_is_kept():
    names = [p.name for p in P.parse({"b": False, "a": (1, 0, 2), "c": "#000000"})]
    assert names == ["b", "a", "c"]


def test_defaults_hand_a_copy_of_the_matrix():
    spec = P.parse({"m": [[1, 0], [0, 1]]})
    P.defaults(spec)["m"][0, 0] = 9
    assert spec[0].default[0, 0] == 1


@pytest.mark.parametrize("value", [
    (1.0, 2.0, 1.0),            # minimum above maximum
    (5.0, 0.0, 1.0),            # default outside the range
    (1.0, 0.0),                 # too few numbers
    (1.0, 0.0, 1.0, 0.0),       # a step of zero
    "ff8000",                   # colour without its hash
    "#ff80",                    # colour of the wrong length
    "#gg8000",                  # colour that is not hexadecimal
    [[1, 2, 3], [4, 5, 6]],     # matrix that is not square
    [[1, "a"], [2, 3]],         # matrix with a word in it
    {"a": 1},                   # a kind there is no widget for
    [],                         # an empty list
    1.5,                        # a bare number
])
def test_what_has_no_widget_is_refused_with_its_name(value):
    with pytest.raises(P.ParamError, match="^x:"):
        one(value)


def test_a_name_that_is_not_an_identifier_is_refused():
    with pytest.raises(P.ParamError):
        P.parse({"two words": False})
