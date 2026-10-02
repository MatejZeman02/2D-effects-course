"""Reads an effect's PARAMS into widget descriptions, and widget values back.

PARAMS is a plain dict, so the same block runs in the notebook (sara.live)
and in this template without importing either. Its order is the order of the
widgets, and the kind of each value decides its widget:

    PARAMS = {
        "amount": (1.0, 0.0, 1.0),          # slider: default, minimum, maximum
        "width": (16.0, 2.0, 64.0, 0.5),    # slider with a step
        "steps": (4, 1, 8),                 # whole numbers, all three are int
        "tint": "#ff8c32",                  # colour, "#rrggbbaa" adds alpha
        "mode": ["Multiply", "Screen"],     # choice, the first is the default
        "invert": False,                    # on or off
        "weights": [[0, 1, 0],              # matrix, square, rows of numbers
                    [1, -4, 1],
                    [0, 1, 0]],
    }

What apply() gets for each kind: a float, an int, a tuple of three or four
floats from 0 to 1 in sRGB as the picker shows them, the chosen label, a bool,
and an (n, n) float32 array.

Nothing here imports Qt or Krita, so it runs in plain Python too.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SLIDER = "slider"
WHOLE = "whole"
COLOUR = "colour"
CHOICE = "choice"
TOGGLE = "toggle"
MATRIX = "matrix"

FORMS = """The values PARAMS takes:
  (default, minimum, maximum)          slider
  (default, minimum, maximum, step)    slider with a step
  (4, 1, 8)                            whole numbers, when all three are whole
  "#ff8c32" or "#ff8c32cc"             colour, the second form with alpha
  ["Multiply", "Screen"]               choice, the first is the default
  True or False                        checkbox
  [[0, 1, 0], [1, -4, 1], [0, 1, 0]]   square matrix"""


class ParamError(ValueError):
    """A PARAMS entry this template cannot turn into a widget."""


@dataclass
class Param:
    """One widget: its name, kind, default and what the kind needs."""

    name: str
    kind: str
    default: Any
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.0
    labels: list[str] = field(default_factory=list)
    alpha: bool = False
    size: int = 0


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_whole(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _slider(name: str, value: tuple) -> Param:
    if len(value) not in (3, 4) or not all(_is_number(v) for v in value):
        raise ParamError(f"{name}: a slider needs three or four numbers, got {value!r}.")
    default, low, high = value[:3]
    if not low < high:
        raise ParamError(f"{name}: the minimum {low} must be less than the maximum {high}.")
    if not low <= default <= high:
        raise ParamError(f"{name}: the default {default} lies outside the range {low} to {high}.")
    whole = all(_is_whole(v) for v in value)
    if len(value) == 4:
        step = value[3]
        if step <= 0:
            raise ParamError(f"{name}: the step {step} must be positive.")
    else:
        step = 1 if whole else (high - low) / 100.0
    if whole:
        return Param(name, WHOLE, int(default), int(low), int(high), int(step))
    return Param(name, SLIDER, float(default), float(low), float(high), float(step))


def parse_colour(name: str, text: str) -> tuple[float, ...]:
    """'#rrggbb' or '#rrggbbaa' as floats from 0 to 1, as the picker shows them."""
    digits = text[1:] if text.startswith("#") else ""
    if len(digits) not in (6, 8):
        raise ParamError(f"{name}: a colour is written \"#rrggbb\" or \"#rrggbbaa\", got {text!r}.")
    try:
        return tuple(int(digits[i:i + 2], 16) / 255.0 for i in range(0, len(digits), 2))
    except ValueError:
        raise ParamError(f"{name}: {text!r} is not a hexadecimal colour.") from None


def _matrix(name: str, rows: list) -> Param:
    size = len(rows)
    if not all(isinstance(r, (list, tuple)) and len(r) == size for r in rows):
        raise ParamError(f"{name}: the matrix must be square, {size} rows of {size} numbers.")
    if not all(_is_number(v) for r in rows for v in r):
        raise ParamError(f"{name}: the matrix may hold numbers only.")
    return Param(name, MATRIX, np.array(rows, dtype=np.float32), size=size)


def parse(params: dict) -> list[Param]:
    """The widgets PARAMS asks for, in its order. Raises ParamError on a bad entry."""
    if not isinstance(params, dict):
        raise ParamError("PARAMS must be a dict, {\"name\": value, ...}.")
    out = []
    for name, value in params.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ParamError(f"{name!r}: a parameter name must be a valid Python variable name.")
        # bool before numbers, because True is also an int in Python.
        if isinstance(value, bool):
            out.append(Param(name, TOGGLE, value))
        elif isinstance(value, tuple):
            out.append(_slider(name, value))
        elif isinstance(value, str):
            colour = parse_colour(name, value)
            out.append(Param(name, COLOUR, colour, alpha=len(colour) == 4))
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            out.append(Param(name, CHOICE, value[0], labels=list(value)))
        elif isinstance(value, list) and value and all(isinstance(v, (list, tuple)) for v in value):
            out.append(_matrix(name, value))
        else:
            raise ParamError(f"{name}: the value {value!r} cannot become a control.\n\n{FORMS}")
    return out


def defaults(spec: list[Param]) -> dict[str, Any]:
    """The keyword arguments apply() gets before any widget moves."""
    return {p.name: (p.default.copy() if p.kind == MATRIX else p.default) for p in spec}
