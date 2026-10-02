"""Loads the student's effect and runs it on a layer, with no Qt in sight.

The effect lives in effect.py: TITLE, PARAMS and apply(img, **params). It is
loaded again every time the filter opens, so an edit to effect.py shows up
without restarting Krita. A mistake in it becomes an EffectError whose text
is the traceback, which the dialog shows instead of crashing.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import params as P
from . import pixels


class EffectError(RuntimeError):
    """effect.py could not be loaded, or apply() failed."""


@dataclass
class Effect:
    title: str
    spec: list[P.Param]
    apply: Callable[..., np.ndarray]


def load() -> Effect:
    """effect.py, loaded fresh, its PARAMS already read into widgets."""
    try:
        from . import effect
        effect = importlib.reload(effect)
    except Exception:
        raise EffectError("effect.py cannot be loaded:\n\n" + traceback.format_exc(limit=-3)) from None
    apply = getattr(effect, "apply", None)
    if not callable(apply):
        raise EffectError("effect.py has no function apply(img, ...).")
    try:
        spec = P.parse(getattr(effect, "PARAMS", {}))
    except P.ParamError as error:
        raise EffectError(f"PARAMS v effect.py:\n\n{error}") from None
    return Effect(getattr(effect, "TITLE", "Filtr"), spec, apply)


def run(effect: Effect, img: np.ndarray, values: dict[str, Any]) -> np.ndarray:
    """apply() on a copy of *img*, its result checked. Raises EffectError with the traceback."""
    try:
        result = effect.apply(img.copy(), **values)
    except Exception:
        raise EffectError("apply() raised an error:\n\n" + traceback.format_exc(limit=-3)) from None
    try:
        return pixels.check_result(result, img.shape)
    except pixels.PixelError as error:
        raise EffectError(str(error)) from None


def apply_to_layer(doc, node, effect: Effect, values: dict[str, Any]):
    """Filters the selection, or the canvas, of *node* and returns the layer that replaced it.

    Writing pixels through Krita's Python API leaves no undo step, but adding
    and removing a layer does. So the result goes into a copy of the layer,
    the copy is added above it and the original is removed. Ctrl+Z twice puts
    the original back.
    """
    x, y, w, h = pixels.region(doc)
    original = pixels.read_layer(node, x, y, w, h)
    result = run(effect, original, values)
    final = pixels.blend(original, result, pixels.selection_mask(doc, x, y, w, h))
    copy = node.duplicate()
    copy.setName(node.name())
    pixels.write_layer(copy, final, x, y)
    node.parentNode().addChildNode(copy, node)
    node.remove()
    doc.setActiveNode(copy)
    doc.refreshProjection()
    return copy
