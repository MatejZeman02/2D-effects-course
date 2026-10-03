"""`sara.live`: a plain NumPy function with sliders, rerun on every move and written to a layer.

    import sara
    doc, layer, pixels = sara.init()
    doc.import_image("img/balls.png")

    PARAMS = {"amount": (1.0, 0.0, 1.0), "mode": ["Multiply", "Screen"], "invert": False}
    TITLE = "Goal NumPy"

    def apply(img, amount, mode, invert):
        ...                               # img is float32 (h, w, 4) RGBA, the answer the same shape
        return out

    sara.live(apply, PARAMS, source="balls")

The function and its `PARAMS` are the ones the Krita plugin template takes unchanged, and
`sara_params.py` reads the dict the way the template does: a bool is a checkbox, a tuple a slider, a
`"#rrggbb[aa]"` string a colour picker (with an alpha slider beside it for the eight-digit form), a list of
str a drop-down and an n by n list of lists a grid of number fields. `apply(img, **values)` runs once with the
defaults and then on every move, debounced so a drag runs it at most once a frame, and what it answers is
written to the layer called *target*, which is `TITLE` of the notebook when none is given, a layer of that name
being reused, so running the cell again leaves one. The source is read once when the cell runs, and each call
gets a copy of it, so a function that edits `img` in place never compounds across moves.

A module of its own beside `sara_notebook.py`, whose debounce and connection it shares, so `sara.py` stays
free of `ipywidgets` and `sara.live` imports this at its first call.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

import ipywidgets
import numpy as np
from IPython.display import display

import sara
import sara_notebook
from sara_params import BOOL, CHOICE, COLOR, FLOAT, INT, Param, parse

# The layer a run writes when neither the call nor the notebook's `TITLE` names one.
DEFAULT_TARGET = "Live NumPy"
GRID_WIDTH = "6em"
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


class ResultError(ValueError):
    """What `apply` answered when it is not an array of the picture's shape holding finite numbers."""


def checked(result: object, shape: tuple[int, ...]) -> np.ndarray:
    """*result* when it is an ndarray of *shape* with finite numbers in it, a `ResultError` saying why when not."""
    if not isinstance(result, np.ndarray):
        raise ResultError(f"apply answered {type(result).__name__}, an ndarray of shape {shape} is wanted")
    if result.shape != shape:
        raise ResultError(f"apply answered shape {result.shape}, the picture's shape {shape} is wanted")
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ResultError(f"apply answered {result.dtype} values that are not all finite numbers")
    return result


class Control:
    """The widgets of one parameter, the ones a move changes and the value `apply` gets from them."""

    def __init__(self, param: Param) -> None:
        self.param = param
        self.alpha: ipywidgets.FloatSlider | None = None
        self.fields: list[list[ipywidgets.FloatText]] = []
        self.main: ipywidgets.Widget
        self.last = param.default
        name = param.name
        if param.kind == BOOL:
            self.main = ipywidgets.Checkbox(value=param.default, description=name)
            self.box: ipywidgets.Widget = self.main
        elif param.kind == INT:
            self.main = ipywidgets.IntSlider(value=param.default, min=param.low, max=param.high, step=param.step, description=name)
            self.box = self.main
        elif param.kind == FLOAT:
            self.main = ipywidgets.FloatSlider(value=param.default, min=param.low, max=param.high, step=param.step, description=name)
            self.box = self.main
        elif param.kind == COLOR:
            self.main = ipywidgets.ColorPicker(value=param.default[:7].lower(), description=name)
            self.box = self.main
            if param.alpha:
                # An eight-digit colour carries its alpha in eight bits, so the slider steps by that.
                tail = int(param.default[7:9], 16) / 255.0
                self.alpha = ipywidgets.FloatSlider(value=tail, min=0.0, max=1.0, step=1.0 / 255.0, description="alpha")
                self.box = ipywidgets.HBox([self.main, self.alpha])
        elif param.kind == CHOICE:
            self.main = ipywidgets.Dropdown(options=list(param.labels), value=param.default, description=name)
            self.box = self.main
        else:
            layout = ipywidgets.Layout(width=GRID_WIDTH)
            self.fields = [[ipywidgets.FloatText(value=cell, layout=layout) for cell in row] for row in param.default]
            self.box = ipywidgets.VBox([ipywidgets.Label(name), *[ipywidgets.HBox(row) for row in self.fields]])

    def inputs(self) -> list[ipywidgets.Widget]:
        """The widgets whose `value` a move changes."""
        if self.fields:
            return [field for row in self.fields for field in row]
        return [self.main] if self.alpha is None else [self.main, self.alpha]

    def value(self) -> Any:
        """What `apply` gets for the widgets as they stand now."""
        kind = self.param.kind
        if kind == COLOR:
            # A name the picker was typed over with is not a colour yet, so the last hex stands.
            if _HEX.match(self.main.value):
                self.last = self.main.value.lower()
            tail = f"{round(self.alpha.value * 255):02x}" if self.alpha is not None else ""
            return self.param.value(self.last[:7] + tail)
        if self.fields:
            return self.param.value([[field.value for field in row] for row in self.fields])
        return self.param.value(self.main.value)


class Live:
    """One `sara.live` call: the function, its controls and the layer each run is written to."""

    def __init__(self, document: sara.Document, apply: Callable[..., np.ndarray], params: list[Param], source: sara.Layer, target: str) -> None:
        self.document = document
        self.apply = apply
        self.source = source
        self.target = target
        self.image = source.read()
        self.controls = {param.name: Control(param) for param in params}
        self.values: dict[str, Any] = {param.name: param.initial() for param in params}
        self.status = ipywidgets.Label()
        self.output = ipywidgets.Output()
        self.debounce = sara_notebook.Debounce(self.rerun)
        for name, control in self.controls.items():
            for widget in control.inputs():
                widget.observe(lambda change, name=name: self.moved(name, change), names="value")
        self.box = ipywidgets.VBox([*[control.box for control in self.controls.values()], self.status, self.output])

    def run(self, values: dict[str, Any]) -> sara.Layer:
        """Calls `apply` on a copy of the picture and a copy of each grid, writes its answer to the target layer.

        A refused answer raises a `ResultError` before anything is written."""
        given = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in values.items()}
        began = time.perf_counter()
        result = checked(self.apply(self.image.copy(), **given), self.image.shape)
        applied = time.perf_counter()
        layer = self.document.new_layer(self.target, result, step=f"live {self.target}")
        written = time.perf_counter()
        self.status.value = f"apply {(applied - began) * 1000.0:.0f} ms, written {(written - applied) * 1000.0:.0f} ms"
        return layer

    def moved(self, name: str, change: dict) -> None:
        """A widget moved: its parameter's value joins the rest and the debounce decides when to run."""
        self.values[name] = self.controls[name].value()
        self.debounce(self.values)

    def rerun(self, values: dict[str, Any]) -> None:
        """The debounce's call: a refusal or a failure of `apply` is said under the widgets and the next move tries again."""
        try:
            self.show(self.run(values))
        except Exception as error:  # noqa: BLE001 a student's function may fail any way it likes, and a drag goes on
            self.status.value = f"{type(error).__name__}: {error}"

    def show(self, layer: sara.Layer) -> None:
        """Draws the written layer under the widgets, replacing the last, in `show()`'s small form."""
        with self.output:
            self.output.clear_output(wait=True)
            display(layer.show())


# The last call's controls, which a script or a test reads the widgets of.
last: Live | None = None


def live(
    apply: Callable[..., np.ndarray],
    params: dict,
    source: sara.Layer | str | int | None = None,
    target: str | None = None,
) -> None:
    """Runs *apply* on *source* with a widget for each of *params*, and writes the answer to *target*.

    *source* is a layer or its name or index, and the layer `sara.init()` read when none is given, a `Layer`
    also naming the document the run uses, which a script outside IPython passes. *target* is
    the layer written, `TITLE` of the notebook the function was defined in when none is given. The cell runs
    `apply` with the defaults at once, so a refused `PARAMS` or answer raises there, under the cell."""
    global last
    specs = parse(params)
    if not callable(apply):
        raise TypeError(f"apply is a function of the picture and the parameters, not {apply!r}")
    # The door serves one client, so a Layer brings its own document and a script never dials a second.
    layer = sara.layer_of(source)
    title = getattr(apply, "__globals__", {}).get("TITLE")
    name = target or (title if isinstance(title, str) and title else DEFAULT_TARGET)
    run = Live(layer.document, apply, specs, layer, name)
    written = run.run(run.values)
    display(run.box)
    run.show(written)
    last = run
