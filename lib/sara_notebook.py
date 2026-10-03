"""The `%%gmacs` cell magic and a slider for each of a kernel's parameters.

    import sara
    doc, layer, pixels = sara.init()   # connects, and registers the magic on that document
    %load_ext sara_notebook            # or this alone, which connects at the first cell

    %%gmacs Background -> Cells scale=12
    kernel cells

    image src: readonly
    image dst

    params:
        float scale: hint_range(2, 64)

    def cells():
        ...

The body may instead be the short pixel form a module's pixel function takes,
`uniform` lines and one `def pixel(ivec2 at) -> vec4` with no `kernel` line,
which the app wraps into a kernel called `pixels`:

    %%gmacs Background -> Dithered
    uniform float levels = 4.0    # range 2 16

    def pixel(ivec2 at) -> vec4:
        return floor(src(at) * levels + 0.5) / levels

The first line names the layer to read, the layer to write, which a new layer
named after the kernel is when the line names none, and the values of the
parameters. A word `<image>=<layer>` names a layer for each further readonly
image the kernel declares, as `%%gmacs Background src1=Sky -> Mixed` for a
kernel with `image src1: readonly`, and the kernel reads it under that name, a
number being the layer's index as for the first two. The body is the kernel, sent as source to the app's `kernel` step,
and the cell shows the written layer with `show()`. Every scalar parameter the
compiled kernel answers becomes a slider, its range from the
`hint_range(<min>, <max>[, <step>])` on its line in the params block or on its
`uniform` line, the way a Godot shader declares one, opening at the default the
app answers, which is the `uniform` line's own value in the short form, and
moving one reruns the kernel, debounced so a drag sends at most one request a
frame, and setting one from code, `cell.sliders["scale"].value = 24.0`, reruns
it at once. The `# range <min> <max>` comment that came first is still read, so a
saved cell keeps its sliders. An `int` with a `hint_enum("<label>", ...)` is a
drop-down of its labels instead, the kernel getting the picked entry's index as
in Godot, and a `bool` is a checkbox. A matrix, `float weights[9]` on either
line, is a grid of number fields, 3 by 3 for nine numbers, 5 by 5 for
twenty-five and 7 by 7 for forty-nine, read row by row and opening on the
numbers the app answers, a one in the middle where the line gives none. Under
the widgets a line says what the last run cost, `GPU 0.21 ms, round trip 38 ms`:
the dispatch alone as the app timed it, and the wall time of the request,
without the picture's own read. A `vec3` or
`vec4` tagged `source_color`, as `vec3 tint: source_color`, becomes a colour
picker, a `vec4` with an alpha slider beside it, and the kernel gets the very
numbers the picker shows, sRGB as `src()` hands a pixel, with none of Godot's
conversion to linear light.

A module of its own beside `sara.py`, so the client a student copies needs
neither IPython nor `ipywidgets`, and this one needs both.
"""

from __future__ import annotations

import functools
import math
import re
import shlex
import threading
import time
from typing import Any, Callable, NamedTuple

import ipywidgets
from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.display import display

import sara

# A frame at sixty a second, the most often a dragged slider sends.
FRAME_S = 1.0 / 60.0
# The range a parameter with no range of its own gets.
FLOAT_RANGE = (0.0, 1.0)
INT_RANGE = (0, 10)
INT_TYPES = ("int", "uint")
COLOUR_TYPES = ("vec3", "vec4")

# The kernel the app wraps a short pixel form into, never `pixel`, the dialect's own word.
PIXEL_KERNEL = "pixels"

_KERNEL_NAME = re.compile(r"^\s*kernel\s+(\w+)", re.MULTILINE)
_PIXEL_SIGNATURE = re.compile(r"^def\s+pixel\s*\(", re.MULTILINE)
_READONLY_IMAGE = re.compile(r"^\s*image\s+(\w+)\s*:\s*readonly\b", re.MULTILINE)
# A field of the params block: its type, its name, the hints after a colon and
# the comment after a hash. The hints stop at an equals sign, so a value after
# them is not read as one.
_PARAM_LINE = re.compile(r"^[ \t]+(\w+)[ \t]+(\w+)[ \t]*(?::([^#=]*))?(?:=[^#]*)?(?:#(.*))?$")
# A `uniform` line of the short pixel form, read into the same four groups.
_UNIFORM_LINE = re.compile(r"^[ \t]*uniform[ \t]+(\w+)[ \t]+(\w+)[ \t]*(?::([^#=]*))?(?:=[^#]*)?(?:#(.*))?$")
_HINT_RANGE = re.compile(r"\bhint_range\s*\(([^()]*)\)")
_RANGE_COMMENT = re.compile(r"^\s*range\s+(\S+)\s+(\S+)\s*$")
_SOURCE_COLOR = re.compile(r"\bsource_color\b")
_HINT_ENUM = re.compile(r"\bhint_enum\s*\((.*)\)")
_LABEL = re.compile(r'\s*"([^"]*)"\s*(?:,|$)')


# How wide a matrix's number field is, room for a sign and four decimals.
ENTRY_WIDTH = "5.5em"


def matrix_side(param: dict) -> int | None:
    """The side of the grid *param* draws, None when it is no matrix.

    A matrix is a `float` array whose length is a square, as the app answers it:
    `length` beside the type, 9 for `float weights[9]`.
    """
    length = param.get("length") or 0
    if param.get("type") != "float" or length < 1:
        return None
    side = math.isqrt(length)
    return side if side * side == length else None


class Range(NamedTuple):
    """The numbers a parameter's slider takes, `step` None where it named none."""

    low: float
    high: float
    step: float | None = None


class CellLine:
    """The first line of a `%%gmacs` cell: the layer read, the layer written, the values.

    A word `<image>=<layer>` whose name is one of *images*, the further readonly
    images the kernel declares beside `src`, names a layer the kernel reads
    under that name, and the `kernel` step gets it in `inputs`."""

    def __init__(self, line: str, images: set[str] | frozenset[str] = frozenset()) -> None:
        self.layer: str | int | None = None
        self.into: str | int | None = None
        self.values: dict[str, float | bool] = {}
        self.rect: list[int] | None = None
        self.inputs: dict[str, str | int] = {}
        layers: list[str | int] = []
        for word in shlex.split(line):
            if word == "->":
                continue
            key, equals, value = word.partition("=")
            if not equals:
                layers.append(int(word) if word.isdigit() else word)
            elif key in images:
                self.inputs[key] = int(value) if value.isdigit() else value
            elif key == "rect":
                self.rect = [int(v) for v in value.split(",")]
            elif value in ("true", "false"):
                self.values[key] = value == "true"
            else:
                try:
                    self.values[key] = float(value)
                except ValueError:
                    raise ValueError(f"{word} is neither a value nor a layer for a readonly image the kernel declares") from None
        if len(layers) > 2:
            raise ValueError(f"a %%gmacs line names a layer to read and one to write, not {layers}")
        self.layer = layers[0] if layers else None
        self.into = layers[1] if len(layers) > 1 else None


def images(source: str) -> set[str]:
    """The readonly images *source* declares beside `src`, each a layer the first line may name."""
    return {name for name in _READONLY_IMAGE.findall(source) if name != "src"}


def kernel_name(source: str) -> str:
    """The name the source's `kernel <name>` line declares, `pixels` for the short pixel form."""
    found = _KERNEL_NAME.search(source)
    if found is not None:
        return found.group(1)
    if _PIXEL_SIGNATURE.search(source) is not None:
        return PIXEL_KERNEL
    raise ValueError("a %%gmacs cell's body starts with `kernel <name>` or holds `def pixel(ivec2 at) -> vec4`")


def ranges(source: str) -> dict[str, Range]:
    """The range each params or `uniform` line declares, by the parameter's name.

    `hint_range(<min>, <max>[, <step>])` after the name's colon, or the
    `# range <min> <max>` comment that came first, the hint winning where a
    line has both. A line with neither, or a hint that is not two or three
    numbers, keeps the default range.
    """
    found: dict[str, Range] = {}
    for line in source.splitlines():
        match = _PARAM_LINE.match(line) or _UNIFORM_LINE.match(line)
        if match is None:
            continue
        hinted = _hint_range(match.group(3) or "")
        commented = _RANGE_COMMENT.match(match.group(4) or "")
        if hinted is not None:
            found[match.group(2)] = hinted
        elif commented is not None:
            found[match.group(2)] = Range(float(commented.group(1)), float(commented.group(2)))
    return found


def colours(source: str) -> set[str]:
    """The names of the params or `uniform` lines tagged `source_color`, which become a colour picker."""
    found: set[str] = set()
    for line in source.splitlines():
        match = _PARAM_LINE.match(line) or _UNIFORM_LINE.match(line)
        if match is not None and _SOURCE_COLOR.search(match.group(3) or ""):
            found.add(match.group(2))
    return found


def hex_of(numbers: list[float]) -> str:
    """The `#rrggbb` a picker shows for the first three of *numbers*, each clamped to 0 to 1."""
    return "#" + "".join(f"{round(min(max(float(n), 0.0), 1.0) * 255):02x}" for n in numbers[:3])


def numbers_of(picked: str) -> list[float]:
    """The three numbers a picker's `#rrggbb` or `#rgb` shows, from 0 to 1, as the kernel gets them."""
    digits = picked.lstrip("#")
    if len(digits) == 3:
        digits = "".join(d * 2 for d in digits)
    if len(digits) != 6:
        raise ValueError(f"a colour picker answers #rrggbb, not {picked!r}")
    return [int(digits[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _hint_range(hints: str) -> Range | None:
    """The `hint_range` among a line's *hints*, or None when it has none it can read."""
    match = _HINT_RANGE.search(hints)
    if match is None:
        return None
    try:
        numbers = [float(part) for part in match.group(1).split(",")]
    except ValueError:
        return None
    if len(numbers) not in (2, 3) or (len(numbers) == 3 and numbers[2] <= 0):
        return None
    return Range(*numbers)


def choices(source: str) -> dict[str, list[str]]:
    """The labels of each params or `uniform` line's `hint_enum`, by the parameter's name.

    A line whose labels are not all quoted strings, or that has none, is left
    out and its parameter keeps a slider.
    """
    found: dict[str, list[str]] = {}
    for line in source.splitlines():
        match = _PARAM_LINE.match(line) or _UNIFORM_LINE.match(line)
        hint = _HINT_ENUM.search(match.group(3) or "") if match is not None else None
        if hint is None:
            continue
        labels, at, text = [], 0, hint.group(1)
        while at < len(text):
            label = _LABEL.match(text, at)
            if label is None:
                break
            labels.append(label.group(1))
            at = label.end()
        if labels and at >= len(text):
            found[match.group(2)] = labels
    return found


class Debounce:
    """Sends the latest values at most once a frame, the rest of a drag waits for the next.

    A call with *at_once* sends now whatever the last send's time, and drops a
    send the timer still held, as its values are older."""

    def __init__(self, send: Callable[[dict], None], interval: float = FRAME_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.send = send
        self.interval = interval
        self.clock = clock
        self.last = -interval
        self.pending: dict | None = None
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def __call__(self, values: dict, at_once: bool = False) -> None:
        with self.lock:
            self.pending = dict(values)
            wait = self.last + self.interval - self.clock()
            if at_once:
                wait = 0.0
            elif self.timer is not None:
                return
            if wait > 0:
                self.timer = threading.Timer(wait, self.flush)
                self.timer.daemon = True
                self.timer.start()
                return
        self.flush()

    def flush(self) -> None:
        """Sends what waits, now: the timer's call, and a value set from code."""
        with self.lock:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            values, self.pending = self.pending, None
            if values is None:
                return
            self.last = self.clock()
            self.send(values)


class Cell:
    """One `%%gmacs` cell run: the kernel, its sliders and the picture they rerun."""

    def __init__(self, document: sara.Document, line: str, source: str) -> None:
        self.document = document
        self.line = CellLine(line, images(source))
        self.source = source
        self.name = kernel_name(source)
        self.into = self.line.into
        self.values: dict[str, Any] = dict(self.line.values)
        self.output = ipywidgets.Output()
        self.timing = ipywidgets.Label()
        self.sliders: dict[str, ipywidgets.Widget] = {}
        self.grids: dict[str, ipywidgets.GridBox] = {}
        # A tagged colour's picker, and its alpha slider when it is a `vec4`.
        self.pickers: dict[str, ipywidgets.ColorPicker] = {}
        self.alphas: dict[str, ipywidgets.FloatSlider] = {}
        # The row each tagged colour shows, the picker or the picker and its alpha.
        self.colours: dict[str, ipywidgets.Widget] = {}
        self._debounce = Debounce(self.rerun)

    def run(self) -> dict:
        """Sends the kernel with the current values, keeps the layer it wrote and says what it cost."""
        step: dict[str, Any] = {"source": self.source, "name": self.name, "params": dict(self.values)}
        if self.line.layer is not None:
            step["layer"] = self.line.layer
        if self.into is not None:
            step["into"] = self.into
        if self.line.rect is not None:
            step["rect"] = self.line.rect
        if self.line.inputs:
            step["inputs"] = dict(self.line.inputs)
        said = self.document.kernel(step)
        # A rerun writes where the first run did, never a new layer a slider step.
        self.into = said.get("into", self.into)
        timing = sara.Timing(said.get("gpu_ms"), said["round_trip_ms"])
        self.timing.value = timing.line(said.get("gpu_unknown", ""))
        return said

    def build(self, params: list[dict]) -> ipywidgets.Widget:
        """A widget per scalar, tagged colour or matrix the kernel answered, the line of what a run cost, then the picture.

        A `bool` is a checkbox, an `int` with a `hint_enum` a drop-down whose
        value is the picked entry's index, a `vec3` or `vec4` tagged
        `source_color` a colour picker, a `float` array of a square length a grid
        of number fields, and every other scalar a slider.
        """
        given, labelled, tagged = ranges(self.source), choices(self.source), colours(self.source)
        for param in params:
            kind, name = param.get("type"), param["name"]
            side = matrix_side(param)
            if side is not None:
                self.sliders[name] = self.grid(name, side, param.get("default"))
                continue
            # An array that is no square matrix has no widget, the way an untagged vector has none.
            if param.get("length"):
                continue
            if kind in COLOUR_TYPES and name in tagged:
                self.colours[name] = self._colour(name, kind, param.get("default"))
                continue
            if kind == "bool":
                self._add(name, ipywidgets.Checkbox(value=bool(self.values.get(name, param.get("default", False))), description=name))
                continue
            if kind == "int" and name in labelled:
                labels = labelled[name]
                index = min(max(int(self.values.get(name, param.get("default", 0))), 0), len(labels) - 1)
                if name in self.values:
                    self.values[name] = index
                options = [(label, i) for i, label in enumerate(labels)]
                self._add(name, ipywidgets.Dropdown(options=options, value=index, description=name))
                continue
            if kind != "float" and kind not in INT_TYPES:
                continue
            integer = kind in INT_TYPES
            low, high, step = given.get(name, Range(*(INT_RANGE if integer else FLOAT_RANGE)))
            value = self.values.get(name, param.get("default", low))
            value = min(max(value, low), high)
            if name in self.values:
                self.values[name] = int(value) if integer else float(value)
            if integer:
                whole = max(round(step), 1) if step is not None else 1
                slider = ipywidgets.IntSlider(value=int(value), min=int(low), max=int(high), step=whole, description=name)
            else:
                fine = step if step is not None else (high - low) / 100.0 or 0.01
                slider = ipywidgets.FloatSlider(value=float(value), min=low, max=high, step=fine, description=name)
            self._add(name, slider)
        shown = [self.sliders.get(p["name"]) or self.colours.get(p["name"]) for p in params]
        return ipywidgets.VBox([*[w for w in shown if w is not None], self.timing, self.output])

    def _colour(self, name: str, kind: str, default: Any) -> ipywidgets.Widget:
        """The picker of a tagged colour, with an alpha slider beside it for a `vec4`.

        `ColorPicker` holds `#rrggbb` and no alpha, so a `vec4`'s fourth number
        has a slider of its own. The picker opens on the cell line's value or
        the kernel's default, which a params block declares as nought.
        """
        width = 4 if kind == "vec4" else 3
        held = self.values.get(name, default)
        numbers = [float(n) for n in held] if isinstance(held, (list, tuple)) else [0.0] * width
        numbers = (numbers + [1.0] * width)[:width]
        picker = ipywidgets.ColorPicker(value=hex_of(numbers), concise=False, description=name)
        picker.observe(lambda change: self.recoloured(name, change), names="value")
        self.pickers[name] = picker
        if width == 4:
            alpha = ipywidgets.FloatSlider(value=numbers[3], min=0.0, max=1.0, step=0.01, description=f"{name} alpha")
            alpha.observe(lambda change: self.recoloured(name, change), names="value")
            self.alphas[name] = alpha
            return ipywidgets.HBox([picker, alpha])
        return picker

    def recoloured(self, name: str, change: dict) -> None:
        """A picker or its alpha moved: the colour joins the rest as the numbers the picker shows."""
        numbers = numbers_of(self.pickers[name].value)
        if name in self.alphas:
            numbers.append(float(self.alphas[name].value))
        self.values[name] = numbers
        self.send(change)

    def _add(self, name: str, widget: ipywidgets.Widget) -> None:
        """Keeps *widget* as *name*'s, and reruns the kernel when it moves."""
        widget.observe(self.moved, names="value")
        self.sliders[name] = widget

    def grid(self, name: str, side: int, opening: list | None) -> ipywidgets.Widget:
        """A matrix's number fields, *side* to a row in the order the kernel reads them, under its name."""
        numbers = [float(number) for number in self.values.get(name, opening or [])]
        if len(numbers) != side * side:
            numbers = [0.0] * (side * side)
        entries = []
        for at, number in enumerate(numbers):
            entry = ipywidgets.FloatText(value=number, layout=ipywidgets.Layout(width=ENTRY_WIDTH))
            entry.observe(functools.partial(self.entry_moved, name, at), names="value")
            entries.append(entry)
        columns = ipywidgets.Layout(grid_template_columns=f"repeat({side}, {ENTRY_WIDTH})")
        self.grids[name] = ipywidgets.GridBox(entries, layout=columns)
        return ipywidgets.VBox([ipywidgets.Label(name), self.grids[name]])

    def entry_moved(self, name: str, at: int, change: dict) -> None:
        """A matrix's field changed: the whole matrix joins the values, and the debounce decides when to send."""
        numbers = [float(entry.value) for entry in self.grids[name].children]
        numbers[at] = float(change["new"])
        self.values[name] = numbers
        self.send(change)

    def moved(self, change: dict) -> None:
        """A widget moved: its value joins the rest and the debounce decides when to send."""
        self.values[change["owner"].description] = change["new"]
        self.send(change)

    def send(self, change: dict) -> None:
        """Hands the values to the debounce, at once when *change* was made in code.

        A move from the page, a drag, arrives through the widget's `set_state`,
        which holds the name in `_property_lock` while the observers run. A value
        set from code does not, and reruns at once, so a script reads the new
        picture on its next line without reaching into the debounce."""
        self._debounce(self.values, at_once=change["name"] not in change["owner"]._property_lock)

    def rerun(self, values: dict) -> None:
        self.values = dict(values)
        self.show(self.run())

    def show(self, said: dict) -> None:
        """Draws the written layer in the cell's picture, replacing the last.

        `show()`'s small form, made in Sara on the GPU, so a slider move reads back a few
        hundred KB rather than the layer in float32."""
        with self.output:
            self.output.clear_output(wait=True)
            display(self.document.layer(said.get("into", self.into)).show())


@magics_class
class GmacsMagics(Magics):
    """The `%%gmacs` magic, on the document `use()` gave or the Sara `sara.connect()` reaches."""

    document: sara.Document | None = None
    last: Cell | None = None

    @cell_magic
    def gmacs(self, line: str, cell: str) -> Cell | None:
        if GmacsMagics.document is None:
            GmacsMagics.document = sara.connect()
        run = Cell(GmacsMagics.document, line, cell)
        try:
            said = run.run()
        except sara.SaraError as error:
            print(error)
            return None
        widget = run.build(said.get("params", []))
        run.show(said)
        display(widget)
        GmacsMagics.last = run
        return None


def use(document: sara.Document) -> None:
    """Runs every `%%gmacs` cell on *document* rather than connecting on the first."""
    GmacsMagics.document = document


def register(shell: Any = None, document: sara.Document | None = None) -> None:
    """Registers `%%gmacs` with *shell*, the running IPython when none is given, and runs its cells on
    *document* when one is given, which `sara.init()` does so no second connection is made."""
    if shell is None:
        from IPython import get_ipython

        shell = get_ipython()
    shell.register_magics(GmacsMagics)
    if document is not None:
        use(document)


def load_ipython_extension(shell: Any) -> None:
    """What `%load_ext sara_notebook` calls."""
    register(shell)
