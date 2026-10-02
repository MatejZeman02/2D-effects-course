"""The `%%gmacs` cell magic and a slider for each of a kernel's parameters.

    %load_ext sara_notebook            # or: import sara_notebook; sara_notebook.register()

    %%gmacs Background -> Cells scale=12
    kernel cells

    image src: readonly
    image dst

    params:
        float scale    # range 2 64

    def cells():
        ...

The first line names the layer to read, the layer to write, which a new layer
named after the kernel is when the line names none, and the values of the
parameters. The body is the kernel, sent as source to the app's `kernel` step,
and the cell shows the written layer with `show()`. Every scalar parameter the
compiled kernel answers becomes a slider, its range from a `# range <min> <max>`
comment on its line in the params block, and moving one reruns the kernel,
debounced so a drag sends at most one request a frame.

A module of its own beside `sara.py`, so the client a student copies needs
neither IPython nor `ipywidgets`, and this one needs both.
"""

from __future__ import annotations

import re
import shlex
import threading
import time
from typing import Any, Callable

import ipywidgets
from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.display import display

import sara

# A frame at sixty a second, the most often a dragged slider sends.
FRAME_S = 1.0 / 60.0
# The range a parameter with no `# range` comment gets.
FLOAT_RANGE = (0.0, 1.0)
INT_RANGE = (0, 10)
INT_TYPES = ("int", "uint")

_KERNEL_NAME = re.compile(r"^\s*kernel\s+(\w+)", re.MULTILINE)
_PARAM_LINE = re.compile(r"^\s+(\w+)\s+(\w+)\s*#\s*range\s+(\S+)\s+(\S+)\s*$", re.MULTILINE)


class CellLine:
    """The first line of a `%%gmacs` cell: the layer read, the layer written, the values."""

    def __init__(self, line: str) -> None:
        self.layer: str | int | None = None
        self.into: str | int | None = None
        self.values: dict[str, float] = {}
        self.rect: list[int] | None = None
        layers: list[str | int] = []
        for word in shlex.split(line):
            if word == "->":
                continue
            key, equals, value = word.partition("=")
            if not equals:
                layers.append(int(word) if word.isdigit() else word)
            elif key == "rect":
                self.rect = [int(v) for v in value.split(",")]
            else:
                self.values[key] = float(value)
        if len(layers) > 2:
            raise ValueError(f"a %%gmacs line names a layer to read and one to write, not {layers}")
        self.layer = layers[0] if layers else None
        self.into = layers[1] if len(layers) > 1 else None


def kernel_name(source: str) -> str:
    """The name the source's `kernel <name>` line declares."""
    found = _KERNEL_NAME.search(source)
    if found is None:
        raise ValueError("a %%gmacs cell's body starts with `kernel <name>`")
    return found.group(1)


def ranges(source: str) -> dict[str, tuple[float, float]]:
    """The `# range <min> <max>` each params line carries, by the parameter's name."""
    found: dict[str, tuple[float, float]] = {}
    for match in _PARAM_LINE.finditer(source):
        found[match.group(2)] = (float(match.group(3)), float(match.group(4)))
    return found


class Debounce:
    """Sends the latest values at most once a frame, the rest of a drag waits for the next."""

    def __init__(self, send: Callable[[dict], None], interval: float = FRAME_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.send = send
        self.interval = interval
        self.clock = clock
        self.last = -interval
        self.pending: dict | None = None
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def __call__(self, values: dict) -> None:
        with self.lock:
            self.pending = dict(values)
            wait = self.last + self.interval - self.clock()
            if self.timer is not None:
                return
            if wait > 0:
                self.timer = threading.Timer(wait, self.flush)
                self.timer.daemon = True
                self.timer.start()
                return
        self.flush()

    def flush(self) -> None:
        """Sends what waits, now: the timer's call, and a test's."""
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
        self.line = CellLine(line)
        self.source = source
        self.name = kernel_name(source)
        self.into = self.line.into
        self.values: dict[str, float] = dict(self.line.values)
        self.output = ipywidgets.Output()
        self.sliders: dict[str, ipywidgets.Widget] = {}
        self.debounce = Debounce(self.rerun)

    def run(self) -> dict:
        """Sends the kernel with the current values, and keeps the layer it wrote."""
        step: dict[str, Any] = {"source": self.source, "name": self.name, "params": dict(self.values)}
        if self.line.layer is not None:
            step["layer"] = self.line.layer
        if self.into is not None:
            step["into"] = self.into
        if self.line.rect is not None:
            step["rect"] = self.line.rect
        said = self.document.door.verdict({"kernel": step})
        if not isinstance(said, dict) or not said.get("ok", False):
            raise sara.SaraError(f"{sara.VERDICT}kernel {said}")
        # A rerun writes where the first run did, never a new layer a slider step.
        self.into = said.get("into", self.into)
        return said

    def build(self, params: list[dict]) -> ipywidgets.Widget:
        """A slider per scalar parameter the kernel answered, then the picture."""
        given = ranges(self.source)
        for param in params:
            kind, name = param.get("type"), param["name"]
            if kind != "float" and kind not in INT_TYPES:
                continue
            integer = kind in INT_TYPES
            low, high = given.get(name, INT_RANGE if integer else FLOAT_RANGE)
            value = self.values.get(name, param.get("default", low))
            value = min(max(value, low), high)
            if name in self.values:
                self.values[name] = int(value) if integer else float(value)
            if integer:
                slider = ipywidgets.IntSlider(value=int(value), min=int(low), max=int(high), description=name)
            else:
                slider = ipywidgets.FloatSlider(value=float(value), min=low, max=high, step=(high - low) / 100.0 or 0.01, description=name)
            slider.observe(self.moved, names="value")
            self.sliders[name] = slider
        return ipywidgets.VBox([*self.sliders.values(), self.output])

    def moved(self, change: dict) -> None:
        """A slider moved: its value joins the rest and the debounce decides when to send."""
        self.values[change["owner"].description] = change["new"]
        self.debounce(self.values)

    def rerun(self, values: dict) -> None:
        self.values = dict(values)
        self.show(self.run())

    def show(self, said: dict) -> None:
        """Draws the written layer in the cell's picture, replacing the last."""
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


def register(shell: Any = None) -> None:
    """Registers `%%gmacs` with *shell*, the running IPython when none is given."""
    if shell is None:
        from IPython import get_ipython

        shell = get_ipython()
    shell.register_magics(GmacsMagics)


def load_ipython_extension(shell: Any) -> None:
    """What `%load_ext sara_notebook` calls."""
    register(shell)
