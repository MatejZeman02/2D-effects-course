"""The client a student copies: reach the Sara on the screen and its layers.

    import sara
    doc, layer, a = sara.init()        # a new sRGB document in the Sara open, its layer, its canvas
    doc, layer, a = sara.init(width=512, height=512)  # the same at 512 by 512 pixels
    doc, layer, a = sara.init(attach=True)  # or the document already open, the layer selected in it
    doc = sara.connect()               # or just the document, never a new Sara
    layer = doc.layer("Background")    # or doc.layer(0), the lowest, or doc.layer() the selected
    a = layer.read()                   # float32, shape (h, w, 4), the whole canvas, document space
    layer.write(a * 0.5, name="darker")
    chart = doc.import_image("img/color_chart.png")  # a layer called color_chart, which live reads next
    doc.undo_step("fourier magnitude") # names the next write's history step
    layer.show()                       # a cell ending in this shows it, 512 px on the GPU
    layer.show(max_size=None)          # the same at full size
    sara.bench(doc.layer("cells"))     # the GPU time and round trip of the kernel that wrote it
    sara.live(apply, PARAMS)           # a NumPy function with sliders, see sara_live.py
    doc.layers()                       # the stack bottom first, a dict a layer
    copy = layer.duplicate()           # the copy is a Layer and is selected, duplicate("Ink blurred") names it
    copy.opacity = 0.5                 # copy.visible too, both leave the selection alone
    copy.delete()                      # one undo step, and layer.select() picks a layer
    sara.check(layer, mine)            # how far the layer is from an array, printed and answered

One file, the standard library and NumPy, so it is copied into a student's own
folder rather than installed. It speaks the door's HTTP itself, on one
connection it keeps: it reads the file Sara writes when its door listens,
connects over the Unix socket where Python has one and the loopback port
otherwise, which is Windows, and posts each step with the file's token as a
Bearer. With no door to reach and no Sara running, it starts the `sara` on the
`PATH` with its door and waits for it, and otherwise raises a guide to opening
the door, which a notebook shows without a traceback. Its first request says it attaches, so
closing hangs up without `quit` and Sara stays open. Sara drops a client idle
for ten minutes, and the next call attaches again, reading the file afresh. A
call that was sent and never answered raises, since it may have run. An error
the app answers is a `SaraError` saying the app's sentence, its verdict as `said`.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

# The way in this Python has: a Windows CPython has no `AF_UNIX`.
TRANSPORT = "unix" if hasattr(socket, "AF_UNIX") else "tcp"
# Where Sara writes where it listens, in its user folder, unless this names one.
DOOR_FILE_ENV = "SARA_DOOR_FILE"
DOOR_FILE_NAME = "door.json"
LOOPBACK = "127.0.0.1"
TIMEOUT_S = 120.0
VERDICT = "SaCapture: "
# The words of the `layer` step's `done` lines the wrappers read.
DUPLICATED = "duplicated as "
KEPT = "kept the last layer"
# How long `connect` waits for the door of a Sara it started, past which the guide is raised.
START_BOUND_S = 60.0
# The names a running Sara's process goes by, the Linux build behind its wrapper and the Windows one.
PROCESS_NAMES = ("sara.x86_64", "sara.exe")
# The error colour IPython's own tracebacks use, and the colour after it.
RED, PLAIN = "\x1b[0;31m", "\x1b[0m"
# The Saras `connect` started, kept so a child that outlives the call is never collected while it runs.
STARTED: list[subprocess.Popen] = []
# The document `init()` answered, which `layer_of` falls back to until it closes.
_held: Document | None = None


class SaraError(RuntimeError):
    """What the app answered when a step did not do what it was asked, its verdict as `said` when it gave one."""

    def __init__(self, message: str = "", said: dict | None = None) -> None:
        super().__init__(message)
        self.said = said


class NoDoorError(SaraError):
    """No Sara answers where the known file says, which is the guide to opening a door."""

    def __init__(self, found: str = "") -> None:
        super().__init__("\n".join(guide(found)))


def guide(found: str = "", platform: str | None = None) -> list[str]:
    """The lines a cell shows when no door answers: *found*, or that Sara has none open, the two ways
    to open one, the command for *platform*, and to run the cell again."""
    platform = platform or sys.platform
    command = "sara-with-door.cmd in Sara's folder" if platform.startswith("win") else "sara -- --agent-door"
    return [
        found or "Sara has no door open for this Python to reach.",
        "In a running Sara, ticking Modules › Local Server opens it.",
        f"Or start Sara with its door: {command}",
        "Then run the cell again.",
    ]


def door_file(platform: str | None = None) -> Path:
    """The known file, `user://door.json` of an exported Sara on this platform."""
    chosen = os.environ.get(DOOR_FILE_ENV)
    if chosen:
        return Path(chosen).expanduser()
    platform = platform or sys.platform
    if platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Godot"
    elif platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Godot"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "godot"
    return base / "app_userdata" / "Sara" / DOOR_FILE_NAME


def connect(path: str | Path | None = None, *, start: bool = True) -> Document:
    """The document of the Sara whose door the known file, or *path*, names. With no door to reach,
    the Sara on the `PATH` is started with its door when no Sara runs and *start* is left on."""
    known = Path(path) if path is not None else door_file()
    try:
        return Document(Door(known))
    except NoDoorError as error:
        if not start:
            raise
        found = error
    return Document(_start(known, found))


def _start(known: Path, found: NoDoorError) -> Door:
    """Starts `sara` from the `PATH` detached with `-- --agent-door` and attaches once its door
    answers. A Sara that runs is never given a second one beside it, and nor is one that cannot be
    told from none, so both raise *found*, as does a `PATH` with no `sara` on it. A stale known file
    counts as no Sara, since the Sara that wrote it is looked for by its process and not by the file."""
    program = shutil.which("sara")
    if program is None or _sara_runs() is not False:
        raise found
    print("Starting Sara with its door…", flush=True)
    child = subprocess.Popen([program, "--", "--agent-door"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_detached())
    STARTED.append(child)
    deadline = time.monotonic() + START_BOUND_S
    while True:
        try:
            return Door(known)
        except SaraError:
            # A file from the last run names a dead door, and a file being written may not parse yet.
            pass
        ended = child.poll()
        if ended:
            raise NoDoorError(f"Sara started from {program} ended with {ended} and opened no door.") from None
        if time.monotonic() > deadline:
            raise NoDoorError(f"Sara started from {program} opened no door in {START_BOUND_S:g} s.") from None
        time.sleep(0.25)


def _detached() -> dict:
    """What `Popen` takes for a child that outlives this Python and its notebook."""
    if sys.platform.startswith("win"):
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _sara_runs(platform: str | None = None) -> bool | None:
    """Whether a Sara runs here, read off its process, and None where the platform cannot say."""
    platform = platform or sys.platform
    if platform.startswith("linux"):
        try:
            ids = [entry for entry in os.listdir("/proc") if entry.isdigit()]
        except OSError:
            return None
        return any(_is_sara(pid) for pid in ids)
    if platform.startswith("win"):
        try:
            # The console's code page is not UTF-8, and the names looked for are ASCII.
            listed = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, encoding="utf-8", errors="replace", timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        names = [line.split(",")[0].strip('"').lower() for line in listed.stdout.splitlines()]
        # A Godot may be running Sara from the editor, which `tasklist` cannot say.
        return True if any(name in PROCESS_NAMES for name in names) else None if any(name.startswith("godot") for name in names) else False
    return None


def _is_sara(pid: str) -> bool:
    """Whether Linux process *pid* is an exported Sara, or a Godot the editor runs Sara's project in."""
    try:
        name = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
        if not name.startswith("godot"):
            return name in PROCESS_NAMES
        args = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        if "--remote-debug" not in args or "--path" not in args[:-1]:
            return False
        project = Path(args[args.index("--path") + 1].replace("%20", " ")) / "project.godot"
        return 'config/name="Sara"' in project.read_text(encoding="utf-8")
    except OSError:
        return False


def init(
    path: str | Path | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    attach: bool = False,
    start: bool = True,
) -> tuple[Document, Layer, np.ndarray]:
    """What a lesson opens with: connect, and answer the document, the layer selected in Sara and
    its pixels, the whole canvas as `(h, w, 4)`. Inside IPython `%%gmacs` is registered on that document.
    The document is a new sRGB one at the size File > New offers, which **replaces the one open in
    Sara**, so `layer.read()` hands back the numbers a picture holds and not Sara's own Oklab ones.
    `width=512, height=512` sizes it instead, both or neither, named so that NumPy's `(h, w)` order
    is never confused with a picture's width by height.
    `attach=True` keeps the document already open and answers its selected layer instead.
    That layer is the document's `home`, which `live`, `check` and `bench` read when given no source,
    until `import_image` moves it to the picture it opens. With no door to reach it starts Sara when it can,
    as `connect` says, and `start=False` turns that off."""
    global _held
    size = _canvas_size(width, height, attach)
    document = connect(path, start=start)
    try:
        _register_magic(document)
        if not attach:
            _open_srgb_document(document, size)
        layer = document.layer()
        document.home = layer
        _held = document
        return document, layer, layer.read()
    except Exception:
        # The door serves one client at a time, so a call that failed hangs up for the next.
        document.close()
        raise


def _canvas_size(width: int | None, height: int | None, attach: bool) -> list[int] | str:
    """What the `document` step's `size` says for *width* and *height*: `[w, h]`, or `"default"` for neither.
    Checked before anything connects, so a wrong pair starts no Sara."""
    if width is None and height is None:
        return "default"
    if attach:
        raise ValueError("attach=True keeps the open document, so it takes no width or height")
    if width is None or height is None:
        raise ValueError("a canvas takes both width and height, or neither for File > New's size")
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} is a whole number of pixels, at least 1, not {value!r}")
    return [int(width), int(height)]


def _open_srgb_document(document: Document, size: list[int] | str = "default") -> None:
    """Replaces the document open in Sara with a new sRGB one at *size*, File > New's when "default", which
    the `document` step's verdict says it made, a canvas it found nowhere to put it in being a verdict as well."""
    said = document.door.verdict({"document": "srgb", "size": size})
    if not isinstance(said, str) or not said.startswith("srgb, a new "):
        raise SaraError(f"{VERDICT}document {said}")


def live(apply: Callable[..., np.ndarray], params: dict, source: Layer | str | int | None = None, target: str | None = None) -> None:
    """A NumPy function with a widget for each of *params*, as `sara_live.py` says, imported here for `ipywidgets`."""
    import sara_live

    sara_live.live(apply, params, source, target)


def layer_of(source: Layer | str | int | None) -> Layer:
    """*source* as a Layer, a Layer as it is. A name or an index is on the document `init()` answered,
    held here until it closes so a script dials no second connection to a door serving one client, or
    else the one `%%gmacs` runs on, connected and handed to the notebook when there is none. None is the
    document's `home`, the layer `init()` read or the picture `import_image` opened since, or the one
    selected in Sara when there is neither."""
    if isinstance(source, Layer):
        return source
    document = _held
    if document is None:
        import sara_notebook

        document = sara_notebook.GmacsMagics.document
        if document is None:
            document = connect()
            sara_notebook.use(document)
    return document.layer(source) if source is not None else document.home or document.layer()


def _shell() -> object | None:
    """The IPython shell this runs in, looked for among the modules already loaded and never
    imported, so a script that did not start IPython stays free of it."""
    ipython = sys.modules.get("IPython")
    return ipython.get_ipython() if ipython is not None else None


def _register_handler() -> None:
    """Shows a `SaraError` in IPython as its own lines in the error colour with no traceback. Every
    other exception keeps IPython's traceback, and a plain script keeps the ordinary one for all."""
    shell = _shell()
    if shell is not None:
        shell.set_custom_exc((SaraError,), _show_error)


def _show_error(shell: object, kind: type, error: BaseException, _traceback: object, tb_offset: int | None = None) -> list[str]:
    """Shows *error*'s own lines as IPython shows a traceback, which a kernel sends as the cell's error
    output. IPython drops what a handler answers when the code ran, so the handler shows them itself."""
    lines = [f"{RED}{line}{PLAIN}" for line in str(error).splitlines()]
    show = getattr(shell, "_showtraceback", None)
    if show is not None:
        show(kind, error, lines)
    else:
        print("\n".join(lines), file=sys.stderr)
    return lines


def _register_magic(document: Document) -> None:
    """Registers `%%gmacs` on *document* when this runs in IPython."""
    shell = _shell()
    if shell is None:
        return
    try:
        import sara_notebook
    except ImportError as error:
        print(f"sara.init: %%gmacs is not registered, {error}", file=sys.stderr)
        return
    sara_notebook.register(shell, document)


class _UnixConnection(HTTPConnection):
    """`http.client` over the Unix socket the door listens on in Linux and macOS."""

    def __init__(self, path: str) -> None:
        super().__init__("sara", timeout=TIMEOUT_S)
        self.socket_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


# What a connection closed at once says. Sara serves one client at a time, and a second is closed unheard.
HELD = (
    "Sara closed the connection at once. It serves one client at a time and another holds it: an earlier "
    "sara.init() in this kernel, or another notebook's kernel. Restart this kernel and shut the other one "
    "down, then run this cell again."
)


class _NotSent(SaraError):
    """A request that never went out, so Sara ran none of it."""


class Door:
    """One HTTP connection to a listening Sara, kept open, a step posted and its answer read. Sara
    drops a client idle for ten minutes, so a call that finds the connection closed attaches again."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._next_id = 0
        self.conn: HTTPConnection | None = None
        self._attach()

    def _attach(self) -> None:
        """Reads the known file again, a Sara opened since has a new token, then connects and attaches."""
        self._hang_up()
        path = self.path
        if not path.exists():
            raise NoDoorError()
        try:
            written = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SaraError(f"{path} does not say where Sara listens ({error})") from error
        self.token = str(written.get("token") or "")
        unix = TRANSPORT == "unix" and bool(written.get("socket"))
        if not self.token or not (unix or written.get("port")):
            raise SaraError(f"{path} names no way in this Python can use ({TRANSPORT})")
        self.transport, port = ("unix" if unix else "tcp"), int(written.get("port") or 0)
        self.conn = _UnixConnection(written["socket"]) if unix else HTTPConnection(LOOPBACK, port, timeout=TIMEOUT_S)
        try:
            self.conn.connect()
        except OSError as error:
            self._hang_up()
            raise NoDoorError(f"Sara has no door open: {path} names one that does not answer ({error}).") from error
        try:
            self._write([{"attach": True}])
            self._read([{"attach": True}])
        except SaraError as error:
            # The door closes a second client at once, before it reads a line, so the attach ran nowhere.
            if isinstance(error.__cause__, ConnectionError):
                raise SaraError(HELD) from error.__cause__
            raise

    def send(self, step: dict) -> dict:
        """Sends *step* and answers the app's reply, raising on a refusal."""
        return self.batch([step])[0]

    def batch(self, steps: list[dict]) -> list[dict]:
        """Sends *steps* as one request and answers their replies in order, raising on a refusal. A
        connection Sara closed meanwhile is attached again first, and a request that never went out is
        sent once more. One that went out and was not answered may have run, so it raises instead."""
        fresh = self.conn is None or self._closed()
        if fresh:
            self._attach()
        try:
            self._write(steps)
        except _NotSent:
            if fresh:
                raise
            self._attach()
            self._write(steps)
        return self._read(steps)

    def _closed(self) -> bool:
        """Whether Sara closed the connection while it sat idle. Sara says nothing unasked, so a socket
        with anything to read, an end of stream above all, is not one to send on."""
        if self.conn.sock is None:
            return True
        with selectors.DefaultSelector() as selector:
            selector.register(self.conn.sock, selectors.EVENT_READ)
            return bool(selector.select(0))

    def _write(self, steps: list[dict]) -> None:
        body = [{"id": self._next_id + n, **step} for n, step in enumerate(steps, 1)]
        self._next_id += len(steps)
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        try:
            self.conn.request("POST", "/step", json.dumps(body).encode(), headers)
        except (OSError, HTTPException) as error:
            self._hang_up()
            raise _NotSent(f"Sara hung up before a step went out ({error!r})") from error

    def _read(self, steps: list[dict]) -> list[dict]:
        try:
            reply = self.conn.getresponse()
            answers = json.loads(reply.read().decode())
        except (OSError, ValueError, HTTPException) as error:
            self._hang_up()
            raise SaraError(f"Sara hung up with a step outstanding, which may have run ({error!r})") from error
        if reply.status != 200 or not isinstance(answers, list):
            self._hang_up()
            raise SaraError(f"Sara answered {reply.status}: {answers}")
        for step, answer in zip(steps, answers):
            if not answer.get("ok", True):
                raise SaraError(" ".join(_lines(answer)) or f"Sara refused {json.dumps(step)}")
        return answers

    def _hang_up(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def verdict(self, step: dict) -> str | dict:
        """The step's own verdict, parsed when it is JSON, raising when it says an error.

        The error says the app's own sentence after the step's name, `kernel: no layer called Ink.`,
        since a notebook shows it to a student who never sees the JSON around it. The whole verdict
        stays on the exception as `said`."""
        kind = next(iter(step))
        answer = self.send(step)
        for line in (line for line in _lines(answer) if VERDICT + kind + " " in line):
            at = line.find(VERDICT + kind + " ")
            said = line[at + len(VERDICT) + len(kind) + 1 :].strip()
            try:
                parsed = json.loads(said)
            except ValueError:
                return said
            if isinstance(parsed, dict) and parsed.get("error"):
                raise SaraError(f"{kind}: {str(parsed['error']).strip()}", said=parsed)
            return parsed
        raise SaraError(f"Sara answered {kind} with no verdict line")

    def close(self) -> None:
        """Hangs up and leaves Sara running: an attached client never sends `quit`."""
        self._hang_up()


def _lines(answer: dict) -> list[str]:
    """Every line the answer carries, wherever the door put them."""
    values = [value if isinstance(value, list) else [value] for value in answer.values()]
    return [item for value in values for item in value if isinstance(item, str)]


class Document:
    """The document open in Sara, what `connect()` answers."""

    def __init__(self, door: Door) -> None:
        self.door = door
        self._step_name: str | None = None
        # The layer `init()` read or `import_image` opened last, which `live` runs on when it is given no source.
        self.home: Layer | None = None
        # The last `kernel` step that wrote each layer, by its name, written by `kernel` alone.
        self.kernels: dict[str, dict] = {}
        # The layers whose last kernel read the layer it wrote, which a rerun changes each time.
        self.in_place: set[str] = set()

    def layer(self, key: str | int | None = None) -> Layer:
        """A layer by its name, or by its index from the bottom of the stack, 0 the lowest, or the
        layer selected in Sara when none is given."""
        if key is None:
            return self._selected()
        if not isinstance(key, (str, int)) or isinstance(key, bool):
            raise TypeError(f"a layer is named by a string or an index, not {key!r}")
        return Layer(self, key)

    def layers(self) -> list[dict]:
        """The stack bottom first, one dict a layer: `i`, `name`, `kind`, `visible`, `opacity`, `blend`,
        `active`, `group` (its group's name or None) and `bounds` (`[x, y, w, h]` of its chunks or None)."""
        said = self.door.verdict({"layers": True})
        if not isinstance(said, list):
            raise SaraError(f"{VERDICT}layers {said}")
        return said

    def _selected(self) -> Layer:
        """The layer the layers report marks active, by its name, or by its index when another shares the name."""
        stack = self.layers()
        active = next((entry for entry in stack if entry.get("active")), None)
        if active is None:
            raise SaraError("no layer is selected in Sara")
        shared = sum(entry["name"] == active["name"] for entry in stack) > 1
        return Layer(self, active["i"] if shared else active["name"])

    def new_layer(
        self,
        name: str,
        array: np.ndarray | None = None,
        rect: list[int] | None = None,
        step: str | None = None,
        history: bool = True,
    ) -> Layer:
        """Adds a layer called *name* above the active one and answers it, holding *array* at *rect*
        when one is given, the add and the pixels then being one undo step called *step*. A layer
        already called *name* is the one answered and written, so running a cell again leaves one,
        and *history* False writes it as `Layer.write` says."""
        if array is None:
            said = self.door.verdict({"layer": {"add": name}})
            return Layer(self, str(said.get("layer", name)) if isinstance(said, dict) else name)
        said = _write(self, {"new_layer": name}, array, rect, step, history)
        return Layer(self, said.get("layer", name))

    def import_image(self, path: str | Path) -> Layer:
        """Opens the picture at *path* as a layer named after the file, decoded into the document's
        space, and answers it. The path is resolved against this working directory, because Sara
        runs in its own, and a layer of that name gives way to it, so a rerun leaves one. The picture
        becomes the document's `home`, so `live`, `check` and `bench` given no source read it, as a lesson
        that opens a chart after `init()` means. A layer selected in Sara afterwards does not move it."""
        said = self.door.verdict({"import": os.path.abspath(os.path.expanduser(str(path)))})
        if not isinstance(said, dict) or "layer" not in said:
            raise SaraError(f"{VERDICT}import {said}")
        self.home = Layer(self, str(said["layer"]))
        return self.home

    def undo_step(self, name: str) -> None:
        """Names the history step the next write records."""
        self._step_name = name

    def take_step_name(self) -> str | None:
        """The name `undo_step` gave, once: the write that records it spends it."""
        name, self._step_name = self._step_name, None
        return name

    def kernel(self, step: dict) -> dict:
        """Runs the `kernel` step and answers its verdict, `round_trip_ms` beside the app's `gpu_ms`:
        the wall time of the request, the dispatch and the answer. Kept under the layer it wrote, without
        the keys that say how a rerun is recorded, which the rerun's own sender chooses."""
        began = time.perf_counter()
        said = self.door.verdict({"kernel": step})
        round_trip_ms = (time.perf_counter() - began) * 1000.0
        if not isinstance(said, dict) or not said.get("ok", False):
            raise SaraError(f"{VERDICT}kernel {said}")
        said["round_trip_ms"] = round_trip_ms
        if "into" in said:
            kept = {key: value for key, value in step.items() if key not in ("history", "step")}
            self.kernels[said["into"]] = dict(kept, into=said["into"])
            (self.in_place.add if said.get("in_place") else self.in_place.discard)(said["into"])
        return said

    def undo(self, steps: int = 1) -> None:
        """Takes back *steps* steps of the app's history."""
        self.door.send({"undo": int(steps)})

    def show(self, max_size: int | None = 512) -> Picture:
        """The composited document as Sara's own `look` draws it, *max_size* px on the long side, None for full."""
        folder = Path(tempfile.mkdtemp(prefix="sara-look-"))
        path = folder / "look.png"
        said = self.door.verdict({"look": {"path": str(path), "long_edge": int(max_size or 0)}})
        if not path.exists():
            raise SaraError(f"{VERDICT}look {said}")
        data = path.read_bytes()
        path.unlink()
        folder.rmdir()
        return Picture(data)

    def close(self) -> None:
        global _held
        if _held is self:
            _held = None
        self.door.close()

    def __enter__(self) -> Document:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _write(
    document: Document, target: dict, array: np.ndarray, rect: list[int] | None, name: str | None, history: bool = True
) -> dict:
    """The `write_pixels` step for *target*, `layer` or `new_layer`, and what it answered. Sent with a
    `settle` of 0, since the step prints its verdict before it returns and a slider's drag cannot spare
    the two frames the door waits by default."""
    pixels = np.asarray(array, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"a layer takes (h, w, 4) or (h, w, 3), not {pixels.shape}")
    if pixels.shape[2] == 3:
        pixels = np.concatenate([pixels, np.ones(pixels.shape[:2] + (1,), np.float32)], axis=2)
    height, width = pixels.shape[:2]
    rect = [int(v) for v in rect] if rect is not None else [0, 0, width, height]
    if rect[2:] != [width, height]:
        raise ValueError(f"the array is {width} by {height}, the rectangle {rect[2]} by {rect[3]}")
    handle, path = tempfile.mkstemp(prefix="sara-write-", suffix=".npy")
    os.close(handle)
    try:
        np.save(path, np.ascontiguousarray(pixels))
        step = dict(target, rect=rect, file=path)
        if not history:
            step["history"] = False
        pending = document.take_step_name() if history else None
        chosen = name or pending
        if chosen:
            step["name"] = chosen
        said = document.door.verdict({"write_pixels": step, "settle": 0})
    finally:
        Path(path).unlink(missing_ok=True)
    return said if isinstance(said, dict) else {"said": said}


class Layer:
    """One layer of the document, its pixels read and written as float32 arrays."""

    def __init__(self, document: Document, key: str | int) -> None:
        self.document = document
        self.key = key

    def read(self, rect: list[int] | None = None) -> np.ndarray:
        """The texels of *rect*, `[x, y, w, h]`, or of the whole canvas, as `(h, w, 4)`, zeros where the layer holds nothing."""
        return self._load({"layer": self.key}, rect).astype(np.float32, copy=False)

    def _load(self, step: dict, rect: list[int] | None) -> np.ndarray:
        step = {**step, "rect": [int(v) for v in rect] if rect is not None else "canvas"}
        said = self.document.door.verdict({"read_pixels": step})
        if not isinstance(said, dict) or "path" not in said:
            raise SaraError(f"{VERDICT}read_pixels {said}")
        return np.load(said["path"])

    def write(self, array: np.ndarray, rect: list[int] | None = None, name: str | None = None, history: bool = True) -> dict:
        """Writes *array* at *rect*, the array's size at the origin when none is given, as one undo step.

        With *history* False the write records no step and Sara keeps what it overwrote, so the next
        write with history on records every write since as its one step, the way a slider's drag and
        its let go are one undo. A step recorded, undone or redone in between lets that go."""
        return _write(self.document, {"layer": self.key}, array, rect, name, history)

    def show(self, rect: list[int] | None = None, max_size: int | None = 512) -> Picture:
        """*rect* of the layer, the whole canvas when none is given, in eight bits for the screen, at most
        *max_size* px on the long side, or at full size when it is None. The small form is made on the GPU
        by `read_pixels`'s `max_size`, so a few hundred KB cross where a 1920 by 1080 layer in float32 is
        33 MB. Full size tone-maps the float read here."""
        if max_size is None:
            return Picture(png_bytes(tone_map(self.read(rect))))
        return Picture(png_bytes(self._load({"layer": self.key, "max_size": int(max_size)}, rect)))

    def _act(self, **keys: object) -> dict:
        """One `layer` step aimed at this layer by `of`, so the painter's selection stays where it was."""
        said = self.document.door.verdict({"layer": {"of": self.key, **keys}})
        if not isinstance(said, dict):
            raise SaraError(f"{VERDICT}layer {said}")
        return said

    def _entry(self) -> dict:
        """This layer's row of `Document.layers()`, which is where `visible` and `opacity` are read."""
        stack = self.document.layers()
        if isinstance(self.key, int):
            if not 0 <= self.key < len(stack):
                raise SaraError(f"there is no layer {self.key}, the stack holds {len(stack)}")
            return stack[self.key]
        for entry in stack:
            if entry["name"] == self.key:
                return entry
        raise SaraError(f"there is no layer named '{self.key}'")

    def select(self) -> None:
        """Makes this the layer the brush paints on."""
        name = self._entry()["name"]
        if self._act(activate=name)["active"] != name:
            raise SaraError(f"Sara did not select {name}")

    def duplicate(self, name: str | None = None) -> Layer:
        """Copies the layer above itself as one undo step and answers the copy, which Sara selects, called
        *name* in the same step when given. A copy is a new layer, so a name the stack holds raises."""
        for line in self._act(duplicate=True if name is None else str(name))["done"]:
            if line.startswith(DUPLICATED) and line != DUPLICATED + "nothing":
                return Layer(self.document, line[len(DUPLICATED) :])
        raise SaraError(f"Sara did not duplicate {self.key}")

    def delete(self) -> None:
        """Takes the layer out of the stack as one undo step. The only layer a document has is cleared
        instead, as the menu does, and a locked one is kept, which raises."""
        if KEPT in self._act(delete=True)["done"]:
            raise SaraError(KEPT)

    @property
    def visible(self) -> bool:
        return bool(self._entry()["visible"])

    @visible.setter
    def visible(self, shown: bool) -> None:
        self._act(visible=bool(shown))

    @property
    def opacity(self) -> float:
        """From 0 to 1, read to three places as the stack lists it."""
        return float(self._entry()["opacity"])

    @opacity.setter
    def opacity(self, value: float) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"opacity runs from 0 to 1, not {value!r}")
        self._act(opacity=float(value))


class Timing(NamedTuple):
    """A kernel's GPU time, its dispatch alone, and the round trip Python saw, in milliseconds."""

    gpu_ms: float | None
    round_trip_ms: float

    def line(self, unknown: str = "") -> str:
        """The two as a cell shows them, `GPU 0.21 ms, round trip 38 ms`, with *unknown*, the app's
        reason, where there is no GPU time."""
        gpu = f"GPU {_ms(self.gpu_ms)} ms" if self.gpu_ms is not None else f"GPU time not known ({unknown})"
        return f"{gpu}, round trip {self.round_trip_ms:.0f} ms"


def _ms(value: float) -> str:
    """Two decimals, and two figures for a kernel under a tenth of a millisecond."""
    return f"{value:.2f}" if value >= 0.1 else f"{value:.2g}"


def bench(layer: Layer | str | int, runs: int = 20) -> Timing:
    """Reruns the kernel that last wrote *layer*, a Layer or its name or index, with its values, *runs*
    times, a request each so each has its round trip, prints the medians of both times and answers them.
    The reruns write what the kernel wrote already, so they leave no undo step. A kernel that read the
    layer it wrote changes it each time, and its reruns are one step called "Bench <name>"."""
    layer = layer_of(layer)
    # A kernel is kept under the name it wrote, so an index is looked up in the stack.
    name = layer.key if isinstance(layer.key, str) else layer._entry()["name"]
    step = layer.document.kernels.get(name)
    if step is None:
        raise SaraError(f"no kernel has written {name!r} since connect(), run its cell and name its layer")
    in_place = name in layer.document.in_place
    said = []
    for n in range(max(1, int(runs))):
        # An in-place rerun records once, the first, and the step undoes and redoes all of them.
        record = dict(step, step=f"Bench {step['name']}") if in_place and n == 0 else dict(step, history=False)
        said.append(layer.document.kernel(record))
    gpu = [s["gpu_ms"] for s in said if s.get("gpu_ms") is not None]
    medians = Timing(statistics.median(gpu) if gpu else None, statistics.median(s["round_trip_ms"] for s in said))
    print(f"{medians.line(said[-1].get('gpu_unknown', ''))} (median of {len(said)})")
    return medians


class Difference(NamedTuple):
    """How far a layer is from an array: the largest difference of a channel, the pixel it is in as
    `(x, y)`, None where nothing differs, and the per cent of pixels past the tolerance."""

    largest: float
    where: tuple[int, int] | None
    percent: float


def check(layer: Layer | str | int, array: np.ndarray, tolerance: float = 0.01) -> Difference:
    """Reads *layer*, a Layer or its name or index, and prints how far it is from *array*, `(h, w, 4)`, `(h, w, 3)` for the colour
    alone or `(h, w)` for a grey, and answers the same. Feedback rather than a grade, so it never
    raises on a difference: a pixel on a level boundary that rounds the other way on the GPU is a
    whole level off, and the share past *tolerance* says that it is one pixel in thousands."""
    mine = np.asarray(array, dtype=np.float32)
    if mine.ndim == 2:
        mine = mine[..., None]
    pixels = layer_of(layer).read()
    if mine.ndim != 3 or mine.shape[:2] != pixels.shape[:2] or mine.shape[2] not in (1, 3, 4):
        raise ValueError(f"the layer is {pixels.shape[1]} by {pixels.shape[0]}, the array's shape {np.shape(array)}")
    channels = 3 if mine.shape[2] == 1 else mine.shape[2]
    # A NaN on either side is as far as can be, never quietly equal.
    apart = np.nan_to_num(np.abs(pixels[..., :channels] - mine), nan=np.inf).max(axis=2)
    largest = float(apart.max())
    y, x = np.unravel_index(int(apart.argmax()), apart.shape)
    found = Difference(largest, (int(x), int(y)) if largest > 0 else None, 100.0 * float(np.mean(apart > tolerance)))
    at = f" at x {found.where[0]}, y {found.where[1]}" if found.where else ""
    said = f"largest difference {_figures(largest)}{at}, {_figures(found.percent)} %"
    print(f"{said} of pixels differ by more than {_figures(tolerance)}")
    return found


def _figures(value: float) -> str:
    """Two significant figures with no exponent, so one pixel in two million reads `0.000048 %`."""
    return np.format_float_positional(value, precision=2, fractional=False, trim="-")


class Picture:
    """What a notebook shows inline: a cell that ends in one draws the PNG."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def _repr_png_(self) -> bytes:
        return self.data


def tone_map(pixels: np.ndarray) -> np.ndarray:
    """Eight bits for a screen: each channel clipped to [0, 1] and rounded, a copy."""
    return np.rint(np.clip(np.nan_to_num(pixels), 0.0, 1.0) * 255.0).astype(np.uint8)


def png_bytes(rgba: np.ndarray) -> bytes:
    """An eight bit RGBA PNG of *rgba*, `(h, w, 4)` of `uint8`, by `zlib` and `struct` alone."""
    height, width = rgba.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), rgba.reshape(height, width * 4)], axis=1)

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes()))
        + chunk(b"IEND", b"")
    )


_register_handler()
