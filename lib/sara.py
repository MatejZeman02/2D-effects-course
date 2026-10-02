"""The client a student copies: reach the Sara on the screen and its layers.

    import sara
    doc = sara.connect()               # the Sara already open, never a new one
    layer = doc.layer("Background")    # or doc.layer(0), the lowest
    a = layer.read()                   # float32, shape (h, w, 4), document space
    layer.write(a * 0.5, name="darker")
    doc.undo_step("fourier magnitude") # names the next write's history step
    layer.show()                       # a cell ending in this shows it inline

One file, the standard library and NumPy, so it is copied into a student's own
folder rather than installed. It speaks the door's HTTP itself, on one
connection it keeps: it reads the file Sara writes when its door listens,
connects over the Unix socket where Python has one and the loopback port
otherwise, which is Windows, and posts each step with the file's token as a
Bearer. It never launches anything. Its first request says it attaches, so
closing hangs up without `quit` and Sara stays open. An error the app answers is
a `SaraError` carrying the line.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import tempfile
import zlib
from http.client import HTTPConnection, HTTPException
from pathlib import Path

import numpy as np

# The way in this Python has: a Windows CPython has no `AF_UNIX`.
TRANSPORT = "unix" if hasattr(socket, "AF_UNIX") else "tcp"
# Where Sara writes where it listens, in its user folder, unless this names one.
DOOR_FILE_ENV = "SARA_DOOR_FILE"
DOOR_FILE_NAME = "door.json"
LOOPBACK = "127.0.0.1"
TIMEOUT_S = 120.0
VERDICT = "SaCapture: "


class SaraError(RuntimeError):
    """What the app answered when a step did not do what it was asked."""


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


def connect(path: str | Path | None = None) -> Document:
    """The document of the Sara whose door the known file, or *path*, names."""
    return Document(Door(Path(path) if path is not None else door_file()))


class _UnixConnection(HTTPConnection):
    """`http.client` over the Unix socket the door listens on in Linux and macOS."""

    def __init__(self, path: str) -> None:
        super().__init__("sara", timeout=TIMEOUT_S)
        self.socket_path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class Door:
    """One HTTP connection to a listening Sara, kept open, a step posted and its answer read."""

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise SaraError(f"no Sara is listening, {path} is missing: start Sara with its door")
        try:
            written = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise SaraError(f"{path} does not say where Sara listens ({error})") from error
        self._next_id = 0
        self.token = str(written.get("token") or "")
        unix = TRANSPORT == "unix" and bool(written.get("socket"))
        if not self.token or not (unix or written.get("port")):
            raise SaraError(f"{path} names no way in this Python can use ({TRANSPORT})")
        self.transport, port = ("unix" if unix else "tcp"), int(written.get("port") or 0)
        self.conn = _UnixConnection(written["socket"]) if unix else HTTPConnection(LOOPBACK, port, timeout=TIMEOUT_S)
        try:
            self.conn.connect()
        except OSError as error:
            raise SaraError(f"Sara is not listening where {path} says ({error}): start Sara with its door") from error
        self.send({"attach": True})

    def send(self, step: dict) -> dict:
        """Sends *step* and answers the app's reply, raising on a refusal."""
        return self.batch([step])[0]

    def batch(self, steps: list[dict]) -> list[dict]:
        """Sends *steps* as one request and answers their replies in order, raising on a refusal."""
        body = [{"id": self._next_id + n, **step} for n, step in enumerate(steps, 1)]
        self._next_id += len(steps)
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        try:
            self.conn.request("POST", "/step", json.dumps(body).encode(), headers)
            reply = self.conn.getresponse()
            answers = json.loads(reply.read().decode())
        except (OSError, ValueError, HTTPException) as error:
            raise SaraError(f"Sara hung up with a step outstanding ({error!r})") from error
        if reply.status != 200 or not isinstance(answers, list):
            self.conn.close()
            raise SaraError(f"Sara answered {reply.status}: {answers}")
        for step, answer in zip(steps, answers):
            if not answer.get("ok", True):
                raise SaraError(" ".join(_lines(answer)) or f"Sara refused {json.dumps(step)}")
        return answers

    def verdict(self, step: dict) -> str | dict:
        """The step's own verdict, parsed when it is JSON, raising when it says an error."""
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
                raise SaraError(line[at:])
            return parsed
        raise SaraError(f"Sara answered {kind} with no verdict line")

    def close(self) -> None:
        """Hangs up and leaves Sara running: an attached client never sends `quit`."""
        self.conn.close()


def _lines(answer: dict) -> list[str]:
    """Every line the answer carries, wherever the door put them."""
    values = [value if isinstance(value, list) else [value] for value in answer.values()]
    return [item for value in values for item in value if isinstance(item, str)]


class Document:
    """The document open in Sara, what `connect()` answers."""

    def __init__(self, door: Door) -> None:
        self.door = door
        self._step_name: str | None = None

    def layer(self, key: str | int) -> Layer:
        """A layer by its name, or by its index from the bottom of the stack, 0 the lowest."""
        if not isinstance(key, (str, int)) or isinstance(key, bool):
            raise TypeError(f"a layer is named by a string or an index, not {key!r}")
        return Layer(self, key)

    def new_layer(
        self, name: str, array: np.ndarray | None = None, rect: list[int] | None = None, step: str | None = None
    ) -> Layer:
        """Adds a layer called *name* above the active one and answers it, holding *array* at *rect*
        when one is given, the add and the pixels then being one undo step called *step*."""
        if array is None:
            self.door.verdict({"layer": {"add": name}})
            return Layer(self, name)
        said = _write(self, {"new_layer": name}, array, rect, step)
        return Layer(self, said.get("layer", name))

    def undo_step(self, name: str) -> None:
        """Names the history step the next write records."""
        self._step_name = name

    def take_step_name(self) -> str | None:
        """The name `undo_step` gave, once: the write that records it spends it."""
        name, self._step_name = self._step_name, None
        return name

    def undo(self, steps: int = 1) -> None:
        """Takes back *steps* steps of the app's history."""
        self.door.send({"undo": int(steps)})

    def show(self) -> Picture:
        """The composited document as Sara's own `look` draws it for the screen."""
        folder = Path(tempfile.mkdtemp(prefix="sara-look-"))
        path = folder / "look.png"
        said = self.door.verdict({"look": {"path": str(path)}})
        if not path.exists():
            raise SaraError(f"{VERDICT}look {said}")
        data = path.read_bytes()
        path.unlink()
        folder.rmdir()
        return Picture(data)

    def close(self) -> None:
        self.door.close()

    def __enter__(self) -> Document:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _write(document: Document, target: dict, array: np.ndarray, rect: list[int] | None, name: str | None) -> dict:
    """The `write_pixels` step for *target*, `layer` or `new_layer`, and what it answered."""
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
        pending = document.take_step_name()
        chosen = name or pending
        if chosen:
            step["name"] = chosen
        said = document.door.verdict({"write_pixels": step})
    finally:
        Path(path).unlink(missing_ok=True)
    return said if isinstance(said, dict) else {"said": said}


class Layer:
    """One layer of the document, its pixels read and written as float32 arrays."""

    def __init__(self, document: Document, key: str | int) -> None:
        self.document = document
        self.key = key

    def read(self, rect: list[int] | None = None) -> np.ndarray:
        """The texels of *rect*, `[x, y, w, h]`, or of what the layer holds, as `(h, w, 4)`."""
        step: dict = {"layer": self.key}
        if rect is not None:
            step["rect"] = [int(v) for v in rect]
        said = self.document.door.verdict({"read_pixels": step})
        if not isinstance(said, dict) or "path" not in said:
            raise SaraError(f"{VERDICT}read_pixels {said}")
        return np.load(said["path"]).astype(np.float32, copy=False)

    def write(self, array: np.ndarray, rect: list[int] | None = None, name: str | None = None) -> dict:
        """Writes *array* at *rect*, the array's size at the origin when none is given, as one undo step."""
        return _write(self.document, {"layer": self.key}, array, rect, name)

    def show(self, rect: list[int] | None = None) -> Picture:
        """The layer tone-mapped to eight bits for the screen, the float array untouched."""
        return Picture(png_bytes(tone_map(self.read(rect))))


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
