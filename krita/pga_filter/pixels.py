"""Krita's pixel bytes to a float32 RGBA array and back.

Krita hands a layer's pixels in the layer's own depth and channel order. 8-bit
and 16-bit RGB come as BGRA, half and full float as RGBA, alpha straight and
never premultiplied. apply() always gets an (h, w, 4) float32 array in RGBA
order, 0 to 1 for the integer depths. Float layers may hold values above 1,
and in a linear profile their numbers are linear light, which this module
passes through untouched.

The only Krita calls are in read_layer, write_layer and selection_mask. The
rest is NumPy and runs in plain Python, which is how the tests check it.
"""

from __future__ import annotations

import numpy as np

# depth -> (NumPy type, the value of full intensity or None for float, BGRA order)
DEPTHS = {
    "U8": (np.uint8, 255.0, True),
    "U16": (np.uint16, 65535.0, True),
    "F16": (np.float16, None, False),
    "F32": (np.float32, None, False),
}


class PixelError(ValueError):
    """A layer this template cannot read or write."""


def check_layer(node) -> None:
    """Raises PixelError, in words a student can act on, for a layer the template cannot change."""
    if node is None:
        raise PixelError("No layer is selected.")
    if node.type() != "paintlayer":
        raise PixelError(
            f"Layer \"{node.name()}\" is a {node.type()}. The filter changes paint layers only. "
            "Select one in the Layers docker, or convert the group to a paint layer.")
    if node.locked():
        raise PixelError(f"Layer \"{node.name()}\" is locked. Unlock it in the Layers docker.")
    if node.colorModel() != "RGBA":
        raise PixelError(
            f"Layer \"{node.name()}\" uses the {node.colorModel()} colour model. The filter works on RGB, "
            "convert the image through Image > Convert Image Color Space.")
    if node.colorDepth() not in DEPTHS:
        raise PixelError(f"The filter does not know the bit depth {node.colorDepth()}.")


def to_float(data: bytes, w: int, h: int, depth: str) -> np.ndarray:
    """Krita's bytes for a w by h rectangle as an (h, w, 4) float32 RGBA array."""
    kind, full, bgra = DEPTHS[depth]
    img = np.frombuffer(data, dtype=kind).reshape(h, w, 4).astype(np.float32)
    if full is not None:
        img /= full
    if bgra:
        img = img[..., [2, 1, 0, 3]]
    return np.ascontiguousarray(img)


def to_bytes(img: np.ndarray, depth: str) -> bytes:
    """An (h, w, 4) RGBA array as Krita's bytes for *depth*, clipped for the integer depths."""
    kind, full, bgra = DEPTHS[depth]
    out = np.asarray(img, dtype=np.float32)
    if bgra:
        out = out[..., [2, 1, 0, 3]]
    if full is not None:
        out = np.rint(np.clip(out, 0.0, 1.0) * full)
    return np.ascontiguousarray(out.astype(kind)).tobytes()


def check_result(result, shape: tuple) -> np.ndarray:
    """apply()'s return value as float32, or a PixelError that says what went wrong."""
    if not isinstance(result, np.ndarray):
        raise PixelError(f"apply() must return a NumPy array, it returned {type(result).__name__}.")
    if result.shape != shape:
        raise PixelError(f"apply() returned an array of shape {result.shape}, expected {shape}, "
                         "the same as its input.")
    if not np.all(np.isfinite(result)):
        raise PixelError("apply() returned an array with NaN or infinity, perhaps after a division by zero.")
    return result.astype(np.float32, copy=False)


def blend(original: np.ndarray, result: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """*result* where the selection is full, *original* where it is empty, mixed in between."""
    if mask is None:
        return result
    m = mask[..., None]
    return original * (1.0 - m) + result * m


def region(doc) -> tuple[int, int, int, int]:
    """The rectangle to filter: the selection's bounds inside the canvas, or the whole canvas."""
    x0, y0, x1, y1 = 0, 0, doc.width(), doc.height()
    sel = doc.selection()
    if sel is not None and sel.width() > 0 and sel.height() > 0:
        x0, y0 = max(x0, sel.x()), max(y0, sel.y())
        x1, y1 = min(x1, sel.x() + sel.width()), min(y1, sel.y() + sel.height())
    if x1 <= x0 or y1 <= y0:
        raise PixelError("The selection lies entirely outside the canvas.")
    return x0, y0, x1 - x0, y1 - y0


def selection_mask(doc, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """How selected each pixel is, 0 to 1, or None when nothing is selected."""
    sel = doc.selection()
    if sel is None or sel.width() == 0:
        return None
    data = bytes(sel.pixelData(x, y, w, h))
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0


def read_layer(node, x: int, y: int, w: int, h: int) -> np.ndarray:
    return to_float(bytes(node.pixelData(x, y, w, h)), w, h, node.colorDepth())


def write_layer(node, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    if not node.setPixelData(to_bytes(img, node.colorDepth()), x, y, w, h):
        raise PixelError("Krita refused to write the pixels into the layer.")


def downscale(img: np.ndarray, longest: int) -> np.ndarray:
    """A smaller copy whose longer side is at most *longest*, by averaging blocks of pixels."""
    h, w = img.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / longest)))
    if step == 1:
        return img.copy()
    hh, ww = (h // step) * step, (w // step) * step
    if hh == 0 or ww == 0:
        return img[::step, ::step].copy()
    blocks = img[:hh, :ww].reshape(hh // step, step, ww // step, step, 4)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def centre_crop(img: np.ndarray, size: int) -> np.ndarray:
    """The middle *size* by *size* pixels at full resolution, for judging detail."""
    h, w = img.shape[:2]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return img[top:top + size, left:left + size].copy()


def to_display(img: np.ndarray, checker: int = 8) -> np.ndarray:
    """An RGBA float array as opaque 8-bit RGB over a grey checkerboard, for the preview."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // checker) + (xx // checker)) % 2 == 0, 0.8, 0.6).astype(np.float32)
    a = np.clip(img[..., 3:4], 0.0, 1.0)
    rgb = np.clip(img[..., :3], 0.0, 1.0) * a + board[..., None] * (1.0 - a)
    return np.ascontiguousarray(np.rint(rgb * 255.0).astype(np.uint8))
