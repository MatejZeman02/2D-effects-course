"""The effect this plugin runs. This is the only file to edit.

Paste PARAMS and apply() from the notebook here unchanged. TITLE names the
dialog. The file is read again every time the filter opens, so there is no
need to restart Krita after an edit.

img is an (h, w, 4) float32 array in RGBA order, 0 to 1. apply() returns an
array of the same shape. The kinds PARAMS can hold are listed in params.py.
"""

import numpy as np

TITLE = "Invert"


def mix(a, b, t):
    """Same as mix in GLSL: (1 - t) * a + t * b."""
    return (1.0 - t) * a + t * b


PARAMS = {
    "amount": (1.0, 0.0, 1.0),   # (default, minimum, maximum)
}


def apply(img, amount):
    out = img.copy()
    rgb = img[..., :3]
    out[..., :3] = mix(rgb, 1.0 - rgb, amount)
    return out
