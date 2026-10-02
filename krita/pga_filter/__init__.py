"""Krita loads this when the plugin is enabled: NumPy's folder joins the path,
then the extension registers its menu entry."""

import os
import sys

_VENDOR = os.path.join(os.path.dirname(__file__), "vendor",
                       "windows" if sys.platform == "win32" else "linux")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from krita import Krita  # noqa: E402

from .extension import FilterExtension  # noqa: E402

Krita.instance().addExtension(FilterExtension(Krita.instance()))
