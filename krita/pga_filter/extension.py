"""The menu entry under Tools > Scripts, and what it does when chosen.

NumPy and the effect are imported only when the entry is chosen, so a missing
NumPy or a mistake in effect.py shows up as a message instead of a plugin
that silently fails to load.

The action takes its id from this package's folder name and its text from
the effect's TITLE. A copy of the whole folder under another name, with the
.desktop file renamed to match, is therefore a second filter that does not
clash with the first.
"""

from __future__ import annotations

import sys

from krita import Extension, Krita

try:
    from PyQt6.QtWidgets import QMessageBox
except ImportError:
    from PyQt5.QtWidgets import QMessageBox

PACKAGE = __package__ or "pga_filter"


def _numpy_problem() -> str:
    """Why NumPy cannot be imported, or an empty string when it can."""
    try:
        import numpy  # noqa: F401
    except ImportError as error:
        return (f"Krita could not find NumPy for its Python {sys.version.split()[0]}.\n\n"
                f"The plugin's vendor folder must hold NumPy for this Python version "
                f"and this system ({sys.platform}).\n\n{error}")
    return ""


class FilterExtension(Extension):
    def __init__(self, parent) -> None:
        super().__init__(parent)

    def setup(self) -> None:
        pass

    def createActions(self, window) -> None:
        title = PACKAGE
        if not _numpy_problem():
            try:
                from . import engine
                title = engine.load().title
            except Exception:
                pass
        action = window.createAction(PACKAGE, f"{title}…", "tools/scripts")
        action.triggered.connect(self.open)

    def open(self) -> None:
        app = Krita.instance()
        window = app.activeWindow()
        parent = window.qwindow() if window else None
        problem = _numpy_problem()
        if problem:
            QMessageBox.warning(parent, PACKAGE, problem)
            return
        from . import engine, pixels
        from .dialog import FilterDialog

        doc = app.activeDocument()
        if doc is None:
            QMessageBox.information(parent, PACKAGE, "Open an image first.")
            return
        node = doc.activeNode()
        try:
            pixels.check_layer(node)
            effect = engine.load()
            x, y, w, h = pixels.region(doc)
            original = pixels.read_layer(node, x, y, w, h)
        except (engine.EffectError, pixels.PixelError) as error:
            QMessageBox.warning(parent, PACKAGE, str(error))
            return
        FilterDialog(effect, doc, node, original, parent).exec()
