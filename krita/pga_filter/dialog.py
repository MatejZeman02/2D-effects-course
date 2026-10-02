"""The filter's dialog: a control for each parameter, a live preview and Apply.

Written for PyQt6, which Krita 6 ships. Every enum is spelled in full and
exec() is used rather than exec_(), so the same file runs on PyQt5 in
Krita 5.2 too.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QColor, QImage, QPixmap
    from PyQt6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDialog,
                                 QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                                 QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)
except ImportError:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QImage, QPixmap
    from PyQt5.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDialog,
                                 QDoubleSpinBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                                 QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from . import engine
from . import params as P
from . import pixels

PREVIEW = 420        # the longest side of the preview, in pixels
WAIT_MS = 120        # how long the controls rest before the preview is redrawn


class SliderControl(QWidget):
    """A slider with a number box beside it, whole numbers or decimals."""

    def __init__(self, param: P.Param, changed) -> None:
        super().__init__()
        self.param = param
        self.steps = max(1, round((param.maximum - param.minimum) / param.step))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.steps)
        if param.kind == P.WHOLE:
            self.box = QSpinBox()
        else:
            self.box = QDoubleSpinBox()
            self.box.setDecimals(min(4, max(1, 1 - math.floor(math.log10(param.step)))))
        self.box.setRange(param.minimum, param.maximum)
        self.box.setSingleStep(param.step)
        self.box.setMinimumWidth(80)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.slider, 1)
        row.addWidget(self.box)
        self.set_value(param.default)
        self.slider.valueChanged.connect(self._from_slider)
        self.box.valueChanged.connect(self._from_box)
        self.changed = changed

    def _from_slider(self, i: int) -> None:
        self.box.blockSignals(True)
        self.box.setValue(self.param.minimum + i * self.param.step)
        self.box.blockSignals(False)
        self.changed()

    def _from_box(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round((value - self.param.minimum) / self.param.step))
        self.slider.blockSignals(False)
        self.changed()

    def set_value(self, value) -> None:
        for widget in (self.slider, self.box):
            widget.blockSignals(True)
        self.box.setValue(value)
        self.slider.setValue(round((value - self.param.minimum) / self.param.step))
        for widget in (self.slider, self.box):
            widget.blockSignals(False)

    def value(self):
        return int(self.box.value()) if self.param.kind == P.WHOLE else float(self.box.value())


class ColourControl(QPushButton):
    """A button painted in the colour, which opens a colour picker."""

    def __init__(self, param: P.Param, changed) -> None:
        super().__init__()
        self.param = param
        self.changed = changed
        self.setMinimumWidth(80)
        self.set_value(param.default)
        self.clicked.connect(self._pick)

    def _pick(self) -> None:
        if self.param.alpha:
            colour = QColorDialog.getColor(self.colour, self, self.param.name,
                                           QColorDialog.ColorDialogOption.ShowAlphaChannel)
        else:
            colour = QColorDialog.getColor(self.colour, self, self.param.name)
        if colour.isValid():
            self.set_value(colour.getRgbF()[:4 if self.param.alpha else 3])
            self.changed()

    def set_value(self, value) -> None:
        self._value = tuple(float(v) for v in value)
        r, g, b = self._value[:3]
        a = self._value[3] if self.param.alpha else 1.0
        self.colour = QColor.fromRgbF(r, g, b, a)
        # Written as PARAMS writes it, alpha last. Qt's own name puts alpha first.
        self.setText("#" + "".join(f"{round(v * 255):02x}" for v in self._value))
        ink = "#000000" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#ffffff"
        self.setStyleSheet(f"background-color: {QColor.fromRgbF(r, g, b).name()}; color: {ink};")

    def value(self):
        return self._value


class ChoiceControl(QComboBox):
    def __init__(self, param: P.Param, changed) -> None:
        super().__init__()
        self.addItems(param.labels)
        self.currentIndexChanged.connect(lambda _: changed())

    def set_value(self, value) -> None:
        self.setCurrentText(value)

    def value(self):
        return self.currentText()


class ToggleControl(QCheckBox):
    def __init__(self, param: P.Param, changed) -> None:
        super().__init__()
        self.setChecked(param.default)
        self.toggled.connect(lambda _: changed())

    def set_value(self, value) -> None:
        self.setChecked(value)

    def value(self):
        return self.isChecked()


class MatrixControl(QWidget):
    """A square grid of number boxes, read row by row."""

    def __init__(self, param: P.Param, changed) -> None:
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        self.boxes = []
        for i in range(param.size):
            for j in range(param.size):
                box = QDoubleSpinBox()
                box.setRange(-10000.0, 10000.0)
                box.setDecimals(3)
                box.setSingleStep(0.1)
                box.setMinimumWidth(64)
                box.valueChanged.connect(lambda _: changed())
                grid.addWidget(box, i, j)
                self.boxes.append(box)
        self.size = param.size
        self.set_value(param.default)

    def set_value(self, value) -> None:
        for box, v in zip(self.boxes, np.asarray(value).ravel()):
            box.blockSignals(True)
            box.setValue(float(v))
            box.blockSignals(False)

    def value(self):
        return np.array([b.value() for b in self.boxes], dtype=np.float32).reshape(self.size, self.size)


CONTROLS = {
    P.SLIDER: SliderControl,
    P.WHOLE: SliderControl,
    P.COLOUR: ColourControl,
    P.CHOICE: ChoiceControl,
    P.TOGGLE: ToggleControl,
    P.MATRIX: MatrixControl,
}


class FilterDialog(QDialog):
    """Shows the effect on a small copy of the region while the controls move."""

    def __init__(self, effect: engine.Effect, doc, node, original: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.effect, self.doc, self.node = effect, doc, node
        self.sources = [pixels.downscale(original, PREVIEW), pixels.centre_crop(original, PREVIEW)]
        self.setWindowTitle(effect.title)

        form = QFormLayout()
        self.controls = {}
        for param in effect.spec:
            control = CONTROLS[param.kind](param, self.schedule)
            self.controls[param.name] = control
            form.addRow(param.name, control)
        if not effect.spec:
            form.addRow(QLabel("The effect has no parameters."))

        self.mode = QComboBox()
        self.mode.addItems(["Whole region, scaled down", "Centre at full size"])
        self.mode.currentIndexChanged.connect(lambda _: self.redraw())
        self.picture = QLabel()
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(PREVIEW, PREVIEW)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message.setStyleSheet("color: #d04040; font-family: monospace;")

        reset = QPushButton("Defaults")
        reset.clicked.connect(self.reset)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply_to_layer)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)

        side = QVBoxLayout()
        side.addLayout(form)
        side.addStretch(1)
        view = QVBoxLayout()
        view.addWidget(self.mode)
        view.addWidget(self.picture, 1)
        view.addWidget(self.message)
        body = QHBoxLayout()
        body.addLayout(side, 1)
        body.addLayout(view)
        buttons = QHBoxLayout()
        buttons.addWidget(reset)
        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(WAIT_MS)
        self.timer.timeout.connect(self.redraw)
        self.redraw()

    def values(self) -> dict:
        return {name: control.value() for name, control in self.controls.items()}

    def schedule(self) -> None:
        self.timer.start()

    def reset(self) -> None:
        for param in self.effect.spec:
            self.controls[param.name].set_value(param.default)
        self.redraw()

    def redraw(self) -> None:
        source = self.sources[self.mode.currentIndex()]
        try:
            result = engine.run(self.effect, source, self.values())
        except engine.EffectError as error:
            self.message.setText(str(error))
            self.apply_button.setEnabled(False)
            return
        self.message.setText("")
        self.apply_button.setEnabled(True)
        self.shown = pixels.to_display(result)
        h, w = self.shown.shape[:2]
        image = QImage(self.shown.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.picture.setPixmap(QPixmap.fromImage(image))

    def apply_to_layer(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            engine.apply_to_layer(self.doc, self.node, self.effect, self.values())
        except (engine.EffectError, pixels.PixelError) as error:
            self.message.setText(str(error))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.accept()
