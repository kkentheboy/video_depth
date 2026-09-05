# -*- coding: utf-8 -*-
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QCursor, QPainter, QPainterPath, QColor, QPen, QLinearGradient, QImage
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton

from depth_fusion_workers import NoWheelSlider
class SliderValue(QWidget):
    valueChanged = Signal()

    def __init__(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float = 1.0,
        decimals: int = 0,
        suffix: str = "",
        label_width: int = 64,
    ) -> None:
        super().__init__()
        self._decimals = max(0, int(decimals))
        self._scale = 10 ** self._decimals
        self._suffix = suffix
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = max(float(step), 1.0 / self._scale)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setRange(self._to_int(self._minimum), self._to_int(self._maximum))
        self.slider.setSingleStep(max(1, self._to_int(self._step)))
        self.slider.setPageStep(max(1, self._to_int(self._step * 5)))

        # Clickable value label — click to enter numeric edit mode
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setMinimumWidth(label_width)
        self.value_label.setCursor(QCursor(Qt.IBeamCursor))
        self.value_label.setToolTip("点击直接输入数值")
        self.value_label.setObjectName("sliderValueLabel")

        # Inline editor (hidden by default)
        self._editor = QLineEdit()
        self._editor.setAlignment(Qt.AlignRight)
        self._editor.setMinimumWidth(label_width)
        self._editor.setMaximumWidth(label_width + 12)
        self._editor.setObjectName("sliderValueEditor")
        self._editor.hide()
        self._editor.returnPressed.connect(self._commit_edit)
        self._editor.editingFinished.connect(self._commit_edit)
        self._editing = False

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        layout.addWidget(self._editor)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.setValue(value)

        # Direct numeric edit on value label
        self.value_label.mousePressEvent = self._on_label_click  # type: ignore[method-assign]
        self.value_label.mouseDoubleClickEvent = self._on_label_click  # type: ignore[method-assign]

    def _on_label_click(self, event) -> None:  # noqa: ANN001
        if hasattr(event, "button") and event.button() != Qt.LeftButton:
            return
        self._start_edit()

    def _start_edit(self) -> None:
        if self._editing:
            return
        self._editing = True
        raw = self._format_value(self.value())
        if self._suffix and raw.endswith(self._suffix):
            raw = raw[: -len(self._suffix)]
        self._editor.setText(raw)
        self._editor.selectAll()
        self.value_label.hide()
        self._editor.show()
        self._editor.setFocus()

    def _commit_edit(self) -> None:
        if not self._editing:
            return
        text = self._editor.text().strip()
        try:
            v = float(text)
            v = max(self._minimum, min(self._maximum, v))
            self.setValue(v)
        except ValueError:
            pass
        self._editing = False
        self._editor.hide()
        self.value_label.show()

    def _to_int(self, value: float) -> int:
        return int(round(float(value) * self._scale))

    def _from_int(self, value: int) -> float:
        result = float(value) / self._scale
        if self._decimals == 0:
            return int(round(result))
        return round(result, self._decimals)

    def _format_value(self, value: float) -> str:
        if self._decimals == 0:
            text = str(int(round(value)))
        else:
            text = f"{float(value):.{self._decimals}f}"
        return f"{text}{self._suffix}"

    def _on_slider_changed(self, _value: int) -> None:
        self.value_label.setText(self._format_value(self.value()))
        self.valueChanged.emit()

    def value(self):  # noqa: ANN201
        return self._from_int(self.slider.value())

    def setValue(self, value: float) -> None:
        ivalue = self._to_int(value)
        ivalue = max(self.slider.minimum(), min(self.slider.maximum(), ivalue))
        self.slider.setValue(ivalue)
        self.value_label.setText(self._format_value(self.value()))

    def setRange(self, minimum: float, maximum: float) -> None:
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self.slider.setRange(self._to_int(self._minimum), self._to_int(self._maximum))
        self.setValue(self.value())

    def setSingleStep(self, step: float) -> None:
        self._step = max(float(step), 1.0 / self._scale)
        self.slider.setSingleStep(max(1, self._to_int(self._step)))
        self.slider.setPageStep(max(1, self._to_int(self._step * 5)))

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.value_label.setEnabled(enabled)
        self._editor.setEnabled(enabled)

    def blockSignals(self, block: bool) -> bool:  # noqa: N802
        old = super().blockSignals(block)
        self.slider.blockSignals(block)
        return old



class ToneWheelCard(QWidget):
    """Slim five-zone tone card backed by the existing SliderValue controls."""

    def __init__(self, name: str, base_gray: int, offset_slider: SliderValue, exposure_slider: SliderValue, contrast_slider: SliderValue) -> None:
        super().__init__()
        self.setObjectName("toneWheelCard")
        self.offset_slider = offset_slider
        self.exposure_slider = exposure_slider
        self.contrast_slider = contrast_slider
        self.setMinimumWidth(210)
        self.setMinimumHeight(132)
        self.setMaximumHeight(142)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(4)
        title = QLabel(name)
        title.setObjectName("toneWheelTitle")
        title.setStyleSheet("color: #b0bdc8; font-weight: bold;")
        self._swatch = QLabel()
        self._swatch.setObjectName("toneBandSwatch")
        self._swatch.setFixedSize(18, 7)
        self._swatch.setStyleSheet(
            f"QLabel#toneBandSwatch {{ background: rgb({base_gray}, {base_gray}, {base_gray}); border-radius: 3px; border: 1px solid #3d4654; }}"
        )
        reset = QPushButton("↺")
        reset.setObjectName("tinyResetBtn")
        reset.setFixedSize(22, 20)
        reset.setToolTip(f"重置{name}")
        reset.clicked.connect(self.reset)
        top.addWidget(title)
        top.addWidget(self._swatch)
        top.addStretch(1)
        top.addWidget(reset)
        layout.addLayout(top)

        self._prepare_slider(offset_slider)
        self._prepare_slider(exposure_slider)
        self._prepare_slider(contrast_slider)
        layout.addLayout(self._row("偏移", offset_slider))
        layout.addLayout(self._row("曝光", exposure_slider))
        layout.addLayout(self._row("对比度", contrast_slider))

    def _prepare_slider(self, slider: SliderValue) -> None:
        slider.setMinimumHeight(24)
        slider.value_label.setMinimumWidth(44)
        slider.value_label.setMaximumWidth(50)

    def _row(self, name: str, slider: SliderValue) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        lbl = QLabel(name)
        lbl.setObjectName("toneWheelParam")
        lbl.setStyleSheet("color: #7f8ca0;")
        lbl.setFixedWidth(34 if name != "对比度" else 44)
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        return row

    def reset(self) -> None:
        self.offset_slider.setValue(0)
        self.exposure_slider.setValue(0)
        self.contrast_slider.setValue(0)

