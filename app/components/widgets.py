# -*- coding: utf-8 -*-
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QCursor, QPainter, QPainterPath, QColor, QPen, QLinearGradient, QImage,
    QDragEnterEvent, QDropEvent, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
)
class DropLineEdit(QLineEdit):
    dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setReadOnly(True)
        self.setProperty("dragging", False)

    def _set_dragging(self, active: bool) -> None:
        self.setProperty("dragging", bool(active))
        try:
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
        except Exception:
            pass

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self._set_dragging(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: ANN001, N802
        self._set_dragging(False)
        try:
            event.accept()
        except Exception:
            pass

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_dragging(False)
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        self.dropped.emit(path)

class NoWheelSlider(QSlider):
    """Slider that ignores mouse wheel so page scrolling stays predictable."""

    def wheelEvent(self, event) -> None:  # noqa: ANN001, N802
        event.ignore()

class NoWheelSpinBox(QSpinBox):
    """SpinBox whose value can only change via keyboard/arrows, never the wheel."""

    def wheelEvent(self, event) -> None:  # noqa: ANN001, N802
        event.ignore()

class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """DoubleSpinBox that ignores the mouse wheel."""

    def wheelEvent(self, event) -> None:  # noqa: ANN001, N802
        event.ignore()

class NoWheelComboBox(QComboBox):
    """ComboBox that ignores the mouse wheel so selection stays deliberate."""

    def wheelEvent(self, event) -> None:  # noqa: ANN001, N802
        event.ignore()

class PreviewImageLabel(QLabel):
    double_clicked = Signal()
    dragged = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._overlay_text = ""
        self._interaction_mode = "pan"
        self._source_pixmap: Optional[QPixmap] = None
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._panning = False
        self._last_pos = QPointF(0.0, 0.0)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)

    def _stop_pointer_drag(self) -> None:
        if self._panning:
            self._panning = False
        try:
            self.releaseMouse()
        except Exception:
            pass
        self.setCursor(Qt.OpenHandCursor)

    def setInteractionMode(self, mode: str) -> None:  # noqa: N802
        mode = str(mode or "pan").lower()
        self._interaction_mode = "rotate" if mode == "rotate" else "pan"
        self.setCursor(Qt.OpenHandCursor)

    def setOverlayText(self, text: str) -> None:  # noqa: N802
        self._overlay_text = text
        self.update()

    def setImagePixmap(self, pixmap: QPixmap) -> None:  # noqa: N802
        old_size = self._source_pixmap.size() if self._source_pixmap is not None and not self._source_pixmap.isNull() else None
        new_size = pixmap.size()
        self._source_pixmap = pixmap
        QLabel.clear(self)
        if old_size != new_size:
            self.resetView()
        else:
            self._clamp_offset()
            self.update()

    def clearImage(self, text: str = "") -> None:  # noqa: N802
        self._source_pixmap = None
        self._overlay_text = ""
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        QLabel.clear(self)
        if text:
            self.setText(text)
        self.update()

    def resetView(self) -> None:  # noqa: N802
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self.update()

    def _fit_size(self) -> tuple[float, float]:
        if self._source_pixmap is None or self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return 0.0, 0.0
        sw = float(self._source_pixmap.width())
        sh = float(self._source_pixmap.height())
        scale = min(float(self.width()) / sw, float(self.height()) / sh)
        return sw * scale, sh * scale

    def _image_rect(self, zoom: Optional[float] = None, offset: Optional[QPointF] = None) -> QRectF:
        fw, fh = self._fit_size()
        z = self._zoom if zoom is None else float(zoom)
        off = self._offset if offset is None else offset
        dw = fw * z
        dh = fh * z
        x = (self.width() - dw) / 2.0 + off.x()
        y = (self.height() - dh) / 2.0 + off.y()
        return QRectF(x, y, dw, dh)

    def _clamp_offset(self) -> None:
        rect = self._image_rect()
        max_x = max(0.0, (rect.width() - self.width()) / 2.0)
        max_y = max(0.0, (rect.height() - self.height()) / 2.0)
        if max_x <= 0.0:
            ox = 0.0
        else:
            ox = max(-max_x, min(max_x, self._offset.x()))
        if max_y <= 0.0:
            oy = 0.0
        else:
            oy = max(-max_y, min(max_y, self._offset.y()))
        self._offset = QPointF(ox, oy)

    def wheelEvent(self, event) -> None:  # noqa: ANN001, N802
        if self._source_pixmap is None or self._source_pixmap.isNull():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        old_rect = self._image_rect()
        if old_rect.width() <= 1.0 or old_rect.height() <= 1.0:
            event.ignore()
            return
        pos = event.position()
        rel_x = (pos.x() - old_rect.x()) / old_rect.width()
        rel_y = (pos.y() - old_rect.y()) / old_rect.height()
        factor = 1.12 if delta > 0 else 1.0 / 1.12
        new_zoom = max(1.0, min(8.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            event.accept()
            return
        fw, fh = self._fit_size()
        new_w = fw * new_zoom
        new_h = fh * new_zoom
        centered_x = (self.width() - new_w) / 2.0
        centered_y = (self.height() - new_h) / 2.0
        new_x = pos.x() - rel_x * new_w
        new_y = pos.y() - rel_y * new_h
        self._zoom = new_zoom
        self._offset = QPointF(new_x - centered_x, new_y - centered_y)
        self._clamp_offset()
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001, N802
        if event.button() == Qt.LeftButton and self._source_pixmap is not None and not self._source_pixmap.isNull():
            pos = event.position()
            if not self._image_rect().contains(pos):
                self._stop_pointer_drag()
                super().mousePressEvent(event)
                return
            self._panning = True
            self._last_pos = pos
            self.grabMouse()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001, N802
        # Only rotate/pan while the left mouse button is physically held down.
        # This protects against missed release events after dragging outside the widget/window.
        if self._panning and not (event.buttons() & Qt.LeftButton):
            self._stop_pointer_drag()
            event.accept()
            return
        if self._panning:
            pos = event.position()
            delta = pos - self._last_pos
            self._last_pos = pos
            if self._interaction_mode == "rotate":
                self.dragged.emit(float(delta.x()), float(delta.y()))
            else:
                self._offset = self._offset + delta
                self._clamp_offset()
                self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001, N802
        if event.button() == Qt.LeftButton and self._panning:
            self._stop_pointer_drag()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: ANN001, N802
        if self._panning and not (QApplication.mouseButtons() & Qt.LeftButton):
            self._stop_pointer_drag()
        super().leaveEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: ANN001, N802
        self._stop_pointer_drag()
        super().focusOutEvent(event)

    def hideEvent(self, event) -> None:  # noqa: ANN001, N802
        self._stop_pointer_drag()
        super().hideEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: ANN001, N802
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001, N802
        super().resizeEvent(event)
        self._clamp_offset()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001, N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            painter.drawPixmap(self._image_rect(), self._source_pixmap, QRectF(0, 0, self._source_pixmap.width(), self._source_pixmap.height()))
        if self._overlay_text:
            painter.setRenderHint(QPainter.Antialiasing, True)
            overlay = QRectF(0, 0, self.width(), self.height())
            painter.fillRect(overlay, QColor(10, 12, 16, 128))
            badge_w = min(180.0, max(96.0, self.width() * 0.34))
            badge_h = 34.0
            badge = QRectF(
                (self.width() - badge_w) / 2.0,
                (self.height() - badge_h) / 2.0,
                badge_w,
                badge_h,
            )
            path = QPainterPath()
            path.addRoundedRect(badge, 8, 8)
            painter.fillPath(path, QColor(37, 40, 48, 220))
            painter.setPen(QPen(QColor(229, 138, 34), 1.2))
            painter.drawPath(path)
            painter.setPen(QColor(220, 225, 232))
            painter.drawText(badge, Qt.AlignCenter, self._overlay_text)
        painter.end()


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

