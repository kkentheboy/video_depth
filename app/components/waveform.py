# -*- coding: utf-8 -*-
from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QCursor, QPainter, QPainterPath, QColor, QPen, QLinearGradient, QImage
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton

from depth_fusion_core import normalize_curve_points, build_curve_lut, tone_range_reference_bands
class CurveWaveformPanel(QWidget):
    levelsChanged = Signal(int, int, float, int, int)
    curveChanged = Signal(object)

    _C_BG = QColor(0x14, 0x1a, 0x22)
    _C_GRID = QColor(0x2c, 0x36, 0x44)
    _C_LABEL = QColor(0x88, 0x95, 0xa8)
    _C_NUM = QColor(0xc4, 0xcc, 0xd8)
    _C_CURVE = QColor(0xd8, 0xde, 0xe8)
    _C_CURVE_SOFT = QColor(0xd8, 0xde, 0xe8, 48)
    _C_POINT = QColor(0xff, 0xa6, 0x3d)
    _C_WAVE = QColor(0xe8, 0xee, 0xf7)
    _C_CLIP = QColor(0xd0, 0x40, 0x40, 42)
    _C_ACTIVE = QColor(0x30, 0x6d, 0xb0, 26)
    _MAX_CURVE_POINTS = 64

    def __init__(self) -> None:
        super().__init__()
        # Fixed-height parent gives this widget about 250-270px.
        # Geometry below uses wide rectangles, not square graphs.
        self.setMinimumHeight(230)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip("点击曲线区域新增控制点；拖动控制点调整；首尾点可横向/纵向移动；选中中间点后点 × 或按 Delete 删除。")
        self.waveform = np.zeros((512, 512), dtype=np.float32)
        self.waveform_scope_label = "全图"

        # Legacy Levels controls are still part of the post pipeline.
        self.in_black = 0
        self.in_white = 255
        self.gamma = 1.0
        self.out_black = 0
        self.out_white = 255
        self.black_pct = 0.0
        self.white_pct = 100.0
        self.invert = False
        self.tone_values = {
            "black": 0, "shadow": 0, "mid": 0, "light": 0, "white": 0,
            "black_shift": 0, "shadow_shift": 0, "mid_shift": 0, "light_shift": 0, "white_shift": 0,
            "black_contrast": 0, "shadow_contrast": 0, "mid_contrast": 0, "light_contrast": 0, "white_contrast": 0,
        }

        # Photoshop-style free curve points, normalized 0..1.
        self.curve_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._selected_curve_index: Optional[int] = None
        self._hover_curve_index: Optional[int] = None
        self._drag_curve_index: Optional[int] = None
        self._drag_level_target: Optional[str] = None
        self._hover_level_target: Optional[str] = None
        self._hover_delete_button = False

    # ── geometry ───────────────────────────────────────────────────────────

    def _graph_layout(self) -> tuple[QRectF, QRectF, QRectF]:
        margin_l, margin_t, margin_r, gap = 56.0, 26.0, 10.0, 22.0
        bottom_reserved = 62.0
        available_w = max(520.0, float(self.width()) - margin_l - margin_r - gap)
        graph_h = max(150.0, min(220.0, float(self.height()) - margin_t - bottom_reserved))
        curve_w = max(260.0, available_w * 0.50)
        wave_w = max(260.0, available_w - curve_w)
        cr = QRectF(margin_l, margin_t, curve_w, graph_h)
        wr = QRectF(cr.right() + gap, margin_t, wave_w, graph_h)
        gr = QRectF(cr.left(), cr.bottom() + 30.0, cr.width(), 12.0)
        return cr, wr, gr

    def _curve_rect(self) -> QRectF:
        return self._graph_layout()[0]

    def _waveform_rect(self) -> QRectF:
        return self._graph_layout()[1]

    def _grad_rect(self) -> QRectF:
        return self._graph_layout()[2]

    def _x_from_value(self, value: float) -> float:
        r = self._curve_rect()
        return r.left() + (max(0.0, min(255.0, float(value))) / 255.0) * r.width()

    def _value_from_x(self, x: float) -> int:
        r = self._curve_rect()
        if r.width() <= 0:
            return 0
        v = (float(x) - r.left()) / r.width() * 255.0
        return int(round(max(0.0, min(255.0, v))))

    def _point_pos(self, index: int) -> tuple[float, float]:
        cr = self._curve_rect()
        x, y = self.curve_points[index]
        return cr.left() + x * cr.width(), cr.bottom() - y * cr.height()

    def _pos_to_curve_point(self, pos) -> tuple[float, float]:  # noqa: ANN001
        cr = self._curve_rect()
        x = (float(pos.x()) - cr.left()) / max(1.0, cr.width())
        y = (cr.bottom() - float(pos.y())) / max(1.0, cr.height())
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def _mid_value(self) -> float:
        span = max(1.0, self.in_white - self.in_black)
        g = max(0.05, min(5.0, self.gamma))
        return self.in_black + span * (0.5 ** (1.0 / g))

    def _gamma_from_mid_value(self, mid: float) -> float:
        span = max(1.0, self.in_white - self.in_black)
        t = max(0.02, min(0.98, (mid - self.in_black) / span))
        return max(0.05, min(5.0, float(np.log(0.5) / np.log(t))))

    # ── public setters ─────────────────────────────────────────────────────

    def setHistogramFromGray(self, gray: Optional[np.ndarray], subject_mask: Optional[np.ndarray] = None) -> None:  # noqa: N802
        """Build grayscale waveform data. X=image position, Y=gray/depth value.

        When subject_mask is provided, waveform statistics ignore the
        environment and only plot pixels inside the processed subject mask.
        The final rendered depth is not cropped; this is a mask-aware analysis
        scope for Curves/Levels/Tone decisions.
        """
        self.waveform = np.zeros((512, 512), dtype=np.float32)
        self.waveform_scope_label = "全图"
        if gray is not None:
            arr = np.asarray(gray)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            arr = np.squeeze(arr)
            if arr.ndim == 2 and arr.size > 0:
                if arr.dtype != np.uint8:
                    a = arr.astype(np.float32)
                    if float(np.nanmax(a)) <= 1.5:
                        a = a * 255.0
                    arr_u8 = np.clip(a, 0, 255).astype(np.uint8)
                else:
                    arr_u8 = arr
                wf_w = 512
                wf_h = 512
                small = cv2.resize(arr_u8, (wf_w, wf_h), interpolation=cv2.INTER_AREA)
                mask_small = None
                if subject_mask is not None:
                    try:
                        mask_arr = np.asarray(subject_mask, dtype=np.float32)
                        if mask_arr.ndim == 3:
                            mask_arr = mask_arr[..., 0]
                        mask_arr = np.squeeze(mask_arr)
                        if mask_arr.ndim == 2 and mask_arr.size > 0:
                            if mask_arr.shape[:2] != arr_u8.shape[:2]:
                                mask_arr = cv2.resize(mask_arr, (arr_u8.shape[1], arr_u8.shape[0]), interpolation=cv2.INTER_LINEAR)
                            mask_small = cv2.resize(mask_arr, (wf_w, wf_h), interpolation=cv2.INTER_AREA)
                            mask_small = np.asarray(mask_small, dtype=np.float32) > 0.08
                            if int(mask_small.sum()) < 16:
                                mask_small = None
                            else:
                                self.waveform_scope_label = "主体Mask"
                    except Exception:
                        mask_small = None
                wf = np.zeros((wf_h, wf_w), dtype=np.float32)
                xs_full = np.tile(np.arange(wf_w, dtype=np.int32), small.shape[0])
                vals_full = small.reshape(-1).astype(np.int32)
                if mask_small is not None:
                    valid = mask_small.reshape(-1)
                    xs = xs_full[valid]
                    vals = vals_full[valid]
                else:
                    xs = xs_full
                    vals = vals_full
                if vals.size > 0:
                    ys = (wf_h - 1) - np.clip(np.round(vals.astype(np.float32) / 255.0 * (wf_h - 1)), 0, wf_h - 1).astype(np.int32)
                    np.add.at(wf, (ys, xs), 1.0)
                    wf = cv2.GaussianBlur(wf, (0, 0), sigmaX=0.9, sigmaY=1.8)
                    wf = np.log1p(wf)
                    max_wf = float(wf.max())
                    if max_wf > 1e-6:
                        wf /= max_wf
                    self.waveform = np.clip(wf, 0.0, 1.0).astype(np.float32)
        self.update()

    def setValues(
        self,
        in_black: int,
        in_white: int,
        gamma: float,
        out_black: int,
        out_white: int,
        emit: bool = False,
    ) -> None:  # noqa: A002
        self.in_black = max(0, min(254, int(in_black)))
        self.in_white = max(1, min(255, int(in_white)))
        if self.in_white <= self.in_black:
            self.in_white = min(255, self.in_black + 1)
        self.gamma = max(0.05, min(5.0, float(gamma)))
        self.out_black = max(0, min(255, int(out_black)))
        self.out_white = max(0, min(255, int(out_white)))
        self.update()
        if emit:
            self.levelsChanged.emit(self.in_black, self.in_white, self.gamma, self.out_black, self.out_white)

    def setCurvePoints(self, points, emit: bool = False) -> None:  # noqa: ANN001, N802
        pts = list(normalize_curve_points(points))
        if len(pts) > self._MAX_CURVE_POINTS:
            middle_keep = max(0, self._MAX_CURVE_POINTS - 2)
            pts = [pts[0], *pts[1:-1][:middle_keep], pts[-1]]
        self.curve_points = [(float(x), float(y)) for x, y in normalize_curve_points(pts)]
        self._selected_curve_index = None
        self._hover_curve_index = None
        self._drag_curve_index = None
        self.update()
        if emit:
            self.curveChanged.emit(self.getCurvePoints())

    def getCurvePoints(self) -> list[list[float]]:  # noqa: N802
        return [[round(float(x), 6), round(float(y), 6)] for x, y in normalize_curve_points(self.curve_points)]

    def resetCurve(self, emit: bool = True) -> None:  # noqa: N802
        self.setCurvePoints([(0.0, 0.0), (1.0, 1.0)], emit=emit)

    def setNormalize(self, black_pct: float, white_pct: float, invert: bool) -> None:  # noqa: N802
        self.black_pct = float(black_pct)
        self.white_pct = float(white_pct)
        self.invert = bool(invert)
        self.update()

    def setToneValues(
        self,
        tone_black: int,
        tone_shadow: int,
        tone_mid: int,
        tone_light: int,
        tone_white: int,
        tone_black_shift: int,
        tone_shadow_shift: int,
        tone_mid_shift: int,
        tone_light_shift: int,
        tone_white_shift: int,
        tone_black_contrast: int = 0,
        tone_shadow_contrast: int = 0,
        tone_mid_contrast: int = 0,
        tone_light_contrast: int = 0,
        tone_white_contrast: int = 0,
    ) -> None:  # noqa: N802
        self.tone_values = {
            "black": int(tone_black),
            "shadow": int(tone_shadow),
            "mid": int(tone_mid),
            "light": int(tone_light),
            "white": int(tone_white),
            "black_shift": int(tone_black_shift),
            "shadow_shift": int(tone_shadow_shift),
            "mid_shift": int(tone_mid_shift),
            "light_shift": int(tone_light_shift),
            "white_shift": int(tone_white_shift),
            "black_contrast": int(tone_black_contrast),
            "shadow_contrast": int(tone_shadow_contrast),
            "mid_contrast": int(tone_mid_contrast),
            "light_contrast": int(tone_light_contrast),
            "white_contrast": int(tone_white_contrast),
        }
        self.update()

    # ── drawing helpers ────────────────────────────────────────────────────

    def _mapping_curve(self) -> np.ndarray:
        return build_curve_lut(self.curve_points, 256)

    def _draw_panel_bg(self, p: QPainter, r: QRectF, title: str) -> None:
        path = QPainterPath()
        path.addRoundedRect(r, 3, 3)
        p.fillPath(path, self._C_BG)
        p.setPen(QPen(self._C_GRID, 1))
        for i in range(1, 4):
            gx = r.left() + r.width() * i / 4.0
            p.drawLine(QPointF(gx, r.top()), QPointF(gx, r.bottom()))
        for level in (0, 64, 128, 192, 255):
            gy = r.bottom() - (level / 255.0) * r.height()
            p.drawLine(QPointF(r.left(), gy), QPointF(r.right(), gy))
        p.setPen(self._C_LABEL)
        p.drawText(int(r.left() + 6), int(r.top() + 15), title)
        p.setPen(QColor(0x7f, 0x8c, 0xa3))
        p.drawText(int(r.left() + 6), int(r.top() + 31), "255")
        p.drawText(int(r.left() + 6), int(r.center().y() + 4), "128")
        p.drawText(int(r.left() + 6), int(r.bottom() - 5), "0")

    def _draw_circle_handle(self, p: QPainter, x: float, y: float, radius: float, color: QColor, active: bool) -> None:
        p.setBrush(color)
        p.setPen(QPen(QColor(0x80, 0x8a, 0x98) if active else QColor(0x56, 0x60, 0x70), 1.2))
        p.drawEllipse(QPointF(x, y), radius + (1.5 if active else 0), radius + (1.5 if active else 0))

    def _draw_tone_reference_bands(self, p: QPainter, r: QRectF) -> None:
        """Draw dynamic five-zone references without turning them into thick bars.

        The zones are computed from the same helper used by processing.  Center
        lines show each zone center; short top spans show the effective width.
        This keeps the curve readable while still making the tone sliders
        spatially understandable.
        """
        shifts = [
            int(self.tone_values.get("black_shift", 0)),
            int(self.tone_values.get("shadow_shift", 0)),
            int(self.tone_values.get("mid_shift", 0)),
            int(self.tone_values.get("light_shift", 0)),
            int(self.tone_values.get("white_shift", 0)),
        ]
        bands = tone_range_reference_bands(*shifts)
        labels = ["黑", "暗", "中", "亮", "高"]
        p.save()
        p.setClipRect(r.adjusted(1, 1, -1, -1))
        for (center, width), label in zip(bands, labels):
            c = max(0.0, min(1.0, float(center)))
            left_n = max(0.0, c - float(width))
            right_n = min(1.0, c + float(width))
            left = r.left() + left_n * r.width()
            right = r.left() + right_n * r.width()
            x = r.left() + c * r.width()

            # Influence width: a small cap only, not a full-height colored strip.
            cap_y = r.top() + 20.0
            p.setPen(QPen(QColor(0xe5, 0x8a, 0x22, 72), 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(left, cap_y), QPointF(right, cap_y))
            p.setPen(QPen(QColor(0xe5, 0x8a, 0x22, 82), 1.0, Qt.DashLine))
            p.drawLine(QPointF(x, r.top() + 24.0), QPointF(x, r.bottom() - 1.0))

            p.setPen(QColor(0xe8, 0xb0, 0x72, 145))
            p.drawText(QRectF(x - 10, r.top() + 3, 20, 13), Qt.AlignCenter, label)
        p.restore()

    def _delete_button_rect(self) -> Optional[QRectF]:
        idx = self._selected_curve_index
        if idx is None or idx <= 0 or idx >= len(self.curve_points) - 1:
            return None
        x, y = self._point_pos(idx)
        return QRectF(x + 9.0, y - 25.0, 20.0, 20.0)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        cr = self._curve_rect()
        wr = self._waveform_rect()
        gr = self._grad_rect()

        # Left: Photoshop-like editable LUT curve.
        self._draw_panel_bg(p, cr, "灰度曲线")

        clip_l = self._x_from_value(self.in_black)
        clip_r = self._x_from_value(self.in_white)
        if self.in_black > 0:
            p.fillRect(QRectF(cr.left(), cr.top(), clip_l - cr.left(), cr.height()), self._C_CLIP)
        if self.in_white < 255:
            p.fillRect(QRectF(clip_r, cr.top(), cr.right() - clip_r, cr.height()), self._C_CLIP)
        p.fillRect(QRectF(clip_l, cr.top(), clip_r - clip_l, cr.height()), self._C_ACTIVE)

        # Diagonal reference.
        p.setPen(QPen(QColor(0xff, 0xff, 0xff, 42), 1, Qt.DashLine))
        p.drawLine(QPointF(cr.left(), cr.bottom()), QPointF(cr.right(), cr.top()))

        # Five-zone tone reference: dynamic centers/widths from the same helper used by processing.
        self._draw_tone_reference_bands(p, cr)

        curve = self._mapping_curve()
        curve_path = QPainterPath()
        for i, v in enumerate(curve):
            x_pos = cr.left() + (i / 255.0) * cr.width()
            y_pos = cr.bottom() - float(v) * cr.height()
            if i == 0:
                curve_path.moveTo(QPointF(x_pos, y_pos))
            else:
                curve_path.lineTo(QPointF(x_pos, y_pos))
        p.setPen(QPen(self._C_CURVE_SOFT, 5.0))
        p.drawPath(curve_path)
        p.setPen(QPen(self._C_CURVE, 2.0))
        p.drawPath(curve_path)

        # Free curve points.
        for i, _pt in enumerate(self.curve_points):
            px, py = self._point_pos(i)
            active = i == self._selected_curve_index or i == self._hover_curve_index or i == self._drag_curve_index
            color = self._C_POINT if active else QColor(0x1c, 0x21, 0x2a)
            p.setBrush(color)
            p.setPen(QPen(QColor(0xff, 0xe1, 0xa8) if active else self._C_POINT, 1.4))
            p.drawEllipse(QPointF(px, py), 5.8 + (1.5 if active else 0.0), 5.8 + (1.5 if active else 0.0))

        del_rect = self._delete_button_rect()
        if del_rect is not None:
            del_path = QPainterPath()
            del_path.addRoundedRect(del_rect, 4, 4)
            p.fillPath(del_path, QColor(0x60, 0x22, 0x24) if self._hover_delete_button else QColor(0x38, 0x1e, 0x22))
            p.setPen(QPen(QColor(0xff, 0x8a, 0x8a), 1.4))
            p.drawText(del_rect, Qt.AlignCenter, "×")

        # Curve border.
        p.setPen(QPen(QColor(0x3a, 0x40, 0x4c), 1))
        p.setBrush(Qt.NoBrush)
        border = QPainterPath()
        border.addRoundedRect(cr, 3, 3)
        p.drawPath(border)

        # Legacy input handles.
        y_in = cr.bottom() + 14.0
        r_in = 6.5
        x_blk = self._x_from_value(self.in_black)
        x_wht = self._x_from_value(self.in_white)
        x_mid = self._x_from_value(self._mid_value())
        p.setPen(QPen(QColor(0, 0, 0, 60), 1))
        p.drawLine(QPointF(x_blk, cr.bottom() + 2), QPointF(x_blk, y_in))
        p.drawLine(QPointF(x_wht, cr.bottom() + 2), QPointF(x_wht, y_in))

        self._draw_circle_handle(p, x_blk, y_in, r_in, QColor(0x30, 0x34, 0x3c), self._hover_level_target == "in_black" or self._drag_level_target == "in_black")
        self._draw_circle_handle(p, x_mid, y_in, r_in, QColor(0x9a, 0x9e, 0xaa), self._hover_level_target == "gamma" or self._drag_level_target == "gamma")
        self._draw_circle_handle(p, x_wht, y_in, r_in, QColor(0xee, 0xf0, 0xf4), self._hover_level_target == "in_white" or self._drag_level_target == "in_white")

        # Output gradient and handles.
        grad = QLinearGradient(gr.left(), 0, gr.right(), 0)
        grad.setColorAt(0.0, QColor(0, 0, 0))
        grad.setColorAt(1.0, QColor(255, 255, 255))
        grad_path = QPainterPath()
        grad_path.addRoundedRect(gr, 2, 2)
        p.fillPath(grad_path, grad)
        p.setPen(QPen(QColor(0x28, 0x2c, 0x34), 1))
        p.drawPath(grad_path)

        y_out = gr.bottom() + 12.0
        r_out = 6.0
        x_ob = self._x_from_value(self.out_black)
        x_ow = self._x_from_value(self.out_white)
        self._draw_circle_handle(p, x_ob, y_out, r_out, QColor(0x30, 0x34, 0x3c), self._hover_level_target == "out_black" or self._drag_level_target == "out_black")
        self._draw_circle_handle(p, x_ow, y_out, r_out, QColor(0xee, 0xf0, 0xf4), self._hover_level_target == "out_white" or self._drag_level_target == "out_white")

        # Right: grayscale waveform information panel.
        self._draw_panel_bg(p, wr, f"亮度波形图 · {self.waveform_scope_label}")
        waveform_inner = wr.adjusted(1.0, 1.0, -1.0, -1.0)
        p.fillRect(waveform_inner, QColor(0x00, 0x00, 0x00))
        if self.waveform.size > 0 and float(self.waveform.max()) > 0:
            wf = np.clip(self.waveform, 0.0, 1.0)
            density = np.power(wf, 0.60)
            density = np.clip((density - 0.025) / 0.975, 0.0, 1.0)
            white_mix = np.clip((density - 0.42) / 0.58, 0.0, 1.0)
            white_mix = white_mix * white_mix * (3.0 - 2.0 * white_mix)

            white = np.array([0xf8, 0xfa, 0xff], dtype=np.float32)
            rgb = white[None, None, :] * np.clip(0.32 + 0.68 * white_mix[..., None], 0.0, 1.0)

            alpha = np.clip(density * 248.0, 0, 248).astype(np.uint8)
            img = np.zeros((wf.shape[0], wf.shape[1], 4), dtype=np.uint8)
            img[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
            img[..., 3] = alpha
            qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGBA8888)
            p.drawImage(wr, qimg)
            p.setPen(QColor(0xe8, 0xb0, 0x72, 120))
            p.drawText(int(wr.left() + 6), int(wr.top() + 15), f"亮度波形图 · {self.waveform_scope_label}")
            p.setPen(QColor(0x75, 0x82, 0x96))
            p.drawText(int(wr.left() + 6), int(wr.top() + 31), "255")
            p.drawText(int(wr.left() + 6), int(wr.center().y() + 4), "128")
            p.drawText(int(wr.left() + 6), int(wr.bottom() - 5), "0")
        else:
            p.setPen(QColor(0x68, 0x75, 0x88))
            p.drawText(wr, Qt.AlignCenter, "未读取当前帧")

        wf_border = QPainterPath()
        wf_border.addRoundedRect(wr, 3, 3)
        p.setPen(QPen(QColor(0x3a, 0x40, 0x4c), 1))
        p.setBrush(Qt.NoBrush)
        p.drawPath(wf_border)

        # Keep the graph area clean; detailed numeric controls live below the panel.
        p.end()

    # ── interaction ────────────────────────────────────────────────────────

    def _level_handle_positions(self) -> dict[str, tuple[float, float]]:
        cr = self._curve_rect()
        gr = self._grad_rect()
        y_in = cr.bottom() + 14.0
        y_out = gr.bottom() + 12.0
        return {
            "in_black": (self._x_from_value(self.in_black), y_in),
            "gamma": (self._x_from_value(self._mid_value()), y_in),
            "in_white": (self._x_from_value(self.in_white), y_in),
            "out_black": (self._x_from_value(self.out_black), y_out),
            "out_white": (self._x_from_value(self.out_white), y_out),
        }

    def _nearest_level_handle(self, pos, radius: float = 20.0) -> Optional[str]:  # noqa: ANN001
        px, py = float(pos.x()), float(pos.y())
        best_name: Optional[str] = None
        best_dist = radius
        for name, (hx, hy) in self._level_handle_positions().items():
            d = ((px - hx) ** 2 + (py - hy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name

    def _nearest_curve_point(self, pos, radius: float = 13.0) -> Optional[int]:  # noqa: ANN001
        px, py = float(pos.x()), float(pos.y())
        best_idx: Optional[int] = None
        best_dist = radius
        for i in range(len(self.curve_points)):
            hx, hy = self._point_pos(i)
            d = ((px - hx) ** 2 + (py - hy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _add_curve_point(self, pos) -> int:  # noqa: ANN001
        if len(self.curve_points) >= self._MAX_CURVE_POINTS:
            return max(0, len(self.curve_points) - 1)
        x, y = self._pos_to_curve_point(pos)
        points = list(self.curve_points)
        insert_at = 1
        for i, (px, _py) in enumerate(points):
            if x > px:
                insert_at = i + 1
        insert_at = max(1, min(len(points) - 1, insert_at))
        left_x = points[insert_at - 1][0]
        right_x = points[insert_at][0]
        min_gap = 1.0 / 255.0
        x = max(left_x + min_gap, min(right_x - min_gap, x))
        points.insert(insert_at, (x, y))
        self.curve_points = [(float(px), float(py)) for px, py in normalize_curve_points(points)]
        self._selected_curve_index = insert_at
        self._emit_curve_changed()
        return insert_at

    def _delete_selected_curve_point(self) -> bool:
        idx = self._selected_curve_index
        if idx is None or idx <= 0 or idx >= len(self.curve_points) - 1:
            return False
        points = list(self.curve_points)
        del points[idx]
        self.curve_points = [(float(px), float(py)) for px, py in normalize_curve_points(points)]
        self._selected_curve_index = min(idx, len(self.curve_points) - 2)
        if self._selected_curve_index <= 0:
            self._selected_curve_index = None
        self._emit_curve_changed()
        return True

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        pos = event.position()
        self.setFocus(Qt.MouseFocusReason)
        if event.button() == Qt.RightButton:
            idx = self._nearest_curve_point(pos)
            if idx is not None:
                self._selected_curve_index = idx
                if self._delete_selected_curve_point():
                    event.accept()
                    return
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            del_rect = self._delete_button_rect()
            if del_rect is not None and del_rect.contains(pos):
                if self._delete_selected_curve_point():
                    event.accept()
                    return
            idx = self._nearest_curve_point(pos)
            if idx is not None:
                self._selected_curve_index = idx
                self._drag_curve_index = idx
                self._apply_curve_drag(pos)
                event.accept()
                return
            if self._curve_rect().contains(pos):
                idx = self._add_curve_point(pos)
                self._drag_curve_index = idx
                event.accept()
                return
            self._drag_level_target = self._nearest_level_handle(pos)
            if self._drag_level_target:
                self._apply_level_drag(pos.x())
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        pos = event.position()
        if self._drag_curve_index is not None:
            self._apply_curve_drag(pos)
            event.accept()
            return
        if self._drag_level_target:
            self._apply_level_drag(pos.x())
            event.accept()
            return
        del_rect = self._delete_button_rect()
        delete_hit = bool(del_rect is not None and del_rect.contains(pos))
        hit_curve = self._nearest_curve_point(pos)
        hit_level = self._nearest_level_handle(pos)
        if delete_hit != self._hover_delete_button or hit_curve != self._hover_curve_index or hit_level != self._hover_level_target:
            self._hover_delete_button = delete_hit
            self._hover_curve_index = hit_curve
            self._hover_level_target = hit_level
            if delete_hit:
                self.setCursor(QCursor(Qt.PointingHandCursor))
            elif hit_curve is not None:
                self.setCursor(QCursor(Qt.OpenHandCursor))
            elif hit_level is not None:
                self.setCursor(QCursor(Qt.SizeHorCursor))
            elif self._curve_rect().contains(pos):
                self.setCursor(QCursor(Qt.CrossCursor))
            else:
                self.setCursor(QCursor(Qt.ArrowCursor))
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        self._drag_curve_index = None
        self._drag_level_target = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: ANN001
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._delete_selected_curve_point():
                event.accept()
                return
        super().keyPressEvent(event)

    def _apply_curve_drag(self, pos) -> None:  # noqa: ANN001
        idx = self._drag_curve_index
        if idx is None or idx < 0 or idx >= len(self.curve_points):
            return
        x, y = self._pos_to_curve_point(pos)
        points = list(self.curve_points)
        min_gap = 1.0 / 255.0
        if idx == 0:
            x = max(0.0, min(points[1][0] - min_gap, x))
        elif idx == len(points) - 1:
            x = min(1.0, max(points[idx - 1][0] + min_gap, x))
        else:
            x = max(points[idx - 1][0] + min_gap, min(points[idx + 1][0] - min_gap, x))
        points[idx] = (x, y)
        self.curve_points = [(float(px), float(py)) for px, py in normalize_curve_points(points)]
        self._selected_curve_index = idx
        self._emit_curve_changed()

    def _apply_level_drag(self, x: float) -> None:
        value = self._value_from_x(x)
        if self._drag_level_target == "in_black":
            self.in_black = max(0, min(value, self.in_white - 1))
        elif self._drag_level_target == "in_white":
            self.in_white = min(255, max(value, self.in_black + 1))
        elif self._drag_level_target == "gamma":
            self.gamma = self._gamma_from_mid_value(max(self.in_black + 1, min(value, self.in_white - 1)))
        elif self._drag_level_target == "out_black":
            self.out_black = max(0, min(value, self.out_white - 1))
        elif self._drag_level_target == "out_white":
            self.out_white = min(255, max(value, self.out_black + 1))
        self.update()
        self.levelsChanged.emit(self.in_black, self.in_white, self.gamma, self.out_black, self.out_white)

    def _emit_curve_changed(self) -> None:
        self.update()
        self.curveChanged.emit(self.getCurvePoints())


