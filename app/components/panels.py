# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
    QScrollArea,
    QPushButton,
    QStackedWidget,
    QRadioButton,
    QButtonGroup,
    QSlider,
    QProgressBar,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Design tokens
# ═══════════════════════════════════════════════════════════════════════════════

_CLR_BG = "#09090b"
_CLR_CARD = "#111113"
_CLR_CARD_BORDER = "#1e1e22"
_CLR_TEXT = "#e4e4e7"
_CLR_TEXT_DIM = "#a1a1aa"
_CLR_TEXT_MUTED = "#71717a"
_CLR_HINT = "#52525b"
_CLR_ACCENT = "#2563eb"
_CLR_ACCENT_HOVER = "#3b82f6"
_CLR_ACCENT_BORDER = "#60a5fa"
_CLR_GREEN = "#22c55e"

_CARD_RADIUS = 10
_PAGE_MARGIN = 20
_CARD_SPACING = 14
_INNER_SPACING = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Layout primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("flowCard")
    frame.setStyleSheet(
        f"QFrame#flowCard {{ "
        f"background: {_CLR_CARD}; "
        f"border: 1px solid {_CLR_CARD_BORDER}; "
        f"border-radius: {_CARD_RADIUS}px; "
        f"}}"
    )
    return frame


def _page_title(text: str, hint: str = "") -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(4)
    title = QLabel(text)
    title.setStyleSheet(
        f"font-size: 18px; font-weight: 700; color: {_CLR_TEXT}; "
        "letter-spacing: 0.3px;"
    )
    lay.addWidget(title)
    if hint:
        lay.addWidget(_hint(hint))
    sep = QFrame()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {_CLR_CARD_BORDER};")
    lay.addWidget(sep)
    return w


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"font-weight: 700; color: {_CLR_TEXT}; font-size: 13px; "
        f"padding-bottom: 3px; margin-bottom: 1px;"
    )
    return lbl


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: {_CLR_HINT}; font-size: 11px; line-height: 15px; "
        f"border-left: 2px solid #27272a; padding-left: 6px;"
    )
    return lbl


def _status_text(text: str = "") -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px; line-height: 17px;")
    return lbl


def _badge(text: str, *, color: str = "#93c5fd", border: str = "#1d4ed8") -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background: #0f172a; color: {color}; border: 1px solid {border}; "
        "border-radius: 9px; font-size: 10px; font-weight: bold; padding: 2px 8px;"
    )
    return lbl


def _row(*widgets, stretch_first: bool = False) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    for idx, widget in enumerate(widgets):
        lay.addWidget(widget, 1 if stretch_first and idx == 0 else 0)
    lay.addStretch(1)
    return w


def _field_row(label: str, widget: QWidget) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(10)
    lbl = QLabel(label)
    lbl.setMinimumWidth(72)
    lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px;")
    lay.addWidget(lbl)
    lay.addWidget(widget, 1)
    return w


def _scroll_page() -> tuple[QScrollArea, QVBoxLayout]:
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(_PAGE_MARGIN, _PAGE_MARGIN, _PAGE_MARGIN, _PAGE_MARGIN)
    lay.setSpacing(_CARD_SPACING)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(page)
    return scroll, lay


def _primary_style(btn: QPushButton) -> None:
    btn.setObjectName("primaryButton")
    btn.setStyleSheet(
        "QPushButton#primaryButton {"
        f"  background: {_CLR_ACCENT}; color: #ffffff; border: 1px solid {_CLR_ACCENT_BORDER}; "
        f"  border-radius: 8px; padding: 9px 16px; font-weight: 700; font-size: 13px;"
        "}"
        f"QPushButton#primaryButton:hover {{ background: {_CLR_ACCENT_HOVER}; }}"
        "QPushButton#primaryButton:pressed { background: #1d4ed8; }"
        "QPushButton#primaryButton:disabled {"
        "  background: #1a1a1f; color: #52525b; border-color: #27272a;"
        "}"
    )


def _secondary_style(btn: QPushButton) -> None:
    btn.setObjectName("secondaryButton")
    btn.setStyleSheet(
        "QPushButton#secondaryButton {"
        f"  background: #1a1a1f; color: #d4d4d8; border: 1px solid #27272a; "
        f"  border-radius: 7px; padding: 7px 12px; font-size: 12px;"
        "}"
        "QPushButton#secondaryButton:hover {"
        f"  background: #27272a; border-color: {_CLR_ACCENT_BORDER}; color: #f4f4f5;"
        "}"
        "QPushButton#secondaryButton:disabled {"
        "  background: #131316; color: #3f3f46; border-color: #1e1e22;"
        "}"
    )


def _dark_button_style(btn: QPushButton, active: bool = False, state: str = "available") -> None:
    state = "active" if active else str(state or "available")
    obj = {
        "active": "navActive", "completed": "navDone",
        "locked": "navLocked", "available": "navBtn",
    }.get(state, "navBtn")
    btn.setObjectName(obj)
    styles = {
        "active": (
            f"QPushButton#{obj} {{ "
            f"background: #18181b; color: #ffffff; border: none; "
            f"border-left: 3px solid {_CLR_ACCENT}; border-radius: 4px; "
            f"padding: 9px 10px; text-align: left; font-weight: 700; font-size: 13px; }}"
        ),
        "completed": (
            f"QPushButton#{obj} {{ "
            f"background: transparent; color: {_CLR_TEXT_DIM}; border: none; "
            f"border-left: 3px solid {_CLR_GREEN}; border-radius: 4px; "
            f"padding: 9px 10px; text-align: left; font-weight: 600; font-size: 13px; }}"
            f"QPushButton#{obj}:hover {{ background: #18181b; color: {_CLR_TEXT}; }}"
        ),
        "locked": (
            f"QPushButton#{obj} {{ "
            f"background: transparent; color: #3f3f46; border: none; "
            f"border-left: 3px solid transparent; "
            f"border-radius: 4px; padding: 9px 10px; text-align: left; font-size: 13px; }}"
        ),
        "available": (
            f"QPushButton#{obj} {{ "
            f"background: transparent; color: {_CLR_TEXT_DIM}; border: none; "
            f"border-left: 3px solid transparent; "
            f"border-radius: 4px; padding: 9px 10px; text-align: left; font-size: 13px; }}"
            f"QPushButton#{obj}:hover {{ background: #18181b; color: {_CLR_TEXT}; }}"
            f"QPushButton#{obj}:disabled {{ color: #3f3f46; }}"
        ),
    }
    btn.setStyleSheet(styles.get(state, styles["available"]))


def _footer(*buttons: QPushButton) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 6, 0, 0)
    lay.addStretch(1)
    for btn in buttons:
        lay.addWidget(btn)
    return w


def _make_next_button(window, text: str, step: int, before=None) -> QPushButton:
    btn = QPushButton(text)
    btn.setMinimumHeight(40)
    _primary_style(btn)

    def _go() -> None:
        if before is not None:
            ok = before()
            if ok is False:
                return
        window.set_workflow_step(step)

    btn.clicked.connect(_go)
    return btn


def _divider() -> QFrame:
    """Thin horizontal divider inside a card."""
    sep = QFrame()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {_CLR_CARD_BORDER};")
    return sep


# ═══════════════════════════════════════════════════════════════════════════════
# Hidden compatibility controls
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_hidden_source_controls(window) -> None:
    window.source_cutout_radio = QRadioButton("主视频自带 Alpha")
    window.source_matanyone_radio = QRadioButton("MatAnything / MatAnyone")
    window.source_external_mask_radio = QRadioButton("独立 Mask")
    window.source_mode_group = QButtonGroup(window)
    for btn in (window.source_cutout_radio, window.source_matanyone_radio, window.source_external_mask_radio):
        window.source_mode_group.addButton(btn)
        btn.setVisible(False)
    window.source_cutout_radio.setChecked(True)

    defaults = (
        ("input_cutout_mask_check", True),
        ("external_mask_enable_check", False),
        ("matting_enable_check", False),
        ("pointcloud_enable_check", True),
        ("pointcloud_usd_check", True),
        ("pointcloud_remove_outliers_check", False),
        ("pointcloud_voxel_check", False),
        ("pointcloud_temporal_check", True),
        ("external_depth_enable_check", False),
    )
    for name, checked in defaults:
        widget = getattr(window, name, None)
        if widget is not None:
            widget.setChecked(bool(checked))
            widget.setVisible(False)

    legacy_names = (
        "model_combo", "device_combo", "batch_spin", "color_combo", "invert_check",
        "copy_audio_check", "normalize_mode_combo", "encoder_combo", "adjust_mode_combo",
        "adjust_stack", "detail_boost_spin", "anti_banding_spin", "depth_smooth_spin",
        "edge_preserve_spin", "normal_enable_check", "normal_strength_spin", "normal_refine_spin",
        "external_mask_path_edit", "external_mask_pick_btn", "external_mask_invert_check",
        "external_depth_path_edit", "external_depth_pick_btn", "external_depth_weight_spin",
        "external_depth_invert_check", "pointcloud_stride_spin", "pointcloud_max_points_spin",
        "pointcloud_alpha_threshold_spin", "pointcloud_template_ratio_spin",
        "pointcloud_template_conf_spin", "pointcloud_hand_ratio_spin", "pointcloud_hand_conf_spin",
        "pointcloud_resume_check", "pointcloud_obj_check", "pointcloud_abc_check",
        "pointcloud_normal_relief_check", "pointcloud_normal_relief_strength_spin",
        "pointcloud_normal_relief_edge_spin", "pointcloud_normal_relief_min_alpha_spin",
        "pointcloud_color_combo", "pointcloud_mode_combo", "pointcloud_outlier_sigma_spin",
        "pointcloud_voxel_size_spin",
    )
    for name in legacy_names:
        try:
            getattr(window, name).setVisible(False)
        except Exception:
            pass
    try:
        window.pointcloud_color_combo.setCurrentText("无颜色XYZ")
        window.pointcloud_mode_combo.setCurrentText("结构身体XYZ")
        window.pointcloud_density_combo.setCurrentText("中")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Page builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_environment_page(window) -> QScrollArea:
    page, lay = _scroll_page()
    lay.addWidget(_page_title("准备页：环境部署", "检查依赖、模型权重和第三方仓库。环境通过后进入第 1 步导入视频。"))

    # ── 运行环境 ──
    env_card = _card()
    el = QVBoxLayout(env_card)
    el.setContentsMargins(18, 16, 18, 16)
    el.setSpacing(_INNER_SPACING)
    el.addWidget(_section_label("运行环境"))
    window.deployment_env_label = _status_text("环境：未检查")
    el.addWidget(window.deployment_env_label)
    btn_row = QHBoxLayout()
    btn_row.setSpacing(8)
    window.deployment_env_check_btn = QPushButton("检查环境")
    window.deployment_install_deps_btn = QPushButton("安装依赖 / 下载模型")
    for btn in (window.deployment_env_check_btn, window.deployment_install_deps_btn):
        _primary_style(btn)
        btn.setMinimumHeight(38)
        btn_row.addWidget(btn)
    btn_row.addStretch(1)
    el.addLayout(btn_row)
    lay.addWidget(env_card)

    # ── 模型资源 ──
    model_card = _card()
    ml = QVBoxLayout(model_card)
    ml.setContentsMargins(18, 16, 18, 16)
    ml.setSpacing(_INNER_SPACING)
    ml.addWidget(_section_label("模型资源"))
    ml.addWidget(_status_text(
        "主体结构：SMPL + 4DHumans 是默认主线核心。\n"
        "画面分割：FASHN Human Parser，用于衣服和头发 mask 约束。\n"
        "未生成分割缓存时明确 Body Only，不再生成假衣服 / 假头发。"
    ))
    if hasattr(window, "segmentation_status_label"):
        ml.addWidget(window.segmentation_status_label)
    seg_row = QHBoxLayout()
    seg_row.setSpacing(8)
    for btn_name in ("segmentation_open_btn", "segmentation_test_btn"):
        try:
            btn = getattr(window, btn_name)
            _secondary_style(btn)
            seg_row.addWidget(btn)
        except Exception:
            pass
    seg_row.addStretch(1)
    ml.addLayout(seg_row)
    lay.addWidget(model_card)

    # ── 工具 ──
    tool_card = _card()
    tl = QVBoxLayout(tool_card)
    tl.setContentsMargins(18, 14, 18, 14)
    tl.setSpacing(8)
    tl.addWidget(_section_label("工具"))
    tool_row = QHBoxLayout()
    tool_row.setSpacing(8)
    window.deployment_models_btn = QPushButton("打开 models")
    window.deployment_logs_btn = QPushButton("打开 logs")
    window.deployment_cache_btn = QPushButton("打开 cache")
    for btn in (window.deployment_models_btn, window.deployment_logs_btn, window.deployment_cache_btn):
        _secondary_style(btn)
        tool_row.addWidget(btn)
    tool_row.addStretch(1)
    tl.addLayout(tool_row)
    lay.addWidget(tool_card)

    lay.addStretch(1)

    # ── Connections ──
    window.deployment_env_check_btn.clicked.connect(window.refresh_deployment_environment_status)
    window.deployment_install_deps_btn.clicked.connect(window.install_deployment_python_dependencies)
    window.deployment_models_btn.clicked.connect(window.open_models_folder)
    window.deployment_logs_btn.clicked.connect(window.open_log_dir)
    try:
        window.segmentation_open_btn.clicked.connect(window.open_segmentation_models_folder)
        window.segmentation_test_btn.clicked.connect(window.test_current_segmentation_frame)
        window.segmentation_cache_btn.clicked.connect(window.start_segmentation_cache_generation)
    except Exception:
        pass
    try:
        window.deployment_cache_btn.clicked.connect(window.open_cache_manager)
    except Exception:
        pass
    return page


def _build_import_page(window) -> QScrollArea:
    page, lay = _scroll_page()
    lay.addWidget(_page_title("1. 导入视频 / 选择范围", "第一步只做输入和处理范围。主界面先变轻，后面的结构、分割、导出分开处理。"))

    body = QHBoxLayout()
    body.setSpacing(_CARD_SPACING)

    # 左侧：导入和范围
    left_card = _card()
    left = QVBoxLayout(left_card)
    left.setContentsMargins(18, 16, 18, 16)
    left.setSpacing(_INNER_SPACING)
    left.addWidget(_section_label("主视频"))
    window.path_edit.setMinimumHeight(150)
    window.path_edit.setPlaceholderText("拖入视频文件，或点击下方按钮")
    window.path_edit.setObjectName("dropZone")
    left.addWidget(window.path_edit, 1)

    window.pick_btn.setText("导入视频")
    window.pick_btn.setMinimumHeight(40)
    _primary_style(window.pick_btn)
    left.addWidget(window.pick_btn, 0, Qt.AlignHCenter)
    left.addWidget(_divider())

    left.addWidget(_section_label("处理范围"))
    left.addWidget(_hint("入点 / 出点会影响 4DHumans、WHAM、分割缓存和最终导出。重新打开项目后会恢复。"))
    if hasattr(window, "processing_range_progress"):
        window.processing_range_progress.setVisible(False)
    row_in = QHBoxLayout()
    row_in.setSpacing(8)
    in_lbl = QLabel("入点")
    in_lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px;")
    row_in.addWidget(in_lbl)
    row_in.addWidget(window.processing_start_slider, 1)
    row_in.addWidget(window.processing_start_spin)
    left.addLayout(row_in)
    row_out = QHBoxLayout()
    row_out.setSpacing(8)
    out_lbl = QLabel("出点")
    out_lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px;")
    row_out.addWidget(out_lbl)
    row_out.addWidget(window.processing_end_slider, 1)
    row_out.addWidget(window.processing_end_spin)
    left.addLayout(row_out)
    left.addWidget(window.processing_range_label)

    body.addWidget(left_card, 42)

    # 右侧：原视频预览
    right_card = _card()
    right = QVBoxLayout(right_card)
    right.setContentsMargins(18, 16, 18, 16)
    right.setSpacing(_INNER_SPACING)
    right.addWidget(_section_label("视频预览"))
    right.addWidget(window.info_label)
    frame_row = QHBoxLayout()
    frame_row.setSpacing(8)
    frame_lbl = QLabel("帧")
    frame_lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px;")
    frame_row.addWidget(frame_lbl)
    frame_row.addWidget(window.input_preview_frame_slider, 1)
    frame_row.addWidget(window.input_preview_frame_spin)
    if hasattr(window, "input_preview_play_btn"):
        _secondary_style(window.input_preview_play_btn)
        frame_row.addWidget(window.input_preview_play_btn)
    right.addLayout(frame_row)
    window.preview_original_label.setMinimumSize(520, 520)
    window.preview_original_label.setMaximumHeight(16777215)
    window.preview_original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window.preview_original_label.setText("导入后显示原视频帧")
    right.addWidget(window.preview_original_label, 1)
    right.addWidget(window.external_status_label)
    body.addWidget(right_card, 58)

    lay.addLayout(body, 1)

    next_btn = _make_next_button(window, "下一步：人体结构 →", 1,
                                 before=lambda: window.validate_main_video_alpha_chain(silent=False))
    next_btn.setFixedWidth(210)
    lay.addWidget(_footer(next_btn))
    return page

def _build_generation_page(window) -> QScrollArea:
    page, lay = _scroll_page()
    lay.addWidget(_page_title("2. 人体结构", "独立运行 4DHumans 或 WHAM。两个方案分开缓存，处理过的方案重新打开项目后仍可查看。"))

    body = QHBoxLayout()
    body.setSpacing(_CARD_SPACING)

    # 左侧：两个独立方案
    left_card = _card()
    left = QVBoxLayout(left_card)
    left.setContentsMargins(18, 16, 18, 16)
    left.setSpacing(_INNER_SPACING)
    left.addWidget(_section_label("结构方案"))
    left.addWidget(_hint("默认使用最优化参数。需要比较时可以分别生成 4D 和 WHAM，它们不会互相覆盖。"))

    scheme_row = QHBoxLayout()
    scheme_row.setSpacing(12)
    for title, desc, gen_btn, view_btn, status_lbl in (
        ("4DHumans", "默认优先。适合普通人体动作，生成稳定身体 Mesh。", window.generate_4dhumans_btn, window.preview_4dhumans_btn, window.structure_4d_status_label),
        ("WHAM", "轨迹锚定。适合需要保留运动轨迹的片段。", window.generate_wham_btn, window.preview_wham_btn, window.structure_wham_status_label),
    ):
        card = _card()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 12, 14, 12)
        cl.setSpacing(8)
        cl.addWidget(_section_label(title))
        cl.addWidget(_status_text(desc))
        cl.addWidget(status_lbl)
        gen_btn.setMinimumHeight(40)
        gen_btn.setText("生成 4D 人体" if title == "4DHumans" else "生成 WHAM")
        view_btn.setEnabled(False)
        view_btn.setToolTip("该方案生成缓存后可查看。")
        _primary_style(gen_btn)
        _secondary_style(view_btn)
        btns = QHBoxLayout()
        btns.setSpacing(8)
        btns.addWidget(gen_btn)
        btns.addWidget(view_btn)
        cl.addLayout(btns)
        scheme_row.addWidget(card, 1)
    left.addLayout(scheme_row)

    left.addWidget(_divider())
    left.addWidget(_section_label("可调参数"))
    left.addWidget(_field_row("身体细分", window.mesh_dense_segments_combo))
    window.pointcloud_temporal_check.setText("时序稳定")
    window.pointcloud_temporal_check.setVisible(True)
    left.addWidget(window.pointcloud_temporal_check)
    left.addWidget(_hint("4DHumans / WHAM 使用各自默认优化输入。这里不显示深度流程的推理分辨率，避免误导。"))
    left.addWidget(_divider())
    left.addWidget(_section_label("运行状态"))
    left.addWidget(window.structure_cache_status_label)
    left.addWidget(window.structure_progress)
    left.addStretch(1)
    body.addWidget(left_card, 42)

    # 右侧：人体结构预览
    right_card = _card()
    right = QVBoxLayout(right_card)
    right.setContentsMargins(18, 16, 18, 16)
    right.setSpacing(_INNER_SPACING)
    right.addWidget(_section_label("结构预览"))
    right.addWidget(_hint("拖动预览图可旋转视角；拖动帧条可查看对应帧身体模型。"))
    frame_row = QHBoxLayout()
    frame_row.setSpacing(8)
    frame_lbl = QLabel("帧")
    frame_lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px;")
    frame_row.addWidget(frame_lbl)
    frame_row.addWidget(window.structure_preview_frame_slider, 1)
    frame_row.addWidget(window.structure_preview_frame_spin)
    right.addLayout(frame_row)
    view_row = QHBoxLayout()
    view_row.setSpacing(8)
    window.preview_structure_btn.setText("预览身体")
    _secondary_style(window.preview_structure_btn)
    view_row.addWidget(window.preview_structure_btn)
    view_row.addStretch(1)
    right.addLayout(view_row)
    window.structure_mesh_preview_label.setMinimumSize(480, 320)
    window.structure_mesh_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window.structure_mesh_preview_label.setText("生成 4D 或 WHAM 后，可在这里查看身体模型")
    right.addWidget(window.structure_mesh_preview_label, 1)
    body.addWidget(right_card, 58)

    lay.addLayout(body, 1)

    window.generation_next_btn = _make_next_button(window, "下一步：衣服和头发 →", 2,
                                                   before=lambda: window._has_structure_cache())
    window.generation_next_btn.setFixedWidth(230)
    lay.addWidget(_footer(window.generation_next_btn))
    return page

def _build_tuning_page(window) -> QScrollArea:
    page, lay = _scroll_page()
    lay.addWidget(_page_title("3. 衣服和头发", "先生成逐帧分割缓存，再生成 / 预览衣服、头发和组合网格。"))

    body = QHBoxLayout()
    body.setSpacing(_CARD_SPACING)

    # 左侧：分割和生成参数
    left_card = _card()
    left = QVBoxLayout(left_card)
    left.setContentsMargins(18, 16, 18, 16)
    left.setSpacing(_INNER_SPACING)
    left.addWidget(_section_label("分割缓存"))
    if hasattr(window, "segmentation_cache_status_label"):
        left.addWidget(window.segmentation_cache_status_label)
    if hasattr(window, "segmentation_status_label"):
        left.addWidget(window.segmentation_status_label)
    if hasattr(window, "segmentation_provider_combo"):
        window.segmentation_provider_combo.setCurrentText("Auto")
        window.segmentation_provider_combo.setVisible(False)
    if hasattr(window, "segmentation_enable_check"):
        window.segmentation_enable_check.setChecked(True)
        window.segmentation_enable_check.setVisible(False)
    left.addWidget(_hint("分割模型和衣服/头发壳层使用默认设置。这里不再暴露多余选项，生成分割缓存后直接预览。"))
    btn_row = QHBoxLayout()
    btn_row.setSpacing(8)
    if hasattr(window, "segmentation_cache_btn"):
        window.segmentation_cache_btn.setText("生成分割缓存")
        window.segmentation_cache_btn.setMinimumHeight(40)
        _primary_style(window.segmentation_cache_btn)
        try:
            window.segmentation_cache_btn.clicked.disconnect()
        except Exception:
            pass
        start_seg = getattr(window, "start_segmentation_cache_generation", None)
        if callable(start_seg):
            window.segmentation_cache_btn.clicked.connect(start_seg)
        else:
            window.segmentation_cache_btn.setEnabled(False)
            window.segmentation_cache_btn.setToolTip("分割缓存入口未注册。")
        btn_row.addWidget(window.segmentation_cache_btn)
    if hasattr(window, "segmentation_test_btn"):
        window.segmentation_test_btn.setText("测试当前帧")
        _secondary_style(window.segmentation_test_btn)
        try:
            window.segmentation_test_btn.clicked.disconnect()
        except Exception:
            pass
        test_seg = getattr(window, "test_current_segmentation_frame", None)
        if callable(test_seg):
            window.segmentation_test_btn.clicked.connect(test_seg)
        else:
            window.segmentation_test_btn.setEnabled(False)
            window.segmentation_test_btn.setToolTip("分割测试入口未注册。")
        btn_row.addWidget(window.segmentation_test_btn)
    btn_row.addStretch(1)
    left.addLayout(btn_row)

    if hasattr(window, "garment_shell_check"):
        window.garment_shell_check.setChecked(True)
        window.garment_shell_check.setVisible(False)
    if hasattr(window, "hair_shell_check"):
        window.hair_shell_check.setChecked(True)
        window.hair_shell_check.setVisible(False)
    if hasattr(window, "garment_shell_offset_spin"):
        window.garment_shell_offset_spin.setVisible(False)
    if hasattr(window, "hair_shell_offset_spin"):
        window.hair_shell_offset_spin.setVisible(False)
    left.addStretch(1)
    body.addWidget(left_card, 38)

    # 右侧：分割/网格预览
    right_card = _card()
    right = QVBoxLayout(right_card)
    right.setContentsMargins(18, 16, 18, 16)
    right.setSpacing(_INNER_SPACING)
    right.addWidget(_section_label("预览"))
    right.addWidget(_hint("拖动预览图可旋转视角；滚轮缩放。拖动帧条可查看对应帧模型。"))
    frame_row = QHBoxLayout()
    frame_row.setSpacing(8)
    frame_lbl = QLabel("帧")
    frame_lbl.setStyleSheet(f"color: {_CLR_TEXT_DIM}; font-size: 12px;")
    frame_row.addWidget(frame_lbl)
    frame_row.addWidget(window.preview_frame_slider, 1)
    frame_row.addWidget(window.preview_frame_spin)
    right.addLayout(frame_row)

    btn_row2 = QHBoxLayout()
    btn_row2.setSpacing(6)
    window.preview_garment_btn.setText("衣服")
    window.preview_hair_btn.setText("头发")
    window.preview_detail_btn.setText("组合")
    for btn in (window.preview_garment_btn, window.preview_hair_btn, window.preview_detail_btn):
        _secondary_style(btn)
        btn_row2.addWidget(btn)

    def _set_active_layer_preview_btn(active_btn) -> None:
        for _btn in (window.preview_garment_btn, window.preview_hair_btn, window.preview_detail_btn):
            if _btn is active_btn:
                _primary_style(_btn)
            else:
                _secondary_style(_btn)

    window.preview_garment_btn.clicked.connect(lambda _checked=False: _set_active_layer_preview_btn(window.preview_garment_btn))
    window.preview_hair_btn.clicked.connect(lambda _checked=False: _set_active_layer_preview_btn(window.preview_hair_btn))
    window.preview_detail_btn.clicked.connect(lambda _checked=False: _set_active_layer_preview_btn(window.preview_detail_btn))
    btn_row2.addStretch(1)
    right.addLayout(btn_row2)
    window.layer_mesh_preview_label.setMinimumSize(520, 340)
    window.layer_mesh_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window.layer_mesh_preview_label.setText("生成分割缓存后，查看衣服 / 头发 / 组合网格")
    right.addWidget(window.layer_mesh_preview_label, 1)
    right.addWidget(window.preview_status_label)
    body.addWidget(right_card, 62)

    lay.addLayout(body, 1)
    next_btn = _make_next_button(window, "下一步：导出 →", 3,
                                 before=lambda: window._has_structure_cache())
    next_btn.setFixedWidth(190)
    lay.addWidget(_footer(next_btn))
    return page

def _build_export_page(window) -> QScrollArea:
    page, lay = _scroll_page()
    lay.addWidget(_page_title("4. 导出", "配置输出项并导出网格。"))

    # ── 输出内容 ──
    content_card = _card()
    cl = QVBoxLayout(content_card)
    cl.setContentsMargins(18, 16, 18, 16)
    cl.setSpacing(_INNER_SPACING)
    cl.addWidget(_section_label("导出格式 / 内容"))
    window.mesh_export_check.setText("低模身体 USDA")
    window.detail_mesh_export_check.setText("Body / Garment / Hair / Combined USDA")
    if hasattr(window, "pointcloud_usd_check"):
        window.pointcloud_usd_check.setText("稳定点云 USDA（可选）")
    output_hints = [
        (window.mesh_export_check, "用于检查 4DHumans / WHAM 的人体结构稳定性，最快。"),
        (window.detail_mesh_export_check, "最终主输出：身体、衣服、头发、组合网格。"),
    ]
    if hasattr(window, "pointcloud_usd_check"):
        output_hints.append((window.pointcloud_usd_check, "可选点云输出。只需要网格时保持关闭。"))
    for w, hint_text in output_hints:
        w.setVisible(True)
        cl.addWidget(w)
        cl.addWidget(_hint(hint_text))
    lay.addWidget(content_card)

    def _refresh_export_scope() -> None:
        try:
            any_output = bool(window.mesh_export_check.isChecked() or window.detail_mesh_export_check.isChecked() or (hasattr(window, "pointcloud_usd_check") and window.pointcloud_usd_check.isChecked()))
            if hasattr(window, "preview_status_label") and not any_output:
                window.preview_status_label.setText("至少选择一个输出项。")
            if hasattr(window, "refresh_workflow_action_gates"):
                window.refresh_workflow_action_gates()
        except Exception:
            pass

    for _w in (window.mesh_export_check, window.detail_mesh_export_check, getattr(window, "pointcloud_usd_check", None)):
        if _w is not None:
            _w.toggled.connect(lambda _checked=False: _refresh_export_scope())
    _refresh_export_scope()

    # ── 输出路径 ──
    path_card = _card()
    ol = QVBoxLayout(path_card)
    ol.setContentsMargins(18, 16, 18, 16)
    ol.setSpacing(_INNER_SPACING)
    ol.addWidget(_section_label("输出路径"))
    window.output_path_edit.setVisible(True)
    ol.addWidget(window.output_path_edit)
    out_btns = QHBoxLayout()
    out_btns.setSpacing(8)
    window.output_pick_btn.setText("选择路径")
    window.output_open_btn.setText("打开目录")
    for btn in (window.output_pick_btn, window.output_open_btn):
        _secondary_style(btn)
        out_btns.addWidget(btn)
    out_btns.addStretch(1)
    ol.addLayout(out_btns)
    ol.addWidget(window.out_size_label)
    lay.addWidget(path_card)

    # ── 导出控制 ──
    export_card = _card()
    ex = QVBoxLayout(export_card)
    ex.setContentsMargins(18, 16, 18, 16)
    ex.setSpacing(_INNER_SPACING)
    ex.addWidget(_section_label("导出控制"))
    ex.addWidget(window.stage_status_label)
    ex.addWidget(window.progress)
    window.start_btn.setText("开始导出")
    window.cancel_btn.setText("停止")
    window.start_btn.setMinimumHeight(42)
    _primary_style(window.start_btn)
    _secondary_style(window.cancel_btn)
    action_row = QHBoxLayout()
    action_row.setSpacing(10)
    action_row.addWidget(window.start_btn)
    action_row.addWidget(window.cancel_btn)
    action_row.addStretch(1)
    ex.addLayout(action_row)
    lay.addWidget(export_card)

    lay.addStretch(1)
    return page


# ═══════════════════════════════════════════════════════════════════════════════
# Main layout assembly
# ═══════════════════════════════════════════════════════════════════════════════

def build_main_layout(window) -> None:
    central = QWidget()
    outer = QVBoxLayout(central)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    root = QHBoxLayout()
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    _setup_hidden_source_controls(window)

    # ── Sidebar ──
    sidebar = QFrame()
    sidebar.setFixedWidth(200)
    sidebar.setObjectName("sidebar")
    sidebar.setStyleSheet(
        "QFrame#sidebar {"
        f"  background: #0c0c0e; border: none; "
        f"  border-right: 1px solid {_CLR_CARD_BORDER};"
        "}"
    )
    side_lay = QVBoxLayout(sidebar)
    side_lay.setContentsMargins(12, 16, 12, 12)
    side_lay.setSpacing(6)

    title = QLabel("视频人体网格重建")
    title.setStyleSheet(f"color: {_CLR_TEXT}; font-weight: 700; font-size: 14px;")
    side_lay.addWidget(title)
    subtitle = QLabel("Video Human Mesh")
    subtitle.setStyleSheet(f"color: {_CLR_HINT}; font-size: 10px;")
    side_lay.addWidget(subtitle)
    side_lay.addSpacing(12)

    # Sidebar separator
    sep = QFrame()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {_CLR_CARD_BORDER};")
    side_lay.addWidget(sep)
    side_lay.addSpacing(8)

    window.workflow_buttons = []

    cat_flow = QLabel("工作流")
    cat_flow.setStyleSheet(f"color: {_CLR_HINT}; font-size: 10px; font-weight: 600; text-transform: uppercase;")
    side_lay.addWidget(cat_flow)

    for idx, text in enumerate(("1  导入范围", "2  人体结构", "3  衣服头发", "4  导出"), start=0):
        btn = QPushButton(text)
        btn.setProperty("base_text", text)
        btn.clicked.connect(lambda _checked=False, i=idx: window.set_workflow_step(i))
        side_lay.addWidget(btn)
        window.workflow_buttons.append(btn)

    side_lay.addStretch(1)

    ver_label = QLabel("v1.0")
    ver_label.setStyleSheet(f"color: #27272a; font-size: 10px;")
    ver_label.setAlignment(Qt.AlignCenter)
    side_lay.addWidget(ver_label)

    root.addWidget(sidebar)

    # ── Page stack ──
    stack = QStackedWidget()
    window.workflow_stack = stack
    stack.addWidget(_build_import_page(window))
    stack.addWidget(_build_generation_page(window))
    stack.addWidget(_build_tuning_page(window))
    stack.addWidget(_build_export_page(window))
    root.addWidget(stack, 1)

    # ── Workflow state logic ──

    def _chain_state() -> tuple[bool, bool]:
        main_ready = False
        cache_ready = False
        try:
            main_ready = bool(window.validate_main_video_alpha_chain(silent=True))
        except Exception:
            main_ready = False
        if main_ready:
            try:
                cache_ready = bool(window._has_structure_cache())
            except Exception:
                cache_ready = False
        return main_ready, cache_ready

    def _seg_ready_cached(cache_ready: bool) -> bool:
        if not cache_ready:
            return False
        try:
            state_func = getattr(window, "_current_layer_cache_state", None)
            if callable(state_func):
                state = state_func()
                ready = bool(state.get("ready", False))
                msg = str(state.get("message", ""))
                if hasattr(window, "segmentation_cache_status_label"):
                    if ready:
                        model = str(state.get("model", "") or "").upper()
                        suffix = f"（{model}）" if model else ""
                        window.segmentation_cache_status_label.setText("分割缓存：已生成" + suffix + ("\n" + msg if msg else ""))
                    else:
                        window.segmentation_cache_status_label.setText("分割缓存：未生成" + ("\n" + msg if msg else ""))
                return ready
            from depth_fusion_core import structure_cache_root
            from segmentation_pipeline.segmentation_cache import segmentation_cache_summary, segmentation_summary_path
            root = structure_cache_root(window.make_config())
            summary_file = segmentation_summary_path(root)
            key = (str(summary_file), summary_file.stat().st_mtime_ns if summary_file.exists() else -1)
            cached = getattr(window, "_workflow_seg_summary_cache", None)
            if isinstance(cached, dict) and cached.get("key") == key:
                return bool(cached.get("ready", False))
            summary = segmentation_cache_summary(root)
            ready = bool(summary.get("ok", False)) and int(summary.get("cached_frames", 0) or 0) > 0
            window._workflow_seg_summary_cache = {"key": key, "ready": bool(ready)}
            return bool(ready)
        except Exception:
            window._workflow_seg_summary_cache = {"key": None, "ready": False}
            return False

    def _refresh_action_gates(active_index: int) -> None:
        main_ready, cache_ready = _chain_state()
        seg_ready = _seg_ready_cached(cache_ready)
        app_busy = bool(
            getattr(window, "thread", None) is not None
            or getattr(window, "preview_thread", None) is not None
            or getattr(window, "structure_cache_thread", None) is not None
            or getattr(window, "segmentation_cache_thread", None) is not None
        )
        shell_garment_on = True
        shell_hair_on = True
        layer_base_ready = bool(cache_ready and seg_ready and not app_busy)
        for btn in (getattr(window, "structure_cache_btn", None), getattr(window, "generate_4dhumans_btn", None), getattr(window, "generate_wham_btn", None)):
            if btn is not None:
                btn.setEnabled(bool(main_ready and not app_busy))
        if hasattr(window, "segmentation_cache_btn"):
            window.segmentation_cache_btn.setEnabled(bool(main_ready and cache_ready and not app_busy))
        if hasattr(window, "segmentation_test_btn"):
            window.segmentation_test_btn.setEnabled(bool(main_ready and cache_ready and not app_busy))
        if hasattr(window, "preview_structure_btn"):
            window.preview_structure_btn.setEnabled(bool(cache_ready and not app_busy))
        for _model, _btn_name in (("4dhumans", "preview_4dhumans_btn"), ("wham", "preview_wham_btn")):
            _btn = getattr(window, _btn_name, None)
            if _btn is not None:
                try:
                    _scheme_ready = bool(window._has_structure_cache_for_model(_model))
                except Exception:
                    try:
                        _scheme_ready = bool(window._has_structure_cache())
                    except Exception:
                        _scheme_ready = False
                _btn.setEnabled(bool(_scheme_ready and not app_busy))
        if hasattr(window, "preview_detail_btn"):
            window.preview_detail_btn.setEnabled(bool(cache_ready and not app_busy))
        garment_btn = getattr(window, "preview_garment_btn", None)
        if garment_btn is not None:
            garment_btn.setEnabled(bool(layer_base_ready and shell_garment_on))
            garment_btn.setToolTip("预览衣服分割生成的壳层。需要：结构缓存 + 逐帧分割缓存。")
        hair_btn = getattr(window, "preview_hair_btn", None)
        if hair_btn is not None:
            hair_btn.setEnabled(bool(layer_base_ready and shell_hair_on))
            hair_btn.setToolTip("预览头发分割生成的壳层。需要：结构缓存 + 逐帧分割缓存。")
        if hasattr(window, "_update_structure_scheme_status_labels"):
            window._update_structure_scheme_status_labels()
        detail_check = getattr(window, "detail_mesh_export_check", None)
        if detail_check is not None:
            try:
                if not seg_ready:
                    if detail_check.isChecked():
                        window._detail_export_auto_disabled_by_gate = True
                        detail_check.blockSignals(True)
                        detail_check.setChecked(False)
                        detail_check.blockSignals(False)
                    detail_check.setEnabled(False)
                    detail_check.setToolTip("需要先生成分割缓存，才可以导出 Body / Garment / Hair / Combined。")
                else:
                    detail_check.setEnabled(True)
                    detail_check.setToolTip("最终主输出：身体、衣服、头发、组合网格。")
                    if bool(getattr(window, "_detail_export_auto_disabled_by_gate", False)) and not detail_check.isChecked():
                        detail_check.blockSignals(True)
                        detail_check.setChecked(True)
                        detail_check.blockSignals(False)
                    window._detail_export_auto_disabled_by_gate = False
            except Exception:
                pass
        if hasattr(window, "start_btn"):
            try:
                output_selected = bool(
                    window.mesh_export_check.isChecked()
                    or (seg_ready and window.detail_mesh_export_check.isChecked())
                    or (hasattr(window, "pointcloud_usd_check") and window.pointcloud_usd_check.isChecked())
                )
            except Exception:
                output_selected = True
            window.start_btn.setEnabled(bool(cache_ready and output_selected and not app_busy))
        if hasattr(window, "cancel_btn"):
            window.cancel_btn.setEnabled(bool(window.thread is not None))

        enabled = [True, main_ready, cache_ready, cache_ready]
        completed = [main_ready, cache_ready, seg_ready, False]
        for i, btn in enumerate(window.workflow_buttons):
            can_enter = bool(enabled[i] or i <= active_index)
            btn.setEnabled(can_enter)
            base_text = str(btn.property("base_text") or btn.text()).replace("✓ ", "").replace("🔒 ", "")
            if i == active_index:
                btn.setText(base_text)
                _dark_button_style(btn, active=True, state="active")
            elif bool(completed[i]):
                btn.setText("✓ " + base_text)
                _dark_button_style(btn, active=False, state="completed")
            elif not can_enter:
                btn.setText("🔒 " + base_text)
                _dark_button_style(btn, active=False, state="locked")
            else:
                btn.setText(base_text)
                _dark_button_style(btn, active=False, state="available")
        if hasattr(window, "generation_next_btn"):
            window.generation_next_btn.setEnabled(bool(cache_ready))

    def set_workflow_step(index: int) -> None:
        requested = max(0, min(int(index), stack.count() - 1))
        index = requested
        if requested >= 1:
            if not window.validate_main_video_alpha_chain(silent=True):
                index = 0
                try:
                    window.external_status_label.setText("请先导入主视频。")
                    window.preview_status_label.setText("请先导入主视频，再生成模型。")
                except Exception:
                    pass
        if requested >= 2 and index == requested:
            try:
                has_cache = bool(window._has_structure_cache())
            except Exception:
                has_cache = False
            if not has_cache:
                index = 1
                try:
                    window.preview_status_label.setText("请先生成人体结构，再生成衣服和头发。")
                except Exception:
                    pass
        if hasattr(window, "structure_mesh_preview_label") and hasattr(window, "layer_mesh_preview_label"):
            window.preview_depth_label = window.layer_mesh_preview_label if index == 2 else window.structure_mesh_preview_label
        stack.setCurrentIndex(index)
        _refresh_action_gates(index)
        for i, btn in enumerate(window.workflow_buttons):
            btn.setChecked(i == index) if hasattr(btn, "setChecked") else None
        if hasattr(window, "save_project_state"):
            window.save_project_state(index)

    window.set_workflow_step = set_workflow_step
    window.refresh_workflow_action_gates = lambda: _refresh_action_gates(stack.currentIndex())

    outer.addLayout(root, 1)

    # ── Event console ──
    console_card = QFrame()
    console_card.setObjectName("eventConsoleCard")
    console_card.setStyleSheet(
        "QFrame#eventConsoleCard {"
        f"  background: #0c0c0e; border: none; "
        f"  border-top: 1px solid {_CLR_CARD_BORDER};"
        "}"
    )
    console_lay = QVBoxLayout(console_card)
    console_lay.setContentsMargins(14, 8, 14, 8)
    console_lay.setSpacing(6)

    console_header = QHBoxLayout()
    console_title = QLabel("事件控制台")
    console_title.setStyleSheet(f"color: {_CLR_TEXT}; font-weight: 700; font-size: 12px;")
    console_header.addWidget(console_title)
    console_hint = QLabel("实时日志 · data/logs/events.log")
    console_hint.setStyleSheet(f"color: {_CLR_HINT}; font-size: 10px;")
    console_header.addWidget(console_hint, 1)

    clear_btn = QPushButton("清空")
    open_btn = QPushButton("打开日志")
    collapse_btn = QPushButton("隐藏")
    for btn in (clear_btn, open_btn, collapse_btn):
        _secondary_style(btn)
        btn.setMaximumHeight(26)
        btn.setStyleSheet(btn.styleSheet() + f"\nQPushButton {{ padding: 3px 10px; font-size: 11px; }}")
    clear_btn.clicked.connect(window.clear_event_console)
    open_btn.clicked.connect(window.open_log_dir)
    console_header.addWidget(clear_btn)
    console_header.addWidget(open_btn)
    console_header.addWidget(collapse_btn)
    console_lay.addLayout(console_header)

    window.log_box.setMinimumHeight(120)
    window.log_box.setMaximumHeight(200)
    window.log_box.setVisible(True)
    console_lay.addWidget(window.log_box)
    window.event_console_card = console_card

    def _toggle_global_console() -> None:
        opened = window.log_box.isVisible()
        window.log_box.setVisible(not opened)
        collapse_btn.setText("显示" if opened else "隐藏")

    collapse_btn.clicked.connect(_toggle_global_console)
    outer.addWidget(console_card, 0)
    window.setCentralWidget(central)

    # ── Legacy signal connections ──
    for _btn, _mode in (
        (window.source_cutout_radio, "cutout_video"),
        (window.source_matanyone_radio, "matanyone"),
        (window.source_external_mask_radio, "external_mask"),
    ):
        _btn.toggled.connect(lambda checked=False, mode=_mode: checked and window._apply_source_mode(mode))

    window.pointcloud_density_combo.currentTextChanged.connect(lambda _text="": window._on_density_mode_changed())
    try:
        window._update_conditional_visibility()
        window._on_density_mode_changed()
    except Exception:
        pass
    window.set_workflow_step(0)
