# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import os
import sys
import shutil
import subprocess
import importlib

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QBoxLayout

from depth_fusion_core import (
    APP_NAME, APP_STYLESHEET, AUX_MODEL_SPECS, BUILTIN_PRESETS, CACHE_FORMAT_VERSION, CacheEntry, Callable,
    DEFAULT_MATANYONE_MODEL_PATH, DEFAULT_MATTING_MASK_DIR, ENCODER_MODES, ExternalDepthFusionState,
    FfmpegRawVideoWriter, JobConfig, MAX_SAFE_LONG_SIDE_HINT, MODEL_IDS, NORMALIZE_MODES, Optional,
    PROJECT_CACHE_DIR, PROJECT_DIR, PROJECT_HF_HOME, PROJECT_HF_HUB, PROJECT_LOG_DIR, PROJECT_MODELS_DIR,
    Path, PngSequenceWriter, QApplication, QCheckBox, QColor, QComboBox, QCursor, QDialog, QDragEnterEvent,
    QDropEvent, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QImage, QLabel, QLineEdit,
    QLinearGradient, QMainWindow, QMessageBox, QObject, QPainter, QPainterPath, QPen, QPixmap,
    QPlainTextEdit, QPointF, QProgressBar, QPushButton, QRectF, QScrollArea, QSizePolicy, QSlider, QSpinBox,
    QStackedWidget, QThread, QTimer, QVBoxLayout, QWidget, Qt, RESOURCES_DIR, RotatingFileHandler,
    SAFE_DEFAULT_LONG_SIDE, SAFE_DEFAULT_PROCESS_RES, Signal, TONE_RANGE_BASE_CENTERS,
    TONE_RANGE_MIN_GAPS, TONE_RANGE_WIDTHS, VIDEO_EXTS,
    VideoInfo, apply_anti_banding, apply_curve_lut, apply_depth_output_grade, apply_depth_smoothing,
    apply_detail_boost, apply_input_adjustments_bgr, apply_input_adjustments_from_cfg, apply_levels,
    apply_normal_depth_fusion, apply_subject_background_fill, apply_tone_ranges, atexit, bgr_to_pixmap,
    build_alignment_subject_mask, build_configured_subject_mask, build_curve_lut, build_sliding_ranges,
    clear_all_cache, clear_cache_entry, clear_cache_older_than, clear_memory_model_cache,
    compute_mask_depth_stats, contextlib, cuda_total_memory_gb, current_event_log_path, add_event_listener, remove_event_listener, install_stdio_event_tee,
    curve_points_are_identity, cv2, dataclass, default_output_path, default_structure_output_dir,
    depth_affine_to_reference, depth_percentile_range, describe_real_alpha_source, directory_size_bytes,
    ensure_aux_model_file, ensure_builtin_preset_files, estimate_vram_gb, even_int, event_exception,
    event_log, extract_subject_alpha_from_cutout_frame, extract_subject_alpha_from_depth_frame,
    extract_valid_alpha_from_external_depth_frame, fast_bilateral_float, ffmpeg_has_encoder, format_bytes,
    format_seconds, frame_alpha_path, frame_cache_paths, frame_cache_root, frame_refined_depth_path,
    fuse_base_depth_with_external_reference, gc, get_cached_aux_model, get_cached_model, hashlib, importlib,
    init_event_logger, install_global_event_hooks,
    io, is_direct_depth_video_workflow, is_likely_cutout_frame, is_local_model_ready, is_scene_cut, json,
    lightweight_environment_report, list_cache_entries, load_alpha_mask_for_depth, local_aux_model_file,
    logging, make_base_gray_for_levels, make_diagnostic_preview_bgr, map_aux_frame_index,
    model_cache_dir_from_id, mux_original_audio, normal_guided_depth_refine, normalize_curve_points,
    normalize_depth, np, os, partial_output_path, partial_png_sequence_dir, png_sequence_output_dir,
    postprocess_subject_mask, prepare_da3_input_frame, probe_video, qInstallMessageHandler, queue,
    quiet_third_party_output, read_external_depth_reference, read_external_depth_reference_with_alpha,
    read_external_subject_mask, read_masked_depth_video_frame, read_video_frame_alpha01,
    read_video_frame_bgr, refine_depth_with_human_crop, remove_partial_output, render_depth_frame,
    render_depth_frame_16bit, render_depth_gray_float, resolve_device_mode, robust_global_range,
    save_npy_safely, scaled_size_from_long_side, scene_cut_score, scene_cut_signature, short_error_message,
    shutil, stabilize_subject_mask_temporal, structure_cache_root, subject_bbox_from_mask, subprocess, sys,
    temporal_align_depth_by_mask, temporal_ema_masked, threading, time, tone_range_reference_bands,
    traceback, trim_cuda_allocator_cache, try_load_npy, update_anchor_stats, verbose_third_party_output,
    video_has_real_alpha, warmup_three_models,
)
from depth_fusion_workers import (
    DepthWorker,
    DepthExportWorker,
    MeshExportWorker,
    StructureCacheWorker,
    ModelPreloadWorker,
    PreviewWorker,
    MeshPreviewWorker,
    SegmentationCacheWorker,
    OriginalFrameWorker,
    DropLineEdit,
    NoWheelSlider,
    NoWheelSpinBox,
    NoWheelDoubleSpinBox,
    NoWheelComboBox,
    PreviewImageLabel,
    LocalModelManagerDialog,
    is_structure_xyz_export_config,
    _BaseRebuildWorker,
)


from common.encoder_display import encoder_display_name, encoder_internal_name


from structure_pipeline.structure_cache import load_structure_frame
from geometry_fusion.structure_detail import (
    apply_depth_detail_displacement,
    apply_normal_depth_detail_displacement,
    make_surface_sample_spec,
    robust_geometry_center,
    sample_mesh_normals_with_spec,
    sample_mesh_surface_with_spec,
    stabilize_vertices_by_root,
)
from geometry_fusion.stable_dense_mesh import (
    apply_shell_offsets,
    build_dense_mesh_template,
    conservative_shell_offsets,
    evaluate_dense_normals,
    evaluate_dense_vertices,
    soft_region_weights,
)
from segmentation_pipeline.human_parsing import check_segmentation_environment, run_human_parsing
from segmentation_pipeline.foreground import check_foreground_environment, constrain_by_foreground, read_alpha_foreground
from segmentation_pipeline.mask_quality import classify_mask_quality, quality_to_meta
from segmentation_pipeline.segmentation_cache import save_segmentation_frame, load_segmentation_frame, segmentation_frame_paths, segmentation_cache_summary

from components.widgets import SliderValue, ToneWheelCard

from components.waveform import CurveWaveformPanel


class MainWindow(QMainWindow):
    event_console_line = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            win_w = min(1850, max(1580, avail.width() - 80))
            win_h = min(1040, max(940, avail.height() - 80))
            self.resize(win_w, win_h)
        else:
            self.resize(1580, 940)
        self.setMinimumSize(1580, 900)
        self.current_project_dir: Optional[Path] = None
        self.video_info: Optional[VideoInfo] = None
        self.current_input: Optional[str] = None
        self.thread: Optional[QThread] = None
        self.worker: Optional[DepthWorker] = None
        self.structure_cache_thread: Optional[QThread] = None
        self.structure_cache_worker: Optional[StructureCacheWorker] = None
        self.segmentation_cache_thread: Optional[QThread] = None
        self.segmentation_cache_worker: Optional[SegmentationCacheWorker] = None
        self._structure_dep_autoretry_count: int = 0
        self.preview_thread: Optional[QThread] = None
        self.preview_worker: Optional[PreviewWorker] = None
        self.preload_thread: Optional[QThread] = None
        self.preload_worker: Optional[ModelPreloadWorker] = None
        self.preload_key: Optional[tuple[str, str]] = None
        self.preload_pending_key: Optional[tuple[str, str]] = None
        self.loaded_model_key: Optional[tuple[str, str]] = None
        self._pending_model_action: Optional[str] = None
        self.original_frame_thread: Optional[QThread] = None
        self.original_frame_worker: Optional[OriginalFrameWorker] = None
        self._original_frame_requested: Optional[int] = None
        self._original_frame_running: bool = False
        self._original_frame_pending: bool = False
        self.preview_original_bgr: Optional[np.ndarray] = None
        self.preview_depth: Optional[np.ndarray] = None
        self.preview_subject_mask: Optional[np.ndarray] = None
        self.preview_normal_map: Optional[np.ndarray] = None
        self.preview_original_render_bgr: Optional[np.ndarray] = None
        self.preview_depth_render_bgr: Optional[np.ndarray] = None
        self.preview_base_gray_cache: Optional[np.ndarray] = None
        self.preview_hist_gray_cache: Optional[np.ndarray] = None
        self.preview_base_key: Optional[tuple] = None
        self.preview_depth_version: int = 0
        self._reference_preview_tile_keys: dict[str, tuple] = {}
        self._reference_preview_tile_bgr: dict[str, np.ndarray] = {}
        self.job_started_at: Optional[float] = None
        self._eta_started_at: Optional[float] = None
        self._eta_started_done: int = 0
        self._current_output_path: Optional[str] = None

        # Debounce timer: batches rapid slider changes into a single render call
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(140)  # ms; direct depth gray edits are cheap, keep preview responsive
        self._preview_debounce.timeout.connect(self._do_render_preview)
        self._preview_render_busy = False
        self._preview_render_pending = False


        self._direct_depth_auto_preview_timer = QTimer(self)
        self._direct_depth_auto_preview_timer.setSingleShot(True)
        self._direct_depth_auto_preview_timer.setInterval(220)
        self._direct_depth_auto_preview_timer.timeout.connect(self._auto_render_direct_depth_current_frame)

        self._seek_debounce = QTimer(self)
        self._seek_debounce.setSingleShot(True)
        self._seek_debounce.setInterval(70)
        self._seek_debounce.timeout.connect(self._start_original_frame_read)

        self._mesh_preview_frame_timer = QTimer(self)
        self._mesh_preview_frame_timer.setSingleShot(True)
        self._mesh_preview_frame_timer.setInterval(180)
        self._mesh_preview_frame_timer.timeout.connect(self._refresh_active_mesh_preview_from_timer)
        self._mesh_preview_rotation_timer = QTimer(self)
        self._mesh_preview_rotation_timer.setSingleShot(True)
        self._mesh_preview_rotation_timer.setInterval(140)
        self._mesh_preview_rotation_timer.timeout.connect(self._refresh_active_mesh_preview_from_timer)
        self._last_mesh_preview_mode: str | None = None
        self.mesh_preview_yaw: float = 0.0
        self.mesh_preview_pitch: float = 0.0
        self._preview_playing: bool = False
        self.preview_play_timer = QTimer(self)
        self.preview_play_timer.setInterval(42)
        self.preview_play_timer.timeout.connect(self._advance_preview_playback)

        self._preload_debounce = QTimer(self)
        self._preload_debounce.setSingleShot(True)
        self._preload_debounce.setInterval(600)
        self._preload_debounce.timeout.connect(self.start_model_preload)

        self._auto_mask_debounce = QTimer(self)
        self._auto_mask_debounce.setSingleShot(True)
        self._auto_mask_debounce.setInterval(160)
        self._auto_mask_debounce.timeout.connect(self._apply_auto_mask_controls)

        # Background base-rebuild thread state
        self._base_rebuild_thread: Optional[QThread] = None
        self._base_rebuild_worker: Optional[_BaseRebuildWorker] = None
        self._base_rebuild_pending: bool = False   # rebuild queued while one is running
        self._base_rebuild_restart_key: Optional[tuple] = None

        self.path_edit = DropLineEdit()
        self.path_edit.setPlaceholderText("拖入视频，或点击选择视频")
        self.path_edit.dropped.connect(self.load_video)
        self.pick_btn = QPushButton("选择视频")
        self.pick_btn.clicked.connect(self.pick_video)
        self.model_manager_btn = QPushButton("本地模型")
        self.model_manager_btn.clicked.connect(self.open_model_manager)
        self.cache_manager_btn = QPushButton("缓存管理")
        self.cache_manager_btn.clicked.connect(self.open_cache_manager)
        self.log_dir_btn = QPushButton("日志")
        self.log_dir_btn.clicked.connect(self.open_log_dir)

        self.info_label = QLabel("未导入视频")
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.long_side_spin = SliderValue(128, 4096, SAFE_DEFAULT_LONG_SIDE, step=64, decimals=0, label_width=72)
        self.long_side_spin.valueChanged.connect(self.on_output_geometry_changed)

        self.out_size_label = QLabel("输出: -")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setPlaceholderText("输出路径会随视频和分辨率自动生成")
        self.output_path_edit.setObjectName("outputPathEdit")
        self.output_pick_btn = QPushButton("选择输出")
        self.output_pick_btn.clicked.connect(self.pick_output_path)
        self.output_open_btn = QPushButton("打开目录")
        self.output_open_btn.setToolTip("导入视频并生成输出路径后可打开目录。")
        self.output_open_btn.clicked.connect(self.open_output_dir)
        self.output_open_btn.setEnabled(False)
        self._manual_output_path = False

        self.model_combo = NoWheelComboBox()
        self.model_combo.addItems(list(MODEL_IDS.keys()))
        self.model_combo.setCurrentText("图像驱动网格主流程")

        self.device_combo = NoWheelComboBox()
        self.device_combo.addItems(["CUDA 优先", "CPU"])

        self.batch_spin = SliderValue(1, 8, 1, step=1, decimals=0, label_width=48)

        self.process_res_spin = SliderValue(256, 2048, SAFE_DEFAULT_PROCESS_RES, step=32, decimals=0, label_width=64)


        self.pipeline_hint_label = QLabel("主流程：主视频 → 4D/WHAM 人体 Mesh → 逐帧 FASHN 分割缓存 → Body/Garment/Hair/Combined Mesh 预览 → 导出网格动画。")
        self.pipeline_hint_label.setObjectName("pipelineHint")
        self.pipeline_hint_label.setWordWrap(True)
        self.three_model_state_label = QLabel("主输入只需要主视频；Alpha 自动读取，有就用，没有也继续。Depth / 外部 法线 不进主流程。")
        self.three_model_state_label.setObjectName("pipelineHint")
        self.three_model_state_label.setWordWrap(True)

        self.color_combo = NoWheelComboBox()
        self.color_combo.addItems(["灰度", "伪彩色"])
        self.color_combo.setToolTip("仅影响右侧预览显示；正式导出始终为黑白 depth，不会输出伪彩色。")

        self.invert_check = QCheckBox("反向深度")
        self.cache_enable_check = QCheckBox("使用帧缓存")
        self.cache_enable_check.setChecked(True)
        self.cache_enable_check.setToolTip("缓存结构和预览数据。Depth / 外部 法线 已移出主流程。")
        self.copy_audio_check = QCheckBox("合并原音频")
        self.copy_audio_check.setChecked(True)
        self.copy_audio_check.setToolTip("需要系统可调用 ffmpeg；没有 ffmpeg 时自动跳过，仍会输出无声 MP4。")
        self.normalize_mode_combo = NoWheelComboBox()
        self.normalize_mode_combo.addItems(NORMALIZE_MODES)
        self.normalize_mode_combo.setCurrentText("全局抽帧")
        self.normalize_mode_combo.setToolTip("逐帧最快但容易呼吸；全局抽帧最稳；滑动窗口适合大场景变化。")
        self.encoder_combo = NoWheelComboBox()
        self.encoder_combo.addItems([encoder_display_name(mode) for mode in ENCODER_MODES])
        self.encoder_combo.setCurrentText(encoder_display_name("FFmpeg H.264"))
        self.encoder_combo.setToolTip("黑白 MP4 是默认视频输出；16bit PNG 序列用于 AE/Blender/位移贴图。FFmpeg 只是底层编码实现，不放在主界面命名里。")
        self.encoder_combo.currentTextChanged.connect(self.on_encoder_changed)

        self.pointcloud_enable_check = QCheckBox("启用点云输出")
        self.pointcloud_enable_check.setChecked(True)
        self.pointcloud_enable_check.setToolTip("启用真人视频到动态点云结果输出。默认生成 USDA 时序点云，同时保留 PLY 调试序列。")
        self.structure_solver_combo = NoWheelComboBox()
        self.structure_solver_combo.addItems(["4DHumans 结构补全", "WHAM 轨迹锚定"])
        self.structure_cache_btn = QPushButton("生成结构缓存")
        self.structure_cache_btn.setToolTip("调用外部 4DHumans / WHAM runner，生成 structure cache 后才能启用结构补全。")
        self.structure_cache_btn.clicked.connect(self.start_structure_cache_generation)
        self.generate_4dhumans_btn = QPushButton("生成 4D 人体")
        self.generate_4dhumans_btn.setToolTip("使用 4DHumans 生成身体结构缓存。默认优先方案，适合普通人体动作。")
        self.generate_4dhumans_btn.clicked.connect(self.start_4dhumans_structure_generation)
        self.generate_wham_btn = QPushButton("生成 WHAM 人体")
        self.generate_wham_btn.setToolTip("使用 WHAM 生成身体结构缓存。适合更重视轨迹锚定的片段。")
        self.generate_wham_btn.clicked.connect(self.start_wham_structure_generation)
        self.preview_4dhumans_btn = QPushButton("查看 4D")
        self.preview_4dhumans_btn.setEnabled(False)
        self.preview_4dhumans_btn.setToolTip("4D 缓存生成后可查看。")
        self.preview_4dhumans_btn.clicked.connect(lambda _checked=False: self.select_structure_scheme("4dhumans", preview=True))
        self.preview_wham_btn = QPushButton("查看 WHAM")
        self.preview_wham_btn.setEnabled(False)
        self.preview_wham_btn.setToolTip("WHAM 缓存生成后可查看。")
        self.preview_wham_btn.clicked.connect(lambda _checked=False: self.select_structure_scheme("wham", preview=True))
        self.structure_4d_status_label = QLabel("4D：未生成")
        self.structure_wham_status_label = QLabel("WHAM：未生成")
        for _lbl in (self.structure_4d_status_label, self.structure_wham_status_label):
            _lbl.setWordWrap(True)
            _lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.preview_structure_btn = QPushButton("身体")
        self.preview_structure_btn.setToolTip("只预览稳定人体 Mesh，不带衣服/头发增强。")
        self.preview_structure_btn.clicked.connect(self.preview_current_structure_frame)
        self.preview_garment_btn = QPushButton("Garment")
        self.preview_garment_btn.setToolTip("只预览衣服区域 Mesh，便于检查 Human Parsing 是否贴对。")
        self.preview_garment_btn.clicked.connect(self.preview_current_garment_frame)
        self.preview_hair_btn = QPushButton("Hair")
        self.preview_hair_btn.setToolTip("只预览头发区域 Mesh，便于检查 hair mask 是否贴对。")
        self.preview_hair_btn.clicked.connect(self.preview_current_hair_frame)
        self.preview_detail_btn = QPushButton("Combined")
        self.preview_detail_btn.setToolTip("预览人体 + 衣服 + 头发组合后的最终 Mesh。")
        self.preview_detail_btn.clicked.connect(self.preview_current_detail_frame)
        self.structure_cache_status_label = QLabel("结构缓存：未生成")
        self.structure_cache_status_label.setWordWrap(True)
        self.structure_cache_status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.structure_progress = QProgressBar()
        self.structure_progress.setRange(0, 100)
        self.structure_progress.setValue(0)
        self.structure_progress.setFormat("未开始")
        self.workflow_guide_label = QLabel("流程：1 主视频 → 2 稳定 Mesh 预览 → 3 细节 Mesh 预览 → 4 可选点云导出")
        self.workflow_guide_label.setWordWrap(True)
        self.workflow_guide_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.pointcloud_density_combo = NoWheelComboBox()
        self.pointcloud_density_combo.addItems(["低", "中", "高"])
        self.pointcloud_density_combo.setCurrentText("中")
        self.pointcloud_density_combo.setToolTip("低≈5万点/帧，中≈12万点/帧，高≈20万点/帧。")
        self.pointcloud_usd_check = QCheckBox("稳定点云（可选）")
        self.pointcloud_usd_check.setChecked(False)
        self.pointcloud_usd_check.setToolTip("可选：从稳定细节 Mesh 固定采样，导出 USDA 时序点云。只要网格时保持关闭。")
        self.mesh_export_check = QCheckBox("低模Mesh")
        self.mesh_export_check.setChecked(True)
        self.mesh_export_check.setToolTip("导出 Root稳定/时序去抖后的低模人体 mesh，用来检查 4D/WHAM 本体是否稳定。")
        self.detail_mesh_export_check = QCheckBox("Garment / Hair / Combined Mesh")
        self.detail_mesh_export_check.setChecked(True)
        self.detail_mesh_export_check.setToolTip("导出 mesh_garment.usda、mesh_hair.usda、mesh_combined.usda；均为固定拓扑网格动画。")
        self.mesh_dense_segments_combo = NoWheelComboBox()
        self.mesh_dense_segments_combo.addItems(["低 1x", "中 2x", "高 3x"])
        self.mesh_dense_segments_combo.setCurrentText("中 2x")
        self.mesh_dense_segments_combo.setToolTip("固定拓扑细分密度。越高越大越慢；不会逐帧 remesh，点 ID 稳定。")
        self.garment_shell_check = QCheckBox("衣服轮廓壳")
        self.garment_shell_check.setChecked(True)
        self.garment_shell_check.setToolTip("实验项：只做轻量轮廓外扩，不是视频衣服真实还原。默认开启；只有存在有效分割缓存时才会生效。")
        self.hair_shell_check = QCheckBox("头发轮廓壳")
        self.hair_shell_check.setChecked(True)
        self.hair_shell_check.setToolTip("实验项：只做头顶轮廓外扩，不是发丝/发型重建。默认开启；只有存在有效分割缓存时才会生效。")
        self.garment_shell_offset_spin = SliderValue(0, 25, 20, step=1, decimals=0, suffix=" mm", label_width=58)
        self.garment_shell_offset_spin.setToolTip("衣服轮廓壳外扩厚度。默认 20mm，并会按人体高度二次限幅。")
        self.hair_shell_offset_spin = SliderValue(0, 40, 35, step=1, decimals=0, suffix=" mm", label_width=58)
        self.hair_shell_offset_spin.setToolTip("头发轮廓壳外扩厚度。默认 35mm，并会按人体高度二次限幅。")
        self.segmentation_enable_check = QCheckBox("画面分割约束")
        self.segmentation_enable_check.setChecked(True)
        self.segmentation_enable_check.setToolTip("启用后优先用 Human Parsing / Hair mask 约束衣服和头发区域；模型缺失或分割不可用时明确 Body Only，不再生成假衣服/假头发。")
        self.segmentation_provider_combo = NoWheelComboBox()
        self.segmentation_provider_combo.addItems(["Auto", "FASHN Human Parser", "Off"])
        self.segmentation_provider_combo.setCurrentText("Auto")
        self.segmentation_provider_combo.setToolTip("默认 Auto：使用 models/segmentation/fashn_human_parser。没有模型时明确降级，不放未实现模型假选项。")
        self.garment_shell_check.toggled.connect(self._on_mesh_layer_dependency_changed)
        self.hair_shell_check.toggled.connect(self._on_mesh_layer_dependency_changed)
        self.segmentation_enable_check.toggled.connect(self._on_mesh_layer_dependency_changed)
        self.segmentation_provider_combo.currentTextChanged.connect(self._on_mesh_layer_dependency_changed)
        self.segmentation_status_label = QLabel("分割状态：未检查")
        self.segmentation_status_label.setWordWrap(True)
        self.segmentation_status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.segmentation_test_btn = QPushButton("测试当前帧分割")
        self.segmentation_cache_btn = QPushButton("生成逐帧分割缓存")
        self.segmentation_open_btn = QPushButton("打开 segmentation")
        self.segmentation_cache_status_label = QLabel("分割缓存：未生成")
        self.segmentation_cache_status_label.setWordWrap(True)
        self.segmentation_cache_status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.pointcloud_remove_outliers_check = QCheckBox("去飞点")
        self.pointcloud_remove_outliers_check.setChecked(True)
        self.pointcloud_remove_outliers_check.setToolTip("用稳健统计剔除离群点，减少人体边缘和错误 depth 的爆点。")
        self.pointcloud_voxel_check = QCheckBox("体素降采样")
        self.pointcloud_voxel_check.setChecked(True)
        self.pointcloud_voxel_check.setToolTip("按空间体素合并近邻点，减少重复点和闪烁。")
        self.pointcloud_temporal_check = QCheckBox("时序稳定")
        self.pointcloud_temporal_check.setChecked(True)
        self.pointcloud_temporal_check.setToolTip("时序稳定：结构 XYZ 主流程下会对 root 锁定后的人体做去抖和平滑，并轻度平滑最终细节，减少抽搐感。")

        self.smooth_spin = SliderValue(0, 95, 8, step=1, decimals=0, suffix=" %", label_width=58)

        self.black_pct_spin = SliderValue(0.0, 49.0, 0.0, step=0.5, decimals=1, suffix=" %", label_width=64)

        self.white_pct_spin = SliderValue(51.0, 100.0, 100.0, step=0.5, decimals=1, suffix=" %", label_width=64)

        self.gamma_spin = SliderValue(0.20, 3.00, 1.00, step=0.01, decimals=2, label_width=58)

        self.detail_boost_spin = SliderValue(0, 100, 0, step=1, decimals=0, suffix=" %", label_width=58)
        self.human_refine_spin = SliderValue(0, 100, 0, step=1, decimals=0, suffix=" %", label_width=58)
        self.human_refine_spin.setToolTip("人体 bbox 二次 模型深度 精炼。人物占画面小但需要人体细节时开启，耗时和显存会上升。")

        self.normal_strength_spin = SliderValue(0, 100, 70, step=1, decimals=0, suffix=" %", label_width=58)
        self.normal_refine_spin = SliderValue(0, 30, 0, step=1, decimals=0, suffix=" %", label_width=58)
        self.normal_strength_spin.setToolTip("法线 细节强度。推荐 55-80；过高会让衣服细节抖动。Depth 仍只做约束，不直接拉伸人体。")
        self.normal_refine_spin.setToolTip("保留给旧深度预览。结构 XYZ 主流程通常保持 0。")

        self.matting_model_path_edit = QLineEdit(str(DEFAULT_MATANYONE_MODEL_PATH))
        self.matting_model_path_edit.setPlaceholderText("models/matanyone/matanyone.pth")
        default_mask_path = ""
        if DEFAULT_MATTING_MASK_DIR.exists():
            for _p in sorted(DEFAULT_MATTING_MASK_DIR.glob("*.png")):
                default_mask_path = str(_p)
                break
        self.matting_mask_path_edit = QLineEdit(default_mask_path)
        self.matting_mask_path_edit.setPlaceholderText("第一帧人物 mask PNG：白=穿衣人物，黑=背景")
        self.matting_status_label = QLabel("MatAnyone 关闭")
        self.matting_status_label.setObjectName("pipelineHint")
        self.matting_status_label.setWordWrap(True)
        self.auto_mask_feather_spin = SliderValue(0, 48, 3, step=1, decimals=0, suffix=" px", label_width=58)
        self.auto_mask_feather_spin.setToolTip("Alpha 边缘羽化。0 最硬，3-6 通常够用。")
        self.auto_mask_expand_spin = SliderValue(-48, 48, 4, step=1, decimals=0, suffix=" px", label_width=58)
        self.auto_mask_expand_spin.setToolTip("Alpha 范围修正：负值收缩，正值扩张。建议先外扩 3-8px，再配合羽化。")
        self.background_mode_combo = NoWheelComboBox()
        self.background_mode_combo.addItems(["背景白", "背景灰", "背景黑", "保留场景Depth"])
        self.background_mode_combo.setToolTip("控制主体 Mask 外的最终输出。默认背景白：使用反相 Mask 以 Add 方式叠到 Depth，外部白、人物保留深度，方向与远白近黑一致。")
        self.background_gray_spin = SliderValue(0, 255, 255, step=1, decimals=0, label_width=58)
        self.background_gray_spin.setToolTip("背景模式为“背景灰”时使用的灰度值。0=黑，255=白。默认白背景不使用这个值。")
        self.background_gray_spin.setEnabled(False)

        self.input_cutout_mask_check = QCheckBox("使用原视频 Alpha")
        self.input_cutout_mask_check.setToolTip("原视频必须带真实 Alpha 通道；不需要第二个 Mask 文件。")
        self.input_cutout_mask_check.setChecked(True)
        self.external_mask_path_edit = QLineEdit()
        self.external_mask_path_edit.setPlaceholderText("Legacy：当前主流程不使用")
        self.external_mask_pick_btn = QPushButton("Mask")
        self.external_mask_pick_btn.clicked.connect(self.pick_external_mask_path)
        self.external_mask_invert_check = QCheckBox("反相Mask")
        self.external_mask_invert_check.setToolTip("如果预览里人物和背景反了，再启用。")
        self.external_depth_path_edit = QLineEdit()
        self.external_depth_path_edit.setPlaceholderText("隐藏兼容：主流程不需要参考深度")
        self.external_depth_pick_btn = QPushButton("参考深度")
        self.external_depth_pick_btn.clicked.connect(self.pick_external_depth_path)
        self.external_depth_weight_spin = SliderValue(0, 100, 35, step=1, decimals=0, suffix=" %", label_width=58)
        self.external_depth_weight_spin.setToolTip("Depth 约束强度。它只限制 法线 细节的幅度/遮挡可信度；0 表示不加细节，默认 35%。")
        self.external_depth_invert_check = NoWheelComboBox()
        self.external_depth_invert_check.addItems(["自动方向", "不反相", "反相"])
        self.external_depth_invert_check.setToolTip("参考深度方向控制。结构XYZ模式没有 旧深度 base 可做相关性判断，自动方向会保守等同不反相；方向反了就手动选反相。")
        self.external_status_label = QLabel("参考深度：未导入")
        self.external_status_label.setObjectName("pipelineHint")
        self.external_status_label.setWordWrap(True)
        self.external_chain_label = QLabel("操作链：主视频 → structure cache → 稳定 Mesh / Shell → Mesh 或点云导出。")
        self.external_chain_label.setObjectName("pipelineHint")
        self.external_chain_label.setWordWrap(True)
        self.top_main_chain_label = QLabel("主视频：未加载")
        self.top_main_chain_label.setObjectName("topChainBadge")
        self.top_main_chain_label.setMinimumWidth(160)
        self.top_external_depth_label = QLabel("Alpha：未检测")
        self.top_external_depth_label.setObjectName("topChainBadge")
        self.top_external_depth_label.setMinimumWidth(150)
        self.external_validate_btn = QPushButton("校验素材")
        self.external_validate_btn.setToolTip("检查原视频是否保留真实 Alpha 通道。")
        self.external_validate_btn.clicked.connect(self.validate_external_reference_chain)

        # Unified image-adjustment panel. 原图 = 模型深度 推理前；Depth = 推理后。
        self.adjust_mode_combo = NoWheelComboBox()
        self.adjust_mode_combo.addItems(["深度视频灰度"])
        self.adjust_mode_combo.setToolTip("曲线、灰阶、五区参数作用在深度视频 RGB 转出的灰度 depth；不会改 Alpha。")
        self.adjust_mode_combo.currentTextChanged.connect(self.on_adjust_mode_changed)
        self.adjust_stack = QStackedWidget()

        self.input_brightness_spin = SliderValue(-100, 100, 0, step=1, decimals=0, label_width=54)
        self.input_contrast_spin = SliderValue(-100, 100, 0, step=1, decimals=0, label_width=54)
        self.input_gamma_spin = SliderValue(0.20, 3.00, 1.00, step=0.01, decimals=2, label_width=54)
        self.input_shadow_spin = SliderValue(-100, 100, 0, step=1, decimals=0, label_width=54)
        self.input_highlight_spin = SliderValue(-100, 100, 0, step=1, decimals=0, label_width=54)
        self.input_sharpen_spin = SliderValue(0, 100, 0, step=1, decimals=0, suffix=" %", label_width=54)
        self.input_denoise_spin = SliderValue(0, 100, 0, step=1, decimals=0, suffix=" %", label_width=54)
        for _spin, _tip in [
            (self.input_brightness_spin, "模型深度 推理前亮度，只影响模型看到的画面。"),
            (self.input_contrast_spin, "模型深度 推理前对比度。过高可能让 depth 更硬、更容易断层。"),
            (self.input_gamma_spin, "模型深度 推理前 Gamma。用于提亮暗部或压回高光。"),
            (self.input_shadow_spin, "模型深度 推理前暗部修正。适合让人物边缘和衣服暗纹被模型看见。"),
            (self.input_highlight_spin, "模型深度 推理前高光修正。避免亮面区域过曝丢结构。"),
            (self.input_sharpen_spin, "模型深度 推理前轻锐化。建议低值，过高会把纹理变成伪深度。"),
            (self.input_denoise_spin, "模型深度 推理前降噪。减少噪点被模型当作起伏。"),
        ]:
            _spin.setToolTip(_tip)

        self.levels_in_black_spin = SliderValue(0, 254, 0, step=1, decimals=0, label_width=52)

        self.levels_in_white_spin = SliderValue(1, 255, 255, step=1, decimals=0, label_width=52)

        self.levels_out_black_spin = SliderValue(0, 255, 0, step=1, decimals=0, label_width=52)

        self.levels_out_white_spin = SliderValue(0, 255, 255, step=1, decimals=0, label_width=52)

        self.normal_strength_spin.setToolTip("法线 模型参与强度：控制身体起伏和局部法线感。")
        self.normal_refine_spin.setToolTip("法线 几何细化：比普通 法线 引导更强；未启用 MatAnyone 时使用 模型深度 临时人物蒙版限制范围。")
        self.detail_boost_spin.setToolTip("Depth 细节增强：提升灰度局部对比，不重新推理模型。")
        self.anti_banding_spin = SliderValue(0, 100, 35, step=1, decimals=0, suffix=" %", label_width=58)
        self.depth_smooth_spin = SliderValue(0, 100, 45, step=1, decimals=0, suffix=" %", label_width=58)
        self.edge_preserve_spin = SliderValue(0, 100, 75, step=1, decimals=0, suffix=" %", label_width=58)
        def _make_tone_slider(minimum: int, maximum: int, value: int = 0, label_width: int = 46) -> SliderValue:
            spin = SliderValue(minimum, maximum, value, step=1, decimals=0, label_width=label_width)
            spin.setToolTip("分段色调微调：拖动看预览，也可点击数值精确输入。")
            return spin

        self.tone_black_spin = _make_tone_slider(-100, 100, 0)
        self.tone_shadow_spin = _make_tone_slider(-100, 100, 0)
        self.tone_mid_spin = _make_tone_slider(-100, 100, 0)
        self.tone_light_spin = _make_tone_slider(-100, 100, 0)
        self.tone_white_spin = _make_tone_slider(-100, 100, 0)
        self.tone_black_shift_spin = _make_tone_slider(-40, 40, 0)
        self.tone_shadow_shift_spin = _make_tone_slider(-40, 40, 0)
        self.tone_mid_shift_spin = _make_tone_slider(-40, 40, 0)
        self.tone_light_shift_spin = _make_tone_slider(-40, 40, 0)
        self.tone_white_shift_spin = _make_tone_slider(-40, 40, 0)
        self.tone_black_contrast_spin = _make_tone_slider(-100, 100, 0)
        self.tone_shadow_contrast_spin = _make_tone_slider(-100, 100, 0)
        self.tone_mid_contrast_spin = _make_tone_slider(-100, 100, 0)
        self.tone_light_contrast_spin = _make_tone_slider(-100, 100, 0)
        self.tone_white_contrast_spin = _make_tone_slider(-100, 100, 0)

        self.levels_panel = CurveWaveformPanel()
        self.levels_panel.levelsChanged.connect(self.on_levels_panel_changed)
        self.levels_panel.curveChanged.connect(self.on_curve_panel_changed)
        self.curve_reset_btn = QPushButton("重置曲线")
        self.curve_reset_btn.clicked.connect(self.reset_free_curve)
        self.curve_hint_label = QLabel("点击曲线加点；拖动点调整；选中中间点后点 × / Delete 删除")
        self.curve_hint_label.setObjectName("hintLabel")
        self._syncing_levels = False

        self.preset_human_btn = QPushButton("人体动作推荐")
        self.preset_human_btn.clicked.connect(self.apply_human_motion_preset)
        self.preset_neutral_btn = QPushButton("恢复稳妥默认")
        self.preset_neutral_btn.clicked.connect(self.apply_neutral_preset)
        self.preset_displacement_btn = QPushButton("人体位移")
        self.preset_displacement_btn.clicked.connect(lambda: self.apply_builtin_preset("human_displacement"))
        self.preset_high_png_btn = QPushButton("高精PNG")
        self.preset_high_png_btn.clicked.connect(lambda: self.apply_builtin_preset("high_precision_png"))
        self.preset_low_mem_btn = QPushButton("低显存")
        self.preset_low_mem_btn.clicked.connect(lambda: self.apply_builtin_preset("low_memory"))
        self.preset_import_btn = QPushButton("导入预设")
        self.preset_import_btn.clicked.connect(self.import_preset_json)
        self.preset_export_btn = QPushButton("导出预设")
        self.preset_export_btn.clicked.connect(self.export_preset_json)

        self.preview_frame_slider = NoWheelSlider(Qt.Horizontal)
        self.preview_frame_slider.setRange(0, 0)
        self.preview_frame_slider.setSingleStep(1)
        self.preview_frame_slider.setPageStep(24)
        self.preview_frame_slider.valueChanged.connect(self.on_preview_frame_slider_changed)

        # Numeric frame input only. The horizontal frame slider above is the real seek bar;
        # using SliderValue here creates a second, redundant seek slider in the run bar.
        self.preview_frame_spin = NoWheelSpinBox()
        self.preview_frame_spin.setRange(0, 0)
        self.preview_frame_spin.setMinimumWidth(76)
        self.preview_frame_spin.setMaximumWidth(92)
        self.preview_frame_spin.setObjectName("frameNumberSpin")
        self.preview_frame_spin.setToolTip("当前帧编号，可直接输入。")
        self.preview_frame_spin.valueChanged.connect(lambda _value=0: self.on_preview_frame_spin_changed())

        self.input_preview_frame_slider = NoWheelSlider(Qt.Horizontal)
        self.input_preview_frame_spin = NoWheelSpinBox()
        self.structure_preview_frame_slider = NoWheelSlider(Qt.Horizontal)
        self.structure_preview_frame_spin = NoWheelSpinBox()
        for _sl in (self.input_preview_frame_slider, self.structure_preview_frame_slider):
            _sl.setRange(0, 0)
            _sl.setSingleStep(1)
            _sl.setPageStep(24)
            _sl.valueChanged.connect(lambda v, _self=self: _self._apply_preview_frame_value(int(v)))
        for _sp in (self.input_preview_frame_spin, self.structure_preview_frame_spin):
            _sp.setRange(0, 0)
            _sp.setMinimumWidth(76)
            _sp.setMaximumWidth(92)
            _sp.valueChanged.connect(lambda v, _self=self: _self._apply_preview_frame_value(int(v)))
        self.input_preview_play_btn = QPushButton("播放")
        self.input_preview_play_btn.setToolTip("播放当前处理范围内的原视频预览。")
        self.input_preview_play_btn.clicked.connect(self.toggle_preview_playback)

        self.preview_frame_label = QLabel("第 0 帧 / 0:00")
        self.preview_frame_label.setMinimumWidth(150)

        # Processing range controls. They drive structure/segmentation/export caches,
        # not just preview. Two sliders are used instead of a fake progress bar so
        # the user can set exact in/out frames without a custom range-slider widget.
        self.processing_start_slider = NoWheelSlider(Qt.Horizontal)
        self.processing_end_slider = NoWheelSlider(Qt.Horizontal)
        for _s in (self.processing_start_slider, self.processing_end_slider):
            _s.setRange(0, 0)
            _s.setSingleStep(1)
            _s.setPageStep(24)
        self.processing_start_spin = NoWheelSpinBox()
        self.processing_end_spin = NoWheelSpinBox()
        for _sp in (self.processing_start_spin, self.processing_end_spin):
            _sp.setRange(0, 0)
            _sp.setMinimumWidth(76)
            _sp.setMaximumWidth(92)
        self.processing_range_label = QLabel("处理范围：0 - 0")
        self.processing_range_label.setMinimumWidth(180)
        self.processing_range_progress = QProgressBar()
        self.processing_range_progress.setRange(0, 100)
        self.processing_range_progress.setValue(100)
        self.processing_range_progress.setFormat("处理范围：全部")
        self.processing_range_progress.setVisible(False)  # 双滑条已表达入点/出点位置，避免用单值进度条误导。
        self.processing_start_slider.valueChanged.connect(self._on_processing_start_slider_changed)
        self.processing_end_slider.valueChanged.connect(self._on_processing_end_slider_changed)
        self.processing_start_spin.valueChanged.connect(self._on_processing_start_spin_changed)
        self.processing_end_spin.valueChanged.connect(self._on_processing_end_spin_changed)

        self.preview_btn = QPushButton("读取当前帧 / Mask")
        self.preview_btn.clicked.connect(self.refresh_alpha_preview)
        self.preview_big_btn = QPushButton("打开大图预览")
        self.preview_big_btn.setToolTip("先渲染当前帧生成预览后可打开大图。")
        self.preview_big_btn.setEnabled(False)
        self.preview_big_btn.clicked.connect(self.open_large_preview)
        self.preview_layout_mode = "horizontal"
        self.preview_layout_buttons: dict[str, QPushButton] = {}
        for _layout, _text, _tip in [
            ("horizontal", "左右", "主预览和原图辅助左右并排，适合 9:16 人体视频对比。"),
            ("vertical", "上下", "主预览和原图辅助上下并排，适合窗口较窄或横版素材。"),
        ]:
            _btn = QPushButton(_text)
            _btn.setCheckable(True)
            _btn.setObjectName("previewSwitchBtn")
            _btn.setToolTip(_tip)
            _btn.clicked.connect(lambda _checked=False, m=_layout: self.set_preview_layout_mode(m))
            self.preview_layout_buttons[_layout] = _btn
        self.preview_canvas_layout = None
        self.preview_depth_tile = None
        self.preview_original_tile = None
        self.preview_status_label = QLabel("未生成预览")
        self.preview_original_label = PreviewImageLabel()
        self.preview_original_label.setText("原视频参考")
        self.preview_external_depth_label = PreviewImageLabel()
        self.preview_external_depth_label.setText("原视频 Alpha")
        self.preview_da3_label = PreviewImageLabel()
        self.preview_da3_label.setText("结构缓存")
        self.preview_subject_alpha_label = PreviewImageLabel()
        self.preview_subject_alpha_label.setText("原视频Alpha")
        self.preview_normal_label = PreviewImageLabel()
        self.preview_normal_label.setText("法线 参考")
        self.preview_depth_label = PreviewImageLabel()
        self.preview_depth_label.setText("当前帧预览")
        _reference_preview_defs = (
            ("main", "原视频参考", self.preview_original_label),
            ("external_depth", "原视频 Alpha", self.preview_external_depth_label),
            ("da3_depth", "结构缓存", self.preview_da3_label),
            ("subject_alpha", "原视频Alpha", self.preview_subject_alpha_label),
            ("normal_map", "法线 参考", self.preview_normal_label),
        )
        for _key, _title, _preview_label in _reference_preview_defs:
            _preview_label.double_clicked.connect(lambda k=_key, t=_title: self.open_single_reference_preview(k, t))
            _preview_label.setObjectName("previewImage")
            _preview_label.setMinimumSize(120, 92)
            _preview_label.setMaximumHeight(125)
            _preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            _preview_label.setAlignment(Qt.AlignCenter)
        self.preview_depth_label.double_clicked.connect(self.open_large_preview)
        self.preview_depth_label.setObjectName("previewImage")
        self.preview_depth_label.setMinimumSize(420, 240)
        self.preview_depth_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_depth_label.setAlignment(Qt.AlignCenter)

        # Page-local mesh preview canvases. Do not reuse a QLabel across pages:
        # Qt widgets can only have one parent/layout. preview_depth_label remains
        # a compatibility pointer and is switched to the active canvas before mesh
        # preview workers write an image.
        self.structure_mesh_preview_label = PreviewImageLabel()
        self.structure_mesh_preview_label.setText("人体结构预览")
        self.structure_mesh_preview_label.double_clicked.connect(self.open_large_preview)
        self.structure_mesh_preview_label.setObjectName("previewImage")
        self.structure_mesh_preview_label.setMinimumSize(420, 240)
        self.structure_mesh_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.structure_mesh_preview_label.setAlignment(Qt.AlignCenter)
        try:
            self.structure_mesh_preview_label.setInteractionMode("rotate")
            self.structure_mesh_preview_label.dragged.connect(self.on_mesh_preview_dragged)
        except Exception:
            pass

        self.layer_mesh_preview_label = PreviewImageLabel()
        self.layer_mesh_preview_label.setText("衣服 / 头发预览")
        self.layer_mesh_preview_label.double_clicked.connect(self.open_large_preview)
        self.layer_mesh_preview_label.setObjectName("previewImage")
        self.layer_mesh_preview_label.setMinimumSize(420, 240)
        self.layer_mesh_preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layer_mesh_preview_label.setAlignment(Qt.AlignCenter)
        try:
            self.layer_mesh_preview_label.setInteractionMode("rotate")
            self.layer_mesh_preview_label.dragged.connect(self.on_mesh_preview_dragged)
        except Exception:
            pass

        self.preview_original_status_line = QFrame()
        self.preview_original_status_line.setObjectName("previewStatusLine")
        self.preview_original_status_line.setProperty("role", "original")
        self.preview_original_status_line.setProperty("busy", "0")
        self.preview_original_status_line.setFixedHeight(3)
        self.preview_depth_status_line = QFrame()
        self.preview_depth_status_line.setObjectName("previewStatusLine")
        self.preview_depth_status_line.setProperty("role", "depth")
        self.preview_depth_status_line.setProperty("busy", "0")
        self.preview_depth_status_line.setFixedHeight(3)

        self.external_mask_path_edit.textChanged.connect(lambda _text="": self.on_external_media_changed())
        self.external_depth_path_edit.textChanged.connect(lambda _text="": self.on_external_media_changed())
        self.external_depth_weight_spin.valueChanged.connect(self.on_external_media_changed)
        self.normal_strength_spin.valueChanged.connect(self._update_three_model_status)
        self.normal_refine_spin.valueChanged.connect(self._update_three_model_status)
        self.black_pct_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.white_pct_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.gamma_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.normalize_mode_combo.currentTextChanged.connect(self.render_preview_from_cache)
        self.detail_boost_spin.valueChanged.connect(self.render_preview_from_cache)
        self.human_refine_spin.valueChanged.connect(self.on_human_refine_changed)
        self.normal_strength_spin.valueChanged.connect(self.render_preview_from_cache)
        self.normal_refine_spin.valueChanged.connect(self.render_preview_from_cache)
        self.auto_mask_feather_spin.valueChanged.connect(self.on_auto_mask_controls_changed)
        self.auto_mask_expand_spin.valueChanged.connect(self.on_auto_mask_controls_changed)
        self.background_mode_combo.currentTextChanged.connect(self.on_background_fill_changed)
        self.background_gray_spin.valueChanged.connect(self.on_background_fill_changed)
        for _spin in (
            self.input_brightness_spin, self.input_contrast_spin, self.input_gamma_spin,
            self.input_shadow_spin, self.input_highlight_spin, self.input_sharpen_spin, self.input_denoise_spin,
        ):
            _spin.valueChanged.connect(self.on_input_adjust_controls_changed)
        self.anti_banding_spin.valueChanged.connect(self.render_preview_from_cache)
        self.depth_smooth_spin.valueChanged.connect(self.render_preview_from_cache)
        self.edge_preserve_spin.valueChanged.connect(self.render_preview_from_cache)
        self.tone_black_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_shadow_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_mid_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_light_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_white_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_black_shift_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_shadow_shift_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_mid_shift_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_light_shift_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_white_shift_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_black_contrast_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_shadow_contrast_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_mid_contrast_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_light_contrast_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.tone_white_contrast_spin.valueChanged.connect(self.on_tone_controls_changed)
        self.levels_in_black_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.levels_in_white_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.levels_out_black_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.levels_out_white_spin.valueChanged.connect(self.on_levels_controls_changed)
        self.invert_check.stateChanged.connect(lambda _state=0: self.on_levels_controls_changed())
        self.color_combo.currentTextChanged.connect(self.render_preview_from_cache)
        self.model_combo.currentTextChanged.connect(self.on_model_device_changed)
        self.device_combo.currentTextChanged.connect(self.on_model_device_changed)
        self.sync_levels_panel_from_controls()

        self.start_btn = QPushButton("开始转换")
        self.start_btn.clicked.connect(self.start_job)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setToolTip("导出任务运行时可取消。")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_job)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.stage_status_label = QLabel("阶段：空闲")
        self.stage_status_label.setObjectName("stageStatus")
        self.stage_status_label.setMinimumWidth(180)
        self.stage_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(10000)
        self.log_box.setPlaceholderText("事件控制台：UI、Worker、第三方模型、stdout/stderr、Qt 警告和异常都会显示在这里。")
        self._event_console_listener_active = False
        self._event_console_recent: list[str] = []
        self._event_console_listener = lambda line: self.event_console_line.emit(str(line))
        self.event_console_line.connect(self._append_event_console_line)
        add_event_listener(self._event_console_listener)
        self._event_console_listener_active = True

        self._build_layout()
        self._apply_style()
        self._update_three_model_status()
        self.log(f"事件日志: {current_event_log_path()}")
        # Torch/CUDA environment probing can take noticeable time on Windows.
        # Schedule it after the window is visible instead of blocking startup.
        QTimer.singleShot(300, self._log_lightweight_environment_report)

    def _log_lightweight_environment_report(self) -> None:
        try:
            report = lightweight_environment_report()
            rows = report if isinstance(report, (list, tuple)) else str(report).splitlines()
            for _row in rows:
                if str(_row).strip():
                    self.log("环境检查: " + str(_row))
        except Exception as exc:  # noqa: BLE001
            self.log(f"环境检查失败: {exc}")

    def _build_layout(self) -> None:
        from components.panels import build_main_layout
        build_main_layout(self)

    def _effective_normal_strength(self) -> int:
        return 0

    def _effective_normal_refine(self) -> int:
        return 0

    def on_three_model_controls_changed(self) -> None:
        self._update_three_model_status()
        if self.preview_depth is not None:
            need_normal = (self._effective_normal_strength() > 0 or self._effective_normal_refine() > 0) and self.preview_normal_map is None
            if need_normal:
                self.preview_base_gray_cache = None
                self.preview_hist_gray_cache = None
                self.preview_base_key = None
                self.preview_status_label.setText("当前预览缺少 法线/Alpha 缓存，点“渲染当前帧”补齐。")
                return
        self.render_preview_from_cache()

    def _update_three_model_status(self) -> None:
        if hasattr(self, "three_model_state_label"):
            solver = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else "4DHumans"
            self.three_model_state_label.setText(f"主线：{solver} → Root稳定/时序去抖 → Dense Mesh → Garment/Hair Shell → Mesh / 可选点云")
        self._update_matting_status_label()
        self._update_external_media_status_label()
        self.refresh_3d_model_status()
        try:
            self.normal_strength_spin.setEnabled(False)
            self.normal_refine_spin.setEnabled(False)
        except Exception:
            pass

    def on_matting_controls_changed(self) -> None:
        pass

    def _update_matting_status_label(self) -> None:
        pass


    def _source_mode_from_current_controls(self) -> str:
        """The structure-XYZ workflow uses the alpha channel embedded in the main video."""
        return "cutout_video"

    def _current_source_mode(self) -> str:
        return "cutout_video"

    def _sync_source_mode_radios(self) -> None:
        if not hasattr(self, "source_cutout_radio"):
            return
        for radio in (self.source_cutout_radio, self.source_matanyone_radio, self.source_external_mask_radio):
            radio.blockSignals(True)
            radio.setChecked(radio is self.source_cutout_radio)
            radio.blockSignals(False)
        self._update_conditional_visibility()

    def _apply_source_mode(self, mode: str) -> None:
        """Force the single-input workflow: main video alpha only."""
        for widget, checked in (
            (self.input_cutout_mask_check, True),
        ):
            widget.blockSignals(True)
            widget.setChecked(bool(checked))
            widget.blockSignals(False)
        self._sync_source_mode_radios()
        self._update_matting_status_label()
        self._update_conditional_visibility()
        self.on_external_media_changed()

    def _effective_pointcloud_stride(self) -> int:
        density = self.pointcloud_density_combo.currentText() if hasattr(self, "pointcloud_density_combo") else "中"
        if density == "低":
            return 5
        if density == "中":
            return 3
        if density == "高":
            return 2
        return 3

    def _on_density_mode_changed(self) -> None:
        is_custom = False
        pointcloud_on = bool(getattr(self, "pointcloud_usd_check", None) is None or self.pointcloud_usd_check.isChecked())
        for name in ("pointcloud_stride_row", "pointcloud_max_points_row"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(bool(pointcloud_on and is_custom))
        for name in ("pointcloud_stride_spin", "pointcloud_max_points_spin"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(bool(is_custom))
                widget.setEnabled(bool(is_custom))

    def _update_conditional_visibility(self) -> None:
        """Keep low-frequency controls out of sight until their mode makes them useful."""
        source_mode = self._current_source_mode() if hasattr(self, "_current_source_mode") else self._source_mode_from_current_controls()
        matting_on = source_mode == "matanyone"
        external_mask_on = source_mode == "external_mask"
        if hasattr(self, "matting_paths_widget"):
            self.matting_paths_widget.setVisible(matting_on)
        if hasattr(self, "external_mask_paths_widget"):
            self.external_mask_paths_widget.setVisible(external_mask_on)
        bg_is_gray = bool(hasattr(self, "background_mode_combo") and self.background_mode_combo.currentText() == "背景灰")
        if hasattr(self, "background_gray_spin"):
            self.background_gray_spin.setEnabled(bg_is_gray)
        if hasattr(self, "background_gray_row"):
            self.background_gray_row.setVisible(bg_is_gray)
        if hasattr(self, "pointcloud_density_combo"):
            self._on_density_mode_changed()

    def pick_matting_model_path(self) -> None:
        start = self.matting_model_path_edit.text().strip() or str(DEFAULT_MATANYONE_MODEL_PATH)
        start_path = Path(start)
        start_dir = start_path if start_path.is_dir() else start_path.parent
        path, _ = QFileDialog.getOpenFileName(self, "选择 MatAnyone 模型", str(start_dir), "PyTorch Model (*.pth);;All Files (*)")
        if path:
            self.matting_model_path_edit.setText(os.path.normpath(path))

    def pick_matting_mask_path(self) -> None:
        start = self.matting_mask_path_edit.text().strip() or str(DEFAULT_MATTING_MASK_DIR)
        path, _ = QFileDialog.getOpenFileName(self, "选择第一帧人物 mask", str(Path(start).parent), "Mask Image (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if path:
            self.matting_mask_path_edit.setText(os.path.normpath(path))

    def pick_external_mask_path(self) -> None:
        start = self.external_mask_path_edit.text().strip() or self.current_input or str(PROJECT_DIR)
        path, _ = QFileDialog.getOpenFileName(self, "选择外部抠像人像 / Mask 视频", str(Path(start).parent), "Video/Image (*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if path:
            self.external_mask_path_edit.setText(os.path.normpath(path))
            # Kept only for legacy compatibility. The main workflow still uses the original video alpha.
            if hasattr(self, "source_cutout_radio"):
                self.source_cutout_radio.setChecked(True)
            self.validate_external_reference_chain(silent=True)

    def pick_external_depth_path(self) -> None:
        start = self.external_depth_path_edit.text().strip() or self.current_input or str(PROJECT_DIR)
        path, _ = QFileDialog.getOpenFileName(self, "选择参考深度 RGBA 视频", str(Path(start).parent), "Video/Image (*.mp4 *.mov *.avi *.mkv *.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if path:
            self.external_depth_path_edit.setText(os.path.normpath(path))
            ready = self.validate_external_reference_chain(silent=True)
            if self.current_input and self.video_info and ready:
                if hasattr(self, "preview_status_label"):
                    self.preview_status_label.setText("参考深度已接入。请手动进入对应步骤。")
            if hasattr(self, "refresh_workflow_action_gates"):
                self.refresh_workflow_action_gates()

    def _is_image_path(self, path_text: str) -> bool:
        return Path(str(path_text or "")).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def _safe_probe_external_media(self, path_text: str) -> tuple[Optional[VideoInfo], str]:
        path = Path(str(path_text or "").strip())
        if not path.is_file():
            return None, "未选择"
        if self._is_image_path(str(path)):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, "图片不可读"
            h, w = img.shape[:2]
            has_alpha = bool(img.ndim == 3 and img.shape[2] >= 4)
            return VideoInfo(path=str(path), width=w, height=h, fps=0.0, frame_count=1, has_alpha=has_alpha), f"图片 {w}x{h}"
        try:
            info = probe_video(str(path))
            return info, f"{info.width}x{info.height} / {info.fps:.3f}fps / {info.frame_count}帧"
        except Exception as exc:  # noqa: BLE001
            return None, f"不可读：{short_error_message(str(exc))}"

    def _main_video_alpha_state(self) -> tuple[bool, bool, str]:
        main_ok = bool(self.current_input and self.video_info)
        alpha_ok = False
        alpha_desc = "未检测"
        if main_ok:
            try:
                alpha_ok, alpha_desc = describe_real_alpha_source(self.current_input or "", int(self.preview_frame_spin.value()))
            except Exception as exc:  # noqa: BLE001
                alpha_ok = False
                alpha_desc = f"Alpha 检测失败：{short_error_message(str(exc))}"
        return main_ok, alpha_ok, alpha_desc

    def validate_main_video_alpha_chain(self, silent: bool = False) -> bool:
        """Validate step 1: main video only. Alpha is optional helper, not a blocker."""
        main_ok, alpha_ok, alpha_desc = self._main_video_alpha_state()
        self.input_cutout_mask_check.setChecked(True)
        self.background_mode_combo.setCurrentText("背景白")
        if hasattr(self, "top_main_chain_label"):
            self.top_main_chain_label.setText("主视频：已加载" if main_ok else "主视频：未加载")
        if hasattr(self, "top_external_depth_label"):
            self.top_external_depth_label.setText("Alpha：黑底合成" if alpha_ok else "Alpha：未检测/可继续")
        detail = "主视频：未导入"
        if main_ok:
            detail = f"主视频：{self.video_info.width}x{self.video_info.height} / {self.video_info.fps:.3f}fps / {self.video_info.frame_count}帧"
        detail += f"\nAlpha：{alpha_desc}"
        if not main_ok:
            detail += "\n注意：先导入主视频。"
        elif alpha_ok:
            detail += "\n输入就绪：Alpha 会自动合成黑色背景。"
        else:
            detail += "\n输入就绪：无 Alpha，将按普通视频处理。"
        if hasattr(self, "external_status_label"):
            self.external_status_label.setText(detail)
        if hasattr(self, "external_chain_label"):
            step1 = "✓主视频" if main_ok else "×主视频"
            step2 = "Alpha黑底" if alpha_ok else "无Alpha继续"
            self.external_chain_label.setText(f"链路：{step1} → {step2} → 结构缓存 → 稳定 Mesh/Shell → 导出")
        if hasattr(self, "preview_status_label"):
            if main_ok:
                self.preview_status_label.setText("主视频已接入。下一步生成结构缓存。")
            elif not silent:
                self.preview_status_label.setText("请先导入主视频。")
        self._refresh_reference_preview_tiles()
        self._update_external_media_status_label()
        if not main_ok and not silent:
            QMessageBox.warning(self, APP_NAME, "请先导入主视频。")
        return bool(main_ok)

    def validate_external_reference_chain(self, silent: bool = False) -> bool:
        """Compatibility alias. Depth/法线 reference is no longer required."""
        return self.validate_main_video_alpha_chain(silent=silent)

    def on_external_media_changed(self) -> None:
        self._update_external_media_status_label()
        self.preview_subject_mask = None
        self.preview_depth_version += 1
        self.preview_depth = None
        self.preview_normal_map = None
        self.preview_depth_render_bgr = None
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_depth_label.clearImage("等待 Mesh 预览")
        self.preview_big_btn.setEnabled(False)
        self._refresh_reference_preview_tiles()
        if hasattr(self, "preview_status_label"):
            self.preview_status_label.setText("输入设置已更新。请检查主视频和结构缓存状态。")
        self.validate_external_reference_chain(silent=True)
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def _update_external_media_status_label(self) -> None:
        if not hasattr(self, "external_status_label"):
            return
        main_loaded = bool(self.current_input and self.video_info)
        alpha_ok = False
        if main_loaded:
            try:
                alpha_ok, _desc = describe_real_alpha_source(self.current_input or "", int(self.preview_frame_spin.value()))
            except Exception:
                alpha_ok = False
        if hasattr(self, "top_main_chain_label"):
            self.top_main_chain_label.setText("主视频：已加载" if main_loaded else "主视频：未加载")
        if hasattr(self, "top_external_depth_label"):
            self.top_external_depth_label.setText("Alpha：黑底合成" if alpha_ok else "Alpha：未检测/可继续")
        parts = ["主视频" if main_loaded else "未导入主视频", "Alpha黑底" if alpha_ok else "无Alpha继续", "structure cache", "稳定Mesh", "Dense/Shell", "Mesh/点云导出"]
        self.external_chain_label.setText("操作链：" + " → ".join(parts))

    def set_preview_layout_mode(self, mode: str) -> None:
        # The top preview is now a fixed horizontal production layout:
        # main final-depth preview on the left, five reference tiles on the right.
        self.preview_layout_mode = "horizontal"
        layout = getattr(self, "preview_canvas_layout", None)
        depth_tile = getattr(self, "preview_depth_tile", None)
        original_tile = getattr(self, "preview_original_tile", None)
        preview_canvas = getattr(self, "preview_canvas", None)
        if layout is None or depth_tile is None or original_tile is None:
            return

        layout.setDirection(QBoxLayout.LeftToRight)
        layout.setSpacing(10)
        if preview_canvas is not None:
            preview_canvas.setMinimumHeight(270)
            preview_canvas.setMaximumHeight(310)

        depth_tile.setMinimumWidth(520)
        depth_tile.setMaximumWidth(760)
        original_tile.setMinimumWidth(520)
        original_tile.setMaximumWidth(680)

        self.preview_depth_label.setMinimumSize(420, 240)
        for _label in (
            self.preview_original_label,
            self.preview_external_depth_label,
            self.preview_da3_label,
            self.preview_subject_alpha_label,
            self.preview_normal_label,
        ):
            _label.setMinimumSize(120, 92)
            _label.setMaximumHeight(125)

        depth_tile.updateGeometry()
        original_tile.updateGeometry()
        self.preview_depth_label.updateGeometry()
        self.preview_original_label.updateGeometry()
        self.preview_external_depth_label.updateGeometry()
        self.preview_da3_label.updateGeometry()
        self.preview_subject_alpha_label.updateGeometry()
        self.preview_normal_label.updateGeometry()

    def _effective_preview_subject_mask_for_shape(self, shape_hw: tuple[int, int]) -> Optional[np.ndarray]:
        """Return the prepared Alpha/Mask for the structure-XYZ workflow."""
        mask = self.preview_subject_mask
        if mask is not None:
            arr = np.asarray(mask, dtype=np.float32)
            if arr.shape[:2] != tuple(shape_hw):
                th, tw = shape_hw
                arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR)
            if arr.size and float(np.nanmax(arr)) > 0.05:
                return np.clip(arr, 0.0, 1.0)
        try:
            cfg = self.make_config()
            frame_idx = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else 0
            alpha = read_external_subject_mask(cfg, frame_idx, shape_hw)
            if alpha is not None and float(np.nanmax(alpha)) > 0.05:
                return np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
        except Exception:
            return None
        return None

    def _input_adjust_key(self) -> tuple:
        return (
            int(self.input_brightness_spin.value()),
            int(self.input_contrast_spin.value()),
            round(float(self.input_gamma_spin.value()), 3),
            int(self.input_shadow_spin.value()),
            int(self.input_highlight_spin.value()),
            int(self.input_sharpen_spin.value()),
            int(self.input_denoise_spin.value()),
        )

    def _input_adjusted_preview_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            return apply_input_adjustments_bgr(
                frame_bgr,
                brightness=int(self.input_brightness_spin.value()),
                contrast=int(self.input_contrast_spin.value()),
                gamma=float(self.input_gamma_spin.value()),
                shadow=int(self.input_shadow_spin.value()),
                highlight=int(self.input_highlight_spin.value()),
                sharpen=int(self.input_sharpen_spin.value()),
                denoise=int(self.input_denoise_spin.value()),
            )
        except Exception as exc:  # noqa: BLE001
            self.preview_status_label.setText(f"原图调节失败: {exc}")
            self.log(f"原图调节失败: {exc}")
            return frame_bgr

    def _show_adjusted_original_preview(self) -> None:
        if self.preview_original_bgr is None:
            return
        adjusted = self._input_adjusted_preview_bgr(self.preview_original_bgr)
        self.preview_original_render_bgr = adjusted
        self.set_label_pixmap(self.preview_original_label, adjusted)
        if self.preview_depth is None:
            # The Curves/Waveform panel represents depth, not the input image.
            # Keep it empty until 模型深度 has produced a depth frame; otherwise users
            # read the source image histogram as a depth waveform.
            self.levels_panel.setHistogramFromGray(None)
            self.sync_levels_panel_from_controls()

    def _invalidate_depth_after_input_adjust(self) -> None:
        self.preview_depth = None
        self.preview_subject_mask = None
        self.preview_normal_map = None
        self.preview_depth_version += 1
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_depth_render_bgr = None
        self.preview_depth_label.clearImage("等待 Mesh 预览")
        self.preview_big_btn.setEnabled(False)
        self._refresh_reference_preview_tiles()

    def on_input_adjust_controls_changed(self) -> None:
        self._show_adjusted_original_preview()
        self._invalidate_depth_after_input_adjust()
        self.preview_status_label.setText("当前主线不使用深度灰阶参数；这里只检查 原视频 Alpha。")

    def on_adjust_mode_changed(self, mode: str) -> None:
        if hasattr(self, "adjust_stack"):
            # Direct-depth workflow: the visible adjustment page is the depth-gray page.
            self.adjust_stack.setCurrentIndex(1 if self.adjust_stack.count() > 1 else 0)
        self._show_adjusted_original_preview()
        self.preview_status_label.setText("当前主线不使用曲线/灰阶/五区调色；只检查 原视频 Alpha。")

    def _apply_style(self) -> None:
        # The current card-based UI sets object names while building panels.
        # Do not overwrite primaryButton / secondaryButton / navButton names here,
        # otherwise their page-specific QSS stops matching after _apply_style().
        for btn in (
            self.model_manager_btn, self.cache_manager_btn, self.log_dir_btn,
            self.external_mask_pick_btn, self.external_depth_pick_btn,
            self.preview_btn, self.preview_big_btn,
            self.preset_human_btn, self.preset_neutral_btn, self.preset_displacement_btn,
            self.preset_high_png_btn, self.preset_low_mem_btn, self.preset_import_btn,
            self.preset_export_btn,
        ):
            if not btn.objectName():
                btn.setObjectName("secondaryAction")
        if not self.path_edit.objectName():
            self.path_edit.setObjectName("pathEdit")
        self.preview_status_label.setObjectName("previewStatusLabel")
        self.info_label.setObjectName("infoLabel")

        self.setStyleSheet(APP_STYLESHEET)
        self._install_button_cursor_policy()

    def _install_button_cursor_policy(self) -> None:
        for btn in self.findChildren(QPushButton):
            btn.installEventFilter(self)
            self._sync_button_cursor(btn)

    def _sync_button_cursor(self, btn: QPushButton) -> None:
        if btn.isEnabled():
            btn.setCursor(QCursor(Qt.PointingHandCursor))
        else:
            btn.setCursor(QCursor(Qt.ArrowCursor))

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001, N802
        if isinstance(obj, QPushButton) and event.type() == QEvent.EnabledChange:
            QTimer.singleShot(0, lambda b=obj: self._sync_button_cursor(b))
        return super().eventFilter(obj, event)


    def _refresh_widget_style(self, widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def _set_depth_preview_busy(self, busy: bool) -> None:
        if hasattr(self.preview_depth_label, "setOverlayText"):
            self.preview_depth_label.setOverlayText("计算中..." if busy else "")
        if hasattr(self, "preview_depth_status_line"):
            self.preview_depth_status_line.setProperty("busy", "1" if busy else "0")
            self._refresh_widget_style(self.preview_depth_status_line)

    def _append_event_console_line(self, text: str) -> None:
        line = str(text).rstrip("\n")
        if not line:
            return
        # Worker log signals can arrive shortly after their event_log line.
        # Keep the console readable by dropping immediate duplicates.
        if line in self._event_console_recent[-12:]:
            return
        self._event_console_recent.append(line)
        if len(self._event_console_recent) > 64:
            self._event_console_recent = self._event_console_recent[-64:]
        self.log_box.appendPlainText(line)
        try:
            bar = self.log_box.verticalScrollBar()
            bar.setValue(bar.maximum())
        except Exception:
            pass

    def _on_worker_log_signal(self, text: str) -> None:
        # Most workers already call event_log() before emitting their legacy
        # log signal. The event listener is the source of truth, so do not log
        # again unless the listener was not installed for some reason.
        if not getattr(self, "_event_console_listener_active", False):
            self._append_event_console_line(str(text))

    def log(self, text: str) -> None:
        event_log(text, channel="UI")

    def clear_event_console(self) -> None:
        self.log_box.clear()
        self._event_console_recent.clear()
        event_log("事件控制台已清空", channel="UI")

    def on_stage_changed(self, text: str) -> None:
        if hasattr(self, "stage_status_label"):
            self.stage_status_label.setText(text)
        self._eta_started_at = None
        self._eta_started_done = 0
        if hasattr(self, "progress"):
            self.progress.setRange(0, 0)
            self.progress.setFormat(str(text).replace("阶段:", "").strip() or "处理中")

    def _has_active_model_task(self) -> bool:
        return any([self.thread is not None, self.preview_thread is not None])

    def _has_background_processing(self) -> bool:
        return any([
            self.thread is not None,
            self.preview_thread is not None,
            self.preload_thread is not None,
            self._base_rebuild_thread is not None,
        ])

    def _set_model_config_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.model_manager_btn,
            self.model_combo,
            self.device_combo,
            self.batch_spin,
            self.process_res_spin,
            self.long_side_spin,
            self.output_pick_btn,
            self.pick_btn,
            self.path_edit,
            self.preset_human_btn,
            self.preset_neutral_btn,
            self.preset_displacement_btn,
            self.preset_high_png_btn,
            self.preset_low_mem_btn,
            self.cache_enable_check,
            self.copy_audio_check,
            self.matting_model_path_edit,
            self.matting_mask_path_edit,
            self.input_cutout_mask_check,
            self.external_mask_path_edit,
            self.external_mask_pick_btn,
            self.external_mask_invert_check,
            self.external_depth_path_edit,
            self.external_depth_pick_btn,
            self.external_depth_weight_spin,
            self.external_depth_invert_check,
        ):
            widget.setEnabled(enabled)
        self._update_three_model_status()

    def _resource_risks(self, cfg: JobConfig, preview: bool = False) -> list[str]:
        risks: list[str] = []
        direct_depth_mode = False
        try:
            direct_depth_mode = is_direct_depth_video_workflow(cfg)
        except Exception:
            direct_depth_mode = False
        long_side = max(cfg.output_width, cfg.output_height)
        if long_side > MAX_SAFE_LONG_SIDE_HINT:
            risks.append(f"{'预览' if preview else '输出'}长边 {long_side} 超过建议上限 {MAX_SAFE_LONG_SIDE_HINT}")
        if not direct_depth_mode:
            est = estimate_vram_gb(cfg)
            total = cuda_total_memory_gb()
            if total > 0.0:
                risks.append(f"检测到显存约 {total:.1f}GB，当前参数估算峰值约 {est:.1f}GB")
                if est > total * 0.82:
                    risks.append("估算显存已接近/超过安全区，容易 CUDA OOM")
            else:
                if cfg.process_res > 1280:
                    risks.append(f"process_res={cfg.process_res} 较高")
                if cfg.batch_size > 1:
                    risks.append(f"批量帧数={cfg.batch_size}，显存占用会增加")
        if cfg.encoder_mode.startswith("FFmpeg") and shutil.which("ffmpeg") is None:
            risks.append("未检测到 ffmpeg，会回退 OpenCV mp4v；勾选原音频时无法合并声音")
        return risks

    def _confirm_resource_risk(self, cfg: JobConfig) -> bool:
        risks = self._resource_risks(cfg, preview=False)
        hard = [r for r in risks if "容易 CUDA OOM" in r or "超过建议上限" in r or "回退" in r]
        if not hard:
            return True
        reply = QMessageBox.question(
            self,
            APP_NAME,
            "当前参数有显存/性能风险：\n\n" + "\n".join(f"- {item}" for item in risks) + "\n\n仍然继续导出？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _confirm_preview_resource_risk(self, cfg: JobConfig) -> bool:
        risks = self._resource_risks(cfg, preview=True)
        hard = [r for r in risks if "容易 CUDA OOM" in r or "超过建议上限" in r]
        if not hard:
            return True
        reply = QMessageBox.question(
            self,
            APP_NAME,
            "当前预览参数有显存/性能风险：\n\n" + "\n".join(f"- {item}" for item in risks) + "\n\n仍然渲染当前帧？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def open_log_dir(self) -> None:
        PROJECT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        event_log(f"打开日志目录: {PROJECT_LOG_DIR}", channel="UI")
        try:
            os.startfile(str(PROJECT_LOG_DIR))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"无法打开日志目录: {exc}")

    def open_cache_manager(self) -> None:
        if self._has_background_processing():
            QMessageBox.warning(self, APP_NAME, "当前有预览、导出或融合重建任务，结束后再管理缓存。")
            return
            
        cache_dir = (self.current_project_dir / "cache") if getattr(self, "current_project_dir", None) else PROJECT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        total_size = directory_size_bytes(cache_dir)
        entries = list_cache_entries(cache_dir, limit=10)
        detail_lines = []
        for entry in entries:
            is_current = ""
            try:
                if self.current_input and self.video_info and frame_cache_root(self.make_config()).resolve() == entry.path.resolve():
                    is_current = "  ← 当前项目"
            except Exception:
                is_current = ""
            detail_lines.append(f"- {entry.path.name}: {format_bytes(entry.size_bytes)}{is_current}")
        detail = "\n".join(detail_lines) if detail_lines else "暂无缓存目录。"

        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Question)
        box.setText(
            f"帧缓存目录：{cache_dir}\n"
            f"总大小：{format_bytes(total_size)}\n\n"
            f"最近缓存：\n{detail}\n\n"
            "请选择清理方式。"
        )
        current_btn = box.addButton("清当前项目", QMessageBox.ActionRole)
        old_btn = box.addButton("清7天前", QMessageBox.ActionRole)
        all_btn = box.addButton("清全部", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked == cancel_btn or clicked is None:
            return
        if clicked == current_btn:
            if not self.current_input or not self.video_info:
                QMessageBox.information(self, APP_NAME, "当前没有已导入视频，无法定位当前项目缓存。")
                return
            try:
                root = frame_cache_root(self.make_config())
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, APP_NAME, f"无法定位当前项目缓存: {exc}")
                return
            removed = clear_cache_entry(root)
            self.log(f"已清理当前项目缓存: {root.name} / {format_bytes(removed)}")
            QMessageBox.information(self, APP_NAME, f"已清理当前项目缓存：{format_bytes(removed)}")
            return
        if clicked == old_btn:
            count, removed = clear_cache_older_than(cache_dir, days=7)
            self.log(f"已清理 7 天前缓存: {count} 项 / {format_bytes(removed)}")
            QMessageBox.information(self, APP_NAME, f"已清理 7 天前缓存：{count} 项，{format_bytes(removed)}")
            return
        if clicked == all_btn:
            confirm = QMessageBox.question(
                self,
                APP_NAME,
                "确认清空全部帧缓存？这不会删除原视频或导出结果。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            removed = clear_all_cache(cache_dir)
            self.log(f"已清空全部帧缓存: {format_bytes(removed)}")
            QMessageBox.information(self, APP_NAME, f"已清空全部帧缓存：{format_bytes(removed)}")

    def _preset_payload(self) -> dict:
        return {
            "version": 3,
            "input_path": self.current_input if hasattr(self, "current_input") and self.current_input else "",
            "output_long_side": int(self.long_side_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "process_res": int(self.process_res_spin.value()),
            "processing_range": {
                "start": int(self._processing_range_values()[0]) if hasattr(self, "_processing_range_values") else 0,
                "end": int(self._processing_range_values()[1]) if hasattr(self, "_processing_range_values") else -1,
            },
            "encoder_mode": encoder_internal_name(self.encoder_combo.currentText()),
            "normalize_mode": self.normalize_mode_combo.currentText(),
            "invert": self.invert_check.isChecked(),
            "cache_enabled": self.cache_enable_check.isChecked(),
            "copy_audio": self.copy_audio_check.isChecked(),
            "smooth": int(self.smooth_spin.value()),
            "human_refine": int(self.human_refine_spin.value()),
            "black_pct": float(self.black_pct_spin.value()),
            "white_pct": float(self.white_pct_spin.value()),
            "gamma": float(self.gamma_spin.value()),
            "detail_boost": int(self.detail_boost_spin.value()),
            "normal_strength": int(self.normal_strength_spin.value()),
            "normal_refine": int(self.normal_refine_spin.value()),
            "auto_mask_feather_px": int(self.auto_mask_feather_spin.value()),
            "auto_mask_expand_px": int(self.auto_mask_expand_spin.value()),
            "background_mode": self.background_mode_combo.currentText(),
            "background_gray": int(self.background_gray_spin.value()),
            "external_reference": {
                "source_mode": "cutout_video",
                "input_cutout_mask_enabled": True,
                "external_mask_enabled": False,
                "external_mask_path": "",
                "external_mask_invert": bool(self.external_mask_invert_check.isChecked()),
                "external_depth_path": self.external_depth_path_edit.text().strip(),
                "external_depth_weight": int(self.external_depth_weight_spin.value()),
                "external_depth_orientation_mode": self.external_depth_invert_check.currentText(),
            },
            "input_adjust": {
                "brightness": int(self.input_brightness_spin.value()),
                "contrast": int(self.input_contrast_spin.value()),
                "gamma": float(self.input_gamma_spin.value()),
                "shadow": int(self.input_shadow_spin.value()),
                "highlight": int(self.input_highlight_spin.value()),
                "sharpen": int(self.input_sharpen_spin.value()),
                "denoise": int(self.input_denoise_spin.value()),
            },
            "structure_model": self._structure_model_key() if hasattr(self, "_structure_model_key") else "4dhumans",
            "pointcloud": {
                "enabled": bool(self.pointcloud_enable_check.isChecked()),
                "density": self.pointcloud_density_combo.currentText(),
                "remove_outliers": bool(self.pointcloud_remove_outliers_check.isChecked()),
                "voxel_downsample": bool(self.pointcloud_voxel_check.isChecked()),
                "obj_sequence": False,
                "usd_sequence": bool(self.pointcloud_usd_check.isChecked()),
                "mesh_export": bool(self.mesh_export_check.isChecked()),
                "detail_mesh_export": bool(self.detail_mesh_export_check.isChecked()),
                "mesh_dense_segments": self.mesh_dense_segments_combo.currentText(),
                "garment_shell": bool(self.garment_shell_check.isChecked()),
                "hair_shell": bool(self.hair_shell_check.isChecked()),
                "segmentation_enabled": bool(self.segmentation_enable_check.isChecked()) if hasattr(self, "segmentation_enable_check") else True,
                "segmentation_provider": self.segmentation_provider_combo.currentText() if hasattr(self, "segmentation_provider_combo") else "Auto",
                "abc_sequence": False,
                "normal_relief_enabled": False,
                "temporal_stabilize": bool(self.pointcloud_temporal_check.isChecked()),
            },
            "anti_banding": int(self.anti_banding_spin.value()),
            "depth_smooth": int(self.depth_smooth_spin.value()),
            "edge_preserve": int(self.edge_preserve_spin.value()),
            "levels_in_black": int(self.levels_in_black_spin.value()),
            "levels_in_white": int(self.levels_in_white_spin.value()),
            "levels_out_black": int(self.levels_out_black_spin.value()),
            "levels_out_white": int(self.levels_out_white_spin.value()),
            "curve_points": self.levels_panel.getCurvePoints(),
            "tone": {
                "black": int(self.tone_black_spin.value()),
                "shadow": int(self.tone_shadow_spin.value()),
                "mid": int(self.tone_mid_spin.value()),
                "light": int(self.tone_light_spin.value()),
                "white": int(self.tone_white_spin.value()),
                "black_shift": int(self.tone_black_shift_spin.value()),
                "shadow_shift": int(self.tone_shadow_shift_spin.value()),
                "mid_shift": int(self.tone_mid_shift_spin.value()),
                "light_shift": int(self.tone_light_shift_spin.value()),
                "white_shift": int(self.tone_white_shift_spin.value()),
                "black_contrast": int(self.tone_black_contrast_spin.value()),
                "shadow_contrast": int(self.tone_shadow_contrast_spin.value()),
                "mid_contrast": int(self.tone_mid_contrast_spin.value()),
                "light_contrast": int(self.tone_light_contrast_spin.value()),
                "white_contrast": int(self.tone_white_contrast_spin.value()),
            },
        }

    def _apply_preset_payload(self, data: dict) -> None:
        def set_if(key: str, widget) -> None:  # noqa: ANN001
            if key in data:
                widget.setValue(data[key])
        set_if("output_long_side", self.long_side_spin)
        set_if("batch_size", self.batch_spin)
        set_if("process_res", self.process_res_spin)
        try:
            rng = data.get("processing_range", {}) if isinstance(data.get("processing_range", {}), dict) else {}
            if rng and hasattr(self, "_set_processing_values"):
                self._set_processing_values(int(rng.get("start", 0)), int(rng.get("end", -1)))
        except Exception:
            pass
        try:
            model_key = str(data.get("structure_model", "") or "").strip().lower()
            if model_key in {"4dhumans", "wham"} and hasattr(self, "structure_solver_combo"):
                self.structure_solver_combo.setCurrentText(self._structure_scheme_text(model_key))
        except Exception:
            pass
        if data.get("color_mode") in [self.color_combo.itemText(i) for i in range(self.color_combo.count())]:
            self.color_combo.setCurrentText(data["color_mode"])
        if data.get("encoder_mode"):
            self._set_encoder_combo_value(str(data.get("encoder_mode")))
        if data.get("normalize_mode") in NORMALIZE_MODES:
            self.normalize_mode_combo.setCurrentText(data["normalize_mode"])
        if "invert" in data:
            self.invert_check.setChecked(bool(data["invert"]))
        if "cache_enabled" in data:
            self.cache_enable_check.setChecked(bool(data["cache_enabled"]))
        if "copy_audio" in data:
            self.copy_audio_check.setChecked(bool(data["copy_audio"]))
        if data.get("background_mode") in [self.background_mode_combo.itemText(i) for i in range(self.background_mode_combo.count())]:
            self.background_mode_combo.setCurrentText(data["background_mode"])
        pointcloud = data.get("pointcloud", {}) if isinstance(data.get("pointcloud", {}), dict) else {}
        # Ignore legacy hidden point-cloud enable/mode values from old presets.
        # The current app flow is always structure-cache -> Mesh/Shell -> optional stable point cloud.
        self.pointcloud_enable_check.setChecked(True)
        if pointcloud.get("density") in [self.pointcloud_density_combo.itemText(i) for i in range(self.pointcloud_density_combo.count())]:
            self.pointcloud_density_combo.setCurrentText(pointcloud["density"])
        if "remove_outliers" in pointcloud:
            self.pointcloud_remove_outliers_check.setChecked(bool(pointcloud["remove_outliers"]))
        if "voxel_downsample" in pointcloud:
            self.pointcloud_voxel_check.setChecked(bool(pointcloud["voxel_downsample"]))
        if "temporal_stabilize" in pointcloud:
            self.pointcloud_temporal_check.setChecked(bool(pointcloud["temporal_stabilize"]))
        if "usd_sequence" in pointcloud:
            self.pointcloud_usd_check.setChecked(bool(pointcloud["usd_sequence"]))
        if "mesh_export" in pointcloud:
            self.mesh_export_check.setChecked(bool(pointcloud["mesh_export"]))
        if "detail_mesh_export" in pointcloud:
            self.detail_mesh_export_check.setChecked(bool(pointcloud["detail_mesh_export"]))
        if pointcloud.get("mesh_dense_segments") in [self.mesh_dense_segments_combo.itemText(i) for i in range(self.mesh_dense_segments_combo.count())]:
            self.mesh_dense_segments_combo.setCurrentText(pointcloud["mesh_dense_segments"])
        if "garment_shell" in pointcloud:
            self.garment_shell_check.setChecked(bool(pointcloud["garment_shell"]))
        if "hair_shell" in pointcloud:
            self.hair_shell_check.setChecked(bool(pointcloud["hair_shell"]))
        if "segmentation_enabled" in pointcloud and hasattr(self, "segmentation_enable_check"):
            self.segmentation_enable_check.setChecked(bool(pointcloud["segmentation_enabled"]))
        if pointcloud.get("segmentation_provider") in [self.segmentation_provider_combo.itemText(i) for i in range(self.segmentation_provider_combo.count())] if hasattr(self, "segmentation_provider_combo") else False:
            self.segmentation_provider_combo.setCurrentText(pointcloud["segmentation_provider"])

        for key, widget in [
            ("smooth", self.smooth_spin),
            ("human_refine", self.human_refine_spin),
            ("black_pct", self.black_pct_spin), ("white_pct", self.white_pct_spin), ("gamma", self.gamma_spin),
            ("detail_boost", self.detail_boost_spin), ("normal_strength", self.normal_strength_spin),
            ("normal_refine", self.normal_refine_spin),
            ("auto_mask_feather_px", self.auto_mask_feather_spin),
            ("auto_mask_expand_px", self.auto_mask_expand_spin),
            ("background_gray", self.background_gray_spin),
            ("anti_banding", self.anti_banding_spin),
            ("depth_smooth", self.depth_smooth_spin), ("edge_preserve", self.edge_preserve_spin),
            ("levels_in_black", self.levels_in_black_spin), ("levels_in_white", self.levels_in_white_spin),
            ("levels_out_black", self.levels_out_black_spin), ("levels_out_white", self.levels_out_white_spin),
        ]:
            set_if(key, widget)
        input_adjust = data.get("input_adjust", {}) if isinstance(data.get("input_adjust", {}), dict) else {}
        input_map = {
            "brightness": self.input_brightness_spin, "contrast": self.input_contrast_spin,
            "gamma": self.input_gamma_spin, "shadow": self.input_shadow_spin,
            "highlight": self.input_highlight_spin, "sharpen": self.input_sharpen_spin,
            "denoise": self.input_denoise_spin,
        }
        for key, widget in input_map.items():
            if key in input_adjust:
                widget.setValue(input_adjust[key])
        self.levels_panel.setCurvePoints(data.get("curve_points", [(0.0, 0.0), (1.0, 1.0)]), emit=False)
        tone = data.get("tone", {}) if isinstance(data.get("tone", {}), dict) else {}
        tone_map = {
            "black": self.tone_black_spin, "shadow": self.tone_shadow_spin, "mid": self.tone_mid_spin,
            "light": self.tone_light_spin, "white": self.tone_white_spin,
            "black_shift": self.tone_black_shift_spin, "shadow_shift": self.tone_shadow_shift_spin, "mid_shift": self.tone_mid_shift_spin,
            "light_shift": self.tone_light_shift_spin, "white_shift": self.tone_white_shift_spin,
            "black_contrast": self.tone_black_contrast_spin, "shadow_contrast": self.tone_shadow_contrast_spin,
            "mid_contrast": self.tone_mid_contrast_spin, "light_contrast": self.tone_light_contrast_spin,
            "white_contrast": self.tone_white_contrast_spin,
        }
        for key, widget in tone_map.items():
            if key in tone:
                widget.setValue(tone[key])
        external_reference = data.get("external_reference", {}) if isinstance(data.get("external_reference", {}), dict) else {}
        if "input_cutout_mask_enabled" in external_reference:
            self.input_cutout_mask_check.setChecked(bool(external_reference.get("input_cutout_mask_enabled")))
        self.external_mask_path_edit.setText("")
        if "external_mask_invert" in external_reference:
            self.external_mask_invert_check.setChecked(bool(external_reference.get("external_mask_invert")))
        if "external_depth_path" in external_reference:
            self.external_depth_path_edit.setText(str(external_reference.get("external_depth_path") or ""))
        if "external_depth_weight" in external_reference:
            self.external_depth_weight_spin.setValue(int(external_reference.get("external_depth_weight") or 0))
        if "external_depth_orientation_mode" in external_reference:
            mode = str(external_reference.get("external_depth_orientation_mode") or "自动方向")
            if mode in [self.external_depth_invert_check.itemText(i) for i in range(self.external_depth_invert_check.count())]:
                self.external_depth_invert_check.setCurrentText(mode)
        # Re-force the hidden compatibility controls after applying old presets.
        self.pointcloud_enable_check.setChecked(True)
        self.external_depth_path_edit.setText("")
        if hasattr(self, "source_cutout_radio"):
            source_mode = "cutout_video"
            self.source_cutout_radio.setChecked(True)
            self._apply_source_mode(source_mode)
        else:
            self.on_external_media_changed()
        try:
            self._on_density_mode_changed()
            self._update_conditional_visibility()
        except Exception:
            pass
        self._on_density_mode_changed()
        self.sync_levels_panel_from_controls()
        self.render_preview_from_cache()


    def apply_builtin_preset(self, preset_key: str) -> None:
        payload = BUILTIN_PRESETS.get(preset_key)
        if not isinstance(payload, dict):
            QMessageBox.warning(self, APP_NAME, f"未找到内置预设: {preset_key}")
            return
        self._apply_preset_payload(dict(payload))
        self.preview_status_label.setText(f"已应用内置预设：{payload.get('name', preset_key)}")
        self.log(f"已应用内置预设: {payload.get('name', preset_key)}")

    def export_preset_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出参数预设", str(PROJECT_DIR / "depth_preset.json"), "JSON (*.json)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".json"
        Path(path).write_text(json.dumps(self._preset_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"已导出预设: {path}")

    def import_preset_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入参数预设", str(PROJECT_DIR), "JSON (*.json);;All Files (*.*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise RuntimeError("预设文件格式错误。")
            self._apply_preset_payload(data)
            self.log(f"已导入预设: {path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"导入预设失败: {exc}")

    def save_project_state(self, current_step: int = -1) -> None:
        """Automatically saves the current project state into the project.vhm file."""
        if not hasattr(self, "current_project_dir") or not self.current_project_dir:
            return
        proj_file = self.current_project_dir / "project.vhm"
        try:
            data = {}
            if proj_file.exists():
                data = json.loads(proj_file.read_text(encoding="utf-8"))
            data["preset"] = self._preset_payload()
            if current_step >= 0:
                data["workflow_step"] = current_step
            proj_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            try:
                self.log(f"保存项目进度失败: {exc}")
            except Exception:
                pass

    def open_model_manager(self) -> None:
        if self._has_background_processing():
            QMessageBox.warning(self, APP_NAME, "当前有预览、导出或融合重建任务，结束后再管理模型。")
            return
        dlg = LocalModelManagerDialog(self)
        dlg.exec()

    def on_output_geometry_changed(self) -> None:
        """Refresh output size and invalidate preview caches when output geometry changes."""
        self.refresh_output_size()
        if not self.current_input or not self.video_info:
            return
        if self._has_background_processing():
            return
        self.preview_depth = None
        self.preview_subject_mask = None
        self.preview_normal_map = None
        self.preview_depth_version += 1
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_depth_render_bgr = None
        self.preview_depth_label.clearImage("等待 Mesh 预览")
        self.preview_big_btn.setEnabled(False)
        self._refresh_reference_preview_tiles()
        self.preview_status_label.setText("输出尺寸已变化，请重新预览 Mesh / 点云。")
        self.show_original_frame_immediately(int(self.preview_frame_spin.value()))

    def pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv);;All Files (*.*)",
        )
        if path:
            self.load_video(path)
    def load_project(self, project_dir: Path) -> None:
        self.current_project_dir = project_dir
        self.setWindowTitle(f"{APP_NAME} - {project_dir.name}")
        self.log(f"进入项目: {project_dir}")
        
        proj_file = project_dir / "project.vhm"
        if proj_file.exists():
            try:
                import json
                with open(proj_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                preset = data.get("preset", {})
                if preset.get("input_path") and os.path.isfile(preset["input_path"]):
                    # Automatically load the video if the project had one
                    self.load_video(preset["input_path"])
                
                self._apply_preset_payload(preset)
                if hasattr(self, "_restore_best_available_structure_scheme"):
                    self._restore_best_available_structure_scheme()
                if hasattr(self, "_update_structure_scheme_status_labels"):
                    self._update_structure_scheme_status_labels()
                if hasattr(self, "refresh_workflow_action_gates"):
                    self.refresh_workflow_action_gates()
                
                # Restore step. Step 0 is valid, so do not use truthiness.
                if "workflow_step" in data:
                    saved_step = int(data.get("workflow_step") or 0)
                    if hasattr(self, "set_workflow_step"):
                        self.set_workflow_step(saved_step)
                if hasattr(self, "_update_structure_scheme_status_labels"):
                    self._update_structure_scheme_status_labels()
                if hasattr(self, "refresh_workflow_action_gates"):
                    self.refresh_workflow_action_gates()
            except Exception as e:
                self.log(f"加载工程文件失败: {e}")

    def load_video(self, path: str) -> None:
        path = os.path.normpath(path)
        event_log(f"导入视频: {path}", channel="UI")
        if not os.path.isfile(path):
            QMessageBox.warning(self, APP_NAME, "不是有效文件。")
            return
        if Path(path).suffix.lower() not in VIDEO_EXTS:
            QMessageBox.warning(self, APP_NAME, "请选择视频文件。")
            return
        try:
            info = probe_video(path)
        except Exception as exc:  # noqa: BLE001
            event_exception("读取视频信息失败", exc, path=path)
            QMessageBox.critical(self, APP_NAME, str(exc))
            return
        self.current_input = path
        self.video_info = info
        self.path_edit.setText(path)
        self.info_label.setText(f"原始: {info.width}x{info.height} / {info.fps:.3f} FPS / {info.frame_count} 帧")
        
        # We assume self.current_project_dir is already set by ProjectManager Dialog
        if hasattr(self, "current_project_dir") and self.current_project_dir:
            self.setWindowTitle(f"{APP_NAME} - {self.current_project_dir.name}")
        else:
            self.setWindowTitle(f"{APP_NAME} - {Path(path).name}")
            
        self._manual_output_path = False
        self.long_side_spin.blockSignals(True)
        safe_long = min(max(info.width, info.height), SAFE_DEFAULT_LONG_SIDE)
        self.long_side_spin.setValue(max(128, even_int(safe_long)))
        self.long_side_spin.blockSignals(False)
        self.refresh_output_size()
        self.preview_original_bgr = None
        self.preview_depth = None
        self.preview_depth_version += 1
        self.preview_original_render_bgr = None
        self.preview_depth_render_bgr = None
        self.preview_original_label.clearImage("原视频未读取")
        self.preview_external_depth_label.clearImage("原视频 Alpha 未接入")
        self.preview_da3_label.clearImage("结构缓存未预览")
        self.preview_subject_alpha_label.clearImage("原视频Alpha未读取")
        self.preview_normal_label.clearImage("Shell 预览在当前帧生成")
        self._reference_preview_tile_keys.clear()
        self._reference_preview_tile_bgr.clear()
        self.preview_depth_label.clearImage("等待 Mesh 预览")
        self.preview_big_btn.setEnabled(False)
        self.set_preview_frame_range(info.frame_count)
        self.preview_status_label.setText("未生成预览")
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.show_original_frame_immediately(self.preview_frame_spin.value())
        self.input_cutout_mask_check.blockSignals(True)
        self.input_cutout_mask_check.setChecked(True)
        self.input_cutout_mask_check.blockSignals(False)
        self.external_mask_path_edit.clear()
        if hasattr(self, "source_cutout_radio"):
            self.source_cutout_radio.setChecked(True)
        self.preview_status_label.setText("已导入主视频。下一步生成结构缓存。")
        self.log("主视频已导入：RGB 用于结构识别；Alpha 若存在则作为主体辅助。Depth / 外部 法线 不参与主流程。")
        self.validate_main_video_alpha_chain(silent=True)

        if hasattr(self, "_update_structure_scheme_status_labels"):
            self._update_structure_scheme_status_labels()
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()
        self.log(f"已导入: {path}")
        self.log(f"默认安全输出长边: {int(self.long_side_spin.value())}，避免直接按原片分辨率导出。")

    def _preview_frame_control_pairs(self) -> tuple[tuple[object | None, object | None], ...]:
        return (
            (getattr(self, "preview_frame_slider", None), getattr(self, "preview_frame_spin", None)),
            (getattr(self, "input_preview_frame_slider", None), getattr(self, "input_preview_frame_spin", None)),
            (getattr(self, "structure_preview_frame_slider", None), getattr(self, "structure_preview_frame_spin", None)),
        )

    def _all_preview_frame_controls(self) -> list[object]:
        controls: list[object] = []
        for slider, spin in self._preview_frame_control_pairs():
            if slider is not None:
                controls.append(slider)
            if spin is not None:
                controls.append(spin)
        return controls

    def set_preview_frame_range(self, frame_count: int) -> None:
        max_frame = max(0, int(frame_count) - 1)
        default_frame = max_frame // 2 if max_frame > 0 else 0
        page_step = max(1, int((self.video_info.fps if self.video_info else 24) or 24))
        for ctrl in self._all_preview_frame_controls():
            try:
                ctrl.blockSignals(True)
                ctrl.setRange(0, max_frame)
                ctrl.setValue(default_frame)
                if hasattr(ctrl, "setPageStep"):
                    ctrl.setPageStep(page_step)
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        if hasattr(self, "export_frame_slider"):
            self.export_frame_slider.blockSignals(True)
            self.export_frame_slider.setRange(0, max_frame)
            self.export_frame_slider.setValue(default_frame)
            self.export_frame_slider.setPageStep(page_step)
            self.export_frame_slider.blockSignals(False)
        self._set_processing_frame_range(0, max_frame, reset_values=True)
        self.update_preview_frame_label(default_frame)

    def _set_processing_frame_range(self, start_min: int, end_max: int, *, reset_values: bool = False) -> None:
        max_frame = max(0, int(end_max))
        widgets = [
            getattr(self, "processing_start_slider", None), getattr(self, "processing_end_slider", None),
            getattr(self, "processing_start_spin", None), getattr(self, "processing_end_spin", None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.blockSignals(True)
                widget.setRange(0, max_frame)
        if reset_values:
            if getattr(self, "processing_start_slider", None) is not None:
                self.processing_start_slider.setValue(0)
            if getattr(self, "processing_start_spin", None) is not None:
                self.processing_start_spin.setValue(0)
            if getattr(self, "processing_end_slider", None) is not None:
                self.processing_end_slider.setValue(max_frame)
            if getattr(self, "processing_end_spin", None) is not None:
                self.processing_end_spin.setValue(max_frame)
        for widget in widgets:
            if widget is not None:
                widget.blockSignals(False)
        self._refresh_processing_range_label()

    def _processing_range_values(self) -> tuple[int, int]:
        if not self.video_info:
            return 0, -1
        max_frame = max(0, int(self.video_info.frame_count) - 1)
        start = int(getattr(self, "processing_start_spin", self.preview_frame_spin).value()) if hasattr(self, "processing_start_spin") else 0
        end = int(getattr(self, "processing_end_spin", self.preview_frame_spin).value()) if hasattr(self, "processing_end_spin") else max_frame
        start = max(0, min(start, max_frame))
        end = max(start, min(end, max_frame))
        return start, end

    def _refresh_processing_range_label(self) -> None:
        start, end = self._processing_range_values()
        total = max(0, int(self.video_info.frame_count) if self.video_info else 0)
        frames = max(0, end - start + 1) if end >= start else 0
        pct = 100 if total <= 0 else int(round(frames * 100 / max(1, total)))
        text = f"帧 {start} — {end}  /  共 {total} 帧  ({frames} 帧)"
        if hasattr(self, "processing_range_label"):
            self.processing_range_label.setText(text)
        if hasattr(self, "processing_range_progress"):
            self.processing_range_progress.setValue(max(0, min(100, pct)))
            self.processing_range_progress.setFormat(f"{max(0, min(100, pct))}%")

    def _set_processing_values(self, start: int | None = None, end: int | None = None) -> None:
        if not self.video_info:
            return
        max_frame = max(0, int(self.video_info.frame_count) - 1)
        cur_start, cur_end = self._processing_range_values()
        new_start = cur_start if start is None else max(0, min(int(start), max_frame))
        new_end = cur_end if end is None else max(0, min(int(end), max_frame))
        if new_start > new_end:
            if start is not None and end is None:
                new_end = new_start
            elif end is not None and start is None:
                new_start = new_end
            else:
                new_start, new_end = min(new_start, new_end), max(new_start, new_end)
        pairs = (
            (getattr(self, "processing_start_slider", None), new_start),
            (getattr(self, "processing_start_spin", None), new_start),
            (getattr(self, "processing_end_slider", None), new_end),
            (getattr(self, "processing_end_spin", None), new_end),
        )
        for widget, value in pairs:
            if widget is not None:
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)
        self._refresh_processing_range_label()
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def _on_processing_start_slider_changed(self, value: int) -> None:
        self._set_processing_values(start=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def _on_processing_end_slider_changed(self, value: int) -> None:
        self._set_processing_values(end=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def _on_processing_start_spin_changed(self, value: int) -> None:
        self._set_processing_values(start=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def _on_processing_end_spin_changed(self, value: int) -> None:
        self._set_processing_values(end=int(value))
        self._apply_preview_frame_value(int(value), refresh_mesh=False)

    def update_preview_frame_label(self, frame_index: int) -> None:
        if not self.video_info:
            text = "第 0 帧 / 0:00"
            self.preview_frame_label.setText(text)
            if hasattr(self, "export_frame_label"):
                self.export_frame_label.setText(text)
            return
        fps = max(1e-3, float(self.video_info.fps or 24.0))
        seconds = frame_index / fps
        total = max(0, self.video_info.frame_count - 1)
        text = f"第 {frame_index}/{total} 帧 / {format_seconds(seconds)}"
        self.preview_frame_label.setText(text)
        if hasattr(self, "export_frame_label"):
            self.export_frame_label.setText(text)
        if hasattr(self, "export_frame_slider") and self.export_frame_slider.value() != int(frame_index):
            self.export_frame_slider.blockSignals(True)
            self.export_frame_slider.setValue(int(frame_index))
            self.export_frame_slider.blockSignals(False)

    def _apply_preview_frame_value(self, value: int, *, refresh_mesh: bool = True) -> None:
        max_frame = max(0, int(self.video_info.frame_count) - 1) if self.video_info else 0
        value = max(0, min(int(value), max_frame))
        for ctrl in self._all_preview_frame_controls():
            try:
                ctrl.blockSignals(True)
                ctrl.setValue(value)
            finally:
                try:
                    ctrl.blockSignals(False)
                except Exception:
                    pass
        self.update_preview_frame_label(value)
        self.show_original_frame_immediately(value)
        if hasattr(self, "preview_status_label"):
            self.preview_status_label.setText("已切换帧，正在读取原视频预览...")
        QTimer.singleShot(80, self._refresh_reference_preview_tiles)
        if refresh_mesh:
            self._schedule_active_mesh_preview_refresh()

    def on_preview_frame_slider_changed(self, value: int) -> None:
        self._apply_preview_frame_value(int(value))

    def on_preview_frame_spin_changed(self) -> None:
        self._apply_preview_frame_value(int(self.preview_frame_spin.value()))

    def toggle_preview_playback(self) -> None:
        if not self.current_input or not self.video_info:
            return
        self._preview_playing = not bool(getattr(self, "_preview_playing", False))
        if self._preview_playing:
            fps = max(1.0, min(60.0, float(self.video_info.fps or 24.0)))
            self.preview_play_timer.setInterval(max(16, int(round(1000.0 / fps))))
            self.preview_play_timer.start()
            if hasattr(self, "input_preview_play_btn"):
                self.input_preview_play_btn.setText("暂停")
        else:
            self.preview_play_timer.stop()
            if hasattr(self, "input_preview_play_btn"):
                self.input_preview_play_btn.setText("播放")

    def _advance_preview_playback(self) -> None:
        if not self.current_input or not self.video_info:
            self._preview_playing = False
            self.preview_play_timer.stop()
            return
        start, end = self._processing_range_values()
        if end < start:
            start, end = 0, max(0, int(self.video_info.frame_count) - 1)
        cur = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else start
        nxt = cur + 1
        if nxt > end:
            nxt = start
        self._apply_preview_frame_value(nxt, refresh_mesh=False)

    def _active_workflow_index(self) -> int:
        stack = getattr(self, "workflow_stack", None)
        try:
            return int(stack.currentIndex()) if stack is not None else 0
        except Exception:
            return 0

    def _active_mesh_preview_mode(self) -> str | None:
        idx = self._active_workflow_index()
        last = str(getattr(self, "_last_mesh_preview_mode", "") or "")
        if idx == 1:
            return last if last in {"stable", "body"} else "stable"
        if idx == 2:
            return last if last in {"garment", "hair", "detail", "combined"} else "combined"
        return None

    def _schedule_active_mesh_preview_refresh(self) -> None:
        mode = self._active_mesh_preview_mode()
        if not mode:
            return
        try:
            if not self._has_structure_cache():
                return
        except Exception:
            return
        self._mesh_preview_frame_timer.start()

    def _refresh_active_mesh_preview_from_timer(self) -> None:
        mode = self._active_mesh_preview_mode()
        if not mode:
            return
        if self.preview_thread is not None:
            self._mesh_preview_frame_timer.start()
            return
        self._start_mesh_preview(mode)

    def on_mesh_preview_dragged(self, dx: float, dy: float) -> None:
        if self._active_workflow_index() not in {1, 2}:
            return
        self.mesh_preview_yaw = float(getattr(self, "mesh_preview_yaw", 0.0)) + float(dx) * 0.45
        self.mesh_preview_pitch = max(-75.0, min(75.0, float(getattr(self, "mesh_preview_pitch", 0.0)) + float(dy) * 0.35))
        canvas = getattr(self, "preview_depth_label", None)
        if hasattr(canvas, "setOverlayText"):
            canvas.setOverlayText("")
        if self.preview_thread is None:
            self._mesh_preview_rotation_timer.start()

    def show_original_frame_immediately(self, frame_index: int) -> None:
        """Schedule raw-frame display safely on a QThread.

        Do not update Qt widgets from a Python background thread. The previous
        version did that during timeline dragging, which could crash after the
        first depth preview. This path uses Qt signals and collapses rapid
        slider moves into the latest request.
        """
        if not self.current_input or not self.video_info:
            return
        frame_index = max(0, min(int(frame_index), max(0, self.video_info.frame_count - 1)))
        self._original_frame_requested = frame_index
        self._seek_debounce.start()

        # Clear stale depth immediately so the user does not read old depth as the selected frame.
        self.preview_depth = None
        self.preview_subject_mask = None
        self.preview_normal_map = None
        self.preview_depth_version += 1
        self.preview_depth_render_bgr = None
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_depth_label.clearImage("等待 Mesh 预览")
        self.preview_big_btn.setEnabled(False)
        self._refresh_reference_preview_tiles()

    def _start_original_frame_read(self) -> None:
        if not self.current_input or not self.video_info or self._original_frame_requested is None:
            return
        if self._original_frame_running:
            self._original_frame_pending = True
            return

        frame_index = int(self._original_frame_requested)
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        self._original_frame_running = True
        self.original_frame_thread = QThread(self)
        self.original_frame_worker = OriginalFrameWorker(self.current_input, frame_index, out_w, out_h)
        self.original_frame_worker.moveToThread(self.original_frame_thread)
        self.original_frame_thread.started.connect(self.original_frame_worker.run)
        self.original_frame_worker.finished.connect(self.on_original_frame_finished)
        self.original_frame_worker.failed.connect(self.on_original_frame_failed)
        self.original_frame_worker.finished.connect(self.original_frame_thread.quit)
        self.original_frame_worker.failed.connect(self.original_frame_thread.quit)
        self.original_frame_worker.finished.connect(self.original_frame_worker.deleteLater)
        self.original_frame_worker.failed.connect(self.original_frame_worker.deleteLater)
        self.original_frame_thread.finished.connect(self.cleanup_original_frame_thread)
        self.original_frame_thread.finished.connect(lambda th=self.original_frame_thread: QTimer.singleShot(0, th.deleteLater))
        self.original_frame_thread.start()

    def on_original_frame_finished(self, frame_index: int, frame_bgr: object) -> None:
        current = int(self.preview_frame_spin.value())
        if frame_index != current:
            self._original_frame_pending = True
            return
        self.preview_original_bgr = frame_bgr  # type: ignore[assignment]
        self._show_adjusted_original_preview()
        self._refresh_reference_preview_tiles()
        if hasattr(self, "preview_status_label"):
            self.preview_status_label.setText("当前帧原视频和 Alpha 已刷新。")

    def on_original_frame_failed(self, frame_index: int, msg: str) -> None:
        if frame_index == int(self.preview_frame_spin.value()):
            self.preview_status_label.setText(f"读取原始帧失败: {msg}")
            self.log(f"读取原始帧失败: {msg}")

    def cleanup_original_frame_thread(self) -> None:
        self.original_frame_worker = None
        self.original_frame_thread = None
        self._original_frame_running = False
        if self._original_frame_pending:
            self._original_frame_pending = False
            self._seek_debounce.start()

    def _switch_to_fusion_preview_for_curve_edit(self) -> bool:
        """Main preview is fixed to the final fused depth result."""
        return False

    def on_tone_controls_changed(self) -> None:
        switched = self._switch_to_fusion_preview_for_curve_edit()
        self.sync_levels_panel_from_controls()
        if not switched:
            self.render_preview_from_cache()

    def _set_slider_silent(self, slider: SliderValue, value: int | float) -> None:
        slider.blockSignals(True)
        try:
            slider.setValue(value)
        finally:
            slider.blockSignals(False)

    def on_levels_controls_changed(self) -> None:
        if self._syncing_levels:
            return
        sender = self.sender()
        ib = int(self.levels_in_black_spin.value())
        iw = int(self.levels_in_white_spin.value())
        if ib >= iw:
            if sender is self.levels_in_white_spin:
                self._set_slider_silent(self.levels_in_black_spin, max(0, iw - 1))
            elif ib >= 255:
                self._set_slider_silent(self.levels_in_black_spin, 254)
            else:
                self._set_slider_silent(self.levels_in_white_spin, min(255, ib + 1))
        ob = int(self.levels_out_black_spin.value())
        ow = int(self.levels_out_white_spin.value())
        if ob >= ow:
            if sender is self.levels_out_white_spin:
                self._set_slider_silent(self.levels_out_black_spin, max(0, ow - 1))
            elif ob >= 255:
                self._set_slider_silent(self.levels_out_black_spin, 254)
            else:
                self._set_slider_silent(self.levels_out_white_spin, min(255, ob + 1))
            self.preview_status_label.setText("已限制出黑小于出白，避免输出静默反转。")
        switched = self._switch_to_fusion_preview_for_curve_edit()
        self.sync_levels_panel_from_controls()
        if not switched:
            self.render_preview_from_cache()

    def on_levels_panel_changed(
        self,
        in_black: int,
        in_white: int,
        gamma: float,
        out_black: int,
        out_white: int,
    ) -> None:
        in_black = max(0, min(254, int(in_black)))
        in_white = max(1, min(255, int(in_white)))
        if in_black >= in_white:
            if in_black >= 255:
                in_black = 254
                in_white = 255
            else:
                in_white = min(255, in_black + 1)
        out_black = max(0, min(255, int(out_black)))
        out_white = max(0, min(255, int(out_white)))
        if out_black >= out_white:
            if out_black >= 255:
                out_black = 254
                out_white = 255
            else:
                out_white = min(255, out_black + 1)
            self.preview_status_label.setText("已限制出黑小于出白，避免输出静默反转。")
        self._syncing_levels = True
        try:
            self.levels_in_black_spin.setValue(in_black)
            self.levels_in_white_spin.setValue(in_white)
            self.gamma_spin.setValue(gamma)
            self.levels_out_black_spin.setValue(out_black)
            self.levels_out_white_spin.setValue(out_white)
        finally:
            self._syncing_levels = False
        switched = self._switch_to_fusion_preview_for_curve_edit()
        self.sync_levels_panel_from_controls()
        if not switched:
            self.render_preview_from_cache()

    def on_curve_panel_changed(self, _points=None) -> None:  # noqa: ANN001
        if self._syncing_levels:
            return
        switched = self._switch_to_fusion_preview_for_curve_edit()
        if not switched:
            self.preview_status_label.setText("自由曲线已更新。")
            self.render_preview_from_cache()

    def reset_free_curve(self) -> None:
        switched = self._switch_to_fusion_preview_for_curve_edit()
        self.levels_panel.resetCurve(emit=True)
        if not switched:
            self.preview_status_label.setText("自由曲线已重置。")

    def sync_levels_panel_from_controls(self) -> None:
        self.levels_panel.setValues(
            int(self.levels_in_black_spin.value()),
            int(self.levels_in_white_spin.value()),
            float(self.gamma_spin.value()),
            int(self.levels_out_black_spin.value()),
            int(self.levels_out_white_spin.value()),
            emit=False,
        )
        self.levels_panel.setNormalize(
            float(self.black_pct_spin.value()),
            float(self.white_pct_spin.value()),
            self.invert_check.isChecked(),
        )
        self.levels_panel.setToneValues(
            int(self.tone_black_spin.value()),
            int(self.tone_shadow_spin.value()),
            int(self.tone_mid_spin.value()),
            int(self.tone_light_spin.value()),
            int(self.tone_white_spin.value()),
            int(self.tone_black_shift_spin.value()),
            int(self.tone_shadow_shift_spin.value()),
            int(self.tone_mid_shift_spin.value()),
            int(self.tone_light_shift_spin.value()),
            int(self.tone_white_shift_spin.value()),
            int(self.tone_black_contrast_spin.value()),
            int(self.tone_shadow_contrast_spin.value()),
            int(self.tone_mid_contrast_spin.value()),
            int(self.tone_light_contrast_spin.value()),
            int(self.tone_white_contrast_spin.value()),
        )

    def _current_model_key(self) -> tuple[str, str]:
        return (MODEL_IDS[self.model_combo.currentText()], self.device_combo.currentText())

    def _reference_depth_detail_ready(self) -> bool:
        path = str(self.external_depth_path_edit.text().strip()) if hasattr(self, "external_depth_path_edit") else ""
        return bool(path and Path(path).is_file())

    def _direct_depth_input_ready(self) -> bool:
        # Legacy-only: in the structure XYZ workflow, the external depth video is
        # not a Direct Depth replacement. It is only sampled during export as a
        # small high-frequency detail displacement layer.
        return bool(self._reference_depth_detail_ready() and self._pointcloud_mode() == "structure_xyz")


    def _auto_render_direct_depth_current_frame(self) -> None:
        if not self._direct_depth_input_ready() or not self.current_input or not self.video_info:
            return
        if self.thread is not None or self.preview_thread is not None or self._base_rebuild_thread is not None:
            self._direct_depth_auto_preview_timer.start()
            return
        self.start_preview()

    def _request_direct_depth_live_preview(self, status: str | None = None) -> None:
        if not self._direct_depth_input_ready() or not self.current_input or not self.video_info:
            return
        if status and hasattr(self, "preview_status_label"):
            self.preview_status_label.setText(status)
        if self.preview_depth is None:
            self._direct_depth_auto_preview_timer.start()
        else:
            self.render_preview_from_cache()

    _MODEL_WEIGHT_SUFFIXES = {".pkl", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pt2"}
    _MODEL_SCAN_SKIP_PARTS = {
        "test", "tests", "testing", "example", "examples", "sample", "samples",
        "demo", "demos", "docs", "doc", "assets", "h36m", "coco", "lsp", "mpi_inf_3dhp",
    }
    _MODEL_SCAN_SKIP_FILENAMES = {
        "cameras.pkl", "test_h36m.npz", "test_h36m_body3d.npz",
    }

    def _looks_like_real_model_weight(self, p: Path, key: str) -> bool:
        """Filter out repo fixtures and keep actual model/checkpoint files.

        External repos often contain small .pkl/.npz files under tests/data. Those are
        not model weights. v18 still counted HaMeR's ViTPose h36m test files as
        HaMeR weights; this filter only accepts files that look like real weights.
        """
        suffix = p.suffix.lower()
        if suffix not in self._MODEL_WEIGHT_SUFFIXES:
            return False
        name = p.name.lower()
        if name in self._MODEL_SCAN_SKIP_FILENAMES:
            return False
        parts = {part.lower() for part in p.parts}
        if parts & self._MODEL_SCAN_SKIP_PARTS:
            return False
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        # Most real neural checkpoints are much larger. Keep small body-model pkl/npz
        # possible for SMPL/MANO, but reject tiny placeholder/test files elsewhere.
        if key in {"smpl", "mano"}:
            if size < 64 * 1024:
                return False
        else:
            if size < 1 * 1024 * 1024:
                return False

        path_text = str(p).replace("\\", "/").lower()
        stem_text = p.stem.lower()
        if key == "smpl":
            return ("smpl" in path_text or "basicmodel" in stem_text) and suffix in {".pkl", ".npz"}
        if key == "mano":
            return "mano" in path_text and suffix in {".pkl", ".npz"}
        if key == "4dhumans":
            # Do not count SMPL body model files copied into 4D-Humans/data as
            # 4D-Humans checkpoints. 4D-Humans/HMR2 is usable only when an
            # actual neural checkpoint exists.
            if suffix not in {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".pt2"}:
                return False
            if "basicmodel" in stem_text or "smpl" in stem_text:
                return False
            return any(token in path_text for token in ("4dhumans", "4d-humans", "hmr2", "hmr_2", "hmr"))
        if key == "wham":
            # Do not count WHAM/data/basicModel_neutral...pkl as a WHAM model.
            # The expected public checkpoint is usually wham_vit_w_3dpw.pth.tar
            # or another large file with wham in the filename/path.
            is_pth_tar = p.name.lower().endswith(".pth.tar")
            if not is_pth_tar and suffix not in {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".pt2"}:
                return False
            if "basicmodel" in stem_text or "smpl" in stem_text:
                return False
            return "wham" in path_text or "wham" in p.name.lower()
        if key == "hamer":
            return any(token in path_text for token in ("hamer", "hamer_ckpt", "hamer_ckpts"))
        return True

    def _candidate_weight_files(self, roots: list[Path], key: str, *, limit: int = 12) -> list[Path]:
        """Return real model weight candidates, ignoring placeholders and test data."""
        found: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                if not root.exists():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file():
                        continue
                    if not self._looks_like_real_model_weight(p, key):
                        continue
                    resolved = p.resolve()
                    dedupe_key = str(resolved).lower()
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    found.append(p)
                    if len(found) >= limit:
                        return found
            except Exception:
                continue
        return found

    def _model_scan_roots(self) -> dict[str, list[Path]]:
        models = PROJECT_DIR / "models"
        ckpt = models / "checkpoints"
        repos = PROJECT_DIR / "data" / "models" / "external_repos"
        return {
            "smpl": [ckpt / "smpl", ckpt / "SMPL", models / "smpl", models / "SMPL"],
            "mano": [ckpt / "mano", ckpt / "MANO", models / "mano", models / "MANO"],
            "4dhumans": [ckpt / "4dhumans", ckpt / "4D-Humans", models / "4dhumans", repos / "4D-Humans"],
            "wham": [ckpt / "wham", ckpt / "WHAM", models / "wham", repos / "WHAM"],
            "hamer": [ckpt / "hamer", ckpt / "HaMeR", models / "hamer", repos / "hamer"],
        }

    def _scan_3d_model_config(self) -> dict[str, object]:
        roots = self._model_scan_roots()
        found = {name: self._candidate_weight_files(paths, name) for name, paths in roots.items()}

        smpl_ok = bool(found["smpl"])
        mano_ok = bool(found["mano"])
        body_solver_ok = smpl_ok and bool(found["4dhumans"] or found["wham"])
        hand_solver_ok = bool(found["hamer"])
        structure_cache_ok = self._has_structure_cache()
        structure_ok = bool(structure_cache_ok)
        hand_ok = mano_ok and hand_solver_ok

        report_path = PROJECT_DIR / "data" / "resources" / "configs" / "model_3d_deploy_report.json"
        report_hint = ""
        try:
            if report_path.exists():
                data = json.loads(report_path.read_text(encoding="utf-8"))
                generated = data.get("generated_at") or "未知时间"
                report_hint = f"部署报告：{generated}"
        except Exception:
            report_hint = "部署报告：存在但读取失败"

        return {
            "found": found,
            "smpl_ok": smpl_ok,
            "mano_ok": mano_ok,
            "body_solver_ok": body_solver_ok,
            "hand_solver_ok": hand_solver_ok,
            "structure_ok": structure_ok,
            "structure_cache_ok": structure_cache_ok,
            "structure_runner_ok": body_solver_ok,
            "hand_ok": hand_ok,
            "full_ok": structure_ok and hand_ok,
            "report_hint": report_hint,
        }

    def _format_3d_scan_lines(self, scan: dict[str, object]) -> list[str]:
        found = scan["found"]
        assert isinstance(found, dict)
        body_ready = bool(scan.get("body_solver_ok"))
        cache_ready = bool(scan.get("structure_cache_ok"))
        has_4dh = bool(found.get("4dhumans"))
        has_wham = bool(found.get("wham"))
        has_smpl = bool(found.get("smpl"))

        lines: list[str] = []
        if cache_ready:
            lines.append("✓ 结构缓存：已生成，可以导出稳定 Mesh / 点云。")
        elif body_ready:
            solver = "4DHumans" if has_4dh else "WHAM" if has_wham else "人体结构模型"
            lines.append(f"! 结构缓存：未生成。点击下方按钮后，会用 {solver} 自动生成。")
        else:
            missing = []
            if not has_smpl:
                missing.append("SMPL")
            if not (has_4dh or has_wham):
                missing.append("4DHumans/WHAM")
            lines.append("× 结构模型：缺少 " + "、".join(missing or ["必要资源"]) + "。")
        lines.append("主流程：主视频 → 结构缓存 → 稳定 Mesh → Dense/Shell → Mesh / 可选点云。")
        lines.append("手部增强：第二阶段再接入，当前不参与导出。")
        return lines

    def _deployment_missing_python_modules(self) -> list[str]:
        """Return safe pip-installable deps for the current mesh + parsing flow.

        Torch/CUDA is intentionally not auto-installed here because the correct
        wheel depends on CUDA/PyTorch index selection. The FASHN model itself is
        not a pip package, but the deployment button can install the downloader
        dependency ``huggingface_hub`` when the local model is still missing.
        """
        modules: list[str] = []
        for module_name in ("cv2", "numpy", "smplx", "yacs", "transformers", "PIL"):
            try:
                if importlib.util.find_spec(module_name) is None:
                    modules.append(module_name)
            except Exception:
                modules.append(module_name)
        try:
            seg = check_segmentation_environment(PROJECT_DIR, "auto")
            if not bool(seg.get("model_found")):
                if importlib.util.find_spec("huggingface_hub") is None:
                    modules.append("huggingface_hub")
        except Exception:
            try:
                if importlib.util.find_spec("huggingface_hub") is None:
                    modules.append("huggingface_hub")
            except Exception:
                modules.append("huggingface_hub")
        # Deduplicate while preserving order.
        out: list[str] = []
        for m in modules:
            if m not in out:
                out.append(m)
        return out

    def install_deployment_python_dependencies(self) -> None:
        """Install Python deps and optionally download the FASHN parser model.

        This button is the deployment action for the final image-driven mesh
        workflow. It must not stop at Python packages: if parsing deps are ready
        but the local FASHN model is absent, it should offer to run the bundled
        downloader so the GUI can actually use image-driven Garment/Hair masks.
        """
        missing = self._deployment_missing_python_modules()
        if missing:
            if hasattr(self, "deployment_env_label"):
                self.deployment_env_label.setText("正在安装 Python 依赖：" + "、".join(missing))
            try:
                QApplication.processEvents()
            except Exception:
                pass
            ok = self._install_missing_python_modules(missing, ask=True)
            if not ok:
                if hasattr(self, "deployment_env_label"):
                    self.deployment_env_label.setText("Python 依赖安装未完成。请查看日志或手动执行 pip 命令。")
                return
        try:
            seg = check_segmentation_environment(PROJECT_DIR, "auto")
        except Exception as exc:  # noqa: BLE001
            seg = {"ok": False, "model_found": False, "message": short_error_message(str(exc))}
        if not bool(seg.get("model_found")):
            self._download_fashn_human_parser_model(ask=True)
            return
        if hasattr(self, "deployment_env_label"):
            self.deployment_env_label.setText(
                "部署依赖已齐：Python 依赖可用，FASHN Human Parser 模型已部署。\n"
                "可以生成逐帧分割缓存，衣服/头发将优先使用图像分割约束。"
            )
        try:
            self.log("部署依赖检查：Python 依赖与 FASHN 模型均已就绪。")
        except Exception:
            pass
        self.refresh_deployment_environment_status()

    def _download_fashn_human_parser_model(self, *, ask: bool = True) -> bool:
        """Run the bundled FASHN downloader with explicit user feedback."""
        script = PROJECT_DIR / "app" / "tools" / "download_fashn_human_parser.py"
        if not script.exists():
            QMessageBox.warning(self, APP_NAME, "缺少下载脚本：" + str(script))
            return False
        cmd = [sys.executable, str(script)]
        readable = " ".join(cmd)
        if ask:
            reply = QMessageBox.question(
                self,
                APP_NAME,
                "FASHN Human Parser 模型未部署。\n\n"
                "是否现在下载到 models/segmentation/fashn_human_parser？\n"
                "下载需要联网，文件较大。\n\n命令：" + readable,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                try:
                    self.log("用户取消下载 FASHN Human Parser 模型。")
                except Exception:
                    pass
                if hasattr(self, "deployment_env_label"):
                    self.deployment_env_label.setText("FASHN 模型未下载。当前会降级为固定几何权重，不是图像驱动衣服/头发。")
                return False
        if hasattr(self, "deployment_env_label"):
            self.deployment_env_label.setText("正在下载 FASHN Human Parser 模型，请等待。\n" + readable)
        if hasattr(self, "segmentation_status_label"):
            self.segmentation_status_label.setText("分割状态：正在下载 FASHN Human Parser 模型...")
        try:
            QApplication.processEvents()
        except Exception:
            pass
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.log("下载 FASHN Human Parser: " + readable)
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            QApplication.restoreOverrideCursor()
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            self.log("FASHN 模型下载失败：\n" + output)
            if hasattr(self, "deployment_env_label"):
                self.deployment_env_label.setText("FASHN 模型下载失败。请查看日志，或手动运行 app/tools/download_fashn_human_parser.py。")
            QMessageBox.warning(self, APP_NAME, "FASHN 模型下载失败。\n\n命令：" + readable + "\n\n详细日志已写入下方日志。")
            return False
        importlib.invalidate_caches()
        self.log("FASHN 模型下载完成：\n" + output)
        if hasattr(self, "deployment_env_label"):
            self.deployment_env_label.setText("FASHN Human Parser 模型下载完成。正在重新检查环境...")
        self.refresh_deployment_environment_status()
        return True

    def _deployment_model_resource_note(self) -> list[str]:
        lines: list[str] = []
        try:
            scan = self._scan_3d_model_config()
            found = scan.get("found", {}) if isinstance(scan, dict) else {}
            if not isinstance(found, dict):
                return lines
            has_smpl = bool(found.get("smpl"))
            has_4dh = bool(found.get("4dhumans"))
            has_wham = bool(found.get("wham"))
            if has_4dh and has_smpl:
                lines.append("主体结构：4DHumans + SMPL 可用，可生成结构缓存。")
            elif not has_smpl:
                lines.append("主体结构：缺 SMPL 权重，需放入 models/checkpoints/smpl 或 models/SMPL。")
            elif not has_4dh and not has_wham:
                lines.append("主体结构：缺 4DHumans/WHAM 资源，需部署第三方仓库和权重。")
            if not has_wham:
                lines.append("WHAM：未部署，仅影响高级轨迹模式，不影响默认 4DHumans。")
            try:
                seg = check_segmentation_environment(PROJECT_DIR, "auto")
                if seg.get("ok"):
                    lines.append("画面分割：已部署，衣服/头发区域会优先使用逐帧 parsing cache。")
                else:
                    lines.append("画面分割：未就绪，衣服/头发区域会降级为固定几何权重。")
                try:
                    fg_env = check_foreground_environment(PROJECT_DIR)
                    lines.append("前景约束：" + str(fg_env.get("message", "Alpha 未检查")))
                except Exception:
                    lines.append("前景约束：Alpha 可用性运行时检测。")
                try:
                    seg_sum = segmentation_cache_summary(structure_cache_root(self.make_config())) if self.current_input else {"message": "未生成逐帧分割缓存"}
                    lines.append("分割缓存：" + str(seg_sum.get("message", "未生成逐帧分割缓存")))
                except Exception:
                    pass
            except Exception:
                lines.append("画面分割：检查失败，导出时会降级为固定几何权重。")
            lines.append("Shell：只负责生成最终外层网格；区域来源优先逐帧分割 cache，缺失时才用几何先验。")
        except Exception:
            pass
        return lines


    def open_segmentation_models_folder(self) -> None:
        path = PROJECT_MODELS_DIR / "segmentation"
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception:
            QMessageBox.information(self, APP_NAME, str(path))

    def test_current_segmentation_frame(self) -> None:
        if not self.current_input or not self.video_info:
            QMessageBox.warning(self, APP_NAME, "请先导入主视频，再测试当前帧分割。")
            return
        frame_index = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else 0
        frame_bgr = read_video_frame_bgr(self.current_input, frame_index)
        if frame_bgr is None:
            QMessageBox.warning(self, APP_NAME, "当前帧读取失败，无法测试分割。")
            return
        provider = self.segmentation_provider_combo.currentText() if hasattr(self, "segmentation_provider_combo") else "Auto"
        if provider == "Off":
            QMessageBox.information(self, APP_NAME, "分割模型已关闭。")
            return
        try:
            self.setCursor(Qt.WaitCursor)
            QApplication.processEvents()
            result = run_human_parsing(frame_bgr, project_root=PROJECT_DIR, provider=provider, log=self.log)
        finally:
            try:
                self.unsetCursor()
            except Exception:
                pass
        if not result.ok:
            msg = "分割不可用：" + result.reason + "\n\n请把 FASHN Human Parser 放到 models/segmentation/fashn_human_parser，或关闭分割使用几何兜底。"
            if hasattr(self, "segmentation_status_label"):
                self.segmentation_status_label.setText("分割状态：不可用\n" + result.reason)
            QMessageBox.warning(self, APP_NAME, msg)
            return
        try:
            alpha_fg = read_alpha_foreground(str(self.current_input or ""), int(frame_index), frame_bgr.shape[:2])
            if alpha_fg is not None:
                result.foreground = constrain_by_foreground(result.foreground, alpha_fg, softness=0.96)
                result.garment = constrain_by_foreground(result.garment, result.foreground, softness=0.96)
                result.hair = constrain_by_foreground(result.hair, result.foreground, softness=0.96)
            quality = classify_mask_quality(
                foreground=result.foreground,
                garment=result.garment,
                hair=result.hair,
                parser_confidence=float(result.confidence),
            )
            cache_root = structure_cache_root(self.make_config())
            meta = save_segmentation_frame(cache_root, frame_index, result, quality_to_meta(quality))
            paths = segmentation_frame_paths(cache_root, frame_index)
            preview = cv2.imread(str(paths["preview"]), cv2.IMREAD_COLOR)
            if preview is not None:
                self.set_label_pixmap(self.preview_depth_label, preview)
                if hasattr(self, "export_preview_label"):
                    self.set_label_pixmap(self.export_preview_label, preview)
            status = f"分割测试成功：frame={frame_index}, provider={meta.get('provider')}, status={meta.get('quality_status')}, confidence={meta.get('confidence', 0.0):.3f}"
            if hasattr(self, "segmentation_status_label"):
                self.segmentation_status_label.setText("分割状态：可用\n" + status)
            self.preview_status_label.setText(status)
            self.log(status)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, "分割结果保存失败：" + short_error_message(str(exc)))

    def _format_deployment_environment_lines(self) -> list[str]:
        lines: list[str] = []
        try:
            py_ver = sys.version.split()[0]
            lines.append(f"Python：{py_ver}  /  {sys.executable}")
        except Exception:
            pass
        try:
            ffmpeg_path = shutil.which("ffmpeg")
            lines.append("FFmpeg：" + (ffmpeg_path if ffmpeg_path else "未找到"))
        except Exception:
            lines.append("FFmpeg：检查失败")
        try:
            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is None:
                lines.append("Torch/CUDA：未安装 torch")
            else:
                import torch  # type: ignore
                cuda_text = "CUDA 可用" if bool(torch.cuda.is_available()) else "CUDA 不可用"
                gpu_text = ""
                try:
                    if torch.cuda.is_available():
                        gpu_text = " / " + str(torch.cuda.get_device_name(0))
                except Exception:
                    pass
                lines.append(f"Torch/CUDA：{getattr(torch, '__version__', 'unknown')} / {cuda_text}{gpu_text}")
        except Exception as exc:  # noqa: BLE001
            lines.append("Torch/CUDA：检查失败 - " + short_error_message(str(exc)))
        try:
            base_rows = []
            for module_name in ("cv2", "numpy", "smplx", "yacs", "transformers", "PIL"):
                base_rows.append(f"{module_name}:{'ok' if importlib.util.find_spec(module_name) else 'missing'}")
            lines.append("Python依赖：" + "  ".join(base_rows))
        except Exception:
            pass
        try:
            scan = self._scan_3d_model_config()
            found = scan.get("found", {}) if isinstance(scan, dict) else {}
            if isinstance(found, dict):
                model_state = []
                for key in ("smpl", "4dhumans", "wham", "mano", "hamer"):
                    model_state.append(f"{key}:{len(found.get(key, []) or [])}")
                lines.append("模型资源：" + "  ".join(model_state))
            if bool(scan.get("structure_cache_ok")):
                lines.append("结构缓存：可用")
            elif bool(scan.get("body_solver_ok")):
                lines.append("结构缓存：未生成，环境可尝试生成")
            else:
                lines.append("结构缓存：缺模型或依赖")
            lines.extend(self._deployment_model_resource_note())
            try:
                seg = check_segmentation_environment(PROJECT_DIR, "auto")
                missing = seg.get("missing_modules", [])
                model_paths = seg.get("model_paths", [])
                if seg.get("ok"):
                    lines.append("画面分割：可用 / " + ", ".join(str(p) for p in model_paths[:1]))
                elif model_paths and missing:
                    lines.append("画面分割：模型已找到，但缺依赖 " + ", ".join(str(m) for m in missing))
                elif model_paths:
                    lines.append("画面分割：模型目录存在，但检查未完全通过")
                else:
                    lines.append("画面分割：未部署，衣服/头发会降级为固定几何权重")
                if hasattr(self, "segmentation_status_label"):
                    self.segmentation_status_label.setText("分割状态：" + str(seg.get("message", "未检查")) + "\n目录：" + str(seg.get("segmentation_root", "")))
            except Exception as exc:
                lines.append("画面分割：检查失败 - " + short_error_message(str(exc)))
        except Exception as exc:  # noqa: BLE001
            lines.append("模型资源：检查失败 - " + short_error_message(str(exc)))
        missing_py = self._deployment_missing_python_modules()
        if missing_py:
            cmd = sys.executable + " -m pip install " + " ".join(self._pip_package_for_module(m) for m in missing_py)
            lines.append("可安装依赖：" + cmd)
        else:
            try:
                seg = check_segmentation_environment(PROJECT_DIR, "auto")
                if not bool(seg.get("model_found")):
                    lines.append("模型动作：点击“安装依赖/下载模型”可下载 FASHN Human Parser。")
                else:
                    lines.append("Python依赖动作：无需安装；分割模型已部署。")
            except Exception:
                lines.append("Python依赖动作：无需安装；分割模型状态需重新检查。")
        return lines


    def start_segmentation_cache_generation(self) -> None:
        if self.segmentation_cache_thread is not None:
            return
        if self.thread is not None:
            QMessageBox.information(self, APP_NAME, "导出任务正在进行，完成后再生成分割缓存。")
            return
        if self.preview_thread is not None:
            QMessageBox.information(self, APP_NAME, "预览正在进行，完成后再生成分割缓存。")
            return
        if not self.current_input:
            QMessageBox.information(self, APP_NAME, "请先导入主视频。")
            return
        if not self._has_structure_cache():
            if hasattr(self, "segmentation_cache_status_label"):
                self.segmentation_cache_status_label.setText("分割缓存：请先生成当前方案和当前范围的结构缓存。")
            QMessageBox.information(self, APP_NAME, "请先在第 2 步生成当前方案和当前范围的人体结构缓存。")
            return
        cfg = self.make_config()
        if not bool(getattr(cfg, "segmentation_enabled", True)):
            QMessageBox.information(self, APP_NAME, "画面分割已关闭，不能生成分割缓存。")
            return
        provider = getattr(cfg, "segmentation_provider", "Auto")
        if str(provider).lower() == "off":
            QMessageBox.information(self, APP_NAME, "分割模型为 Off，请先切换到 Auto 或 FASHN Human Parser。")
            return
        env = check_segmentation_environment(PROJECT_DIR, str(provider or "Auto"))
        if not bool(env.get("ok")):
            msg = "分割模型未就绪，不能生成逐帧缓存。\n\n" + str(env.get("message", ""))
            if env.get("missing_modules"):
                msg += "\n\n缺少 Python 依赖：" + "、".join(env.get("missing_modules")) + "\n请手动安装这些依赖（例如使用 pip install）。注意：如果是 torch 缺失，请根据您的 CUDA 版本安装对应的 PyTorch。"
            elif not env.get("model_found"):
                msg += "\n\n请先回到环境检查页面点击“安装依赖 / 下载模型”，或手动运行 app/tools/download_fashn_human_parser.py。"
            QMessageBox.warning(self, APP_NAME, msg)
            if hasattr(self, "segmentation_cache_status_label"):
                self.segmentation_cache_status_label.setText("分割缓存：模型或依赖未就绪，未生成。")
            return
        self.segmentation_cache_status_label.setText("分割缓存：正在逐帧生成，请等待。")
        self.structure_progress.setValue(0)
        thread = QThread(self)
        worker = SegmentationCacheWorker(cfg)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(lambda text: self.segmentation_cache_status_label.setText("分割缓存：" + str(text)))
        worker.progress_value.connect(self.on_segmentation_cache_progress_value)
        worker.finished.connect(self._on_segmentation_cache_finished)
        worker.failed.connect(self._on_segmentation_cache_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._cleanup_segmentation_cache_thread)
        self.segmentation_cache_thread = thread
        self.segmentation_cache_worker = worker
        if hasattr(self, "segmentation_cache_btn"):
            self.segmentation_cache_btn.setEnabled(False)
        thread.start()

    def on_segmentation_cache_progress_value(self, stage: str, done: int, total: int) -> None:
        total = max(1, int(total or 1))
        done = max(0, min(int(done), total))
        try:
            self.structure_progress.setRange(0, total)
            self.structure_progress.setValue(done)
        except Exception:
            pass
        if hasattr(self, "segmentation_cache_status_label"):
            self.segmentation_cache_status_label.setText(f"分割缓存：{done}/{total} 帧")

    def _on_segmentation_cache_finished(self, summary: dict) -> None:
        msg = str(summary.get("message", "逐帧分割缓存完成")) if isinstance(summary, dict) else "逐帧分割缓存完成"
        self._workflow_seg_summary_cache = None
        if hasattr(self, "segmentation_cache_status_label"):
            self.segmentation_cache_status_label.setText("分割缓存：" + msg)
        if hasattr(self, "segmentation_status_label"):
            self.segmentation_status_label.setText("分割状态：已生成逐帧缓存\n" + msg)
        try:
            if hasattr(self, "save_project_state"):
                stack = getattr(self, "workflow_stack", None)
                self.save_project_state(stack.currentIndex() if hasattr(stack, "currentIndex") else -1)
        except Exception:
            pass
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def _on_segmentation_cache_failed(self, msg: str) -> None:
        if hasattr(self, "segmentation_cache_status_label"):
            self.segmentation_cache_status_label.setText("分割缓存：生成失败。")
        QMessageBox.warning(self, APP_NAME, "逐帧分割缓存生成失败：\n\n" + short_error_message(str(msg), 1600))

    def _cleanup_segmentation_cache_thread(self) -> None:
        self.segmentation_cache_thread = None
        self.segmentation_cache_worker = None
        if hasattr(self, "segmentation_cache_btn"):
            self.segmentation_cache_btn.setEnabled(True)
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def refresh_deployment_environment_status(self) -> None:
        """Refresh the deployment card with immediate visible feedback.

        This slot is connected from the card panel. It must never fail silently:
        if dependency scanning raises, the card itself should show the failure
        instead of making the button look unresponsive.
        """
        label = getattr(self, "deployment_env_label", None)
        btn = getattr(self, "deployment_env_check_btn", None)

        def _set_text(text: str) -> None:
            try:
                if label is not None:
                    label.setText(str(text))
            except Exception:
                pass

        def _do_check() -> None:
            try:
                lines = self._format_deployment_environment_lines()
                text = "\n".join(lines[:12]) if lines else "环境：未检查"
                _set_text(text)
                try:
                    self.log("部署环境检查：" + " | ".join(lines))
                except Exception:
                    pass
            except Exception as exc:  # noqa: BLE001
                msg = "环境检查失败：" + short_error_message(str(exc))
                _set_text(msg)
                try:
                    self.log(msg)
                except Exception:
                    pass
            finally:
                try:
                    if btn is not None:
                        btn.setEnabled(True)
                        btn.setText("检查环境")
                except Exception:
                    pass
                try:
                    self.refresh_workflow_action_gates()
                except Exception:
                    pass

        _set_text("环境：正在检查...")
        try:
            if btn is not None:
                btn.setEnabled(False)
                btn.setText("检查中...")
        except Exception:
            pass
        try:
            QApplication.processEvents()
        except Exception:
            pass
        QTimer.singleShot(0, _do_check)

    def open_models_folder(self) -> None:
        target = PROJECT_DIR / "models"
        try:
            target.mkdir(parents=True, exist_ok=True)
            if hasattr(os, "startfile"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, "无法打开 models 目录：" + short_error_message(str(exc)))

    def _set_badge_state(self, label: QLabel, text: str, color: str, border: str) -> None:
        label.setText(text)
        label.setStyleSheet(
            f"background: #0f172a; color: {color}; border: 1px solid {border}; "
            "border-radius: 9px; font-size: 10px; font-weight: bold; padding: 2px 7px;"
        )

    def _structure_cache_root_for_model(self, model: str):
        cfg = self._config_for_structure_scheme(model) if hasattr(self, "_config_for_structure_scheme") else self.make_config()
        return structure_cache_root(cfg)

    def _layer_cache_state_for_model(self, model: str | None = None) -> dict:
        """Return persisted garment/hair layer-cache state for one structure scheme."""
        try:
            if not self.current_input:
                return {"ready": False, "message": "未导入视频"}
            model_key = str(model or self._structure_model_key() or "4dhumans").lower()
            root = self._structure_cache_root_for_model(model_key)
            summary = segmentation_cache_summary(root)
            cached = int(summary.get("cached_frames", 0) or 0) if isinstance(summary, dict) else 0
            if bool(summary.get("ok", False)) and cached > 0:
                return {
                    "ready": True,
                    "model": model_key,
                    "source": "segmentation_summary",
                    "root": str(root),
                    "message": str(summary.get("message", f"已生成逐帧分割缓存：{cached} 帧")),
                    "cached_frames": cached,
                }
            region_path = Path(root) / "region_weights.npz"
            if region_path.exists():
                try:
                    data = np.load(region_path, allow_pickle=False)
                    g = np.asarray(data["garment"], dtype=np.float32).reshape(-1) if "garment" in data.files else np.zeros((0,), dtype=np.float32)
                    h = np.asarray(data["hair"], dtype=np.float32).reshape(-1) if "hair" in data.files else np.zeros((0,), dtype=np.float32)
                    if (g.size and float(np.nanmax(g)) > 1e-6) or (h.size and float(np.nanmax(h)) > 1e-6):
                        return {
                            "ready": True,
                            "model": model_key,
                            "source": "region_weights",
                            "root": str(root),
                            "message": "已生成衣服/头发区域权重，可预览壳层。",
                            "cached_frames": 0,
                        }
                except Exception:
                    pass
            seg_dir = Path(root) / "segmentation"
            try:
                mask_count = sum(1 for _ in seg_dir.glob("frame_*_parsing_masks.npz")) if seg_dir.exists() else 0
            except Exception:
                mask_count = 0
            if mask_count > 0:
                return {
                    "ready": True,
                    "model": model_key,
                    "source": "segmentation_frames",
                    "root": str(root),
                    "message": f"已找到分割帧缓存：{mask_count} 帧。",
                    "cached_frames": mask_count,
                }
            return {"ready": False, "model": model_key, "root": str(root), "message": "未生成逐帧分割缓存"}
        except Exception as exc:
            return {"ready": False, "message": "分割/壳层缓存状态不可读：" + short_error_message(str(exc))}

    def _current_layer_cache_state(self) -> dict:
        current = self._layer_cache_state_for_model(self._structure_model_key())
        if bool(current.get("ready", False)):
            return current
        for model in ("4dhumans", "wham"):
            if model == str(current.get("model", "")).lower():
                continue
            other = self._layer_cache_state_for_model(model)
            if bool(other.get("ready", False)):
                other["needs_switch"] = True
                return other
        return current

    def _restore_best_available_structure_scheme(self) -> None:
        try:
            current = self._structure_model_key()
            if self._has_structure_cache_for_model(current) or bool(self._layer_cache_state_for_model(current).get("ready", False)):
                return
            for model in ("4dhumans", "wham"):
                if self._has_structure_cache_for_model(model) or bool(self._layer_cache_state_for_model(model).get("ready", False)):
                    if hasattr(self, "structure_solver_combo"):
                        self.structure_solver_combo.setCurrentText(self._structure_scheme_text(model))
                    return
        except Exception:
            pass

    def _has_structure_cache(self) -> bool:
        try:
            if not self.current_input:
                return False
            cfg = self.make_config()
            
            def check_dir(d: Path) -> bool:
                return d.exists() and any(d.glob("frame_*_smpl_vertices.npy")) and any(d.glob("frame_*_smpl_faces.npy"))
                
            root = structure_cache_root(cfg) / "structure"
            if check_dir(root):
                return True
                
            # Do not auto-import old stem-only caches here. 4DHumans / WHAM and
            # different in/out ranges now have separate roots; silently copying a
            # legacy cache would make the UI show the wrong model as available.
            return False
        except Exception:
            return False

    def _pip_package_for_module(self, module: str) -> str:
        name = str(module or "").strip()
        mapping = {
            "cv2": "opencv-python",
            "skimage": "scikit-image",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "pytorch_lightning": "pytorch-lightning",
            "lightning_fabric": "lightning-fabric",
            "detectron2": "detectron2",
            "huggingface_hub": "huggingface_hub",
            "mmcv": "mmcv==1.3.9",
            "mmpose": "mmpose==0.29.0",
        }
        return mapping.get(name, name)

    def _install_missing_python_modules(self, modules: list[str], *, ask: bool = True) -> bool:
        modules = [str(m).strip() for m in modules if str(m).strip()]
        if not modules:
            return True
        packages = []
        for mod in modules:
            pkg = self._pip_package_for_module(mod)
            if pkg not in packages:
                packages.append(pkg)
        cmd = [sys.executable, "-m", "pip", "install", *packages]
        readable = " ".join(cmd)
        if ask:
            reply = QMessageBox.question(
                self,
                APP_NAME,
                "当前流程缺少依赖：" + "、".join(modules)
                + "\n\n是否现在自动安装？\n" + readable,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                self.log("用户取消安装缺失依赖：" + "、".join(modules))
                if hasattr(self, "structure_cache_status_label"):
                    self.structure_cache_status_label.setText("结构缓存：依赖未安装，未启动。")
                return False
        if hasattr(self, "structure_cache_status_label"):
            self.structure_cache_status_label.setText("部署：正在安装依赖，请稍等。")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.log("安装缺失依赖: " + readable)
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        finally:
            QApplication.restoreOverrideCursor()
        output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            self.log("依赖安装失败：\n" + output)
            if hasattr(self, "structure_cache_status_label"):
                self.structure_cache_status_label.setText("部署：依赖安装失败。")
            QMessageBox.warning(
                self,
                APP_NAME,
                "依赖安装失败。\n\n命令：" + readable + "\n\n详细日志已写入下方日志。",
            )
            return False
        importlib.invalidate_caches()
        self.log("依赖安装完成：" + "、".join(modules))
        if hasattr(self, "structure_cache_status_label"):
            self.structure_cache_status_label.setText("部署：依赖安装完成。")
        return True

    def _missing_modules_from_text(self, text: str) -> list[str]:
        """Extract real missing Python module names from mixed UI logs/tracebacks.

        The traceback also contains Python source lines such as:
            raise RuntimeError("...缺少 Python 依赖：" + "、".join(...))
        Those are not dependency names.  Keep this parser strict so auto-pip never
        tries to install tokens like '+', 'join', or pieces of source code.
        """
        payload = str(text or "")
        candidates: list[str] = []

        # Standard Python import error.
        candidates += re.findall(r"No module named ['\"]([^'\"]+)['\"]", payload)

        # JSON event from generate_structure_cache.py: {"missing_modules": ["smplx"]}
        for block in re.findall(r'"missing_modules"\s*:\s*\[([^\]]*)\]', payload):
            candidates += re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_.-]*)['\"]", block)

        # Human-readable RuntimeError lines.  Ignore traceback source-code lines.
        for line in payload.splitlines():
            line = line.strip()
            if "缺少 Python 依赖：" not in line:
                continue
            if line.startswith("raise ") or "join(" in line or "+" in line:
                continue
            tail = line.split("缺少 Python 依赖：", 1)[1]
            candidates.append(tail)

        out: list[str] = []
        for item in candidates:
            for part in re.split(r"[、,;\s]+", str(item)):
                name = part.strip().strip("'\"[](){}")
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:[.-][A-Za-z0-9_]+)*", name):
                    continue
                if name in {"RuntimeError", "File", "line", "module", "missing"}:
                    continue
                if name not in out:
                    out.append(name)
        return out

    def _structure_scheme_text(self, model: str) -> str:
        return "WHAM 轨迹锚定" if str(model).lower() == "wham" else "4DHumans 结构补全"

    def _config_for_structure_scheme(self, model: str) -> JobConfig:
        cfg = self.make_config()
        cfg.structure_model = "wham" if str(model).lower() == "wham" else "4dhumans"
        return cfg

    def _has_structure_cache_for_model(self, model: str) -> bool:
        try:
            cfg = self._config_for_structure_scheme(model)
            root = structure_cache_root(cfg) / "structure"
            return root.exists() and any(root.glob("frame_*_smpl_vertices.npy")) and any(root.glob("frame_*_smpl_faces.npy"))
        except Exception:
            return False

    def _update_structure_scheme_status_labels(self) -> None:
        for model, attr, view_attr in (
            ("4dhumans", "structure_4d_status_label", "preview_4dhumans_btn"),
            ("wham", "structure_wham_status_label", "preview_wham_btn"),
        ):
            label = getattr(self, attr, None)
            view_btn = getattr(self, view_attr, None)
            if label is None and view_btn is None:
                continue
            ok = self._has_structure_cache_for_model(model) if self.current_input else False
            name = "4D" if model == "4dhumans" else "WHAM"
            if label is not None:
                label.setText(f"{name}：已生成，可查看" if ok else f"{name}：未生成")
                label.setStyleSheet(("color: #86efac; font-size: 11px;" if ok else "color: #94a3b8; font-size: 11px;"))
            if view_btn is not None:
                view_btn.setEnabled(bool(ok))

    def select_structure_scheme(self, model: str, *, preview: bool = False) -> None:
        model_key = "wham" if str(model).lower() == "wham" else "4dhumans"
        if hasattr(self, "structure_solver_combo"):
            self.structure_solver_combo.setCurrentText(self._structure_scheme_text(model_key))
        try:
            if hasattr(self, "save_project_state"):
                stack = getattr(self, "workflow_stack", None)
                self.save_project_state(stack.currentIndex() if hasattr(stack, "currentIndex") else -1)
        except Exception:
            pass
        self._update_structure_scheme_status_labels()
        if preview:
            if not self._has_structure_cache_for_model(model_key):
                QMessageBox.information(self, APP_NAME, f"{self._structure_scheme_text(model_key)} 还没有生成缓存。")
                return
            if hasattr(self, "structure_mesh_preview_label"):
                self.preview_depth_label = self.structure_mesh_preview_label
            self.preview_current_structure_frame()

    def start_4dhumans_structure_generation(self) -> None:
        self.select_structure_scheme("4dhumans", preview=False)
        self.start_structure_cache_generation(forced_model="4dhumans")

    def start_wham_structure_generation(self) -> None:
        self.select_structure_scheme("wham", preview=False)
        self.start_structure_cache_generation(forced_model="wham")

    def start_structure_cache_generation(self, *args, auto_retry: bool = False, forced_model: str | None = None) -> None:
        if self.structure_cache_thread is not None:
            QMessageBox.information(self, APP_NAME, "结构缓存正在生成，先等当前任务完成。")
            return
        if not auto_retry:
            self._structure_dep_autoretry_count = 0
        if not self.current_input:
            QMessageBox.warning(self, APP_NAME, "先导入原视频。")
            return
        try:
            cfg = self.make_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        text = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else "4DHumans"
        model = "wham" if str(forced_model or "").lower() == "wham" or "WHAM" in text.upper() else "4dhumans"
        cfg.structure_model = model
        if hasattr(self, "structure_solver_combo"):
            self.structure_solver_combo.setCurrentText(self._structure_scheme_text(model))
        cache_root = structure_cache_root(cfg)
        missing_modules = []
        if model == "4dhumans":
            # Minimal direct imports hit by 4D-Humans demo.py on startup.
            # More modules can still be auto-installed from worker error output.
            for mod in ("yacs", "smplx"):
                if importlib.util.find_spec(mod) is None:
                    missing_modules.append(mod)
        if missing_modules:
            if hasattr(self, "structure_cache_status_label"):
                self.structure_cache_status_label.setText("结构缓存：缺少依赖，等待安装。")
            installed = self._install_missing_python_modules(missing_modules)
            if not installed:
                return
            # Re-enter once after successful installation so the normal dependency check
            # and worker startup path stays exactly the same.
            self.start_structure_cache_generation(auto_retry=True)
            return

        self.structure_cache_status_label.setText(f"结构缓存：正在运行 {model}。")
        if hasattr(self, "structure_progress"):
            self.structure_progress.setRange(0, 0)
            self.structure_progress.setFormat("准备中")
        self.log(f"开始生成结构缓存: model={model}, cache_root={cache_root}")
        thread = QThread(self)
        worker = StructureCacheWorker(
            cfg.input_path,
            str(cache_root),
            model,
            str(PROJECT_DIR),
            int(self.process_res_spin.value()),
            start_frame=int(getattr(cfg, "processing_start_frame", 0) or 0),
            end_frame=int(getattr(cfg, "processing_end_frame", -1) if getattr(cfg, "processing_end_frame", -1) is not None else -1),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._on_worker_log_signal)
        worker.progress.connect(self.on_structure_cache_progress_text)
        if hasattr(worker, "progress_value"):
            worker.progress_value.connect(self.on_structure_cache_progress_value)
        worker.finished.connect(self._on_structure_cache_finished)
        worker.failed.connect(self._on_structure_cache_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._cleanup_structure_cache_thread)
        thread.finished.connect(lambda th=thread: QTimer.singleShot(0, th.deleteLater))
        self.structure_cache_thread = thread
        self.structure_cache_worker = worker
        self.structure_cache_btn.setEnabled(False)
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()
        thread.start()

    def on_structure_cache_progress_text(self, msg: str) -> None:
        text = str(msg or "").strip()
        if not text:
            return
        if hasattr(self, "structure_cache_status_label"):
            self.structure_cache_status_label.setText(f"结构缓存：{text}")
        if hasattr(self, "stage_status_label"):
            self.stage_status_label.setText(f"阶段：结构缓存 / {text}")

    def on_structure_cache_progress_value(self, stage: str, done: int, total: int) -> None:
        stage_text = str(stage or "结构缓存").strip()
        done_i = int(done)
        total_i = int(total)
        if not hasattr(self, "structure_progress"):
            return
        if total_i > 0:
            self.structure_progress.setRange(0, total_i)
            self.structure_progress.setValue(max(0, min(done_i, total_i)))
            self.structure_progress.setFormat(f"{stage_text}：{max(0, done_i)}/{total_i}")
        else:
            self.structure_progress.setRange(0, 0)
            self.structure_progress.setFormat(f"{stage_text}：运行中")

    def _on_structure_cache_finished(self, msg: str) -> None:
        self.log(msg)
        if hasattr(self, "structure_cache_status_label"):
            self.structure_cache_status_label.setText("结构缓存：已生成，可预览/导出稳定 Mesh / 点云。")
        if hasattr(self, "structure_progress"):
            self.structure_progress.setRange(0, 1)
            self.structure_progress.setValue(1)
            self.structure_progress.setFormat("完成")
        self.refresh_3d_model_status()
        self._update_structure_scheme_status_labels()
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()
        QMessageBox.information(self, APP_NAME, "人体结构已生成。请手动进入“衣服和头发”步骤生成分割缓存。")

    def _friendly_structure_cache_error(self, msg: str) -> str:
        lower = str(msg).lower()
        missing = self._missing_modules_from_text(msg)
        if missing:
            cmd = sys.executable + " -m pip install " + " ".join(self._pip_package_for_module(m) for m in missing)
            return "结构缓存生成失败：缺少依赖 " + "、".join(missing) + "。\n\n可执行：\n" + cmd
        if "expected str, bytes or os.pathlike object, not nonetype" in lower and "home" in lower:
            return "结构缓存生成失败：Windows HOME 环境变量未传入 4D-Humans。请覆盖最新修复包后重试。"
        if "no importable outputs" in lower:
            return "结构缓存生成失败：4D-Humans 已运行但没有生成可导入 mesh/npz/pkl。展开调试日志看第三方仓库的具体报错。"
        if "modulenotfounderror" in lower:
            return "结构缓存生成失败：第三方结构模型依赖没有装完整，日志里会标出缺失模块。"
        return "结构缓存生成失败。请展开调试日志查看完整错误。"

    def _on_structure_cache_failed(self, msg: str) -> None:
        self.log("结构缓存生成失败:\n" + msg)
        if hasattr(self, "structure_cache_status_label"):
            self.structure_cache_status_label.setText("结构缓存：生成失败，查看日志。")
        if hasattr(self, "structure_progress"):
            self.structure_progress.setRange(0, 1)
            self.structure_progress.setValue(0)
            self.structure_progress.setFormat("失败")
        missing = self._missing_modules_from_text(msg)
        if missing and self._structure_dep_autoretry_count < 5:
            self._structure_dep_autoretry_count += 1
            self.log("检测到缺失依赖，自动安装后重试：" + "、".join(missing))
            if self._install_missing_python_modules(missing, ask=False):
                # Wait 1s to ensure the previous thread has fully emitted its finished signal and cleaned up
                QTimer.singleShot(1000, lambda: self.start_structure_cache_generation(auto_retry=True))
                return
        QMessageBox.warning(self, APP_NAME, self._friendly_structure_cache_error(msg))

    def _cleanup_structure_cache_thread(self) -> None:
        self.structure_cache_thread = None
        self.structure_cache_worker = None
        if hasattr(self, "structure_cache_btn"):
            self.structure_cache_btn.setEnabled(True)
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def _is_3d_structure_model_configured(self) -> bool:
        return bool(self._scan_3d_model_config().get("structure_ok"))

    def _is_3d_hand_model_configured(self) -> bool:
        return bool(self._scan_3d_model_config().get("hand_ok"))

    def refresh_3d_model_status(self) -> None:
        scan = self._scan_3d_model_config()
        structure_ok = bool(scan.get("structure_ok"))
        hand_ok = bool(scan.get("hand_ok"))
        full_ok = bool(scan.get("full_ok"))

        if hasattr(self, "model_3d_surface_badge"):
            if structure_ok:
                self._set_badge_state(self.model_3d_surface_badge, "缓存可用", "#86efac", "#166534")
            else:
                self._set_badge_state(self.model_3d_surface_badge, "需要缓存", "#fbbf24", "#92400e")

        if hasattr(self, "model_3d_completion_badge"):
            if hand_ok:
                self._set_badge_state(self.model_3d_completion_badge, "核心可用", "#86efac", "#166534")
            elif bool(scan.get("smpl_ok")) or bool(scan.get("mano_ok")):
                self._set_badge_state(self.model_3d_completion_badge, "部分可用", "#93c5fd", "#1d4ed8")
            else:
                self._set_badge_state(self.model_3d_completion_badge, "基础可用", "#fbbf24", "#92400e")

        if hasattr(self, "model_3d_status_label"):
            if structure_ok and hand_ok:
                text = "3D模型状态：Structure cache + MANO / HaMeR 已可用；可导出结构+手部 Mesh / 点云。"
            elif structure_ok:
                text = "3D模型状态：Structure cache 已生成；可导出稳定 Mesh / 点云。"
            elif bool(scan.get("structure_runner_ok")):
                text = "3D模型状态：4DHumans / WHAM 资源存在；请先生成结构缓存。"
            elif hand_ok:
                text = "3D模型状态：MANO / HaMeR 已识别；手部资源已就绪，但结构补全还需要 structure cache。"
            elif bool(scan.get("smpl_ok")) or bool(scan.get("mano_ok")):
                text = "3D模型状态：已识别部分授权模型。已识别部分模型资源；主流程仍需先生成 structure cache。"
            else:
                text = "3D模型状态：主流程需要 structure cache；未生成前不能导出结构点云。"
            self.model_3d_status_label.setText(text)

        lines = self._format_3d_scan_lines(scan)
        if hasattr(self, "model_3d_detail_label"):
            self.model_3d_detail_label.setText("\n".join(lines))
        if hasattr(self, "model_3d_check_btn"):
            self.model_3d_check_btn.setText("重新检查3D模型配置")

        try:
            self.log("3D模型配置检查完成：" + " | ".join(lines))
        except Exception:
            pass

        # Deliberately show a small dialog on manual click: the old button looked
        # like it did nothing because only a small label changed.
        if self.sender() is not None:
            QMessageBox.information(self, APP_NAME, "3D模型配置检查完成。\n\n" + "\n".join(lines))

    def on_model_device_changed(self) -> None:
        self._update_three_model_status()
        self.loaded_model_key = None
        if self._direct_depth_input_ready():
            self.preview_status_label.setText("主流程不使用 旧深度/Depth 作为骨架；请先生成 structure cache。")
            return
        if self._has_active_model_task():
            self.preview_status_label.setText("当前预览/导出任务运行中，结束后再切换模型/设备。")
            return
        if self.current_input:
            self.preview_status_label.setText("结构点云流程不使用 旧深度/法线 预热；先生成 structure cache。")
            self.schedule_model_preload()

    def schedule_model_preload(self) -> None:
        # Structure-XYZ workflow does not preload 旧深度/法线. 4DHumans/WHAM are
        # launched only by the explicit “生成结构缓存” action to protect 12GB VRAM.
        return


    def _queue_model_action(self, action: str) -> None:
        self._pending_model_action = action
        action_label = "预览" if action == "preview" else "导出"
        self.preview_status_label.setText(f"结构缓存任务运行中，已记录{action_label}动作，结束后继续。")

    def _run_pending_model_action(self) -> None:
        action = self._pending_model_action
        self._pending_model_action = None
        if action == "preview":
            self.preview_status_label.setText("继续检查 Alpha。")
            QTimer.singleShot(0, self.refresh_alpha_preview)
        elif action == "export":
            self.preview_status_label.setText("继续导出 XYZ USDA。")
            QTimer.singleShot(0, self.start_job)


    def start_model_preload(self, force: bool = False) -> None:
        self.preload_pending_key = None
        self._pending_model_action = None
        if hasattr(self, "preview_status_label"):
            self.preview_status_label.setText("结构点云流程不预加载 旧深度/法线；请按需生成 structure cache。")
        return


    def on_model_preload_finished(self, msg: str) -> None:
        finished_key = self.preload_key
        self.loaded_model_key = finished_key
        self.log(msg)
        if self.preload_pending_key is None and self._pending_model_action is None:
            self.preview_status_label.setText(msg + "。可以渲染当前帧或直接导出。")

    def on_model_preload_failed(self, msg: str) -> None:
        self.loaded_model_key = None
        self.log("模型预热失败:\n" + msg)
        if self.preload_pending_key is None and self._pending_model_action is None:
            self.preview_status_label.setText("模型预热失败，将在渲染/导出时重新加载。")

    def cleanup_preload_thread(self) -> None:
        self.preload_worker = None
        self.preload_thread = None
        if self.preload_pending_key is not None:
            self.preload_pending_key = None
            QTimer.singleShot(0, lambda: self.start_model_preload(force=True))
            return
        if self._pending_model_action is not None:
            self._run_pending_model_action()
            return
        self.preview_btn.setEnabled(bool(self.current_input))
        self.start_btn.setEnabled(bool(self.current_input))

    def apply_human_motion_preset(self) -> None:
        self.model_combo.setCurrentText("图像驱动网格主流程")
        self.device_combo.setCurrentText("CUDA 优先")
        self.batch_spin.setValue(1)
        self.process_res_spin.setValue(1024)
        self.color_combo.setCurrentText("灰度")
        self.invert_check.setChecked(False)
        self.smooth_spin.setValue(8)
        self.black_pct_spin.setValue(0.0)
        self.white_pct_spin.setValue(100.0)
        self.gamma_spin.setValue(0.95)
        self.detail_boost_spin.setValue(0)
        self.normal_strength_spin.setValue(24)
        self.normal_refine_spin.setValue(0)
        self.anti_banding_spin.setValue(30)
        self.depth_smooth_spin.setValue(40)
        self.edge_preserve_spin.setValue(78)
        self.tone_black_shift_spin.setValue(0)
        self.tone_shadow_shift_spin.setValue(-6)
        self.tone_mid_shift_spin.setValue(0)
        self.tone_light_shift_spin.setValue(4)
        self.tone_white_shift_spin.setValue(0)
        self.tone_black_contrast_spin.setValue(0)
        self.tone_shadow_contrast_spin.setValue(0)
        self.tone_mid_contrast_spin.setValue(0)
        self.tone_light_contrast_spin.setValue(0)
        self.tone_white_contrast_spin.setValue(0)
        self.tone_black_spin.setValue(0)
        self.tone_shadow_spin.setValue(10)
        self.tone_mid_spin.setValue(6)
        self.tone_light_spin.setValue(0)
        self.tone_white_spin.setValue(-4)
        self.levels_in_black_spin.setValue(0)
        self.levels_in_white_spin.setValue(130)
        self.levels_out_black_spin.setValue(35)
        self.levels_out_white_spin.setValue(235)
        self.preview_status_label.setText("已应用人体动作推荐参数。")
        self.render_preview_from_cache()

    def apply_neutral_preset(self) -> None:
        self.model_combo.setCurrentText("图像驱动网格主流程")
        self.device_combo.setCurrentText("CUDA 优先")
        self.batch_spin.setValue(1)
        self.process_res_spin.setValue(768)
        self.color_combo.setCurrentText("灰度")
        self.invert_check.setChecked(False)
        self.smooth_spin.setValue(5)
        self.black_pct_spin.setValue(0.0)
        self.white_pct_spin.setValue(100.0)
        self.gamma_spin.setValue(1.00)
        self.detail_boost_spin.setValue(0)
        self.normal_strength_spin.setValue(0)
        self.normal_refine_spin.setValue(0)
        self.anti_banding_spin.setValue(16)
        self.depth_smooth_spin.setValue(24)
        self.edge_preserve_spin.setValue(84)
        self.tone_black_shift_spin.setValue(0)
        self.tone_shadow_shift_spin.setValue(0)
        self.tone_mid_shift_spin.setValue(0)
        self.tone_light_shift_spin.setValue(0)
        self.tone_white_shift_spin.setValue(0)
        self.tone_black_contrast_spin.setValue(0)
        self.tone_shadow_contrast_spin.setValue(0)
        self.tone_mid_contrast_spin.setValue(0)
        self.tone_light_contrast_spin.setValue(0)
        self.tone_white_contrast_spin.setValue(0)
        self.tone_black_spin.setValue(0)
        self.tone_shadow_spin.setValue(0)
        self.tone_mid_spin.setValue(0)
        self.tone_light_spin.setValue(0)
        self.tone_white_spin.setValue(0)
        self.levels_in_black_spin.setValue(0)
        self.levels_in_white_spin.setValue(255)
        self.levels_out_black_spin.setValue(0)
        self.levels_out_white_spin.setValue(255)
        self.preview_status_label.setText("已恢复稳妥默认参数（预览偏均衡）。")
        self.render_preview_from_cache()

    def _set_encoder_combo_value(self, mode_or_display: str) -> None:
        display = encoder_display_name(encoder_internal_name(mode_or_display))
        if display in [self.encoder_combo.itemText(i) for i in range(self.encoder_combo.count())]:
            self.encoder_combo.setCurrentText(display)

    def _current_encoder_mode(self) -> str:
        return encoder_internal_name(self.encoder_combo.currentText())

    def _is_structure_output_mode(self) -> bool:
        """Current main workflow outputs Mesh/Shell/point-cloud folders, not depth videos."""
        return self._pointcloud_mode() != "structure_xyz"

    def _is_png_sequence_mode(self, encoder_mode: Optional[str] = None) -> bool:
        if self._is_structure_output_mode():
            return False
        mode = encoder_internal_name(encoder_mode if encoder_mode is not None else self.encoder_combo.currentText())
        return mode == "PNG序列 16-bit"

    def _effective_pointcloud_max_points(self) -> int:
        density = self.pointcloud_density_combo.currentText() if hasattr(self, "pointcloud_density_combo") else "中"
        if density == "低":
            return 50000
        if density == "中":
            return 120000
        if density == "高":
            return 200000
        return 120000

    def _structure_model_key(self) -> str:
        text = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else "4DHumans"
        return "wham" if "WHAM" in str(text).upper() else "4dhumans"

    def _pointcloud_mode(self) -> str:
        # Current GUI exposes only the Mesh/Shell main workflow. Older presets may
        # still contain "Legacy可见表面" in the hidden combo; never let that hidden
        # value switch export back to the old Depth/visible-surface branch.
        text = self.structure_solver_combo.currentText() if hasattr(self, "structure_solver_combo") else ""
        if "手" in text:
            return "fused_body_hand"
        return "fused_body"

    def _pointcloud_color_mode(self) -> str:
        return "xyz"

    def _default_output_path_for_encoder(self, out_w: int, out_h: int, encoder_mode: Optional[str] = None) -> str:
        if not self.current_input:
            return ""
        if self._is_structure_output_mode():
            return os.path.normpath(default_structure_output_dir(self.current_input))
        base = Path(default_output_path(self.current_input, out_w, out_h))
        if self._is_png_sequence_mode(encoder_mode):
            return str(png_sequence_output_dir(base))
        return str(base)

    def _coerce_output_path_for_encoder(self, path_text: str, out_w: int, out_h: int, encoder_mode: Optional[str] = None) -> str:
        mode = encoder_mode if encoder_mode is not None else self.encoder_combo.currentText()
        text = (path_text or "").strip()
        if not text:
            return os.path.normpath(self._default_output_path_for_encoder(out_w, out_h, mode))
        path = Path(text)
        if self._is_structure_output_mode():
            if path.suffix:
                path = path.with_suffix("")
            return os.path.normpath(str(path))
        if self._is_png_sequence_mode(mode):
            if path.suffix:
                path = png_sequence_output_dir(path)
            return os.path.normpath(str(path))
        if not path.suffix:
            name = path.name[:-6] if path.name.endswith("_png16") else path.name
            path = path.with_name(name + ".mp4")
        elif path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        return os.path.normpath(str(path))

    def on_encoder_changed(self, _text: str = "") -> None:
        if not self.current_input or not self.video_info:
            return
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        current = self.output_path_edit.text().strip()
        if not self._manual_output_path:
            self.output_path_edit.setText(self._default_output_path_for_encoder(out_w, out_h))
        elif current:
            self.output_path_edit.setText(self._coerce_output_path_for_encoder(current, out_w, out_h))
        if self._is_structure_output_mode():
            mode_note = "当前主流程会输出到 Mesh / 点云文件夹。"
        else:
            mode_note = "PNG 序列会输出到当前显示的文件夹。" if self._is_png_sequence_mode() else "视频模式会输出为 .mp4 文件。"
        self.preview_status_label.setText(mode_note)
        self.output_open_btn.setEnabled(bool(self.output_path_edit.text().strip()))

    def on_auto_mask_controls_changed(self) -> None:
        self._auto_mask_debounce.start()

    def on_background_fill_changed(self) -> None:
        # Output-stage only: no model rebuild needed. Debounced render prevents slider spam.
        self._update_conditional_visibility()
        self._switch_to_fusion_preview_for_curve_edit()
        self.render_preview_from_cache()

    def _apply_auto_mask_controls(self) -> None:
        if self.preview_depth is None:
            self.preview_status_label.setText("主体Mask参数会在下次渲染当前帧/导出时生效。")
            return
        try:
            raw_mask = None
            source_name = "模型深度 临时蒙版"
            try:
                cfg = self.make_config()
                external_mask = read_external_subject_mask(cfg, int(self.preview_frame_spin.value()), self.preview_depth.shape[:2])
                if external_mask is not None:
                    raw_mask = external_mask
                    source_name = "原视频Alpha"
                    cache_root = frame_cache_root(cfg) if cfg.cache_enabled else None
                    if cache_root is not None:
                        raw_mask = load_alpha_mask_for_depth(cache_root, int(self.preview_frame_spin.value()), self.preview_depth.shape[:2])
                    if raw_mask is not None:
                        source_name = "MatAnyone alpha"
            except Exception:
                raw_mask = None
            self.preview_subject_mask = build_alignment_subject_mask(
                self.preview_depth,
                raw_mask,
                auto_mask_feather_px=float(self.auto_mask_feather_spin.value()),
                auto_mask_expand_px=int(self.auto_mask_expand_spin.value()),
            )
            self.preview_depth_version += 1
            self.preview_base_gray_cache = None
            self.preview_hist_gray_cache = None
            self.preview_base_key = None
            self.preview_status_label.setText(
                f"已更新主体Mask（{source_name}）：外扩 {int(self.auto_mask_expand_spin.value())}px / 羽化 {int(self.auto_mask_feather_spin.value())}px"
            )
            self.render_preview_from_cache()
        except Exception as exc:  # noqa: BLE001
            self.preview_status_label.setText(f"主体Mask更新失败: {exc}")
            self.log(f"主体Mask更新失败: {exc}")

    def on_human_refine_changed(self) -> None:
        if self.preview_depth is not None:
            self.preview_status_label.setText("人体精炼会重新跑局部 模型深度；调整后请点“渲染当前帧”查看真实效果。")
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None

    def refresh_output_size(self) -> None:
        if not self.video_info:
            self.out_size_label.setText("输出: -")
            self.output_path_edit.clear()
            self.output_open_btn.setEnabled(False)
            return
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        self.out_size_label.setText(f"{out_w}x{out_h}")
        if self.current_input and not self._manual_output_path:
            self.output_path_edit.setText(self._default_output_path_for_encoder(out_w, out_h))
        self.output_open_btn.setEnabled(bool(self.output_path_edit.text().strip()))

    def pick_output_path(self) -> None:
        if not self.current_input or not self.video_info:
            QMessageBox.warning(self, APP_NAME, "请先导入视频。")
            return
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        current = self.output_path_edit.text().strip() or self._default_output_path_for_encoder(out_w, out_h)
        if self._is_structure_output_mode():
            start_dir = str(Path(current).parent if current else Path(self.current_input).parent)
            path = QFileDialog.getExistingDirectory(self, "选择 Mesh / 点云输出文件夹", start_dir)
        else:
            if self._is_png_sequence_mode():
                title = "选择 PNG 序列输出目录名"
                file_filter = "PNG Sequence Folder (*);;All Files (*.*)"
            else:
                title = "选择输出 MP4"
                file_filter = "MP4 Video (*.mp4);;All Files (*.*)"
            path, _ = QFileDialog.getSaveFileName(self, title, current, file_filter)
        if path:
            self._manual_output_path = True
            self.output_path_edit.setText(self._coerce_output_path_for_encoder(path, out_w, out_h))
            self.output_open_btn.setEnabled(True)

    def open_output_dir(self) -> None:
        path = self.output_path_edit.text().strip()
        if not path:
            try:
                self.preview_status_label.setText("还没有输出路径。请先选择输出位置。")
            except Exception:
                pass
            QMessageBox.warning(self, APP_NAME, "还没有输出路径。请先选择输出位置。")
            return
        path_obj = Path(path)
        folder = path_obj if self._is_structure_output_mode() or self._is_png_sequence_mode() or not path_obj.suffix else path_obj.parent
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, f"无法打开目录: {exc}")

    def make_config(self) -> JobConfig:
        if not self.current_input or not self.video_info:
            raise RuntimeError("请先导入视频。")
        out_w, out_h = scaled_size_from_long_side(
            self.video_info.width,
            self.video_info.height,
            self.long_side_spin.value(),
        )
        pointcloud_mode_value = self._pointcloud_mode()
        structure_xyz_mode = pointcloud_mode_value != "structure_xyz"
        if not structure_xyz_mode:
            if self.black_pct_spin.value() >= self.white_pct_spin.value():
                raise RuntimeError("黑位裁切必须小于白位裁切。")
            if self.levels_in_black_spin.value() >= self.levels_in_white_spin.value():
                raise RuntimeError("曲线输入黑必须小于曲线输入白。")
        encoder_mode = self._current_encoder_mode()
        output_path = self._coerce_output_path_for_encoder(self.output_path_edit.text().strip(), out_w, out_h, encoder_mode)
        output_obj = Path(output_path)
        if os.path.abspath(output_path) == os.path.abspath(self.current_input):
            raise RuntimeError("输出路径不能覆盖原视频。")
        if structure_xyz_mode or self._is_png_sequence_mode(encoder_mode):
            output_obj.mkdir(parents=True, exist_ok=True)
        else:
            output_obj.parent.mkdir(parents=True, exist_ok=True)
        source_mode = "cutout_video"
        matting_enabled = False
        matting_model_path = os.path.normpath(self.matting_model_path_edit.text().strip() or str(DEFAULT_MATANYONE_MODEL_PATH))
        matting_mask_path = os.path.normpath(self.matting_mask_path_edit.text().strip())
        # Current main workflow only needs the main video. Alpha is optional helper; MatAnyone remains hidden compatibility.
        if matting_enabled:
            model_ok = Path(matting_model_path).is_file()
            mask_ok = bool(matting_mask_path) and Path(matting_mask_path).is_file()
            if not (model_ok and mask_ok):
                # Do not block the stable Mesh/Shell export. MatAnyone is optional compatibility.
                matting_enabled = False
                try:
                    self._update_matting_status_label()
                except Exception:
                    pass
                self.log("MatAnyone 未参与：缺少模型或第一帧 mask，已继续使用主视频结构流程。")
                self.preview_status_label.setText("MatAnyone 未参与：已继续主视频结构流程。")
        input_cutout_mask_enabled = True
        external_mask_path = ""
        external_mask_enabled = False
        # Main workflow no longer requires external Depth / 法线.
        # Shell geometry uses the mesh's own normals; Alpha is optional.
        external_depth_path = ""
        external_depth_enabled = False
        model_id = MODEL_IDS[self.model_combo.currentText()]
        normalize_mode_for_job = self.normalize_mode_combo.currentText()
        pointcloud_temporal_enabled = bool(self.pointcloud_temporal_check.isChecked())
        mesh_export_selected = bool(self.mesh_export_check.isChecked())
        detail_mesh_export_selected = bool(self.detail_mesh_export_check.isChecked())
        pointcloud_export_selected = bool(self.pointcloud_usd_check.isChecked())
        if structure_xyz_mode and not (mesh_export_selected or detail_mesh_export_selected or pointcloud_export_selected):
            raise RuntimeError("没有选择任何输出内容。请至少勾选低模 Mesh、细节 Mesh 或稳定点云之一。")
        pointcloud_enabled_value = bool(mesh_export_selected or detail_mesh_export_selected or pointcloud_export_selected) if structure_xyz_mode else bool(self.pointcloud_enable_check.isChecked())
        return JobConfig(
            input_path=self.current_input,
            output_path=output_path,
            output_width=out_w,
            output_height=out_h,
            model_id=model_id,
            device_mode=self.device_combo.currentText(),
            batch_size=int(self.batch_spin.value()),
            process_res=int(self.process_res_spin.value()),
            invert=self.invert_check.isChecked(),
            smooth=int(self.smooth_spin.value()),
            black_pct=float(self.black_pct_spin.value()),
            white_pct=float(self.white_pct_spin.value()),
            gamma=float(self.gamma_spin.value()),
            detail_boost=int(self.detail_boost_spin.value()),
            normal_strength=0,
            levels_in_black=int(self.levels_in_black_spin.value()),
            levels_in_white=int(self.levels_in_white_spin.value()),
            levels_out_black=int(self.levels_out_black_spin.value()),
            levels_out_white=int(self.levels_out_white_spin.value()),
            curve_points=tuple((float(x), float(y)) for x, y in self.levels_panel.getCurvePoints()),
            anti_banding=int(self.anti_banding_spin.value()),
            depth_smooth=int(self.depth_smooth_spin.value()),
            edge_preserve=int(self.edge_preserve_spin.value()),
            tone_black=int(self.tone_black_spin.value()),
            tone_shadow=int(self.tone_shadow_spin.value()),
            tone_mid=int(self.tone_mid_spin.value()),
            tone_light=int(self.tone_light_spin.value()),
            tone_white=int(self.tone_white_spin.value()),
            tone_black_shift=int(self.tone_black_shift_spin.value()),
            tone_shadow_shift=int(self.tone_shadow_shift_spin.value()),
            tone_mid_shift=int(self.tone_mid_shift_spin.value()),
            tone_light_shift=int(self.tone_light_shift_spin.value()),
            tone_white_shift=int(self.tone_white_shift_spin.value()),
            tone_black_contrast=int(self.tone_black_contrast_spin.value()),
            tone_shadow_contrast=int(self.tone_shadow_contrast_spin.value()),
            tone_mid_contrast=int(self.tone_mid_contrast_spin.value()),
            tone_light_contrast=int(self.tone_light_contrast_spin.value()),
            tone_white_contrast=int(self.tone_white_contrast_spin.value()),
            copy_audio=self.copy_audio_check.isChecked(),
            cache_enabled=self.cache_enable_check.isChecked(),
            normalize_mode=normalize_mode_for_job,
            human_refine=0,
            normal_refine=0,
            encoder_mode=encoder_mode,
            input_brightness=int(self.input_brightness_spin.value()),
            input_contrast=int(self.input_contrast_spin.value()),
            input_gamma=float(self.input_gamma_spin.value()),
            input_shadow=int(self.input_shadow_spin.value()),
            input_highlight=int(self.input_highlight_spin.value()),
            input_sharpen=int(self.input_sharpen_spin.value()),
            input_denoise=int(self.input_denoise_spin.value()),
            matting_enabled=matting_enabled,
            matting_mask_path=matting_mask_path,
            matting_model_path=matting_model_path,
            matting_max_size=0,
            auto_mask_feather_px=int(self.auto_mask_feather_spin.value()),
            auto_mask_expand_px=int(self.auto_mask_expand_spin.value()),
            background_mode=self.background_mode_combo.currentText(),
            background_gray=int(self.background_gray_spin.value()),
            external_mask_enabled=external_mask_enabled,
            external_mask_path=external_mask_path,
            external_mask_invert=self.external_mask_invert_check.isChecked(),
            input_cutout_mask_enabled=input_cutout_mask_enabled,
            external_depth_enabled=False,
            external_depth_path="",
            external_depth_weight=0,
            external_depth_invert=False,
            external_depth_orientation_mode="不使用",
            pointcloud_enabled=pointcloud_enabled_value,
            pointcloud_mode=pointcloud_mode_value,
            pointcloud_density=self.pointcloud_density_combo.currentText(),
            pointcloud_stride=int(self._effective_pointcloud_stride()),
            pointcloud_max_points=self._effective_pointcloud_max_points(),
            pointcloud_depth_near_percentile=1.0,
            pointcloud_depth_far_percentile=99.0,
            pointcloud_color_mode="xyz",
            pointcloud_coordinate_mode="blender",
            pointcloud_binary_ply=True,
            pointcloud_alpha_erode_px=1,
            pointcloud_alpha_dilate_px=0,
            pointcloud_alpha_feather_px=3,
            pointcloud_body_bbox_margin_px=12,
            pointcloud_remove_outliers=False if structure_xyz_mode else bool(self.pointcloud_remove_outliers_check.isChecked()),
            pointcloud_voxel_downsample=False if structure_xyz_mode else bool(self.pointcloud_voxel_check.isChecked()),
            pointcloud_temporal_depth_smooth=(0.52 if (structure_xyz_mode and pointcloud_temporal_enabled) else 0.0),
            pointcloud_temporal_center_smooth=(0.68 if (structure_xyz_mode and pointcloud_temporal_enabled) else 0.0),
            pointcloud_temporal_scale_smooth=(0.58 if (structure_xyz_mode and pointcloud_temporal_enabled) else 0.0),
            pointcloud_template_align_strength=1.0,
            pointcloud_obj_sequence=False,
            pointcloud_usd_sequence=pointcloud_export_selected,
            pointcloud_usd_max_points=max(1000, min(int(self._effective_pointcloud_max_points()), 800000)),
            pointcloud_usd_point_width=0.008,
            mesh_export_enabled=mesh_export_selected,
            detail_mesh_export_enabled=detail_mesh_export_selected,
            mesh_dense_segments=int((self.mesh_dense_segments_combo.currentText() or "中 2x").split()[1].replace("x", "")) if hasattr(self, "mesh_dense_segments_combo") else 2,
            garment_shell_enabled=True,
            garment_shell_offset=(float(self.garment_shell_offset_spin.value()) / 1000.0 if hasattr(self, "garment_shell_offset_spin") else 0.020),
            hair_shell_enabled=True,
            hair_shell_offset=(float(self.hair_shell_offset_spin.value()) / 1000.0 if hasattr(self, "hair_shell_offset_spin") else 0.035),
            segmentation_enabled=True,
            segmentation_provider="Auto",
            segmentation_use_cache=True,
            segmentation_fallback_geometry=False,
            pointcloud_abc_sequence=False,
            pointcloud_abc_max_points=120000,
            pointcloud_normal_relief_enabled=False,
            pointcloud_normal_relief_strength=0.0,
            pointcloud_normal_relief_gamma=1.6,
            pointcloud_hand_enabled=(pointcloud_mode_value == "fused_body_hand"),
            structure_model=self._structure_model_key() if structure_xyz_mode else "none",
            hand_model="none",
            occlusion_fill_enabled=structure_xyz_mode,
            processing_start_frame=int(self._processing_range_values()[0]) if hasattr(self, "_processing_range_values") else 0,
            processing_end_frame=int(self._processing_range_values()[1]) if hasattr(self, "_processing_range_values") else -1,
            mesh_preview_yaw=float(getattr(self, "mesh_preview_yaw", 0.0)),
            mesh_preview_pitch=float(getattr(self, "mesh_preview_pitch", 0.0)),
            project_dir=str(self.current_project_dir) if self.current_project_dir else "",
        )

    def refresh_alpha_preview(self) -> None:
        """Refresh only the RGB frame and prepared Alpha/Mask preview.

        The structure-XYZ workflow does not generate depth previews. This button
        reads the selected RGB frame, reads the prepared mask, and updates the
        two visible preview tiles.
        """
        if not self.current_input or not self.video_info:
            QMessageBox.warning(self, APP_NAME, "请先导入带 Alpha 原视频。")
            return
        try:
            self.make_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        frame_index = int(self.preview_frame_spin.value())
        self.preview_depth = None
        self.preview_subject_mask = None
        self.preview_normal_map = None
        self.preview_depth_version += 1
        self.preview_depth_render_bgr = None
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_big_btn.setEnabled(False)
        self.preview_status_label.setText(f"正在读取第 {frame_index} 帧原视频和 Alpha...")
        self.show_original_frame_immediately(frame_index)
        QTimer.singleShot(220, self._refresh_reference_preview_tiles)

    def start_preview(self) -> None:
        if self.thread is not None:
            self.preview_status_label.setText("正在导出，导出完成后再渲染预览。")
            return
        direct_depth_mode = self._direct_depth_input_ready()
        if self.preload_thread is not None and not direct_depth_mode:
            self._queue_model_action("preview")
            return
        if self.preview_thread is not None:
            self.preview_status_label.setText("预览正在进行，先等当前帧完成。")
            return
        if self._base_rebuild_thread is not None:
            self.preview_status_label.setText("深度底图正在更新，完成后再重新渲染当前帧。")
            return
        self._seek_debounce.stop()
        try:
            cfg = self.make_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if not self._confirm_preview_resource_risk(cfg):
            return
        frame_index = int(self.preview_frame_spin.value())
        self._set_model_config_controls_enabled(False)
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.preview_big_btn.setEnabled(False)
        self.preview_status_label.setText(f"预览推理中，第 {frame_index} 帧...")
        self.log(f"开始生成预览帧: 第 {frame_index} 帧")
        # Do not clear loaded model caches before every preview. That makes each
        # "render current frame" reload 模型深度/法线 and feels like a cold start.
        # Only trim the CUDA allocator; full model cache clearing is reserved for OOM
        # recovery and the explicit model-manager action.
        if self.device_combo.currentText() != "CPU":
            trim_cuda_allocator_cache(None)

        self.preview_thread = QThread(self)
        self.preview_worker = PreviewWorker(cfg, frame_index)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.log.connect(self._on_worker_log_signal)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.failed.connect(self.on_preview_failed)
        self.preview_worker.finished.connect(self.preview_thread.quit)
        self.preview_worker.failed.connect(self.preview_thread.quit)
        self.preview_worker.finished.connect(self.preview_worker.deleteLater)
        self.preview_worker.failed.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self.cleanup_preview_thread)
        self.preview_thread.finished.connect(lambda th=self.preview_thread: QTimer.singleShot(0, th.deleteLater))
        self.preview_thread.start()

    def on_preview_finished(self, original_bgr: object, depth: object, subject_mask: object, normal_map: object, elapsed: float, frame_index: int) -> None:
        if frame_index != int(self.preview_frame_spin.value()):
            self.preview_status_label.setText("旧预览已丢弃。请点“渲染当前帧”。")
            self.preview_btn.setEnabled(True)
            return
        self.preview_original_bgr = original_bgr
        self._show_adjusted_original_preview()
        self.preview_depth = depth
        self.preview_subject_mask = subject_mask
        # In the direct cutout workflow, the keyed main video alpha is the only
        # subject range. Recompute it from the frame shown in the UI so stale
        # fallback masks or external-depth masks cannot leak into 模型深度/final views.
        if self.preview_original_bgr is not None and self.preview_depth is not None:
            effective_mask = self._effective_preview_subject_mask_for_shape(np.asarray(self.preview_depth).shape[:2])
            if effective_mask is not None:
                self.preview_subject_mask = effective_mask
        self.preview_normal_map = normal_map
        self.preview_depth_version += 1
        self._refresh_reference_preview_tiles()
        self.preview_base_gray_cache = None
        self.preview_hist_gray_cache = None
        self.preview_base_key = None
        self.preview_status_label.setText(f"预览完成，用时 {elapsed:.2f}s。正在更新深度底图...")
        # Start the post-model rebuild immediately and keep model controls locked
        # until the depth base is ready. This avoids changing 法线/model
        # state in the small gap between 模型深度 inference and fusion rendering.
        self._schedule_preview_render()
        if self._base_rebuild_thread is None:
            self._set_model_config_controls_enabled(True)
            self.preview_btn.setEnabled(True)
            self.start_btn.setEnabled(True)

    def on_preview_failed(self, msg: str) -> None:
        self.log("预览失败:\n" + msg)
        self.preview_status_label.setText("原视频 Alpha 检查失败")
        base_msg = short_error_message(msg)
        if "read_video_frame_bgr" in msg or "not defined" in msg:
            hint = "这是预览帧读取入口没有被正确接入造成的内部链路错误，不是你的视频问题。更新本补丁后重启程序再试。"
        elif "无法读取预览帧" in msg or "随机定位" in msg:
            hint = "这个更像视频解码/随机定位问题。可先拖动到附近帧重试；如果仍失败，用 FFmpeg 重新封装/转码该视频后再导入。"
        elif "Alpha" in msg or "alpha" in msg:
            hint = "先检查原视频是否保留真实 Alpha 通道。"
        else:
            hint = "这是预览链路异常。先确认原视频路径有效且带真实 Alpha；如果只是某一帧失败，拖到附近帧再试。"
        QMessageBox.critical(
            self,
            APP_NAME,
            base_msg + "\n\n" + hint,
        )
        self.preview_big_btn.setEnabled(False)
        self._set_model_config_controls_enabled(True)
        self.preview_btn.setEnabled(True)
        self.start_btn.setEnabled(True)

    def cleanup_preview_thread(self) -> None:
        self.preview_worker = None
        self.preview_thread = None
        if self.thread is None and self.preload_thread is None and self._base_rebuild_thread is None:
            self._set_model_config_controls_enabled(True)
            self.preview_btn.setEnabled(bool(self.current_input))
            self.start_btn.setEnabled(bool(self.current_input))

    def set_label_pixmap(self, label: QLabel, frame_bgr: np.ndarray) -> None:
        pixmap = bgr_to_pixmap(frame_bgr)
        if isinstance(label, PreviewImageLabel):
            label.setImagePixmap(pixmap)
        else:
            scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if hasattr(self, "preview_original_label"):
            self._preview_debounce.start()

    def _preview_base_params(self) -> tuple:
        return (
            self.invert_check.isChecked(),
            float(self.black_pct_spin.value()),
            float(self.white_pct_spin.value()),
            float(self.gamma_spin.value()),
            int(self.detail_boost_spin.value()),
            self._effective_normal_strength(),
            self._effective_normal_refine(),
            int(self.depth_smooth_spin.value()),
            int(self.edge_preserve_spin.value()),
            self.preview_original_bgr.shape[1] if self.preview_original_bgr is not None else 0,
            self.preview_original_bgr.shape[0] if self.preview_original_bgr is not None else 0,
            self.preview_depth_version,
            int(self.auto_mask_feather_spin.value()),
            int(self.auto_mask_expand_spin.value()),
        )

    def _rebuild_preview_base_if_needed(self) -> None:
        if self.preview_original_bgr is None or self.preview_depth is None:
            return
        key = self._preview_base_params()
        if self.preview_base_gray_cache is not None and self.preview_base_key == key:
            return  # Cache hit — nothing to do

        # If a rebuild is already running, mark pending and return.
        # When the running rebuild finishes it will check _base_rebuild_pending
        # and start a new one.
        if self._base_rebuild_thread is not None:
            self._base_rebuild_pending = True
            return

        self._start_base_rebuild(key)

    def _start_base_rebuild(self, key: tuple) -> None:
        """Launch _BaseRebuildWorker in a background thread."""
        if self.preview_original_bgr is None or self.preview_depth is None:
            return
        th, tw = self.preview_original_bgr.shape[0], self.preview_original_bgr.shape[1]
        rebuild_mask = self._effective_preview_subject_mask_for_shape(np.asarray(self.preview_depth).shape[:2])
        worker = _BaseRebuildWorker(
            self.preview_depth,
            rebuild_mask,
            self.preview_normal_map,
            self.invert_check.isChecked(),
            float(self.black_pct_spin.value()),
            float(self.white_pct_spin.value()),
            float(self.gamma_spin.value()),
            int(self.detail_boost_spin.value()),
            self._effective_normal_strength(),
            self._effective_normal_refine(),
            int(self.depth_smooth_spin.value()),
            int(self.edge_preserve_spin.value()),
            (th, tw),
            key,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_base_rebuild_finished)
        worker.failed.connect(self._on_base_rebuild_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._cleanup_base_rebuild_thread)
        thread.finished.connect(lambda th=thread: QTimer.singleShot(0, th.deleteLater))
        self._base_rebuild_thread = thread
        self._base_rebuild_worker = worker
        # Show a non-blocking visual hint while the depth base is rebuilt.
        self._set_depth_preview_busy(True)
        self.preview_status_label.setText("正在刷新 Alpha 预览...")
        thread.start()

    def _on_base_rebuild_finished(self, base_gray: object, hist_gray: object, key: object) -> None:
        """Handle base rebuild result without destroying QThread wrappers early.

        Do not set _base_rebuild_thread/_worker to None here. The worker signal is
        delivered before QThread.finished; dropping the Python wrapper at this point
        can let Qt destroy a QObject while queued events are still unwinding, which
        can abort the process with a QObject::~QObject/_purecall stack trace.
        """
        current_key = self._preview_base_params() if self.preview_depth is not None else None
        if key != current_key:
            if current_key is not None:
                self._base_rebuild_pending = True
                self._base_rebuild_restart_key = current_key
            return

        if base_gray is not None:
            self.preview_base_gray_cache = base_gray  # type: ignore[assignment]
            self.preview_hist_gray_cache = hist_gray   # type: ignore[assignment]
            self.preview_base_key = key                # type: ignore[assignment]
            self.levels_panel.setHistogramFromGray(hist_gray, self._effective_preview_subject_mask_for_shape(np.asarray(hist_gray).shape[:2]))  # type: ignore[arg-type]

        if self._base_rebuild_pending:
            self._base_rebuild_restart_key = self._preview_base_params()
        else:
            self._set_depth_preview_busy(False)
            self.preview_status_label.setText("融合底图已更新。")
            self._schedule_preview_render()

    def _on_base_rebuild_failed(self, msg: str, key: object) -> None:
        # Keep thread/worker references alive until QThread.finished.
        self._base_rebuild_pending = False
        self._base_rebuild_restart_key = None
        self._set_depth_preview_busy(False)
        self.preview_status_label.setText("融合底图重建失败，查看日志。")
        self.log("融合底图重建失败:\n" + msg)

    def _cleanup_base_rebuild_thread(self) -> None:
        self._base_rebuild_thread = None
        self._base_rebuild_worker = None
        restart_key = self._base_rebuild_restart_key
        restart = self._base_rebuild_pending
        self._base_rebuild_pending = False
        self._base_rebuild_restart_key = None

        if restart and restart_key is not None and self.preview_depth is not None:
            self._start_base_rebuild(restart_key)
            return

        if self._base_rebuild_thread is None:
            self._set_depth_preview_busy(False)
        if self.thread is None and self.preview_thread is None and self.preload_thread is None:
            self._set_model_config_controls_enabled(True)
            self.preview_btn.setEnabled(bool(self.current_input))
            self.start_btn.setEnabled(bool(self.current_input))

    def _set_reference_tile_image(self, key: str, label: PreviewImageLabel, frame_bgr: np.ndarray, cache_key: tuple) -> None:
        previous_key = self._reference_preview_tile_keys.get(key)
        if previous_key == cache_key and key in self._reference_preview_tile_bgr:
            return
        self._reference_preview_tile_keys[key] = cache_key
        self._reference_preview_tile_bgr[key] = np.asarray(frame_bgr).copy()
        self.set_label_pixmap(label, frame_bgr)

    def _clear_reference_tile(self, key: str, label: PreviewImageLabel, text: str) -> None:
        self._reference_preview_tile_keys.pop(key, None)
        self._reference_preview_tile_bgr.pop(key, None)
        label.clearImage(text)

    def _mask_to_alpha_bgr(self, mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        alpha = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if alpha.shape[:2] != target_hw:
            th, tw = target_hw
            alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_LINEAR)
        gray = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _normal_map_to_bgr(self, normal_map: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        nm = np.clip(np.asarray(normal_map, dtype=np.float32), -1.0, 1.0)
        if nm.ndim != 3 or nm.shape[2] < 3:
            raise ValueError("法线 Map 维度异常")
        normal_rgb = np.dstack([
            (nm[:, :, 0] + 1.0) * 127.5,
            (nm[:, :, 1] + 1.0) * 127.5,
            np.clip((nm[:, :, 2] + 1.0) * 0.5, 0.0, 1.0) * 255.0,
        ]).astype(np.uint8)
        if normal_rgb.shape[:2] != target_hw:
            th, tw = target_hw
            normal_rgb = cv2.resize(normal_rgb, (tw, th), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR)

    def _refresh_reference_preview_tiles(self) -> None:
        """Update RGB and Alpha/Mask preview tiles for the structure-XYZ workflow."""
        if not all(hasattr(self, name) for name in (
            "preview_original_label",
            "preview_external_depth_label",
            "preview_da3_label",
            "preview_subject_alpha_label",
            "preview_normal_label",
        )):
            return

        if self.preview_original_bgr is None:
            self._clear_reference_tile("main", self.preview_original_label, "原视频未读取")
            self._clear_reference_tile("external_depth", self.preview_external_depth_label, "原视频 Alpha 未读取")
            self._clear_reference_tile("da3_depth", self.preview_da3_label, "structure cache 未检查")
            self._clear_reference_tile("subject_alpha", self.preview_subject_alpha_label, "原视频 Alpha 未读取")
            self._clear_reference_tile("normal_map", self.preview_normal_label, "Shell 预览在当前帧生成")
            return

        target_hw = self.preview_original_bgr.shape[:2]
        frame_index = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else 0
        source_bgr = self.preview_original_render_bgr if self.preview_original_render_bgr is not None else self.preview_original_bgr
        self._set_reference_tile_image(
            "main",
            self.preview_original_label,
            source_bgr,
            ("main", frame_index, source_bgr.shape, self._input_adjust_key()),
        )

        subject_mask = self._effective_preview_subject_mask_for_shape(target_hw)
        if subject_mask is not None:
            try:
                alpha_bgr = self._mask_to_alpha_bgr(subject_mask, target_hw)
                alpha_key = ("subject_alpha", frame_index, tuple(np.asarray(subject_mask).shape[:2]), target_hw)
                self._set_reference_tile_image("subject_alpha", self.preview_subject_alpha_label, alpha_bgr, alpha_key)
                self._set_reference_tile_image("external_depth", self.preview_external_depth_label, alpha_bgr, ("alpha_tile",) + alpha_key)
            except Exception as exc:  # noqa: BLE001
                self._clear_reference_tile("subject_alpha", self.preview_subject_alpha_label, "原视频 Alpha 不可读")
                self._clear_reference_tile("external_depth", self.preview_external_depth_label, "原视频 Alpha 不可读")
                self.log(f"原视频 Alpha 刷新失败: {exc}")
        else:
            self._clear_reference_tile("subject_alpha", self.preview_subject_alpha_label, "原视频 Alpha 未读取")
            self._clear_reference_tile("external_depth", self.preview_external_depth_label, "原视频 Alpha 未读取")

        if self._has_structure_cache():
            self._clear_reference_tile("da3_depth", self.preview_da3_label, "structure cache 已生成")
        else:
            self._clear_reference_tile("da3_depth", self.preview_da3_label, "structure cache 未生成")
        self._clear_reference_tile("normal_map", self.preview_normal_label, "Shell 预览在当前帧生成")


    def _structure_frame_for_preview(self, frame_index: int):  # noqa: ANN202
        cache_root = structure_cache_root(self.make_config())
        idx = int(frame_index)
        frame = load_structure_frame(cache_root, idx)
        if frame is None or not frame.available:
            raise RuntimeError(f"第 {idx} 帧 structure cache 不可读。请重新生成结构缓存，或换一帧预览。")
        return frame

    def _preview_axes_and_origin(self, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int, int, dict]:
        verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(verts) == 0:
            raise RuntimeError("Mesh 顶点为空，无法预览。")
        finite = np.isfinite(verts).all(axis=1)
        if not bool(np.any(finite)):
            raise RuntimeError("Mesh 顶点包含无效数值，无法预览。")
        clean = np.nan_to_num(verts, nan=0.0, posinf=0.0, neginf=0.0)
        lo = np.nanpercentile(clean[finite], 2.0, axis=0)
        hi = np.nanpercentile(clean[finite], 98.0, axis=0)
        extent = np.maximum(hi - lo, 1e-6)
        order = list(np.argsort(extent)[::-1])
        vertical_axis = int(order[0])
        horizontal_axis = int(order[1] if len(order) > 1 else 0)
        if horizontal_axis == vertical_axis:
            horizontal_axis = int((vertical_axis + 1) % 3)

        # Keep mesh and point-cloud preview centred in the same stable body space.
        # The centre is computed from a deterministic surface sample instead of
        # independent x/y normalization, so preview proportions match export logic.
        if len(f) > 0:
            origin_sample_count = min(8000, max(2000, int(self._effective_pointcloud_max_points() // 24)))
            try:
                origin_spec = make_surface_sample_spec(clean, f, origin_sample_count, seed=9100003)
                origin = robust_geometry_center(sample_mesh_surface_with_spec(clean, f, origin_spec))
            except Exception:
                origin = np.nanmedian(clean[finite], axis=0).astype(np.float32)
        else:
            origin = np.nanmedian(clean[finite], axis=0).astype(np.float32)
        meta = {
            "horizontal_axis": horizontal_axis,
            "vertical_axis": vertical_axis,
            "extent": extent,
            "origin": origin.astype(np.float32),
        }
        return origin.astype(np.float32), horizontal_axis, vertical_axis, meta

    def _project_preview_points(self, points: np.ndarray, horizontal_axis: int, vertical_axis: int, origin: np.ndarray, canvas_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, float]:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        canvas_h, canvas_w = int(canvas_shape[0]), int(canvas_shape[1])
        centred = pts - np.asarray(origin, dtype=np.float32).reshape(1, 3)
        finite = np.isfinite(centred).all(axis=1)
        if not bool(np.any(finite)):
            raise RuntimeError("预览点包含无效数值。")
        lo = np.nanpercentile(centred[finite], 2.0, axis=0)
        hi = np.nanpercentile(centred[finite], 98.0, axis=0)
        span_x = max(float(hi[horizontal_axis] - lo[horizontal_axis]), 1e-6)
        span_y = max(float(hi[vertical_axis] - lo[vertical_axis]), 1e-6)
        draw_w, draw_h = canvas_w * 0.78, canvas_h * 0.72
        scale = min(draw_w / span_x, draw_h / span_y) * 0.92
        x = np.clip((canvas_w * 0.50 + centred[:, horizontal_axis] * scale).astype(np.int32), 0, canvas_w - 1)
        y = np.clip((canvas_h * 0.52 - centred[:, vertical_axis] * scale).astype(np.int32), 0, canvas_h - 1)
        return x, y, float(scale)

    def _preview_mesh_bgr(self, vertices: np.ndarray, faces: np.ndarray, title: str, detail: str = "") -> np.ndarray:
        verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        face_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        canvas = np.zeros((620, 720, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 28)
        if len(verts) == 0:
            cv2.putText(canvas, "empty mesh", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 235), 2, cv2.LINE_AA)
            return canvas
        origin, horizontal_axis, vertical_axis, _meta = self._preview_axes_and_origin(verts, face_arr)
        x, y, _scale = self._project_preview_points(verts, horizontal_axis, vertical_axis, origin, canvas.shape)
        valid_faces = face_arr[(face_arr >= 0).all(axis=1) & (face_arr < len(verts)).all(axis=1)] if len(face_arr) else face_arr
        if len(valid_faces) > 26000:
            idx = np.linspace(0, len(valid_faces) - 1, 26000).astype(np.int64)
            valid_faces = valid_faces[idx]

        # Batch draw mesh preview. Per-triangle OpenCV calls make Dense Mesh preview
        # feel frozen; fillPoly/polylines keep preview responsive. Screen-space
        # back-face culling removes most rear triangles, with fallback if culling is ambiguous.
        depth_axis = ({0, 1, 2} - {int(horizontal_axis), int(vertical_axis)}).pop()
        if len(valid_faces):
            px = x[valid_faces]
            py = y[valid_faces]
            signed_area = (
                (px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0])
                - (py[:, 1] - py[:, 0]) * (px[:, 2] - px[:, 0])
            ).astype(np.float32)
            area_abs = np.abs(signed_area)
            keep_area = area_abs > 0.35
            pos_count = int(np.count_nonzero(signed_area[keep_area] > 0))
            neg_count = int(np.count_nonzero(signed_area[keep_area] < 0))
            if pos_count or neg_count:
                front_sign = 1.0 if pos_count >= neg_count else -1.0
                keep_face = keep_area & ((signed_area * front_sign) > 0.0)
                if int(np.count_nonzero(keep_face)) >= max(32, int(0.12 * len(valid_faces))):
                    valid_faces = valid_faces[keep_face]
            if len(valid_faces) > 18000:
                valid_faces = valid_faces[np.linspace(0, len(valid_faces) - 1, 18000).astype(np.int64)]
            depths = np.nanmean((verts[valid_faces] - origin[None, None, :])[:, :, depth_axis], axis=1)
            valid_faces = valid_faces[np.argsort(depths)]
            polys = np.stack([x[valid_faces], y[valid_faces]], axis=2).astype(np.int32)
            if len(polys):
                cv2.fillPoly(canvas, list(polys), (92, 101, 116), cv2.LINE_AA)
                edge_faces = valid_faces
                if len(edge_faces) > 6000:
                    edge_faces = edge_faces[np.linspace(0, len(edge_faces) - 1, 6000).astype(np.int64)]
                edge_polys = np.stack([x[edge_faces], y[edge_faces]], axis=2).astype(np.int32)
                cv2.polylines(canvas, list(edge_polys), True, (42, 52, 68), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (66, 74), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
        cv2.putText(canvas, title[:42], (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 240, 248), 2, cv2.LINE_AA)
        axis_names = ("x", "y", "z")
        cv2.putText(canvas, f"mesh verts={len(verts)} faces={len(face_arr)} axis={axis_names[horizontal_axis]}/{axis_names[vertical_axis]} equal-scale", (26, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 205), 1, cv2.LINE_AA)
        if detail:
            cv2.putText(canvas, detail[:80], (26, 608), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 172, 190), 1, cv2.LINE_AA)
        return canvas

    def _preview_pointcloud_bgr(self, points: np.ndarray, origin_vertices: np.ndarray, origin_faces: np.ndarray, title: str, detail: str = "") -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        origin, horizontal_axis, vertical_axis, _meta = self._preview_axes_and_origin(origin_vertices, origin_faces)
        canvas = np.zeros((620, 720, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 28)
        if len(pts) == 0:
            cv2.putText(canvas, "empty pointcloud", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 220, 235), 2, cv2.LINE_AA)
            return canvas
        x, y, _scale = self._project_preview_points(pts, horizontal_axis, vertical_axis, origin, canvas.shape)
        if len(x) > 22000:
            idx = np.linspace(0, len(x) - 1, 22000).astype(np.int64)
            x, y = x[idx], y[idx]
        canvas[y, x] = (230, 236, 245)
        cv2.rectangle(canvas, (66, 74), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
        cv2.putText(canvas, title[:42], (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 240, 248), 2, cv2.LINE_AA)
        axis_names = ("x", "y", "z")
        cv2.putText(canvas, f"points={len(pts)} axis={axis_names[horizontal_axis]}/{axis_names[vertical_axis]} equal-scale", (26, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 190, 205), 1, cv2.LINE_AA)
        if detail:
            cv2.putText(canvas, detail[:80], (26, 608), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 172, 190), 1, cv2.LINE_AA)
        return canvas

    def _preview_structure_geometry(self, apply_detail: bool) -> tuple[np.ndarray, np.ndarray, dict]:
        frame_index = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else 0
        cfg = self.make_config()
        frame0 = self._structure_frame_for_preview(0)
        frame = self._structure_frame_for_preview(frame_index)
        v0 = np.asarray(frame0.vertices, dtype=np.float32).reshape(-1, 3)
        vt = np.asarray(frame.vertices, dtype=np.float32).reshape(-1, 3)
        f0 = np.asarray(frame0.faces, dtype=np.int64).reshape(-1, 3)
        ft = np.asarray(frame.faces, dtype=np.int64).reshape(-1, 3)
        if f0.shape != ft.shape or not np.array_equal(f0, ft):
            raise RuntimeError("当前 structure cache 拓扑不固定：faces 在帧间发生变化，不能安全预览/导出固定拓扑 Mesh。请重新生成结构缓存。")
        j0 = np.asarray(frame0.joints, dtype=np.float32).reshape(-1, 3) if frame0.joints is not None else None
        jt = np.asarray(frame.joints, dtype=np.float32).reshape(-1, 3) if frame.joints is not None else None
        stable = stabilize_vertices_by_root([v0, vt], [j0, jt])
        svt = np.asarray(stable.vertices[-1], dtype=np.float32).reshape(-1, 3)
        if not apply_detail:
            return svt.astype(np.float32), f0.astype(np.int64), {"method": "stable_low_mesh_preview", "mesh_type": "low"}

        segments = int(np.clip(int(getattr(cfg, "mesh_dense_segments", 2) or 2), 1, 3))
        tmpl = build_dense_mesh_template(f0, segments=segments)
        sv0 = np.asarray(stable.vertices[0], dtype=np.float32).reshape(-1, 3)
        pts0 = evaluate_dense_vertices(sv0, f0, tmpl)
        pts = evaluate_dense_vertices(svt, f0, tmpl)
        nrm = evaluate_dense_normals(svt, f0, tmpl)
        # Region weights must be fixed to dense vertex ID. Recomputing by per-frame
        # bbox makes garment/hair shells slide on the body when the pose changes.
        region = soft_region_weights(pts0)
        garment_w = region["garment"] if getattr(cfg, "garment_shell_enabled", False) else np.zeros((len(pts),), dtype=np.float32)
        hair_w = region["hair"] if getattr(cfg, "hair_shell_enabled", False) else np.zeros((len(pts),), dtype=np.float32)
        shell_offsets = conservative_shell_offsets(
            pts,
            garment_w,
            hair_w,
            garment_offset=float(getattr(cfg, "garment_shell_offset", 0.006)),
            hair_offset=float(getattr(cfg, "hair_shell_offset", 0.010)),
        )
        pts = apply_shell_offsets(pts, nrm, shell_offsets)
        meta = {
            "method": "dense_mesh_shell_preview",
            "mesh_type": "detail",
            "dense_segments": segments,
            "dense_vertices": int(len(pts)),
            "garment_shell": bool(getattr(cfg, "garment_shell_enabled", False)),
            "hair_shell": bool(getattr(cfg, "hair_shell_enabled", False)),
            "mean_shell_offset": float(np.mean(np.abs(shell_offsets))) if len(shell_offsets) else 0.0,
        }
        return pts.astype(np.float32), np.asarray(tmpl.faces, dtype=np.int64).reshape(-1, 3), meta

    def _preview_final_pointcloud_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        vertices, faces, meta = self._preview_structure_geometry(apply_detail=True)
        sample_count = min(int(self._effective_pointcloud_max_points()), 24000)
        sample_count = max(2000, sample_count)
        spec = make_surface_sample_spec(vertices, faces, sample_count, seed=9100003)
        points = sample_mesh_surface_with_spec(vertices, faces, spec)
        out_meta = dict(meta)
        out_meta["method"] = "final_pointcloud_from_detail_mesh_preview"
        out_meta["sample_points"] = int(len(points))
        return points.astype(np.float32), vertices.astype(np.float32), faces.astype(np.int64), out_meta

    def _show_structure_preview_image(self, frame_bgr: np.ndarray) -> None:
        self.set_label_pixmap(self.preview_depth_label, frame_bgr)
        export_label = getattr(self, "export_preview_label", None)
        if export_label is not None:
            self.set_label_pixmap(export_label, frame_bgr)

    def _set_export_preview_status(self, text: str) -> None:
        label = getattr(self, "export_preview_status_label", None)
        if label is not None:
            label.setText(str(text or ""))

    def _set_preview_buttons_busy(self, busy: bool) -> None:
        for btn in (
            getattr(self, "preview_btn", None),
            getattr(self, "preview_structure_btn", None),
            getattr(self, "preview_4dhumans_btn", None),
            getattr(self, "preview_wham_btn", None),
            getattr(self, "preview_garment_btn", None),
            getattr(self, "preview_hair_btn", None),
            getattr(self, "preview_detail_btn", None),
            getattr(self, "preview_final_btn", None),
        ):
            if btn is not None:
                btn.setEnabled(not busy)
        try:
            if busy and not getattr(self, "_preview_wait_cursor_active", False):
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self._preview_wait_cursor_active = True
            elif not busy and getattr(self, "_preview_wait_cursor_active", False):
                QApplication.restoreOverrideCursor()
                self._preview_wait_cursor_active = False
        except Exception:
            self._preview_wait_cursor_active = False
        if not busy and hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def _on_mesh_layer_dependency_changed(self, *_args) -> None:
        try:
            if hasattr(self, "refresh_workflow_action_gates"):
                self.refresh_workflow_action_gates()
        except Exception:
            pass

    def _mesh_layer_preview_ready(self, mode: str, cfg: JobConfig | None = None) -> tuple[bool, str]:
        layer = str(mode or "").lower()
        if layer not in {"garment", "hair"}:
            return True, ""
        if not self._has_structure_cache():
            return False, "请先生成结构缓存。"
        # Layer switches and provider choices are hidden in the simplified workflow.
        # Garment / Hair are always default-on; availability is determined by the segmentation cache.
        try:
            state = self._current_layer_cache_state() if hasattr(self, "_current_layer_cache_state") else {"ready": False}
            if not bool(state.get("ready", False)):
                return False, str(state.get("message") or "还没有逐帧分割缓存。先点“生成分割缓存”，否则 Garment / Hair 会保持 Body Only。")
            found_model = str(state.get("model", "") or "").lower()
            if bool(state.get("needs_switch", False)) and found_model in {"4dhumans", "wham"}:
                if hasattr(self, "structure_solver_combo"):
                    self.structure_solver_combo.setCurrentText(self._structure_scheme_text(found_model))
                try:
                    if hasattr(self, "save_project_state"):
                        stack = getattr(self, "workflow_stack", None)
                        self.save_project_state(stack.currentIndex() if hasattr(stack, "currentIndex") else -1)
                except Exception:
                    pass
        except Exception:
            return False, "分割/壳层缓存状态不可读。请先重新生成分割缓存。"
        return True, ""

    def _show_mesh_layer_unavailable_preview(self, title: str, reason: str) -> None:
        canvas = np.zeros((620, 720, 3), dtype=np.uint8)
        canvas[:] = (18, 22, 28)
        cv2.putText(canvas, str(title or "Layer")[:42], (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (235, 240, 248), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Layer unavailable", (32, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210, 220, 235), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Need segmentation cache + shell enabled", (32, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 172, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Current output is Body Only", (32, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 172, 190), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (66, 230), (654, 552), (70, 80, 95), 1, cv2.LINE_AA)
        self._show_structure_preview_image(canvas)
        self.preview_status_label.setText(str(reason or "Garment / Hair 当前不可预览。"))
        self._set_export_preview_status(str(reason or "Garment / Hair 当前不可预览。"))

    def _start_mesh_preview(self, mode: str) -> None:
        if not self._has_structure_cache():
            if str(mode or "").lower() in {"garment", "hair", "detail", "combined", "pointcloud"}:
                if hasattr(self, "layer_mesh_preview_label"):
                    self.preview_depth_label = self.layer_mesh_preview_label
            elif hasattr(self, "structure_mesh_preview_label"):
                self.preview_depth_label = self.structure_mesh_preview_label
            self._show_mesh_layer_unavailable_preview("Mesh", "还没有可读结构缓存。请先在第 2 步生成 4D 或 WHAM 结构缓存。")
            return
        if self.thread is not None:
            self.preview_status_label.setText("正在导出，导出完成后再预览。")
            return
        if self.preview_thread is not None:
            self.preview_status_label.setText("预览正在进行，先等当前帧完成。")
            return
        try:
            cfg = self.make_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        mode_key = str(mode or "").lower()
        self._last_mesh_preview_mode = "combined" if mode_key == "detail" else (mode_key or "stable")
        if mode_key in {"stable", "body"} and hasattr(self, "structure_mesh_preview_label"):
            self.preview_depth_label = self.structure_mesh_preview_label
        elif mode_key in {"garment", "hair", "detail", "combined", "pointcloud"} and hasattr(self, "layer_mesh_preview_label"):
            self.preview_depth_label = self.layer_mesh_preview_label
        frame_index = int(self.preview_frame_spin.value()) if hasattr(self, "preview_frame_spin") else 0
        try:
            start_f, end_f = self._processing_range_values()
            clamped = max(int(start_f), min(int(frame_index), int(end_f)))
            if clamped != frame_index:
                frame_index = clamped
                self._apply_preview_frame_value(frame_index, refresh_mesh=False)
                self.preview_status_label.setText(f"当前帧不在处理范围内，已切到第 {frame_index} 帧。")
        except Exception:
            pass
        label = {"stable": "Body", "body": "Body", "garment": "Garment", "hair": "Hair", "detail": "Combined", "combined": "Combined", "pointcloud": "PointCloud"}.get(str(mode), "Mesh")
        if str(mode).lower() in {"garment", "hair"}:
            ready, reason = self._mesh_layer_preview_ready(str(mode), cfg)
            if not ready:
                self.log(f"{label} 预览未执行：{reason}")
                self._show_mesh_layer_unavailable_preview(label, reason)
                return
        self._set_preview_buttons_busy(True)
        self.preview_status_label.setText(f"正在后台生成{label}预览...")
        self._set_export_preview_status(f"正在后台生成{label}预览...")
        self.log(f"开始后台 Mesh 预览: mode={mode}, frame={frame_index}")

        self.preview_thread = QThread(self)
        self.preview_worker = MeshPreviewWorker(cfg, frame_index, mode)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.log.connect(self._on_worker_log_signal)
        self.preview_worker.finished.connect(self._on_mesh_preview_finished)
        self.preview_worker.failed.connect(self._on_mesh_preview_failed)
        self.preview_worker.finished.connect(self.preview_thread.quit)
        self.preview_worker.failed.connect(self.preview_thread.quit)
        self.preview_worker.finished.connect(self.preview_worker.deleteLater)
        self.preview_worker.failed.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self.cleanup_preview_thread)
        self.preview_thread.finished.connect(lambda th=self.preview_thread: QTimer.singleShot(0, th.deleteLater))
        self.preview_thread.start()

    def _on_mesh_preview_finished(self, mode: str, frame_bgr: object, status: str, elapsed: float, frame_index: int) -> None:
        try:
            if frame_index != int(self.preview_frame_spin.value()):
                self.preview_status_label.setText("旧 Mesh 预览已丢弃。当前帧已改变，请重新预览。")
                self._set_export_preview_status("旧 Mesh 预览已丢弃。")
                return
            self._show_structure_preview_image(np.asarray(frame_bgr, dtype=np.uint8))
            msg = f"{status} 用时 {elapsed:.2f}s。"
            self.preview_status_label.setText(msg)
            self._set_export_preview_status(msg)
        finally:
            self._set_preview_buttons_busy(False)

    def _on_mesh_preview_failed(self, mode: str, msg: str, frame_index: int) -> None:
        text = str(msg or "")
        lower = text.lower()
        expected_missing = (
            ("structure cache" in lower and ("不可读" in text or "没有可读" in text or "missing" in lower or "no readable" in lower))
            or "结构缓存没有可读帧" in text
            or "当前帧没有可读结构缓存" in text
            or "missing_structure_frame" in lower
        )
        if expected_missing:
            # Missing/partial cache is an expected preview state, not a crash.
            # Keep it in logs/status only and never show a warning dialog with a traceback.
            self.log("Mesh 预览未执行：" + short_error_message(text))
            self.preview_status_label.setText("当前帧没有可读结构缓存。")
            self._set_export_preview_status("当前帧没有可读结构缓存。")
            self._show_mesh_layer_unavailable_preview("Mesh", "当前帧没有可读结构缓存。请移动到处理范围内，或重新生成结构缓存。")
            self._set_preview_buttons_busy(False)
            return
        self.log("Mesh 预览失败:\n" + text)
        self.preview_status_label.setText("Mesh 预览失败。")
        self._set_export_preview_status("Mesh 预览失败。")
        self._set_preview_buttons_busy(False)
        QMessageBox.warning(self, APP_NAME, short_error_message(text))

    def preview_current_structure_frame(self) -> None:
        self._start_mesh_preview("stable")

    def preview_current_garment_frame(self) -> None:
        self._start_mesh_preview("garment")

    def preview_current_hair_frame(self) -> None:
        self._start_mesh_preview("hair")

    def preview_current_detail_frame(self) -> None:
        self._start_mesh_preview("combined")

    def preview_current_final_pointcloud_frame(self) -> None:
        self._start_mesh_preview("pointcloud")


    def render_preview_from_cache(self) -> None:
        """Structure-XYZ GUI has no depth render cache; refresh Alpha tiles only."""
        self._refresh_reference_preview_tiles()


    def _schedule_preview_render(self) -> None:
        if self._preview_render_busy:
            self._preview_render_pending = True
            return
        self._do_render_preview()

    def _do_render_preview(self) -> None:
        if self._preview_render_busy:
            self._preview_render_pending = True
            return
        self._preview_render_busy = True
        self._preview_render_pending = False
        try:
            if self.preview_original_bgr is not None:
                self._show_adjusted_original_preview()
            if self.preview_original_bgr is None or self.preview_depth is None:
                self.sync_levels_panel_from_controls()
                self._refresh_reference_preview_tiles()
                return
            if (self._effective_normal_strength() > 0 or self._effective_normal_refine() > 0) and self.preview_normal_map is None:
                self.preview_status_label.setText("当前主线不使用 法线。")
                self._refresh_reference_preview_tiles()
                return
            expected_key = self._preview_base_params()
            self._rebuild_preview_base_if_needed()
            self.sync_levels_panel_from_controls()
            base_gray = self.preview_hist_gray_cache if self.preview_hist_gray_cache is not None else self.preview_base_gray_cache
            if base_gray is None or self.preview_base_key != expected_key:
                self._refresh_reference_preview_tiles()
                return

            gray = apply_levels(
                base_gray,
                int(self.levels_in_black_spin.value()),
                int(self.levels_in_white_spin.value()),
                int(self.levels_out_black_spin.value()),
                int(self.levels_out_white_spin.value()),
            )
            gray = apply_curve_lut(gray, self.levels_panel.getCurvePoints())
            gray = apply_tone_ranges(
                gray,
                int(self.tone_black_spin.value()),
                int(self.tone_shadow_spin.value()),
                int(self.tone_mid_spin.value()),
                int(self.tone_light_spin.value()),
                int(self.tone_white_spin.value()),
                int(self.tone_black_shift_spin.value()),
                int(self.tone_shadow_shift_spin.value()),
                int(self.tone_mid_shift_spin.value()),
                int(self.tone_light_shift_spin.value()),
                int(self.tone_white_shift_spin.value()),
                int(self.tone_black_contrast_spin.value()),
                int(self.tone_shadow_contrast_spin.value()),
                int(self.tone_mid_contrast_spin.value()),
                int(self.tone_light_contrast_spin.value()),
                int(self.tone_white_contrast_spin.value()),
            )
            render_mask = self._effective_preview_subject_mask_for_shape(np.asarray(gray).shape[:2])
            gray = apply_subject_background_fill(
                gray,
                render_mask,
                self.background_mode_combo.currentText(),
                int(self.background_gray_spin.value()),
            )
            gray = apply_anti_banding(gray, int(self.anti_banding_spin.value()))
            # Waveform follows the rendered final depth result.
            self.levels_panel.setHistogramFromGray(gray, render_mask)
            if self.color_combo.currentText() == "伪彩色":
                depth_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            else:
                depth_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.preview_depth_render_bgr = depth_bgr
            self.set_label_pixmap(self.preview_depth_label, depth_bgr)
            self._refresh_reference_preview_tiles()
            self.preview_big_btn.setEnabled(True)
        except Exception as exc:  # noqa: BLE001
            self.preview_status_label.setText(f"预览失败: {exc}")
            self.log(f"预览失败: {exc}")
        finally:
            self._preview_render_busy = False
            if self._preview_render_pending:
                self._preview_render_pending = False
                self._preview_debounce.start()

    def open_single_reference_preview(self, key: str, title: str) -> None:
        frame_bgr = self._reference_preview_tile_bgr.get(key)
        if frame_bgr is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            dialog.resize(min(980, avail.width() - 80), min(880, avail.height() - 80))
        else:
            dialog.resize(900, 760)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        preview = PreviewImageLabel()
        preview.setObjectName("previewImage")
        preview.setImagePixmap(bgr_to_pixmap(frame_bgr))
        preview.setOverlayText(title)
        preview.setMinimumSize(360, 260)
        preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview.double_clicked.connect(preview.resetView)
        layout.addWidget(preview, 1)
        dialog.exec()

    def open_large_preview(self) -> None:
        if self.preview_original_render_bgr is None:
            return
        original_bgr = self.preview_original_render_bgr
        alpha_bgr = self._reference_preview_tile_bgr.get("subject_alpha")

        dialog = QDialog(self)
        dialog.setWindowTitle("Alpha 检查大图")
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            dialog.resize(min(1280, avail.width() - 80), min(900, avail.height() - 80))
        else:
            dialog.resize(1100, 760)
        layout = QHBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        left = PreviewImageLabel()
        right = PreviewImageLabel()
        left.setObjectName("previewImage")
        right.setObjectName("previewImage")
        left.setImagePixmap(bgr_to_pixmap(original_bgr))
        left.setOverlayText("原视频 RGB")
        if alpha_bgr is not None:
            right.setImagePixmap(bgr_to_pixmap(alpha_bgr))
            right.setOverlayText("原视频 Alpha")
        else:
            right.clearImage("原视频 Alpha 未读取")
        for label in (left, right):
            label.setMinimumSize(320, 240)
            label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label.double_clicked.connect(label.resetView)
        layout.addWidget(left, 1)
        layout.addWidget(right, 1)
        dialog.exec()


    def start_job(self) -> None:
        if self.thread is not None:
            QMessageBox.warning(self, APP_NAME, "导出任务正在进行，请等待完成或先取消。")
            return
        try:
            output_selected = bool(self.mesh_export_check.isChecked() or self.detail_mesh_export_check.isChecked() or self.pointcloud_usd_check.isChecked())
        except Exception:
            output_selected = True
        if not output_selected:
            try:
                self.preview_status_label.setText("至少选择一个输出项：低模 Mesh、细节 Mesh 或稳定点云。")
            except Exception:
                pass
            QMessageBox.warning(self, APP_NAME, "至少选择一个输出项。")
            return
        direct_depth_mode = self._direct_depth_input_ready()
        if self.preload_thread is not None and not direct_depth_mode:
            self._queue_model_action("export")
            return
        if self.preview_thread is not None:
            QMessageBox.warning(self, APP_NAME, "预览推理正在进行，请等待完成后再开始。")
            return
        if self._base_rebuild_thread is not None:
            QMessageBox.warning(self, APP_NAME, "深度底图正在更新，请等待完成后再开始。")
            return
        try:
            cfg = self.make_config()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if not self._has_structure_cache():
            msg = "请先在第 2 步生成当前方案和当前范围的人体结构缓存。"
            self.preview_status_label.setText(msg)
            self._set_export_preview_status(msg)
            QMessageBox.information(self, APP_NAME, msg)
            if hasattr(self, "refresh_workflow_action_gates"):
                self.refresh_workflow_action_gates()
            return
        detail_selected = bool(getattr(self, "detail_mesh_export_check", None) is not None and self.detail_mesh_export_check.isChecked())
        if detail_selected:
            try:
                layer_ready = bool(self._current_layer_cache_state().get("ready", False))
            except Exception:
                layer_ready = False
            if not layer_ready:
                msg = "Body / Garment / Hair / Combined 导出需要先在第 3 步生成分割缓存。请先生成分割缓存，或取消该输出项。"
                self.preview_status_label.setText(msg)
                self._set_export_preview_status(msg)
                QMessageBox.information(self, APP_NAME, msg)
                if hasattr(self, "refresh_workflow_action_gates"):
                    self.refresh_workflow_action_gates()
                return
        self.log("点击导出：已进入导出流程。")
        self.preview_status_label.setText("正在准备导出...")
        if not self._confirm_resource_risk(cfg):
            self.preview_status_label.setText("导出已取消。")
            return
        self._current_output_path = cfg.output_path
        remove_partial_output(cfg.output_path)
        self._set_model_config_controls_enabled(False)
        self.preview_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()
        self.progress.setValue(0)
        self.progress.setFormat("准备开始")
        self.stage_status_label.setText("阶段：准备开始")
        self.job_started_at = time.time()
        self._eta_started_at = None
        self._eta_started_done = 0
        self.log("开始任务")

        self.thread = QThread(self)
        self.worker = MeshExportWorker(cfg) if is_structure_xyz_export_config(cfg) else DepthExportWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self._on_worker_log_signal)
        self.worker.stage_signal.connect(self.on_stage_changed)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.finished.connect(lambda th=self.thread: QTimer.singleShot(0, th.deleteLater))
        self.thread.start()

    def cancel_job(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.stage_status_label.setText("阶段：正在取消")
            self.log("正在取消...")

    def on_progress(self, done: int, total: int) -> None:
        elapsed = time.time() - self.job_started_at if self.job_started_at else 0.0
        stage_text = ""
        if hasattr(self, "stage_status_label"):
            stage_text = self.stage_status_label.text().replace("阶段:", "").replace("阶段：", "").strip()
        prefix = (stage_text + " | ") if stage_text else ""
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
            now = time.time()
            # Ignore model warmup and first encoded frame, otherwise ETA starts wildly high.
            if done <= 1 or self._eta_started_at is None or done < self._eta_started_done:
                self._eta_started_at = now
                self._eta_started_done = max(0, int(done))
                self.progress.setFormat(f"{prefix}{done}/{total} | 计算速度中 | 已用 {format_seconds(elapsed)}")
                return
            work_done = max(0, int(done) - self._eta_started_done)
            span = max(1e-6, now - self._eta_started_at)
            fps = work_done / span if work_done > 0 else 0.0
            eta = (total - done) / fps if fps > 0 else 0.0
            self.progress.setFormat(f"{prefix}{done}/{total} | {fps:.1f} fps | 已用 {format_seconds(elapsed)} | 剩余 {format_seconds(eta)}")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(f"{prefix}运行中 | 已用 {format_seconds(elapsed)}")

    def on_finished(self, output_path: str) -> None:
        elapsed = time.time() - self.job_started_at if self.job_started_at else 0.0
        self._current_output_path = None
        self.stage_status_label.setText("阶段：完成")
        self.log(f"输出完成: {output_path}")
        self.log(f"总用时: {format_seconds(elapsed)}")
        audio_note = ""
        if self.copy_audio_check.isChecked() and shutil.which("ffmpeg") is None:
            audio_note = "\n\n注意：未检测到 ffmpeg，本次输出没有合并原视频音频。"
        QMessageBox.information(self, APP_NAME, f"完成\n用时: {format_seconds(elapsed)}\n{output_path}{audio_note}")
        self._set_model_config_controls_enabled(True)
        self.preview_btn.setEnabled(bool(self.current_input))
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if hasattr(self, "stage_status_label"):
            self.stage_status_label.setText("阶段：空闲")
        if hasattr(self, "progress"):
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat("未开始")
        if hasattr(self, "refresh_workflow_action_gates"):
            self.refresh_workflow_action_gates()

    def on_failed(self, msg: str) -> None:
        self.stage_status_label.setText("阶段：失败")
        self.log("失败:\n" + msg)
        # Delete incomplete output file if job was cancelled mid-write
        if "任务已取消" in msg:
            out = getattr(self, "_current_output_path", None)
            if out:
                remove_partial_output(out)
                p = Path(out)
                if p.exists():
                    try:
                        p.unlink()
                        self.log(f"已删除不完整输出: {out}")
                    except OSError:
                        pass
            self._current_output_path = None
            self.stage_status_label.setText("阶段：已取消")
            self._set_model_config_controls_enabled(True)
            self.preview_btn.setEnabled(bool(self.current_input))
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            if hasattr(self, "refresh_workflow_action_gates"):
                self.refresh_workflow_action_gates()
            return
        out = getattr(self, "_current_output_path", None)
        if out:
            remove_partial_output(out)
            self._current_output_path = None
        short = short_error_message(msg)
        if "显存不足" in short:
            clear_memory_model_cache(self.log)
            reply = QMessageBox.question(
                self,
                APP_NAME,
                short + "\n\n已自动清理内存模型缓存。是否把 process_res、输出长边和批量帧数降到更稳妥的值？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.batch_spin.setValue(1)
                self.process_res_spin.setValue(min(int(self.process_res_spin.value()), SAFE_DEFAULT_PROCESS_RES))
                self.long_side_spin.setValue(min(int(self.long_side_spin.value()), SAFE_DEFAULT_LONG_SIDE))
                self.log("OOM 后已自动降参：batch=1，process_res/输出长边降到安全默认上限。")
        else:
            QMessageBox.critical(self, APP_NAME, short)
        self._set_model_config_controls_enabled(True)
        self.preview_btn.setEnabled(bool(self.current_input))
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def cleanup_thread(self) -> None:
        self.worker = None
        self.thread = None

    def closeEvent(self, event) -> None:  # noqa: ANN001
        running = any([
            self.thread is not None,
            self.preview_thread is not None,
            self.preload_thread is not None,
            self.original_frame_thread is not None,
            self._base_rebuild_thread is not None,
        ])
        if running:
            if self.worker:
                self.worker.cancel()
            self.preview_status_label.setText("后台任务还在结束中，请稍后再关闭。")
            self.log("后台任务还在结束中，已请求取消。")
            event.ignore()
            return
        try:
            remove_event_listener(self._event_console_listener)
            self._event_console_listener_active = False
        except Exception:
            pass
            
        if hasattr(self, "save_project_state"):
            self.save_project_state()
            
        super().closeEvent(event)


def main() -> int:
    install_global_event_hooks()
    install_stdio_event_tee()
    event_log("应用启动", channel="APP")
    app = QApplication(sys.argv)
    
    from components.project_manager_ui import ProjectManagerDialog
    pm_dlg = ProjectManagerDialog()
    if pm_dlg.exec() != QDialog.Accepted or not pm_dlg.selected_project_dir:
        return 0
        
    win = MainWindow()
    selected_project_dir = pm_dlg.selected_project_dir
    win.show()
    # Show the main window first. Project loading may probe video metadata and
    # restore UI state; doing that before show() made startup feel frozen.
    QTimer.singleShot(0, lambda: win.load_project(selected_project_dir))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
