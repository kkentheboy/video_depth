# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

_FRAME_RE = re.compile(r"frame_(\d+)")


def _as_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    try:
        import numpy as np  # type: ignore
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def pointcloud_config_payload(cfg: Any) -> dict[str, Any]:
    """Return only fields that affect pointcloud geometry/output validity.

    UI-only/video-encoder fields are intentionally excluded so cached PLY frames
    are reusable when only the debug depth video settings change.
    """
    keys = [
        "pointcloud_mode",
        "pointcloud_density",
        "pointcloud_stride",
        "pointcloud_max_points",
        "pointcloud_alpha_threshold",
        "pointcloud_depth_near_percentile",
        "pointcloud_depth_far_percentile",
        "pointcloud_color_mode",
        "external_mask_enabled",
        "external_mask_path",
        "external_mask_invert",
        "input_cutout_mask_enabled",
        "pointcloud_coordinate_mode",
        "pointcloud_z_near",
        "pointcloud_z_far",
        "pointcloud_alpha_erode_px",
        "pointcloud_alpha_dilate_px",
        "pointcloud_alpha_feather_px",
        "pointcloud_body_bbox_margin_px",
        "pointcloud_remove_outliers",
        "pointcloud_outlier_sigma",
        "pointcloud_voxel_downsample",
        "pointcloud_voxel_size",
        "pointcloud_temporal_depth_smooth",
        "pointcloud_temporal_center_smooth",
        "pointcloud_temporal_scale_smooth",
        # Direct Depth geometry depends on the authored depth-video range and grading stack.
        "external_depth_path",
        "external_depth_weight",
        "external_depth_invert",
        "external_depth_orientation_mode",
        "pointcloud_structure_algorithm",
        "normalize_mode",
        "invert",
        "black_pct",
        "white_pct",
        "gamma",
        "detail_boost",
        "depth_smooth",
        "edge_preserve",
        "levels_in_black",
        "levels_in_white",
        "levels_out_black",
        "levels_out_white",
        "curve_points",
        "tone_black",
        "tone_shadow",
        "tone_mid",
        "tone_light",
        "tone_white",
        "tone_black_shift",
        "tone_shadow_shift",
        "tone_mid_shift",
        "tone_light_shift",
        "tone_white_shift",
        "tone_black_contrast",
        "tone_shadow_contrast",
        "tone_mid_contrast",
        "tone_light_contrast",
        "tone_white_contrast",
        "pointcloud_template_sample_ratio",
        "pointcloud_template_confidence",
        "pointcloud_template_align_strength",
        "pointcloud_hand_enabled",
        "pointcloud_hand_sample_ratio",
        "pointcloud_hand_confidence",
        "mesh_export_enabled",
        "detail_mesh_export_enabled",
        "mesh_dense_segments",
        "garment_shell_enabled",
        "garment_shell_offset",
        "hair_shell_enabled",
        "hair_shell_offset",
        "structure_model",
        "hand_model",
        "occlusion_fill_enabled",
    ]
    payload = {k: _as_plain(getattr(cfg, k, None)) for k in keys}
    payload["pointcloud_structure_algorithm"] = "stable_mesh_shell_no_depth_v2"
    payload["input_path"] = _as_plain(getattr(cfg, "input_path", ""))
    # Include file stamps so resume will not reuse point clouds when the same path
    # was overwritten with different source/depth media.
    for _key in ("input_path", "external_depth_path", "external_mask_path"):
        _path_text = str(getattr(cfg, _key, "") or "")
        try:
            _p = Path(_path_text)
            if _p.is_file():
                _st = _p.stat()
                payload[f"{_key}_size"] = int(_st.st_size)
                payload[f"{_key}_mtime_ns"] = int(getattr(_st, "st_mtime_ns", int(_st.st_mtime * 1_000_000_000)))
        except Exception:
            pass

    # Structure XYZ output depends on generated structure cache. Include a cheap
    # aggregate stamp so resume cannot reuse points from an older body cache.
    try:
        from depth_fusion_core import structure_cache_root  # local import avoids startup cycle
        structure_root = Path(structure_cache_root(cfg)) / "structure"
        if structure_root.exists():
            newest = 0
            total_size = 0
            count = 0
            for _p in structure_root.glob("frame_*_smpl_*.npy"):
                try:
                    _st = _p.stat()
                    newest = max(newest, int(getattr(_st, "st_mtime_ns", int(_st.st_mtime * 1_000_000_000))))
                    total_size += int(_st.st_size)
                    count += 1
                except Exception:
                    pass
            payload["structure_cache_file_count"] = int(count)
            payload["structure_cache_total_size"] = int(total_size)
            payload["structure_cache_newest_mtime_ns"] = int(newest)
    except Exception:
        pass
    return payload


def pointcloud_config_signature(cfg: Any) -> str:
    raw = json.dumps(pointcloud_config_payload(cfg), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def write_export_state(root: Path, *, status: str, config_signature: str, **fields: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": str(status),
        "config_signature": str(config_signature),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload.update({str(k): _as_plain(v) for k, v in fields.items()})
    with open(root / "pointcloud_export_state.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_frame_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        if not meta_path.is_file():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _file_ok(path: Path | None) -> bool:
    try:
        return path is not None and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def existing_frame_is_complete(
    *,
    meta_path: Path,
    config_signature: str,
    require_fused: bool,
    require_obj_visible: Path | None = None,
    require_obj_fused: Path | None = None,
) -> tuple[bool, dict[str, Any] | None, str]:
    meta = read_frame_meta(meta_path)
    if not meta:
        return False, None, "missing_meta"
    if str(meta.get("config_signature", "")) != str(config_signature):
        return False, meta, "signature_changed"
    # Final workflow no longer validates per-frame PLY/OBJ debug files.
    if require_obj_visible is not None and not _file_ok(require_obj_visible):
        return False, meta, "missing_visible_obj"
    if require_obj_fused is not None and require_fused and not _file_ok(require_obj_fused):
        return False, meta, "missing_fused_obj"
    return True, meta, "ok"


def frame_number_from_path(path: Path) -> int | None:
    m = _FRAME_RE.search(path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def cleanup_stale_output_frames(paths: list[Path | None], *, total_frames: int) -> int:
    """Remove frame files beyond current video length.

    This keeps resume mode safe when a shorter video is exported into a reused
    output folder. It never removes files if total_frames is unknown/invalid.
    """
    if total_frames <= 0:
        return 0
    removed = 0
    for root in paths:
        if root is None or not root.exists() or not root.is_dir():
            continue
        for pattern in ("frame_*.json",):
            for path in root.glob(pattern):
                n = frame_number_from_path(path)
                if n is None:
                    continue
                # frame_000001 maps to render_index 0, so valid max is total_frames.
                if n > total_frames:
                    try:
                        path.unlink()
                        removed += 1
                    except Exception:
                        pass
    return removed


def clean_output_dir_frames(paths: list[Path | None]) -> int:
    removed = 0
    for root in paths:
        if root is None or not root.exists() or not root.is_dir():
            continue
        for pattern in ("frame_*.json",):
            for path in root.glob(pattern):
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


def validate_pointcloud_outputs(
    *,
    visible_dir: Path | None,
    fused_dir: Path | None,
    meta_dir: Path | None,
    expected_frames: int,
    require_fused: bool,
) -> dict[str, Any]:
    def _collect_meta(root: Path | None) -> set[int]:
        if root is None or not root.exists():
            return set()
        values: set[int] = set()
        for path in root.glob("frame_*.json"):
            n = frame_number_from_path(path)
            if n is not None:
                values.add(n)
        return values

    expected = set(range(1, max(0, int(expected_frames)) + 1)) if expected_frames > 0 else set()
    meta = _collect_meta(meta_dir)
    missing_meta = sorted(expected - meta) if expected else []
    ok = not missing_meta
    return {
        "ok": bool(ok),
        "expected_frames": int(expected_frames),
        "frame_meta_count": int(len(meta)),
        "missing_meta_frames": missing_meta[:80],
        "missing_truncated": bool(len(missing_meta) > 80),
        "note": "Final source validates USDA/meta only; PLY/OBJ/ABC debug outputs were removed.",
    }
