# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CACHE_STATE_VERSION = "v9-geometry-cache-state"
CACHE_KINDS = ("depth", "alpha", "normal")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
        os.replace(str(tmp), str(path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return {}


def _file_token(path_text: str | Path | None) -> str:
    if not path_text:
        return ""
    try:
        path = Path(path_text)
        if not path.exists():
            return str(path)
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    except Exception:
        return str(path_text)


def _cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(cfg, name, default)
    except Exception:
        return default


def _signature(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default).encode("utf-8", errors="ignore")
    return hashlib.sha1(blob).hexdigest()[:20]


def depth_cache_signature(cfg: Any) -> str:
    return _signature({
        "version": CACHE_STATE_VERSION,
        "kind": "depth",
        "model_id": _cfg_value(cfg, "model_id", ""),
        "device_mode": _cfg_value(cfg, "device_mode", ""),
        "process_res": int(_cfg_value(cfg, "process_res", 0) or 0),
        "input_brightness": int(_cfg_value(cfg, "input_brightness", 0) or 0),
        "input_contrast": int(_cfg_value(cfg, "input_contrast", 0) or 0),
        "input_gamma": float(_cfg_value(cfg, "input_gamma", 1.0) or 1.0),
        "input_shadow": int(_cfg_value(cfg, "input_shadow", 0) or 0),
        "input_highlight": int(_cfg_value(cfg, "input_highlight", 0) or 0),
        "input_sharpen": int(_cfg_value(cfg, "input_sharpen", 0) or 0),
        "input_denoise": int(_cfg_value(cfg, "input_denoise", 0) or 0),
    })


def alpha_cache_signature(cfg: Any) -> str:
    return _signature({
        "version": CACHE_STATE_VERSION,
        "kind": "alpha",
        "matting_enabled": bool(_cfg_value(cfg, "matting_enabled", False)),
        "matting_model": _file_token(_cfg_value(cfg, "matting_model_path", "")),
        "matting_mask": _file_token(_cfg_value(cfg, "matting_mask_path", "")),
        "matting_max_size": int(_cfg_value(cfg, "matting_max_size", 0) or 0),
        "external_mask_enabled": bool(_cfg_value(cfg, "external_mask_enabled", False)),
        "external_mask": _file_token(_cfg_value(cfg, "external_mask_path", "")),
        "external_mask_invert": bool(_cfg_value(cfg, "external_mask_invert", False)),
        "external_mask_frame_offset": int(_cfg_value(cfg, "external_mask_frame_offset", 0) or 0),
        "input_cutout_mask_enabled": bool(_cfg_value(cfg, "input_cutout_mask_enabled", False)),
        "input_video_for_alpha": _file_token(_cfg_value(cfg, "input_path", "")) if bool(_cfg_value(cfg, "input_cutout_mask_enabled", False)) else "",
        "alpha_decoder": "real_container_alpha_v1",
        "auto_mask_feather_px": int(_cfg_value(cfg, "auto_mask_feather_px", 0) or 0),
        "auto_mask_expand_px": int(_cfg_value(cfg, "auto_mask_expand_px", 0) or 0),
        "fallback": "depth_subject_mask",
    })


def normal_cache_signature(cfg: Any) -> str:
    return _signature({
        "version": CACHE_STATE_VERSION,
        "kind": "normal",
        "device_mode": _cfg_value(cfg, "device_mode", ""),
        "process_res": int(_cfg_value(cfg, "process_res", 0) or 0),
        "input_brightness": int(_cfg_value(cfg, "input_brightness", 0) or 0),
        "input_contrast": int(_cfg_value(cfg, "input_contrast", 0) or 0),
        "input_gamma": float(_cfg_value(cfg, "input_gamma", 1.0) or 1.0),
        "input_shadow": int(_cfg_value(cfg, "input_shadow", 0) or 0),
        "input_highlight": int(_cfg_value(cfg, "input_highlight", 0) or 0),
        "input_sharpen": int(_cfg_value(cfg, "input_sharpen", 0) or 0),
        "input_denoise": int(_cfg_value(cfg, "input_denoise", 0) or 0),
        "normal_model": "sapiens-normal-0.3b",
    })


def cache_signatures_for_job(cfg: Any) -> dict[str, str]:
    return {
        "depth": depth_cache_signature(cfg),
        "alpha": alpha_cache_signature(cfg),
        "normal": normal_cache_signature(cfg),
    }


def cache_meta_dir(cache_root: str | Path) -> Path:
    return Path(cache_root) / "meta"


def cache_manifest_path(cache_root: str | Path, kind: str) -> Path:
    return cache_meta_dir(cache_root) / f"{kind}_manifest.json"


def pipeline_state_path(cache_root: str | Path) -> Path:
    return cache_meta_dir(cache_root) / "pipeline_state.json"


def cache_validation_path(cache_root: str | Path) -> Path:
    return cache_meta_dir(cache_root) / "cache_validation.json"


def _frame_key(frame_index: int) -> str:
    return f"{int(frame_index):08d}"


def _array_stats(arr: np.ndarray | None) -> dict[str, Any]:
    if arr is None:
        return {}
    try:
        a = np.asarray(arr)
        finite = np.isfinite(a)
        finite_count = int(np.count_nonzero(finite))
        payload: dict[str, Any] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "finite_count": finite_count,
        }
        if finite_count > 0:
            vals = a[finite]
            payload.update({
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
            })
        return payload
    except Exception:
        return {}


def _file_info(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {"path": str(path), "size": int(stat.st_size), "mtime": int(stat.st_mtime)}
    except Exception:
        return {"path": str(path), "size": 0, "mtime": 0}


def load_cache_manifest(cache_root: str | Path, kind: str) -> dict[str, Any]:
    data = _read_json(cache_manifest_path(cache_root, kind))
    if not data:
        data = {"version": CACHE_STATE_VERSION, "kind": kind, "signature": "", "frames": {}}
    if not isinstance(data.get("frames"), dict):
        data["frames"] = {}
    return data


def write_pipeline_state(cache_root: str | Path, cfg: Any, total_frames: int | None = None, status: str = "running", extra: dict[str, Any] | None = None) -> None:
    root = Path(cache_root)
    payload = _read_json(pipeline_state_path(root))
    payload.update({
        "version": CACHE_STATE_VERSION,
        "status": status,
        "cache_root": str(root),
        "input_path": str(_cfg_value(cfg, "input_path", "")),
        "total_frames": int(total_frames) if total_frames is not None else payload.get("total_frames"),
        "signatures": cache_signatures_for_job(cfg),
    })
    if extra:
        payload.update(extra)
    _atomic_write_json(pipeline_state_path(root), payload)


def record_cache_frame(
    cache_root: str | Path,
    kind: str,
    frame_index: int,
    path: str | Path,
    signature: str,
    arr: np.ndarray | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> None:
    root = Path(cache_root)
    path_obj = Path(path)
    data = load_cache_manifest(root, kind)
    data.update({"version": CACHE_STATE_VERSION, "kind": kind, "signature": signature})
    frames = data.setdefault("frames", {})
    entry: dict[str, Any] = {
        "index": int(frame_index),
        "status": status,
        "signature": signature,
        "file": _file_info(path_obj),
    }
    entry.update(_array_stats(arr))
    if extra:
        entry.update(extra)
    frames[_frame_key(frame_index)] = entry
    _atomic_write_json(cache_manifest_path(root, kind), data)


def record_cache_error(cache_root: str | Path, kind: str, frame_index: int, signature: str, message: str) -> None:
    data = load_cache_manifest(cache_root, kind)
    data.update({"version": CACHE_STATE_VERSION, "kind": kind, "signature": signature})
    frames = data.setdefault("frames", {})
    frames[_frame_key(frame_index)] = {
        "index": int(frame_index),
        "status": "error",
        "signature": signature,
        "error": str(message),
    }
    _atomic_write_json(cache_manifest_path(cache_root, kind), data)


def cache_entry_matches(cache_root: str | Path, kind: str, frame_index: int, path: str | Path, signature: str, allow_legacy: bool = True) -> bool:
    path_obj = Path(path)
    if not path_obj.is_file():
        return False
    data = load_cache_manifest(cache_root, kind)
    entry = data.get("frames", {}).get(_frame_key(frame_index))
    if not entry:
        return bool(allow_legacy)
    if entry.get("status") != "ok":
        return False
    if str(entry.get("signature", "")) != str(signature):
        return False
    try:
        if int(entry.get("file", {}).get("size", 0)) <= 0:
            return False
    except Exception:
        return False
    return True


def validate_cache_file(path: str | Path, kind: str) -> tuple[bool, str, dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.is_file():
        return False, "missing", {}
    try:
        arr = np.load(str(path_obj), allow_pickle=False, mmap_mode="r")
        shape = tuple(int(v) for v in arr.shape)
        if arr.size <= 0:
            return False, "empty", {"shape": list(shape), "dtype": str(arr.dtype)}
        if kind in {"depth", "alpha"} and len(shape) < 2:
            return False, "bad_shape", {"shape": list(shape), "dtype": str(arr.dtype)}
        if kind == "normal" and (len(shape) != 3 or int(shape[-1]) != 3):
            return False, "bad_shape", {"shape": list(shape), "dtype": str(arr.dtype)}
        return True, "ok", {"shape": list(shape), "dtype": str(arr.dtype), "size": int(arr.size)}
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable:{exc}", {}


def validate_cache_kind(cache_root: str | Path, kind: str, expected_frames: int, signature: str, path_builder: Any) -> dict[str, Any]:
    root = Path(cache_root)
    missing: list[int] = []
    stale: list[int] = []
    unreadable: list[dict[str, Any]] = []
    ok_count = 0
    manifest = load_cache_manifest(root, kind)
    frames = manifest.get("frames", {}) if isinstance(manifest.get("frames"), dict) else {}
    for idx in range(max(0, int(expected_frames))):
        path = Path(path_builder(root, idx))
        if not path.is_file():
            missing.append(idx)
            continue
        entry = frames.get(_frame_key(idx))
        if entry and str(entry.get("signature", "")) != str(signature):
            stale.append(idx)
            continue
        ok, reason, info = validate_cache_file(path, kind)
        if ok:
            ok_count += 1
        else:
            unreadable.append({"frame": idx, "reason": reason, **info})
    return {
        "kind": kind,
        "expected_frames": int(expected_frames),
        "ok_count": ok_count,
        "missing_frames": missing[:200],
        "stale_frames": stale[:200],
        "unreadable_frames": unreadable[:100],
        "missing_count": len(missing),
        "stale_count": len(stale),
        "unreadable_count": len(unreadable),
        "signature": signature,
    }


def validate_geometry_cache(
    cache_root: str | Path,
    cfg: Any,
    expected_frames: int,
    frame_depth_path_builder: Any,
    frame_alpha_path_builder: Any,
    frame_normal_path_builder: Any,
    include_alpha: bool = False,
    include_normal: bool = False,
) -> dict[str, Any]:
    sigs = cache_signatures_for_job(cfg)
    kinds = {
        "depth": (sigs["depth"], frame_depth_path_builder, True),
        "alpha": (sigs["alpha"], frame_alpha_path_builder, bool(include_alpha)),
        "normal": (sigs["normal"], frame_normal_path_builder, bool(include_normal)),
    }
    result: dict[str, Any] = {
        "version": CACHE_STATE_VERSION,
        "expected_frames": int(expected_frames),
        "signatures": sigs,
        "kinds": {},
        "status": "green",
    }
    for kind, (sig, builder, enabled) in kinds.items():
        if not enabled:
            result["kinds"][kind] = {"kind": kind, "enabled": False}
            continue
        info = validate_cache_kind(cache_root, kind, expected_frames, sig, builder)
        info["enabled"] = True
        result["kinds"][kind] = info
        if info["missing_count"] or info["stale_count"] or info["unreadable_count"]:
            result["status"] = "yellow" if kind != "depth" else "red"
    _atomic_write_json(cache_validation_path(cache_root), result)
    return result


def summarize_cache_validation(validation: dict[str, Any]) -> str:
    parts: list[str] = []
    kinds = validation.get("kinds", {}) if isinstance(validation.get("kinds"), dict) else {}
    for kind in CACHE_KINDS:
        info = kinds.get(kind, {})
        if not info or not info.get("enabled", False):
            continue
        parts.append(
            f"{kind}: ok={int(info.get('ok_count', 0))}/{int(info.get('expected_frames', 0))}, "
            f"missing={int(info.get('missing_count', 0))}, stale={int(info.get('stale_count', 0))}, bad={int(info.get('unreadable_count', 0))}"
        )
    return "; ".join(parts)
