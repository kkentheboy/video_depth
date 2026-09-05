# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cache_adapter import normalize_faces, write_external_structure_cache


VERTEX_KEYS = (
    "vertices", "verts", "smpl_vertices", "pred_vertices", "pred_verts",
    "mesh_vertices", "vertices_world", "verts_world", "world_vertices",
)
FACE_KEYS = ("faces", "smpl_faces", "triangles", "f")
JOINT_KEYS = ("joints", "smpl_joints", "pred_joints", "joints3d", "joints_world")
CAMERA_KEYS = (
    "camera", "cam", "pred_cam", "camera_translation", "camera_trans",
    "root_translation", "root_trans", "world_translation", "trans", "translation",
    "global_orient", "pose", "betas",
)


def frame_index_from_name(path: Path, fallback: int = 0) -> int:
    m = re.search(r"(\d+)", path.stem)
    if not m:
        return int(fallback)
    value = int(m.group(1))
    # Accept both frame_000000 and frame_000001 conventions.
    return max(0, value - 1 if value >= 1 else value)


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return arr


def _first_mapping_value(data: dict[str, Any], keys: Iterable[str]) -> Any:
    lower = {str(k).lower(): k for k in data.keys()}
    for key in keys:
        real = lower.get(key.lower())
        if real is not None:
            return data[real]
    return None


def _load_pickle(path: Path) -> Any:
    # Try standard pickle first; fall back to joblib which handles
    # compressed pickles written by joblib.dump (used by WHAM).
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception:
        pass
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass
    # Last resort: re-raise with standard pickle for a clear error.
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _load_any_mapping(path: Path) -> dict[str, Any] | list[Any] | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(path, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}
        if suffix in {".pkl", ".pickle"} or path.name.lower().endswith(".pkl"):
            data = _load_pickle(path)
            return data
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _dicts_from_loaded(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, dict):
        # Some exporters store per-frame predictions as a list under a key.
        for key in ("predictions", "frames", "results", "outputs", "sequence"):
            value = data.get(key)
            if isinstance(value, list):
                return [v for v in value if isinstance(v, dict)]
        # WHAM stores results as {track_id(int): {verts, pose, ...}, ...}.
        # When all top-level keys are integers and values are dicts, unwrap
        # the per-track dicts so the vertex finder can inspect them.
        # Track IDs may be numpy.int32 which is not a subclass of Python int.
        if data and all(isinstance(k, (int, np.integer)) for k in data.keys()):
            sub = [v for v in data.values() if isinstance(v, dict)]
            if sub:
                return sub
        return [data]
    if isinstance(data, list):
        return [v for v in data if isinstance(v, dict)]
    return []


def _split_frame_arrays(vertices: np.ndarray, faces: np.ndarray | None, joints: np.ndarray | None, camera_payload: dict[str, Any]) -> list[tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]]:
    verts = np.asarray(vertices)
    if verts.ndim == 2 and verts.shape[-1] == 3:
        return [(0, verts, faces, joints, camera_payload)]
    if verts.ndim == 3 and verts.shape[-1] == 3:
        out = []
        for i in range(verts.shape[0]):
            j = None
            if joints is not None:
                jj = np.asarray(joints)
                if jj.ndim == 3 and jj.shape[0] == verts.shape[0]:
                    j = jj[i]
                elif jj.ndim == 2:
                    j = jj
            cam = dict(camera_payload)
            cam["sequence_index"] = int(i)
            out.append((i, verts[i], faces, j, cam))
        return out
    return []


def _camera_payload(data: dict[str, Any], path: Path, model_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"source_path": str(path), "importer": "external_output_importer"}
    lower_model = model_name.lower()
    if "wham" in lower_model:
        payload.update({"coordinate_space": "wham_world", "has_world_trajectory": True})
    elif "4d" in lower_model or "hmr" in lower_model:
        payload.update({"coordinate_space": "smpl_camera_or_local", "has_world_trajectory": False})

    for key in CAMERA_KEYS:
        value = _first_mapping_value(data, [key])
        if value is None:
            continue
        try:
            arr = np.asarray(value)
            if arr.size <= 128:
                payload[key] = arr.tolist()
            else:
                payload[key + "_shape"] = list(arr.shape)
        except Exception:
            try:
                json.dumps(value)
                payload[key] = value
            except Exception:
                payload[key] = str(type(value))
    return payload


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for part in line.strip().split()[1:4]:
                    head = part.split("/")[0]
                    if head:
                        idx.append(int(head) - 1)
                if len(idx) == 3:
                    faces.append(idx)
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _find_default_smpl_faces(project_root: Path | None) -> np.ndarray | None:
    if project_root is None:
        return None
    roots = [
        project_root / "models" / "checkpoints" / "smpl",
        project_root / "models" / "source_archives",
        project_root / "data" / "models" / "external_repos",
    ]
    names = {"basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl", "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl", "smpl_neutral.pkl"}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pkl"):
            if p.name.lower() not in names and "neutral" not in p.name.lower():
                continue
            try:
                data = _load_pickle(p)
                if isinstance(data, dict) and "f" in data:
                    return normalize_faces(np.asarray(data["f"]))
                if hasattr(data, "f"):
                    return normalize_faces(np.asarray(data.f))
            except Exception:
                continue
    return None


def import_structure_outputs(
    output_root: str | Path,
    cache_root: str | Path,
    *,
    model_name: str,
    confidence: float = 0.85,
    project_root: str | Path | None = None,
    start_index: int = 0,
) -> dict[str, Any]:
    """Import OBJ/NPZ/PKL structure outputs into this project's structure cache.

    This is intentionally tolerant because 4D-Humans / WHAM output layouts vary by
    repo version and command. It accepts common keys and writes the unified cache
    consumed by the pointcloud fusion exporter.
    """
    output_root = Path(output_root)
    cache_root = Path(cache_root)
    project_root_path = Path(project_root) if project_root is not None else None
    if not output_root.exists():
        return {"ok": False, "written_frames": 0, "reason": "output_root_missing", "output_root": str(output_root)}

    default_faces = _find_default_smpl_faces(project_root_path)
    written = 0
    skipped: list[str] = []

    # OBJ meshes.
    for obj_path in sorted(output_root.rglob("*.obj")):
        low = str(obj_path).lower()
        if any(part in low for part in ("/assets/", "\\assets\\", "/example", "\\example")):
            continue
        try:
            vertices, faces = _read_obj(obj_path)
            if len(vertices) == 0 or len(faces) == 0:
                skipped.append(str(obj_path))
                continue
            write_external_structure_cache(
                cache_root,
                frame_index_from_name(obj_path, written + start_index),
                vertices,
                faces,
                joints=None,
                camera={"source_path": str(obj_path), "coordinate_space": "smpl_camera_or_local"},
                confidence=float(confidence),
                model_name=model_name,
            )
            written += 1
        except Exception:
            skipped.append(str(obj_path))

    # NPZ / PKL sequences or per-frame files.
    for data_path in sorted(list(output_root.rglob("*.npz")) + list(output_root.rglob("*.pkl")) + list(output_root.rglob("*.pickle"))):
        low = str(data_path).lower()
        if any(part in low for part in ("/tests/", "\\tests\\", "/assets/", "\\assets\\")):
            continue
        loaded = _load_any_mapping(data_path)
        for item in _dicts_from_loaded(loaded):
            vertices = _as_array(_first_mapping_value(item, VERTEX_KEYS))
            faces = _as_array(_first_mapping_value(item, FACE_KEYS))
            joints = _as_array(_first_mapping_value(item, JOINT_KEYS))
            if vertices is None:
                continue
            if faces is None:
                faces = default_faces
            if faces is None:
                skipped.append(str(data_path) + ": missing faces")
                continue
            cam = _camera_payload(item, data_path, model_name)
            for offset, v_arr, f_arr, j_arr, c in _split_frame_arrays(vertices, faces, joints, cam):
                v_copy = np.asarray(v_arr).copy()
                # WHAM outputs vertices in camera coordinates (Y down).
                # The app expects Y up. Invert Y and Z to flip it upright while preserving handedness.
                if "wham" in model_name.lower() and v_copy.ndim == 2 and v_copy.shape[-1] == 3:
                    v_copy[:, 1] *= -1.0
                    v_copy[:, 2] *= -1.0
                if j_arr is not None:
                    j_copy = np.asarray(j_arr).copy()
                    if "wham" in model_name.lower() and j_copy.ndim == 2 and j_copy.shape[-1] == 3:
                        j_copy[:, 1] *= -1.0
                        j_copy[:, 2] *= -1.0
                else:
                    j_copy = None

                write_external_structure_cache(
                    cache_root,
                    frame_index_from_name(data_path, written + offset + start_index) if vertices.ndim == 2 else int(offset + start_index),
                    v_copy,
                    np.asarray(f_arr),
                    j_copy,
                    camera=c,
                    confidence=float(confidence),
                    model_name=model_name,
                )
                written += 1

    return {
        "ok": bool(written > 0),
        "written_frames": int(written),
        "skipped": skipped[:30],
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "model_name": model_name,
    }
