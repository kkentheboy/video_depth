# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class StructureFrame:
    frame_index: int
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    joints: np.ndarray | None = None
    camera: dict | None = None
    confidence: float = 0.0
    model_name: str = "none"

    @property
    def available(self) -> bool:
        return self.vertices is not None and self.faces is not None and self.confidence > 0.0


def _stem(frame_index: int) -> str:
    return f"frame_{int(frame_index):06d}"


def structure_frame_paths(cache_root: Path, frame_index: int) -> dict[str, Path]:
    root = Path(cache_root) / "structure"
    stem = _stem(frame_index)
    return {
        "root": root,
        "vertices": root / f"{stem}_smpl_vertices.npy",
        "faces": root / f"{stem}_smpl_faces.npy",
        "joints": root / f"{stem}_smpl_joints.npy",
        "camera": root / f"{stem}_smpl_camera.json",
        "alignment": root / f"{stem}_alignment.json",
    }


def save_structure_frame(cache_root: Path, frame: StructureFrame) -> None:
    paths = structure_frame_paths(cache_root, frame.frame_index)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if frame.vertices is not None:
        np.save(paths["vertices"], np.asarray(frame.vertices, dtype=np.float32))
    if frame.faces is not None:
        np.save(paths["faces"], np.asarray(frame.faces, dtype=np.int32))
    if frame.joints is not None:
        np.save(paths["joints"], np.asarray(frame.joints, dtype=np.float32))
    payload = dict(frame.camera or {})
    payload.update({"confidence": float(frame.confidence), "model_name": frame.model_name})
    with open(paths["camera"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_structure_frame(cache_root: Path, frame_index: int) -> StructureFrame | None:
    paths = structure_frame_paths(cache_root, frame_index)
    if not paths["vertices"].exists() or not paths["faces"].exists():
        return None
    camera: dict = {}
    if paths["camera"].exists():
        try:
            camera = json.loads(paths["camera"].read_text(encoding="utf-8"))
        except Exception:
            camera = {}
    joints = np.load(paths["joints"]) if paths["joints"].exists() else None
    return StructureFrame(
        frame_index=int(frame_index),
        vertices=np.load(paths["vertices"]),
        faces=np.load(paths["faces"]),
        joints=joints,
        camera=camera,
        confidence=float(camera.get("confidence", 1.0)),
        model_name=str(camera.get("model_name", "external")),
    )
