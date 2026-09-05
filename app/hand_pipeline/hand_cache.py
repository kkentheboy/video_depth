# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class HandFrame:
    frame_index: int
    left_vertices: np.ndarray | None = None
    right_vertices: np.ndarray | None = None
    left_faces: np.ndarray | None = None
    right_faces: np.ndarray | None = None
    joints: np.ndarray | None = None
    confidence: float = 0.0
    model_name: str = "none"
    meta: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.confidence > 0 and (self.left_vertices is not None or self.right_vertices is not None)


def _stem(frame_index: int) -> str:
    return f"frame_{int(frame_index):06d}"


def hand_frame_paths(cache_root: Path, frame_index: int) -> dict[str, Path]:
    root = Path(cache_root) / "hand"
    stem = _stem(frame_index)
    return {
        "root": root,
        "left": root / f"{stem}_left_hand_vertices.npy",
        "right": root / f"{stem}_right_hand_vertices.npy",
        "left_faces": root / f"{stem}_left_hand_faces.npy",
        "right_faces": root / f"{stem}_right_hand_faces.npy",
        "joints": root / f"{stem}_hand_joints.npy",
        "meta": root / f"{stem}_hand_meta.json",
    }


def save_hand_frame(cache_root: Path, frame: HandFrame) -> None:
    paths = hand_frame_paths(cache_root, frame.frame_index)
    paths["root"].mkdir(parents=True, exist_ok=True)
    if frame.left_vertices is not None:
        np.save(paths["left"], np.asarray(frame.left_vertices, dtype=np.float32))
    if frame.right_vertices is not None:
        np.save(paths["right"], np.asarray(frame.right_vertices, dtype=np.float32))
    if frame.left_faces is not None:
        np.save(paths["left_faces"], np.asarray(frame.left_faces, dtype=np.int32))
    if frame.right_faces is not None:
        np.save(paths["right_faces"], np.asarray(frame.right_faces, dtype=np.int32))
    if frame.joints is not None:
        np.save(paths["joints"], np.asarray(frame.joints, dtype=np.float32))
    payload = dict(frame.meta or {})
    payload.update({"confidence": float(frame.confidence), "model_name": frame.model_name})
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_hand_frame(cache_root: Path, frame_index: int) -> HandFrame | None:
    paths = hand_frame_paths(cache_root, frame_index)
    if not paths["left"].exists() and not paths["right"].exists():
        return None
    meta: dict = {}
    if paths["meta"].exists():
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return HandFrame(
        frame_index=int(frame_index),
        left_vertices=np.load(paths["left"]) if paths["left"].exists() else None,
        right_vertices=np.load(paths["right"]) if paths["right"].exists() else None,
        left_faces=np.load(paths["left_faces"]) if paths["left_faces"].exists() else None,
        right_faces=np.load(paths["right_faces"]) if paths["right_faces"].exists() else None,
        joints=np.load(paths["joints"]) if paths["joints"].exists() else None,
        confidence=float(meta.get("confidence", 1.0)),
        model_name=str(meta.get("model_name", "external")),
        meta=meta,
    )
