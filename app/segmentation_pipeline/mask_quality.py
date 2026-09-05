# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class MaskQuality:
    status: str
    confidence: float
    clipped_edges: list[str]
    foreground_area: float
    garment_area: float
    hair_area: float
    unstable: bool
    reason: str


def _mask01(mask: Optional[np.ndarray], shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    if mask is None:
        if shape_hw is None:
            return np.zeros((1, 1), dtype=np.float32)
        return np.zeros(shape_hw, dtype=np.float32)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        if shape_hw is None:
            return np.zeros((1, 1), dtype=np.float32)
        return np.zeros(shape_hw, dtype=np.float32)
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    if shape_hw is not None and arr.shape[:2] != shape_hw:
        arr = cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    return arr


def _area(mask: np.ndarray) -> float:
    if mask.size <= 1:
        return 0.0
    return float(np.mean(mask > 0.35))


def _edges(mask: np.ndarray) -> list[str]:
    if mask.size <= 1:
        return []
    m = mask > 0.35
    h, w = m.shape[:2]
    if h <= 2 or w <= 2:
        return []
    out: list[str] = []
    thr_x = max(2, int(round(w * 0.012)))
    thr_y = max(2, int(round(h * 0.012)))
    if int(np.count_nonzero(m[0:2, :])) > thr_x:
        out.append("top")
    if int(np.count_nonzero(m[-2:, :])) > thr_x:
        out.append("bottom")
    if int(np.count_nonzero(m[:, 0:2])) > thr_y:
        out.append("left")
    if int(np.count_nonzero(m[:, -2:])) > thr_y:
        out.append("right")
    return out


def classify_mask_quality(
    *,
    foreground: Optional[np.ndarray],
    garment: Optional[np.ndarray],
    hair: Optional[np.ndarray],
    prev_meta: Optional[dict] = None,
    parser_confidence: float = 0.0,
) -> MaskQuality:
    fg = _mask01(foreground)
    gm = _mask01(garment, fg.shape[:2])
    hr = _mask01(hair, fg.shape[:2])
    fa = _area(fg)
    ga = _area(gm)
    ha = _area(hr)
    clipped = _edges(fg)
    flags = {"unstable": False, "occluded": False, "clipped": bool(clipped), "unknown": False, "partial": False}
    reason_parts: list[str] = []
    if fa <= 0.01:
        flags["unknown"] = True
        reason_parts.append("foreground_too_small")
    if clipped:
        reason_parts.append("clipped:" + ",".join(clipped))
    if prev_meta:
        try:
            pfa = float(prev_meta.get("foreground_area", 0.0) or 0.0)
            pga = float(prev_meta.get("garment_area", 0.0) or 0.0)
            pha = float(prev_meta.get("hair_area", 0.0) or 0.0)
            # If a part suddenly disappears but the foreground is not clipped by
            # the image edge, treat it as occlusion instead of deletion. Collect
            # all flags first, then resolve status by priority so order of checks
            # does not hide a more severe unstable state.
            garment_drop = pga > 0.015 and ga < pga * 0.45
            hair_drop = pha > 0.006 and ha < pha * 0.35
            fg_drop = pfa > 0.04 and fa < pfa * 0.55
            if garment_drop:
                if clipped:
                    flags["unstable"] = True
                    reason_parts.append("garment_area_drop")
                else:
                    flags["occluded"] = True
                    reason_parts.append("garment_occluded")
            if hair_drop:
                if clipped:
                    flags["unstable"] = True
                    reason_parts.append("hair_area_drop")
                else:
                    flags["occluded"] = True
                    reason_parts.append("hair_occluded")
            if fg_drop and not (garment_drop or hair_drop):
                flags["unstable"] = True
                reason_parts.append("foreground_area_drop")
        except Exception:
            pass
    if ga <= 0.002 and ha <= 0.001:
        flags["partial"] = True
    status = "trusted"
    if flags["unknown"]:
        status = "unknown"
    elif flags["unstable"]:
        status = "unstable"
    elif flags["occluded"]:
        status = "occluded"
    elif flags["clipped"]:
        status = "clipped"
    elif flags["partial"]:
        status = "partial"
    conf = float(np.clip(parser_confidence, 0.0, 1.0))
    if status == "trusted":
        quality_conf = max(0.62, conf)
    elif status == "partial":
        quality_conf = max(0.38, conf * 0.75)
    elif status == "clipped":
        quality_conf = max(0.34, conf * 0.65)
    elif status == "unstable":
        quality_conf = max(0.22, conf * 0.50)
    elif status == "occluded":
        quality_conf = max(0.30, conf * 0.58)
    else:
        quality_conf = 0.0
    return MaskQuality(
        status=status,
        confidence=float(quality_conf),
        clipped_edges=clipped,
        foreground_area=fa,
        garment_area=ga,
        hair_area=ha,
        unstable=bool(flags["unstable"]),
        reason=";".join(reason_parts) if reason_parts else "ok",
    )


def quality_to_meta(q: MaskQuality) -> dict:
    return {
        "status": q.status,
        "confidence": float(q.confidence),
        "clipped_edges": list(q.clipped_edges),
        "foreground_area": float(q.foreground_area),
        "garment_area": float(q.garment_area),
        "hair_area": float(q.hair_area),
        "unstable": bool(q.unstable),
        "occluded": str(q.status) == "occluded",
        "reason": str(q.reason),
    }
