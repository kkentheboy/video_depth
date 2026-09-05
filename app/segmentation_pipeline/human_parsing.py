# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import importlib.util
import json

import cv2
import numpy as np


FASHN_REPO_ID = "fashn-ai/fashn-human-parser"


@dataclass(slots=True)
class SegmentationResult:
    ok: bool
    provider: str = "none"
    hair: Optional[np.ndarray] = None
    garment: Optional[np.ndarray] = None
    skin: Optional[np.ndarray] = None
    foreground: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    confidence: float = 0.0
    reason: str = ""
    meta: dict | None = None


def _norm01(mask: np.ndarray | None, shape_hw: tuple[int, int] | None = None) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.size == 0:
        return None
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if float(np.nanmax(arr)) > 1.5:
        arr = arr / 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if shape_hw is not None and arr.shape[:2] != tuple(shape_hw):
        arr = cv2.resize(arr, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
    return arr.astype(np.float32)


def _model_roots(project_root: str | Path, provider: str) -> list[Path]:
    root = Path(project_root)
    seg = root / "data" / "models" / "segmentation"
    low = str(provider or "auto").lower().replace(" ", "_")
    roots: list[Path] = []
    if low in {"auto", "fashn", "fashn_human_parser", "fashn-human-parser", "fashn human parser"}:
        roots += [seg / "fashn_human_parser", seg / "fashn-human-parser", seg / "fashn-ai--fashn-human-parser"]
    # 其他解析模型 is not exposed as a fake option here. If it is added later, it must
    # have a real runner instead of a UI-only placeholder.
    return roots


def _is_transformers_model_dir(path: Path) -> bool:
    return path.exists() and ((path / "config.json").exists() or (path / "preprocessor_config.json").exists())


def check_segmentation_environment(project_root: str | Path, provider: str = "auto") -> dict:
    """Check optional parsing dependencies and local model folders.

    This function never downloads models. It reports what is present so the GUI
    can tell the user whether the workflow is using real image segmentation or
    falling back to the built-in geometric shell.
    """
    missing_modules: list[str] = []
    for module_name in ("torch", "PIL", "transformers"):
        try:
            if importlib.util.find_spec(module_name) is None:
                missing_modules.append(module_name)
        except Exception:
            missing_modules.append(module_name)
    roots = _model_roots(project_root, provider)
    found = [p for p in roots if _is_transformers_model_dir(p)]
    seg_root = Path(project_root) / "data" / "models" / "segmentation"
    return {
        "ok": bool(found) and not missing_modules,
        "provider": provider,
        "missing_modules": missing_modules,
        "model_found": bool(found),
        "model_paths": [str(p) for p in found],
        "checked_paths": [str(p) for p in roots],
        "segmentation_root": str(seg_root),
        "message": "画面分割可用" if bool(found) and not missing_modules else "画面分割未就绪，将回退到几何算法权重",
    }


def _label_groups(id2label: dict) -> tuple[set[int], set[int], set[int], set[int]]:
    hair_ids: set[int] = set()
    garment_ids: set[int] = set()
    skin_ids: set[int] = set()
    fg_ids: set[int] = set()
    for key, value in id2label.items():
        try:
            idx = int(key)
        except Exception:
            try:
                idx = int(value)
                name = str(key).lower()
            except Exception:
                continue
        else:
            name = str(value).lower()
        name = name.replace("-", "_").replace(" ", "_")
        if any(tok in name for tok in ("background", "bg", "void")):
            continue
        fg_ids.add(idx)
        if "hair" in name:
            hair_ids.add(idx)
        if any(tok in name for tok in ("upper", "top", "shirt", "coat", "jacket", "dress", "skirt", "pants", "trouser", "jean", "short", "cloth", "clothes", "scarf", "hat", "shoe", "boot")):
            garment_ids.add(idx)
        if any(tok in name for tok in ("skin", "face", "arm", "hand", "leg", "foot", "torso", "neck")):
            skin_ids.add(idx)
    return hair_ids, garment_ids, skin_ids, fg_ids

_PARSER_CACHE = {}

def _run_transformers_parser(image_bgr: np.ndarray, model_path: Path, log: Callable[[str], None] | None = None) -> SegmentationResult:
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except Exception as exc:  # noqa: BLE001
        return SegmentationResult(ok=False, provider="fashn", reason=f"缺少分割依赖：{exc}")

    h, w = image_bgr.shape[:2]
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        
        model_path_str = str(model_path)
        if model_path_str not in _PARSER_CACHE:
            if log:
                log(f"加载 Human Parser: {model_path}")
            processor = AutoImageProcessor.from_pretrained(model_path_str, local_files_only=True)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_path_str, local_files_only=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            _PARSER_CACHE[model_path_str] = (processor, model, device)
            
        processor, model, device = _PARSER_CACHE[model_path_str]
        with torch.inference_mode():
            inputs = processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            up = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = torch.softmax(up, dim=1)[0]
            labels = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)
            conf = torch.max(probs, dim=0).values.detach().cpu().numpy().astype(np.float32)
        id2label = getattr(model.config, "id2label", {}) or {}
        if not id2label:
            cfg_path = model_path / "config.json"
            if cfg_path.exists():
                try:
                    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
                    id2label = payload.get("id2label", {}) or {}
                except Exception:
                    id2label = {}
        hair_ids, garment_ids, skin_ids, fg_ids = _label_groups(id2label)
        if not fg_ids:
            # Conservative generic fallback: class 0 is usually background.
            max_label = int(labels.max()) if labels.size else 0
            fg_ids = set(range(1, max_label + 1))
        hair = np.isin(labels, list(hair_ids)).astype(np.float32) if hair_ids else np.zeros((h, w), dtype=np.float32)
        garment = np.isin(labels, list(garment_ids)).astype(np.float32) if garment_ids else np.zeros((h, w), dtype=np.float32)
        skin = np.isin(labels, list(skin_ids)).astype(np.float32) if skin_ids else np.zeros((h, w), dtype=np.float32)
        foreground = np.isin(labels, list(fg_ids)).astype(np.float32)
        # Slight cleanup for projection sampling.
        kernel = np.ones((3, 3), np.uint8)
        hair = cv2.morphologyEx(hair, cv2.MORPH_OPEN, kernel).astype(np.float32)
        garment = cv2.morphologyEx(garment, cv2.MORPH_CLOSE, kernel).astype(np.float32)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel).astype(np.float32)
        score = float(np.mean(conf[foreground > 0.5])) if np.any(foreground > 0.5) else float(np.mean(conf))
        return SegmentationResult(
            ok=True,
            provider=model_path.name,
            hair=hair,
            garment=garment,
            skin=skin,
            foreground=foreground,
            labels=labels,
            confidence=score,
            reason="ok",
            meta={"id2label": {str(k): str(v) for k, v in id2label.items()}, "model_path": str(model_path)},
        )
    except Exception as exc:  # noqa: BLE001
        return SegmentationResult(ok=False, provider="fashn", reason=f"Human Parser 推理失败：{exc}")


def run_human_parsing(
    image_bgr: np.ndarray,
    *,
    project_root: str | Path,
    provider: str = "auto",
    log: Callable[[str], None] | None = None,
) -> SegmentationResult:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return SegmentationResult(ok=False, provider=str(provider), reason="输入帧不是 BGR 图像")
    env = check_segmentation_environment(project_root, provider)
    if env.get("missing_modules"):
        return SegmentationResult(ok=False, provider=str(provider), reason="缺少分割依赖：" + ", ".join(env.get("missing_modules", [])))
    for path_text in env.get("model_paths", []):
        result = _run_transformers_parser(image, Path(path_text), log=log)
        if result.ok:
            return result
        if log:
            log(result.reason)
    return SegmentationResult(ok=False, provider=str(provider), reason="未找到可用 Human Parsing 模型，请放入 models/segmentation/fashn_human_parser。")
