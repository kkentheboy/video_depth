# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import Path, np, short_error_message, structure_cache_root
from segmentation_pipeline.segmentation_cache import segmentation_cache_summary

class StructureCacheStateMixin:
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
