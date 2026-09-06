# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import PROJECT_DIR, QApplication, QTimer, importlib, short_error_message, shutil, structure_cache_root, sys
from segmentation_pipeline.foreground import check_foreground_environment
from segmentation_pipeline.human_parsing import check_segmentation_environment
from segmentation_pipeline.segmentation_cache import segmentation_cache_summary

class DeploymentEnvironmentMixin:
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
