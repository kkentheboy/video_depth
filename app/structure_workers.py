# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import json
import shutil
import traceback
from typing import Optional

from depth_fusion_core import JobConfig, PROJECT_DIR, QObject, Signal, event_exception, event_log, structure_cache_root
from segmentation_pipeline.segmentation_cache import generate_segmentation_sequence_cache


class StructureCacheWorker(QObject):
    log = Signal(str)
    progress = Signal(str)
    progress_value = Signal(str, int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        input_path: str,
        cache_root: str,
        model: str,
        project_root: str,
        max_side: int = 720,
        *,
        start_frame: int = 0,
        end_frame: int = -1,
    ) -> None:
        super().__init__()
        self.input_path = str(input_path)
        self.cache_root = str(cache_root)
        self.model = str(model or "4dhumans")
        self.project_root = str(project_root)
        self.max_side = max(256, min(1024, int(max_side or 720)))
        self.start_frame = max(0, int(start_frame or 0))
        self.end_frame = int(end_frame if end_frame is not None else -1)
        if self.end_frame >= 0 and self.end_frame < self.start_frame:
            self.end_frame = self.start_frame
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _log(self, text: str) -> None:
        event_log(text, channel="STRUCTURE")
        self.log.emit(text)

    def run(self) -> None:
        tmp_cache_root: Optional[Path] = None
        try:
            project = Path(self.project_root)
            cache_root = Path(self.cache_root)
            cache_root.mkdir(parents=True, exist_ok=True)
            tmp_cache_root = cache_root / "__structure_tmp__"
            if tmp_cache_root.exists():
                shutil.rmtree(tmp_cache_root, ignore_errors=True)
            tmp_cache_root.mkdir(parents=True, exist_ok=True)
            model_key = str(self.model or "4dhumans").strip().lower()
            self._log(f"启动结构缓存生成: model={model_key}, input={self.input_path}, cache_root={cache_root}, range={self.start_frame}-{self.end_frame if self.end_frame >= 0 else 'end'}")
            self._log("结构缓存会先写入临时目录；第三方模型成功后再替换旧缓存，失败时保留旧缓存。")
            self._log("12G显存策略：结构模型单独运行，最长边限制到 1024，完成后释放 GPU。")

            if model_key == "wham":
                from structure_pipeline.wham_runner import WhamRunner
                runner = WhamRunner()
            else:
                from structure_pipeline.fourdhumans_runner import FourDHumansRunner
                runner = FourDHumansRunner()

            def _runner_log(text: str) -> None:
                if self._cancel:
                    raise RuntimeError("结构缓存生成已取消")
                line = str(text or "").strip()
                if not line:
                    return
                if line.startswith("[Progress]"):
                    payload = line.replace("[Progress]", "", 1).strip()
                    self.progress.emit(payload)
                    try:
                        import re as _re
                        m = _re.search(r"^(.*?)\s+(-?\d+)\s*/\s*(-?\d+)", payload)
                        if m:
                            stage = m.group(1).strip() or "结构缓存"
                            self.progress_value.emit(stage, int(m.group(2)), int(m.group(3)))
                    except Exception:
                        pass
                else:
                    self._log(line)

            result = runner.run_video_to_cache(
                self.input_path,
                tmp_cache_root,
                project,
                start_frame=int(self.start_frame),
                end_frame=int(self.end_frame),
                max_side=int(self.max_side),
                log=_runner_log,
            )
            self._log("结构缓存结果: " + json.dumps(result, ensure_ascii=False, default=str))
            if self._cancel:
                raise RuntimeError("结构缓存生成已取消")
            if not isinstance(result, dict) or not bool(result.get("ok", False)):
                reason = str(result.get("reason") if isinstance(result, dict) else "unknown")
                missing = result.get("missing_modules") if isinstance(result, dict) else None
                if reason == "missing_python_modules" and missing:
                    raise RuntimeError("结构缓存生成失败：缺少 Python 依赖：" + "、".join(str(x) for x in missing))
                if reason == "missing_home_env":
                    raise RuntimeError("结构缓存生成失败：Windows HOME 环境变量未设置。")
                if reason == "repo_missing":
                    raise RuntimeError("结构缓存生成失败：外部模型仓库不存在：" + str(result.get("repo", "")))
                if reason == "no_importable_outputs":
                    raise RuntimeError("结构缓存生成失败：第三方模型已运行，但没有生成可导入的 mesh/npz/pkl 输出。")
                raise RuntimeError("结构缓存生成失败：" + reason)
            new_structure_dir = tmp_cache_root / "structure"
            if not new_structure_dir.exists():
                raise RuntimeError("结构缓存生成失败：临时目录没有生成 structure 数据。")
            final_structure_dir = cache_root / "structure"
            backup_structure_dir = cache_root / "structure_backup_previous"
            if backup_structure_dir.exists():
                shutil.rmtree(backup_structure_dir, ignore_errors=True)
            if final_structure_dir.exists():
                shutil.move(str(final_structure_dir), str(backup_structure_dir))
            shutil.move(str(new_structure_dir), str(final_structure_dir))
            shutil.rmtree(tmp_cache_root, ignore_errors=True)
            self.finished.emit("结构缓存生成完成: " + self.cache_root)
        except Exception as exc:  # noqa: BLE001
            event_exception("结构缓存生成失败", exc, model=self.model, input_path=self.input_path)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ModelPreloadWorker(QObject):
    log = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_id: str, device_mode: str, load_normal: bool = False) -> None:
        super().__init__()
        self.model_id = model_id
        self.device_mode = device_mode
        self.load_normal = bool(load_normal)

    def _log(self, text: str) -> None:
        event_log(text, channel="PRELOAD")
        self.log.emit(text)

    def run(self) -> None:
        try:
            self._log("当前清理版无需预热旧模型；请使用环境页检查 4DHumans / FASHN。")
            self.finished.emit("网格主流程无需旧模型预热。")
        except Exception as exc:  # noqa: BLE001
            event_exception("模型预热状态检查失败", exc)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class SegmentationCacheWorker(QObject):
    progress = Signal(str)
    progress_value = Signal(str, int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, cfg: JobConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def _log(self, text: str) -> None:
        event_log(text, channel="SEGMENTATION_CACHE")
        self.progress.emit(str(text))

    def run(self) -> None:
        try:
            cache_root = structure_cache_root(self.cfg)
            def _progress(done: int, total: int) -> None:
                if self._cancel:
                    raise RuntimeError("分割缓存任务已取消。")
                self.progress_value.emit("segmentation", int(done), int(total))
            summary = generate_segmentation_sequence_cache(
                self.cfg,
                cache_root,
                project_root=PROJECT_DIR,
                log=self._log,
                progress=_progress,
            )
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001
            event_exception("逐帧分割缓存生成失败", exc, input_path=getattr(self.cfg, "input_path", ""))
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
