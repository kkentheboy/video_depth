# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import APP_NAME, PROJECT_DIR, QMessageBox, json

class ModelConfigurationMixin:
    def _looks_like_real_model_weight(self, p: Path, key: str) -> bool:
        """Filter out repo fixtures and keep actual model/checkpoint files.

        External repos often contain small .pkl/.npz files under tests/data. Those are
        not model weights. v18 still counted HaMeR's ViTPose h36m test files as
        HaMeR weights; this filter only accepts files that look like real weights.
        """
        suffix = p.suffix.lower()
        if suffix not in self._MODEL_WEIGHT_SUFFIXES:
            return False
        name = p.name.lower()
        if name in self._MODEL_SCAN_SKIP_FILENAMES:
            return False
        parts = {part.lower() for part in p.parts}
        if parts & self._MODEL_SCAN_SKIP_PARTS:
            return False
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        # Most real neural checkpoints are much larger. Keep small body-model pkl/npz
        # possible for SMPL/MANO, but reject tiny placeholder/test files elsewhere.
        if key in {"smpl", "mano"}:
            if size < 64 * 1024:
                return False
        else:
            if size < 1 * 1024 * 1024:
                return False

        path_text = str(p).replace("\\", "/").lower()
        stem_text = p.stem.lower()
        if key == "smpl":
            return ("smpl" in path_text or "basicmodel" in stem_text) and suffix in {".pkl", ".npz"}
        if key == "mano":
            return "mano" in path_text and suffix in {".pkl", ".npz"}
        if key == "4dhumans":
            # Do not count SMPL body model files copied into 4D-Humans/data as
            # 4D-Humans checkpoints. 4D-Humans/HMR2 is usable only when an
            # actual neural checkpoint exists.
            if suffix not in {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".pt2"}:
                return False
            if "basicmodel" in stem_text or "smpl" in stem_text:
                return False
            return any(token in path_text for token in ("4dhumans", "4d-humans", "hmr2", "hmr_2", "hmr"))
        if key == "wham":
            # Do not count WHAM/data/basicModel_neutral...pkl as a WHAM model.
            # The expected public checkpoint is usually wham_vit_w_3dpw.pth.tar
            # or another large file with wham in the filename/path.
            is_pth_tar = p.name.lower().endswith(".pth.tar")
            if not is_pth_tar and suffix not in {".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".pt2"}:
                return False
            if "basicmodel" in stem_text or "smpl" in stem_text:
                return False
            return "wham" in path_text or "wham" in p.name.lower()
        if key == "hamer":
            return any(token in path_text for token in ("hamer", "hamer_ckpt", "hamer_ckpts"))
        return True

    def _candidate_weight_files(self, roots: list[Path], key: str, *, limit: int = 12) -> list[Path]:
        """Return real model weight candidates, ignoring placeholders and test data."""
        found: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                if not root.exists():
                    continue
                for p in root.rglob("*"):
                    if not p.is_file():
                        continue
                    if not self._looks_like_real_model_weight(p, key):
                        continue
                    resolved = p.resolve()
                    dedupe_key = str(resolved).lower()
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    found.append(p)
                    if len(found) >= limit:
                        return found
            except Exception:
                continue
        return found

    def _model_scan_roots(self) -> dict[str, list[Path]]:
        models = PROJECT_DIR / "models"
        ckpt = models / "checkpoints"
        repos = PROJECT_DIR / "data" / "models" / "external_repos"
        return {
            "smpl": [ckpt / "smpl", ckpt / "SMPL", models / "smpl", models / "SMPL"],
            "mano": [ckpt / "mano", ckpt / "MANO", models / "mano", models / "MANO"],
            "4dhumans": [ckpt / "4dhumans", ckpt / "4D-Humans", models / "4dhumans", repos / "4D-Humans"],
            "wham": [ckpt / "wham", ckpt / "WHAM", models / "wham", repos / "WHAM"],
            "hamer": [ckpt / "hamer", ckpt / "HaMeR", models / "hamer", repos / "hamer"],
        }

    def _scan_3d_model_config(self) -> dict[str, object]:
        roots = self._model_scan_roots()
        found = {name: self._candidate_weight_files(paths, name) for name, paths in roots.items()}

        smpl_ok = bool(found["smpl"])
        mano_ok = bool(found["mano"])
        body_solver_ok = smpl_ok and bool(found["4dhumans"] or found["wham"])
        hand_solver_ok = bool(found["hamer"])
        structure_cache_ok = self._has_structure_cache()
        structure_ok = bool(structure_cache_ok)
        hand_ok = mano_ok and hand_solver_ok

        report_path = PROJECT_DIR / "data" / "resources" / "configs" / "model_3d_deploy_report.json"
        report_hint = ""
        try:
            if report_path.exists():
                data = json.loads(report_path.read_text(encoding="utf-8"))
                generated = data.get("generated_at") or "未知时间"
                report_hint = f"部署报告：{generated}"
        except Exception:
            report_hint = "部署报告：存在但读取失败"

        return {
            "found": found,
            "smpl_ok": smpl_ok,
            "mano_ok": mano_ok,
            "body_solver_ok": body_solver_ok,
            "hand_solver_ok": hand_solver_ok,
            "structure_ok": structure_ok,
            "structure_cache_ok": structure_cache_ok,
            "structure_runner_ok": body_solver_ok,
            "hand_ok": hand_ok,
            "full_ok": structure_ok and hand_ok,
            "report_hint": report_hint,
        }

    def _format_3d_scan_lines(self, scan: dict[str, object]) -> list[str]:
        found = scan["found"]
        assert isinstance(found, dict)
        body_ready = bool(scan.get("body_solver_ok"))
        cache_ready = bool(scan.get("structure_cache_ok"))
        has_4dh = bool(found.get("4dhumans"))
        has_wham = bool(found.get("wham"))
        has_smpl = bool(found.get("smpl"))

        lines: list[str] = []
        if cache_ready:
            lines.append("✓ 结构缓存：已生成，可以导出稳定 Mesh / 点云。")
        elif body_ready:
            solver = "4DHumans" if has_4dh else "WHAM" if has_wham else "人体结构模型"
            lines.append(f"! 结构缓存：未生成。点击下方按钮后，会用 {solver} 自动生成。")
        else:
            missing = []
            if not has_smpl:
                missing.append("SMPL")
            if not (has_4dh or has_wham):
                missing.append("4DHumans/WHAM")
            lines.append("× 结构模型：缺少 " + "、".join(missing or ["必要资源"]) + "。")
        lines.append("主流程：主视频 → 结构缓存 → 稳定 Mesh → Dense/Shell → Mesh / 可选点云。")
        lines.append("手部增强：第二阶段再接入，当前不参与导出。")
        return lines

    def _is_3d_structure_model_configured(self) -> bool:
        return bool(self._scan_3d_model_config().get("structure_ok"))

    def _is_3d_hand_model_configured(self) -> bool:
        return bool(self._scan_3d_model_config().get("hand_ok"))

    def refresh_3d_model_status(self) -> None:
        scan = self._scan_3d_model_config()
        structure_ok = bool(scan.get("structure_ok"))
        hand_ok = bool(scan.get("hand_ok"))
        full_ok = bool(scan.get("full_ok"))

        if hasattr(self, "model_3d_surface_badge"):
            if structure_ok:
                self._set_badge_state(self.model_3d_surface_badge, "缓存可用", "#86efac", "#166534")
            else:
                self._set_badge_state(self.model_3d_surface_badge, "需要缓存", "#fbbf24", "#92400e")

        if hasattr(self, "model_3d_completion_badge"):
            if hand_ok:
                self._set_badge_state(self.model_3d_completion_badge, "核心可用", "#86efac", "#166534")
            elif bool(scan.get("smpl_ok")) or bool(scan.get("mano_ok")):
                self._set_badge_state(self.model_3d_completion_badge, "部分可用", "#93c5fd", "#1d4ed8")
            else:
                self._set_badge_state(self.model_3d_completion_badge, "基础可用", "#fbbf24", "#92400e")

        if hasattr(self, "model_3d_status_label"):
            if structure_ok and hand_ok:
                text = "3D模型状态：Structure cache + MANO / HaMeR 已可用；可导出结构+手部 Mesh / 点云。"
            elif structure_ok:
                text = "3D模型状态：Structure cache 已生成；可导出稳定 Mesh / 点云。"
            elif bool(scan.get("structure_runner_ok")):
                text = "3D模型状态：4DHumans / WHAM 资源存在；请先生成结构缓存。"
            elif hand_ok:
                text = "3D模型状态：MANO / HaMeR 已识别；手部资源已就绪，但结构补全还需要 structure cache。"
            elif bool(scan.get("smpl_ok")) or bool(scan.get("mano_ok")):
                text = "3D模型状态：已识别部分授权模型。已识别部分模型资源；主流程仍需先生成 structure cache。"
            else:
                text = "3D模型状态：主流程需要 structure cache；未生成前不能导出结构点云。"
            self.model_3d_status_label.setText(text)

        lines = self._format_3d_scan_lines(scan)
        if hasattr(self, "model_3d_detail_label"):
            self.model_3d_detail_label.setText("\n".join(lines))
        if hasattr(self, "model_3d_check_btn"):
            self.model_3d_check_btn.setText("重新检查3D模型配置")

        try:
            self.log("3D模型配置检查完成：" + " | ".join(lines))
        except Exception:
            pass

        # Deliberately show a small dialog on manual click: the old button looked
        # like it did nothing because only a small label changed.
        if self.sender() is not None:
            QMessageBox.information(self, APP_NAME, "3D模型配置检查完成。\n\n" + "\n".join(lines))
