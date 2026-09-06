from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from PySide6.QtWidgets import QApplication
from depth_fusion_ui import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    app.processEvents()

    assert window.windowTitle(), "MainWindow title must be initialized"
    assert window.path_edit is not None, "input path control missing"
    assert window.output_path_edit is not None, "output path control missing"
    assert window.preview_thread is None, "preview worker must not auto-start during construction"
    assert window.worker is None, "depth worker must not auto-start during construction"

    window.close()
    app.processEvents()
    print("video_depth Windows UI runtime smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
