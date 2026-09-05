# -*- coding: utf-8 -*-
"""Debug helpers for Depth Fusion GUI.

Activated by DEPTH_FUSION_DEBUG=1, normally through run_gui_debug.bat.
It avoids changing normal startup behavior while giving crash/hang logs enough
information to diagnose render-chain failures.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

_DEBUG_VALUES = {"1", "true", "yes", "on", "debug"}
_CONFIGURED = False


def debug_enabled() -> bool:
    return os.environ.get("DEPTH_FUSION_DEBUG", "").strip().lower() in _DEBUG_VALUES


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _print_debug(line: str) -> None:
    try:
        print(f"[{_ts()}] {line}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _debug_log_path() -> Path | None:
    raw = os.environ.get("DEPTH_FUSION_DEBUG_LOG", "").strip()
    if not raw:
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def configure_debug_logging() -> bool:
    """Install Python-level crash/hang diagnostics when debug is enabled."""
    global _CONFIGURED
    if _CONFIGURED or not debug_enabled():
        return debug_enabled()
    _CONFIGURED = True

    log_path = _debug_log_path()
    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    try:
        faulthandler.enable(all_threads=True)
    except Exception as exc:
        _print_debug(f"[WARN] faulthandler.enable failed: {exc!r}")

    # If the render path freezes instead of throwing, dump all Python thread
    # stacks periodically into the same console/log stream.
    try:
        seconds = int(os.environ.get("DEPTH_FUSION_HANG_DUMP_SECONDS", "90"))
        if seconds > 0:
            faulthandler.dump_traceback_later(seconds, repeat=True)
            _print_debug(f"[debug] hang stack dump enabled: every {seconds}s")
    except Exception as exc:
        _print_debug(f"[WARN] hang stack dump setup failed: {exc!r}")

    def excepthook(exc_type, exc, tb):
        _print_debug("[FATAL] unhandled exception")
        traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
        sys.stderr.flush()

    sys.excepthook = excepthook

    if hasattr(threading, "excepthook"):
        def thread_excepthook(args):
            _print_debug(f"[FATAL] unhandled thread exception: {getattr(args.thread, 'name', '<unknown>')}")
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)
            sys.stderr.flush()

        threading.excepthook = thread_excepthook

    _print_debug("[debug] Depth Fusion debug hooks installed")
    if log_path is not None:
        _print_debug(f"[debug] log file: {log_path}")
    return True


def install_qt_message_handler() -> None:
    """Forward Qt messages into the debug log stream when available."""
    if not debug_enabled():
        return
    try:
        from PySide6.QtCore import qInstallMessageHandler
    except Exception:
        return

    def handler(mode, context, message):
        try:
            file_name = getattr(context, "file", "") or ""
            line = getattr(context, "line", 0) or 0
            suffix = f" ({file_name}:{line})" if file_name else ""
            _print_debug(f"[Qt] {message}{suffix}")
        except Exception:
            pass

    try:
        qInstallMessageHandler(handler)
        _print_debug("[debug] Qt message handler installed")
    except Exception as exc:
        _print_debug(f"[WARN] Qt message handler setup failed: {exc!r}")
