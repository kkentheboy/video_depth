# -*- coding: utf-8 -*-
from __future__ import annotations

from depth_fusion_core import event_log


class EventConsoleStateMixin:
    def _append_event_console_line(self, text: str) -> None:
        line = str(text).rstrip("\n")
        if not line:
            return
        # Worker log signals can arrive shortly after their event_log line.
        # Keep the console readable by dropping immediate duplicates.
        if line in self._event_console_recent[-12:]:
            return
        self._event_console_recent.append(line)
        if len(self._event_console_recent) > 64:
            self._event_console_recent = self._event_console_recent[-64:]
        self.log_box.appendPlainText(line)
        try:
            bar = self.log_box.verticalScrollBar()
            bar.setValue(bar.maximum())
        except Exception:
            pass

    def _on_worker_log_signal(self, text: str) -> None:
        # Most workers already call event_log() before emitting their legacy
        # log signal. The event listener is the source of truth, so do not log
        # again unless the listener was not installed for some reason.
        if not getattr(self, "_event_console_listener_active", False):
            self._append_event_console_line(str(text))

    def log(self, text: str) -> None:
        event_log(text, channel="UI")

    def clear_event_console(self) -> None:
        self.log_box.clear()
        self._event_console_recent.clear()
        event_log("事件控制台已清空", channel="UI")

