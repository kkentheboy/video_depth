# -*- coding: utf-8 -*-
import os

if os.environ.get("DEPTH_FUSION_DEBUG", "").strip().lower() in {"1", "true", "yes", "on", "debug"}:
    try:
        from depth_fusion_debug import configure_debug_logging
        configure_debug_logging()
    except Exception:
        # Do not let debug setup prevent normal application startup.
        pass

from depth_fusion_ui import main

if os.environ.get("DEPTH_FUSION_DEBUG", "").strip().lower() in {"1", "true", "yes", "on", "debug"}:
    try:
        from depth_fusion_debug import install_qt_message_handler
        install_qt_message_handler()
    except Exception:
        pass

if __name__ == "__main__":
    raise SystemExit(main())
