# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from .hand_cache import HandFrame


class NoopHandRunner:
    def infer_frame(self, frame_bgr: np.ndarray, frame_index: int) -> HandFrame:  # noqa: ARG002
        return HandFrame(frame_index=int(frame_index), confidence=0.0, model_name="none")


class HamerRunner(NoopHandRunner):
    """Placeholder adapter name for future HaMeR local deployment."""
