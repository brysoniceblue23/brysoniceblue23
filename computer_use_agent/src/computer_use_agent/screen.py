"""Screen capture.

Thin wrapper around ``mss`` that returns BGR numpy arrays suitable for
OpenCV. Kept intentionally small and dependency-light so it can be mocked
in tests (see ``tests/test_agent.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class Region:
    """An axis-aligned rectangle on the virtual desktop."""

    x: int
    y: int
    width: int
    height: int

    def as_mss_dict(self) -> dict:
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture(Protocol):
    """Anything that can return a BGR ``np.ndarray`` of the current screen."""

    def grab(self, region: Optional[Region] = None) -> np.ndarray:  # pragma: no cover - protocol
        ...


class MSSScreen:
    """Default screen capture backend using ``mss``.

    Instantiating this class lazily imports ``mss`` so the rest of the
    package remains importable on headless CI machines where ``mss``
    cannot open a display.
    """

    def __init__(self) -> None:
        import mss  # noqa: WPS433 (deliberate lazy import)

        self._sct = mss.mss()

    def grab(self, region: Optional[Region] = None) -> np.ndarray:
        monitor = self._sct.monitors[0] if region is None else region.as_mss_dict()
        raw = self._sct.grab(monitor)
        # mss returns BGRA; drop the alpha channel and keep BGR for OpenCV.
        arr = np.asarray(raw, dtype=np.uint8)
        return arr[:, :, :3]

    def size(self) -> tuple[int, int]:
        mon = self._sct.monitors[0]
        return int(mon["width"]), int(mon["height"])


class StaticScreen:
    """A deterministic screen backend that always returns the same image.

    Useful for unit tests and examples where you want to exercise the
    agent loop without a real display.
    """

    def __init__(self, image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("StaticScreen expects a BGR image with shape (H, W, 3)")
        self._image = image.copy()

    def grab(self, region: Optional[Region] = None) -> np.ndarray:
        if region is None:
            return self._image.copy()
        x, y, w, h = region.x, region.y, region.width, region.height
        return self._image[y : y + h, x : x + w].copy()
