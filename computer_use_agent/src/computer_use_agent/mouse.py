"""Human-like mouse control.

PyAutoGUI moves the cursor in a straight line at a constant speed, which
is trivially detectable by anti-bot systems and looks obviously robotic.
This module generates a **cubic Bezier path** with randomized control
points and variable per-step timing (a rough Fitts' law-inspired
profile) so movements look plausibly human.

The Bezier path generator is a pure function and is fully unit-tested
(see ``tests/test_mouse.py``), and :class:`Mouse` takes a pluggable
``backend`` so we can exercise the interaction logic without a real
display.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def _cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    u = 1.0 - t
    x = (u**3) * p0[0] + 3 * (u**2) * t * p1[0] + 3 * u * (t**2) * p2[0] + (t**3) * p3[0]
    y = (u**3) * p0[1] + 3 * (u**2) * t * p1[1] + 3 * u * (t**2) * p2[1] + (t**3) * p3[1]
    return x, y


def bezier_path(
    start: Point,
    end: Point,
    *,
    steps: int = 40,
    curvature: float = 0.25,
    rng: Optional[random.Random] = None,
) -> List[Point]:
    """Sample a cubic Bezier path from ``start`` to ``end``.

    Control points are offset perpendicular to the straight line by a
    random fraction of the distance (``curvature``), producing a gentle
    arc that differs on every call when ``rng`` is seeded differently.
    """

    if steps < 2:
        raise ValueError("steps must be >= 2")
    rng = rng or random.Random()

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return [start, end]

    # Unit perpendicular vector.
    px, py = -dy / dist, dx / dist
    max_off = curvature * dist

    # Two control points at roughly 1/3 and 2/3 along the line, nudged
    # perpendicular by a random signed offset.
    off1 = rng.uniform(-max_off, max_off)
    off2 = rng.uniform(-max_off, max_off)
    p1 = (start[0] + dx / 3 + px * off1, start[1] + dy / 3 + py * off1)
    p2 = (start[0] + 2 * dx / 3 + px * off2, start[1] + 2 * dy / 3 + py * off2)

    # Use an ease-in-out time curve so the cursor accelerates and then
    # decelerates the way a human arm does.
    path: List[Point] = []
    for i in range(steps):
        raw_t = i / (steps - 1)
        t = 0.5 - 0.5 * math.cos(math.pi * raw_t)  # smoothstep-ish
        path.append(_cubic_bezier(start, p1, p2, end, t))
    return path


def step_durations(total_seconds: float, steps: int, *, rng: Optional[random.Random] = None) -> List[float]:
    """Generate per-step sleep durations that sum to ``total_seconds``.

    Slightly jittered so the cursor doesn't move in perfectly regular
    ticks. Always returns ``steps`` positive values.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if total_seconds < 0:
        raise ValueError("total_seconds must be >= 0")
    rng = rng or random.Random()
    base = total_seconds / steps
    jitter = base * 0.25
    raw = [max(0.0, base + rng.uniform(-jitter, jitter)) for _ in range(steps)]
    s = sum(raw) or 1.0
    return [total_seconds * r / s for r in raw]


class MouseBackend(Protocol):
    """Minimal interface :class:`Mouse` needs from PyAutoGUI (or a fake)."""

    def position(self) -> Tuple[int, int]:  # pragma: no cover - protocol
        ...

    def moveTo(self, x: int, y: int) -> None:  # pragma: no cover - protocol
        ...

    def click(self, button: str = "left") -> None:  # pragma: no cover - protocol
        ...

    def doubleClick(self, button: str = "left") -> None:  # pragma: no cover - protocol
        ...

    def typewrite(self, text: str, interval: float = 0.0) -> None:  # pragma: no cover - protocol
        ...


class _PyAutoGUIBackend:
    """Default backend that defers the ``pyautogui`` import."""

    def __init__(self) -> None:
        import pyautogui  # noqa: WPS433

        pyautogui.FAILSAFE = True
        self._p = pyautogui

    def position(self) -> Tuple[int, int]:
        pos = self._p.position()
        return int(pos[0]), int(pos[1])

    def moveTo(self, x: int, y: int) -> None:
        self._p.moveTo(x, y)

    def click(self, button: str = "left") -> None:
        self._p.click(button=button)

    def doubleClick(self, button: str = "left") -> None:
        self._p.doubleClick(button=button)

    def typewrite(self, text: str, interval: float = 0.0) -> None:
        self._p.typewrite(text, interval=interval)


class FakeBackend:
    """Deterministic backend used by tests and dry-run mode.

    Records every call so assertions can be made about what the mouse
    code *would* do without touching a real display.
    """

    def __init__(self, start: Tuple[int, int] = (0, 0)) -> None:
        self._pos = start
        self.events: List[Tuple[str, tuple]] = []

    def position(self) -> Tuple[int, int]:
        return self._pos

    def moveTo(self, x: int, y: int) -> None:
        self._pos = (int(x), int(y))
        self.events.append(("move", (int(x), int(y))))

    def click(self, button: str = "left") -> None:
        self.events.append(("click", (button,)))

    def doubleClick(self, button: str = "left") -> None:
        self.events.append(("double_click", (button,)))

    def typewrite(self, text: str, interval: float = 0.0) -> None:
        self.events.append(("type", (text, interval)))


@dataclass
class MouseConfig:
    base_speed_px_per_s: float = 1800.0
    min_duration: float = 0.15
    max_duration: float = 1.2
    steps_per_100px: float = 4.0
    min_steps: int = 16
    max_steps: int = 120
    curvature: float = 0.18
    click_jitter_px: int = 2


class Mouse:
    """High-level mouse controller that produces human-like motion.

    Pass ``backend=FakeBackend()`` in tests or ``dry_run=True`` scripts
    to avoid touching the real cursor.
    """

    def __init__(
        self,
        backend: Optional[MouseBackend] = None,
        *,
        config: Optional[MouseConfig] = None,
        rng: Optional[random.Random] = None,
        sleep: Optional[callable] = None,
    ) -> None:
        self._backend = backend if backend is not None else _PyAutoGUIBackend()
        self.config = config or MouseConfig()
        self._rng = rng or random.Random()
        self._sleep = sleep if sleep is not None else time.sleep

    @property
    def backend(self) -> MouseBackend:
        return self._backend

    def position(self) -> Tuple[int, int]:
        return self._backend.position()

    def move_to(self, x: int, y: int, duration: Optional[float] = None) -> None:
        start = self._backend.position()
        end = (int(x), int(y))
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        if dist < 1:
            self._backend.moveTo(end[0], end[1])
            return

        dur = duration if duration is not None else self._pick_duration(dist)
        steps = self._pick_steps(dist)
        path = bezier_path(start, end, steps=steps, curvature=self.config.curvature, rng=self._rng)
        sleeps = step_durations(dur, len(path), rng=self._rng)
        for point, sleep_s in zip(path, sleeps):
            self._backend.moveTo(int(round(point[0])), int(round(point[1])))
            if sleep_s > 0:
                self._sleep(sleep_s)
        # Ensure the final pixel is exact.
        self._backend.moveTo(end[0], end[1])

    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        *,
        button: str = "left",
        duration: Optional[float] = None,
    ) -> None:
        if x is not None and y is not None:
            jx = x + self._rng.randint(-self.config.click_jitter_px, self.config.click_jitter_px)
            jy = y + self._rng.randint(-self.config.click_jitter_px, self.config.click_jitter_px)
            self.move_to(jx, jy, duration=duration)
        self._backend.click(button=button)

    def double_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        *,
        button: str = "left",
        duration: Optional[float] = None,
    ) -> None:
        if x is not None and y is not None:
            self.move_to(x, y, duration=duration)
        self._backend.doubleClick(button=button)

    def type_text(self, text: str, *, interval: float = 0.03) -> None:
        self._backend.typewrite(text, interval=interval)

    # ------------------------------------------------------------------
    def _pick_duration(self, distance_px: float) -> float:
        raw = distance_px / self.config.base_speed_px_per_s
        raw *= self._rng.uniform(0.85, 1.2)
        return max(self.config.min_duration, min(self.config.max_duration, raw))

    def _pick_steps(self, distance_px: float) -> int:
        raw = int(self.config.steps_per_100px * distance_px / 100.0)
        return max(self.config.min_steps, min(self.config.max_steps, raw))
