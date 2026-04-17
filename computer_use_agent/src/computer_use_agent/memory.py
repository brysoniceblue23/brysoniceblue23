"""Session-scoped memory for the agent.

Each run of the agent produces a :class:`Session` directory on disk:

    sessions/
      2026-04-17T18-42-13/
        events.jsonl         -- append-only log of plans, actions, results
        screenshots/
          0001.png
          0002.png
          ...

Everything the agent sees and does lands here so you can replay or
debug a run after the fact. :class:`ActionLog` exposes a programmatic
view of the same data in memory.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Event:
    kind: str  # "observation" | "plan" | "action" | "result" | "note"
    ts: float
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "ts": self.ts, "payload": self.payload}


class ActionLog:
    """In-memory event log. Thread-unsafe; use one per Agent."""

    def __init__(self) -> None:
        self._events: List[Event] = []

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterable[Event]:
        return iter(self._events)

    def record(self, kind: str, payload: Optional[dict] = None) -> Event:
        event = Event(kind=kind, ts=time.time(), payload=payload or {})
        self._events.append(event)
        return event

    def events(self, kind: Optional[str] = None) -> List[Event]:
        if kind is None:
            return list(self._events)
        return [e for e in self._events if e.kind == kind]

    def summary(self) -> dict:
        counts = Counter(e.kind for e in self._events)
        return {
            "event_count": len(self._events),
            "by_kind": dict(counts),
            "first_ts": self._events[0].ts if self._events else None,
            "last_ts": self._events[-1].ts if self._events else None,
        }


class Session:
    """On-disk session that persists events and screenshots."""

    def __init__(self, root: Path | str, *, name: Optional[str] = None) -> None:
        root = Path(root)
        name = name or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        self.dir = root / name
        self.screenshots_dir = self.dir / "screenshots"
        self.events_path = self.dir / "events.jsonl"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.log = ActionLog()
        self._screenshot_idx = 0

    def save_screenshot(self, image: np.ndarray) -> Path:
        self._screenshot_idx += 1
        path = self.screenshots_dir / f"{self._screenshot_idx:04d}.png"
        # Write PNG via OpenCV; avoid importing at module load so unit
        # tests that don't need screenshots stay cheap.
        import cv2

        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to write screenshot to {path}")
        return path

    def record(self, kind: str, payload: Optional[dict] = None) -> Event:
        event = self.log.record(kind, payload=payload)
        self._append_jsonl(event)
        return event

    def record_action(self, action: Any) -> Event:
        return self.record("action", _as_payload(action))

    def record_result(self, result: Any) -> Event:
        return self.record("result", _as_payload(result))

    def _append_jsonl(self, event: Event) -> None:
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), default=_json_default) + "\n")


def _as_payload(obj: Any) -> dict:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": repr(obj)}


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)
