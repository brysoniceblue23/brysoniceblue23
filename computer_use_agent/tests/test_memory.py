"""Tests for the memory / session layer."""

from __future__ import annotations

import json

import numpy as np

from computer_use_agent.actions import Click, MoveTo
from computer_use_agent.memory import ActionLog, Session


def test_action_log_records_and_summarizes():
    log = ActionLog()
    log.record("plan", {"action": "click"})
    log.record("result", {"ok": True})
    log.record("plan", {"action": "move"})

    assert len(log) == 3
    assert len(log.events("plan")) == 2
    summary = log.summary()
    assert summary["event_count"] == 3
    assert summary["by_kind"] == {"plan": 2, "result": 1}
    assert summary["first_ts"] is not None


def test_session_persists_events_and_screenshots(tmp_path):
    session = Session(root=tmp_path, name="run-1")
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[5, 5] = (255, 255, 255)

    session.record("note", {"goal": "demo"})
    path = session.save_screenshot(img)
    session.record_action(MoveTo(x=10, y=20))
    session.record_action(Click(x=10, y=20))

    assert path.exists()
    assert path.parent == session.screenshots_dir
    assert session.events_path.exists()

    events = [json.loads(line) for line in session.events_path.read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert kinds == ["note", "action", "action"]
    assert events[0]["payload"] == {"goal": "demo"}
    assert events[1]["payload"]["x"] == 10

    summary = session.log.summary()
    assert summary["by_kind"] == {"note": 1, "action": 2}


def test_session_default_name_is_timestamped(tmp_path):
    s1 = Session(root=tmp_path)
    assert s1.dir.exists()
    # The directory name should contain a 4-digit year prefix.
    assert s1.dir.name[:4].isdigit()
