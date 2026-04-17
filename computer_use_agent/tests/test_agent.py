"""End-to-end tests for the perceive-plan-act loop."""

from __future__ import annotations

import random

import numpy as np

from computer_use_agent.actions import Click, Done, MoveTo, Type, Wait
from computer_use_agent.agent import Agent, AgentConfig
from computer_use_agent.memory import Session
from computer_use_agent.mouse import FakeBackend, Mouse, MouseConfig
from computer_use_agent.planner import CallablePlanner, ScriptedPlanner
from computer_use_agent.screen import StaticScreen


def _make_agent(tmp_path, actions, *, max_steps: int = 20):
    img = np.full((50, 50, 3), 200, dtype=np.uint8)
    backend = FakeBackend(start=(0, 0))
    session = Session(root=tmp_path, name="t")
    mouse = Mouse(
        backend=backend,
        rng=random.Random(0),
        sleep=lambda _s: None,
        config=MouseConfig(click_jitter_px=0),
    )
    agent = Agent(
        goal="test",
        planner=ScriptedPlanner(actions),
        screen=StaticScreen(img),
        mouse=mouse,
        session=session,
        config=AgentConfig(max_steps=max_steps),
        sleep=lambda _s: None,
    )
    return agent, backend, session


def test_agent_runs_scripted_plan(tmp_path):
    actions = [
        MoveTo(10, 10),
        Click(15, 15),
        Type("hello"),
        Wait(0.0),
        Done(success=True, reason="ok"),
    ]
    agent, backend, session = _make_agent(tmp_path, actions)
    results = agent.run()

    assert [type(r.action).__name__ for r in results] == [
        "MoveTo",
        "Click",
        "Type",
        "Wait",
        "Done",
    ]
    assert all(r.ok for r in results)

    kinds = [e.kind for e in session.log]
    assert kinds[0] == "note"
    assert kinds.count("observation") == 5
    assert kinds.count("plan") == 5
    assert kinds.count("result") == 5

    event_types = [e[0] for e in backend.events]
    assert "click" in event_types
    assert "type" in event_types
    assert any(e == ("type", ("hello", 0.03)) for e in backend.events)


def test_agent_stops_at_max_steps(tmp_path):
    # Planner never returns Done.
    planner = CallablePlanner(lambda obs: Wait(0.0))
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    session = Session(root=tmp_path, name="t")
    mouse = Mouse(backend=FakeBackend(), sleep=lambda _s: None)
    agent = Agent(
        goal="loop",
        planner=planner,
        screen=StaticScreen(img),
        mouse=mouse,
        session=session,
        config=AgentConfig(max_steps=3),
        sleep=lambda _s: None,
    )
    results = agent.run()
    # 3 Wait actions + 1 synthetic Done(success=False)
    assert len(results) == 4
    assert results[-1].ok is False
    assert results[-1].action.reason == "max_steps"


def test_screenshots_are_written(tmp_path):
    actions = [Wait(0.0), Done()]
    agent, _, session = _make_agent(tmp_path, actions)
    agent.run()

    pngs = sorted(session.screenshots_dir.glob("*.png"))
    assert len(pngs) == 2


def test_disable_screenshots(tmp_path):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    session = Session(root=tmp_path, name="t")
    mouse = Mouse(backend=FakeBackend(), sleep=lambda _s: None)
    agent = Agent(
        goal="no-shots",
        planner=ScriptedPlanner([Done()]),
        screen=StaticScreen(img),
        mouse=mouse,
        session=session,
        config=AgentConfig(max_steps=2, save_screenshots=False),
        sleep=lambda _s: None,
    )
    agent.run()
    assert list(session.screenshots_dir.glob("*.png")) == []
