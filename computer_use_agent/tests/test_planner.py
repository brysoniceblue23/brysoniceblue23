"""Tests for the planner layer."""

from __future__ import annotations

import pytest

from computer_use_agent.actions import Click, Done, MoveTo
from computer_use_agent.planner import CallablePlanner, Observation, ScriptedPlanner


def _obs(step: int = 1) -> Observation:
    return Observation(step=step)


class TestScriptedPlanner:
    def test_emits_actions_in_order(self):
        plan = ScriptedPlanner([MoveTo(10, 20), Click(10, 20)])
        assert isinstance(plan.plan(_obs(1)), MoveTo)
        assert isinstance(plan.plan(_obs(2)), Click)

    def test_returns_done_when_exhausted(self):
        plan = ScriptedPlanner([MoveTo(0, 0)])
        plan.plan(_obs(1))
        result = plan.plan(_obs(2))
        assert isinstance(result, Done)
        assert result.success is True

    def test_reset_replays(self):
        plan = ScriptedPlanner([MoveTo(0, 0)])
        plan.plan(_obs(1))
        plan.reset()
        assert isinstance(plan.plan(_obs(1)), MoveTo)


class TestCallablePlanner:
    def test_returns_action_from_callable(self):
        planner = CallablePlanner(lambda obs: MoveTo(obs.step, obs.step))
        action = planner.plan(_obs(7))
        assert isinstance(action, MoveTo)
        assert action.x == 7 and action.y == 7

    def test_rejects_non_action_return(self):
        planner = CallablePlanner(lambda obs: "nope")
        with pytest.raises(TypeError):
            planner.plan(_obs())
