"""Planners turn observations into actions.

Two planners ship out of the box:

* :class:`ScriptedPlanner` -- emits a pre-recorded list of actions. Great
  for deterministic demos and tests.
* :class:`CallablePlanner` -- wraps any ``callable(observation) -> Action``
  so you can plug in an LLM, a rule engine, or a human-in-the-loop
  prompt without changing the :class:`~computer_use_agent.agent.Agent`.

The :class:`Planner` Protocol documents the minimum contract: given an
observation, return the next :class:`Action`. Returning
:class:`~computer_use_agent.actions.Done` stops the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Protocol

from computer_use_agent.actions import Action, Done


@dataclass(frozen=True)
class Observation:
    """What the agent sees on a given tick.

    ``screenshot_path`` is the on-disk PNG for the current frame (if the
    session persisted it) and ``metadata`` is a free-form dict where
    planners can stash detections, OCR results, step counters, etc.
    """

    step: int
    screenshot_path: str = ""
    metadata: dict = field(default_factory=dict)


class Planner(Protocol):
    def plan(self, observation: Observation) -> Action:  # pragma: no cover - protocol
        ...


@dataclass
class ScriptedPlanner:
    """Emits a fixed list of actions, then returns :class:`Done`."""

    actions: List[Action]
    _index: int = field(default=0, init=False, repr=False)

    def plan(self, observation: Observation) -> Action:  # noqa: D401
        if self._index >= len(self.actions):
            return Done(success=True, reason="script exhausted")
        action = self.actions[self._index]
        self._index += 1
        return action

    def reset(self) -> None:
        self._index = 0


class CallablePlanner:
    """Adapt any ``(Observation) -> Action`` callable as a :class:`Planner`."""

    def __init__(self, fn: Callable[[Observation], Action]) -> None:
        self._fn = fn

    def plan(self, observation: Observation) -> Action:
        result = self._fn(observation)
        if not _looks_like_action(result):
            raise TypeError(
                f"Planner callable returned {type(result)!r}; expected an Action dataclass"
            )
        return result


def _looks_like_action(obj: Any) -> bool:
    # Structural check keeps us decoupled from the concrete Union type.
    return hasattr(obj, "__dataclass_fields__")
