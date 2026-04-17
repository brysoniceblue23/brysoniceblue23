"""The perceive-plan-act loop.

The :class:`Agent` wires together a :class:`ScreenCapture`, a
:class:`Vision`, a :class:`Mouse`, a :class:`Planner` and a
:class:`Session`. Each tick it:

1. **Perceives** - grabs a screenshot and stores it.
2. **Plans**  - asks the planner what to do next.
3. **Acts**   - dispatches the resulting :class:`Action` to the right
   subsystem and records the outcome.

The loop terminates when the planner returns :class:`Done`, when
``max_steps`` is reached, or when an action raises.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from computer_use_agent.actions import (
    Action,
    ActionResult,
    Click,
    Done,
    DoubleClick,
    MoveTo,
    Screenshot,
    Type,
    Wait,
)
from computer_use_agent.memory import Session
from computer_use_agent.mouse import Mouse
from computer_use_agent.planner import Observation, Planner
from computer_use_agent.screen import ScreenCapture
from computer_use_agent.vision import Vision

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_steps: int = 50
    save_screenshots: bool = True
    step_delay: float = 0.0  # optional pause between ticks
    stop_on_error: bool = True


class Agent:
    def __init__(
        self,
        *,
        goal: str,
        planner: Planner,
        screen: ScreenCapture,
        mouse: Mouse,
        session: Session,
        vision: Optional[Vision] = None,
        config: Optional[AgentConfig] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.goal = goal
        self.planner = planner
        self.screen = screen
        self.mouse = mouse
        self.session = session
        self.vision = vision or Vision()
        self.config = config or AgentConfig()
        self._sleep = sleep if sleep is not None else time.sleep
        self._results: List[ActionResult] = []
        self.session.record("note", {"goal": goal})

    @property
    def results(self) -> List[ActionResult]:
        return list(self._results)

    def run(self) -> List[ActionResult]:
        for step in range(1, self.config.max_steps + 1):
            observation = self._observe(step)
            action = self.planner.plan(observation)
            self.session.record("plan", {"step": step, "action": _as_dict(action)})

            if isinstance(action, Done):
                self._record_result(ActionResult(action=action, ok=action.success))
                logger.info("agent finished: %s", action.reason or "done")
                break

            result = self._execute(action)
            self._record_result(result)
            if not result.ok and self.config.stop_on_error:
                logger.warning("agent aborting after failed action: %s", result.error)
                break

            if self.config.step_delay > 0:
                self._sleep(self.config.step_delay)
        else:
            logger.info("agent hit max_steps=%d", self.config.max_steps)
            self._record_result(
                ActionResult(action=Done(success=False, reason="max_steps"), ok=False)
            )

        return list(self._results)

    # ------------------------------------------------------------------
    def _observe(self, step: int) -> Observation:
        image = self.screen.grab()
        path: Path | str = ""
        if self.config.save_screenshots:
            path = self.session.save_screenshot(image)
            self.session.record(
                "observation",
                {"step": step, "screenshot": str(path), "shape": list(image.shape)},
            )
        else:
            self.session.record("observation", {"step": step, "shape": list(image.shape)})
        return Observation(step=step, screenshot_path=str(path), metadata={"image": image})

    def _execute(self, action: Action) -> ActionResult:
        try:
            if isinstance(action, MoveTo):
                self.mouse.move_to(action.x, action.y, duration=action.duration)
                return ActionResult(action=action, ok=True)
            if isinstance(action, Click):
                self.mouse.click(
                    action.x,
                    action.y,
                    button=action.button,
                    duration=action.duration,
                )
                return ActionResult(action=action, ok=True)
            if isinstance(action, DoubleClick):
                self.mouse.double_click(
                    action.x,
                    action.y,
                    button=action.button,
                    duration=action.duration,
                )
                return ActionResult(action=action, ok=True)
            if isinstance(action, Type):
                self.mouse.type_text(action.text, interval=action.interval)
                return ActionResult(action=action, ok=True)
            if isinstance(action, Wait):
                self._sleep(max(0.0, action.seconds))
                return ActionResult(action=action, ok=True)
            if isinstance(action, Screenshot):
                # The next iteration's _observe will grab a fresh frame.
                return ActionResult(action=action, ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("action %r failed", action)
            return ActionResult(action=action, ok=False, error=str(exc))

        return ActionResult(action=action, ok=False, error=f"unknown action: {type(action).__name__}")

    def _record_result(self, result: ActionResult) -> None:
        self._results.append(result)
        self.session.record("result", _as_dict(result))


def _as_dict(obj) -> dict:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return {"value": repr(obj)}
