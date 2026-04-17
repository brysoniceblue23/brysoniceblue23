"""Action dataclasses for the agent.

These are plain-old data objects the planner produces and the agent executes.
Keeping them as immutable dataclasses makes the perceive-plan-act loop easy
to serialize, log, replay and test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass(frozen=True)
class MoveTo:
    x: int
    y: int
    duration: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class Click:
    x: Optional[int] = None
    y: Optional[int] = None
    button: str = "left"
    duration: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class DoubleClick:
    x: Optional[int] = None
    y: Optional[int] = None
    button: str = "left"
    duration: Optional[float] = None
    reason: str = ""


@dataclass(frozen=True)
class Type:
    text: str
    interval: float = 0.03
    reason: str = ""


@dataclass(frozen=True)
class Wait:
    seconds: float
    reason: str = ""


@dataclass(frozen=True)
class Screenshot:
    """Force a new screenshot before the next plan step."""

    reason: str = ""


@dataclass(frozen=True)
class Done:
    """Signal that the agent has achieved its goal (or given up)."""

    success: bool = True
    reason: str = ""


Action = Union[MoveTo, Click, DoubleClick, Type, Wait, Screenshot, Done]


@dataclass(frozen=True)
class ActionResult:
    """What actually happened when an action was executed."""

    action: Action
    ok: bool
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)
