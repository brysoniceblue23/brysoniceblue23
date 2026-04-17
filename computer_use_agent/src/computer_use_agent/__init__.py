"""computer_use_agent - a minimal AI computer-use agent.

This package provides a small, well-tested foundation for building agents
that can *see* a desktop (via screen capture + computer vision), *think*
(via a pluggable planner), and *act* (via human-like mouse and keyboard
control) while recording everything they do for later inspection.

Subpackages:
    screen   - screen capture
    vision   - template matching and OCR-based element detection
    mouse    - human-like cursor motion and clicks
    memory   - session-scoped action + screenshot log
    actions  - dataclasses describing actions the agent can take
    planner  - turns observations into actions (scripted or callable)
    agent    - the perceive-plan-act loop that glues it all together
"""

from computer_use_agent.actions import (
    Action,
    Click,
    Done,
    DoubleClick,
    MoveTo,
    Screenshot,
    Type,
    Wait,
)
from computer_use_agent.agent import Agent, AgentConfig
from computer_use_agent.memory import ActionLog, Session
from computer_use_agent.planner import CallablePlanner, Planner, ScriptedPlanner
from computer_use_agent.vision import Detection, Vision

__all__ = [
    "Action",
    "ActionLog",
    "Agent",
    "AgentConfig",
    "CallablePlanner",
    "Click",
    "Detection",
    "Done",
    "DoubleClick",
    "MoveTo",
    "Planner",
    "ScriptedPlanner",
    "Screenshot",
    "Session",
    "Type",
    "Vision",
    "Wait",
]

__version__ = "0.1.0"
