"""Example: perceive a screenshot, find a button via template matching, click it.

Run with:

    python examples/find_and_click.py path/to/screenshot.png path/to/button.png

This example does not touch your real mouse; it uses :class:`FakeBackend`
and :class:`StaticScreen` to demonstrate the whole perceive-plan-act loop
deterministically.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

from computer_use_agent import (
    Agent,
    AgentConfig,
    CallablePlanner,
    Click,
    Done,
    Session,
    Vision,
)
from computer_use_agent.actions import Screenshot, Wait
from computer_use_agent.mouse import FakeBackend, Mouse, MouseConfig
from computer_use_agent.planner import Observation
from computer_use_agent.screen import StaticScreen


def build_planner(button_template):
    vision = Vision()
    state = {"clicked": False}

    def plan(obs: Observation):
        if state["clicked"]:
            return Done(success=True, reason="button clicked")
        image = obs.metadata["image"]
        det = vision.find_template(image, button_template, threshold=0.7)
        if det is None:
            # Wait a beat and try again on the next frame.
            return Wait(0.05, reason="button not yet visible")
        state["clicked"] = True
        return Click(*det.center, reason=f"found button at {det.center}")

    return CallablePlanner(plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--sessions-dir", type=Path, default=Path("sessions"))
    args = parser.parse_args(argv)

    scene = cv2.imread(str(args.screenshot))
    tpl = cv2.imread(str(args.template))
    if scene is None or tpl is None:
        print("could not load images", file=sys.stderr)
        return 1

    session = Session(root=args.sessions_dir)
    mouse = Mouse(
        backend=FakeBackend(start=(0, 0)),
        rng=random.Random(0),
        sleep=lambda _s: None,
        config=MouseConfig(click_jitter_px=0),
    )
    agent = Agent(
        goal=f"click {args.template.name}",
        planner=build_planner(tpl),
        screen=StaticScreen(scene),
        mouse=mouse,
        session=session,
        config=AgentConfig(max_steps=10),
        sleep=lambda _s: None,
    )
    results = agent.run()
    print(f"session: {session.dir}")
    print(f"events: {session.log.summary()}")
    for r in results:
        print(" ", type(r.action).__name__, "->", "ok" if r.ok else f"FAIL: {r.error}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
