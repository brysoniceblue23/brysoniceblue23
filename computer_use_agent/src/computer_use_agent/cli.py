"""Command-line entry point.

Usage:
    computer-use-agent screenshot [--out PATH]
    computer-use-agent find-text "Submit" [--region x,y,w,h]
    computer-use-agent run-script SCRIPT.json --goal "Open notepad"

``SCRIPT.json`` is a JSON array of action dictionaries, for example::

    [
      {"op": "move_to", "x": 500, "y": 400, "duration": 0.6},
      {"op": "click",   "x": 500, "y": 400},
      {"op": "type",    "text": "hello world"},
      {"op": "wait",    "seconds": 0.5},
      {"op": "done"}
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

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
from computer_use_agent.memory import Session
from computer_use_agent.mouse import Mouse
from computer_use_agent.planner import ScriptedPlanner
from computer_use_agent.screen import MSSScreen, Region
from computer_use_agent.vision import Vision

logger = logging.getLogger("computer_use_agent")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="computer-use-agent")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("screenshot", help="capture a screenshot")
    s.add_argument("--out", type=Path, default=Path("screenshot.png"))
    s.add_argument("--region", type=str, default=None, help="x,y,w,h")

    f = sub.add_parser("find-text", help="OCR for a string on screen")
    f.add_argument("query")
    f.add_argument("--region", type=str, default=None)
    f.add_argument("--min-confidence", type=float, default=50.0)

    r = sub.add_parser("run-script", help="execute a scripted plan")
    r.add_argument("script", type=Path)
    r.add_argument("--goal", default="scripted run")
    r.add_argument("--sessions-dir", type=Path, default=Path("sessions"))
    r.add_argument("--max-steps", type=int, default=200)
    r.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and log actions but do not move the real mouse",
    )

    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "screenshot":
        return _cmd_screenshot(args)
    if args.command == "find-text":
        return _cmd_find_text(args)
    if args.command == "run-script":
        return _cmd_run_script(args)
    return 2


def _parse_region(spec: str | None) -> Region | None:
    if not spec:
        return None
    try:
        x, y, w, h = (int(v.strip()) for v in spec.split(","))
    except Exception as exc:
        raise SystemExit(f"invalid --region: {spec!r} ({exc})") from exc
    return Region(x=x, y=y, width=w, height=h)


def _cmd_screenshot(args: argparse.Namespace) -> int:
    import cv2

    screen = MSSScreen()
    img = screen.grab(_parse_region(args.region))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), img):
        print(f"failed to write {args.out}", file=sys.stderr)
        return 1
    print(f"wrote {args.out} ({img.shape[1]}x{img.shape[0]})")
    return 0


def _cmd_find_text(args: argparse.Namespace) -> int:
    screen = MSSScreen()
    region = _parse_region(args.region)
    img = screen.grab(region)
    det = Vision().find_text(img, args.query, min_confidence=args.min_confidence)
    if det is None:
        print(f"text {args.query!r} not found")
        return 1
    # Translate back to screen coordinates if a region was supplied.
    ox = region.x if region else 0
    oy = region.y if region else 0
    print(
        f"found {args.query!r} at "
        f"({det.x + ox}, {det.y + oy}) size=({det.width}, {det.height}) "
        f"confidence={det.confidence:.2f}"
    )
    return 0


def _cmd_run_script(args: argparse.Namespace) -> int:
    script = json.loads(args.script.read_text(encoding="utf-8"))
    actions = [_action_from_dict(step) for step in script]
    if not any(isinstance(a, Done) for a in actions):
        actions.append(Done(success=True, reason="end of script"))

    session = Session(root=args.sessions_dir)
    screen = MSSScreen()
    if args.dry_run:
        from computer_use_agent.mouse import FakeBackend

        mouse = Mouse(backend=FakeBackend(start=screen.size() if hasattr(screen, "size") else (0, 0)))
    else:
        mouse = Mouse()
    agent = Agent(
        goal=args.goal,
        planner=ScriptedPlanner(actions=actions),
        screen=screen,
        mouse=mouse,
        session=session,
        config=AgentConfig(max_steps=args.max_steps),
    )
    results = agent.run()
    summary = session.log.summary()
    print(json.dumps({"session": str(session.dir), "summary": summary, "steps": len(results)}, indent=2))
    return 0 if all(r.ok for r in results) else 1


_ACTION_MAP = {
    "move_to": MoveTo,
    "click": Click,
    "double_click": DoubleClick,
    "type": Type,
    "wait": Wait,
    "screenshot": Screenshot,
    "done": Done,
}


def _action_from_dict(step: dict) -> Action:
    op = step.get("op")
    if op not in _ACTION_MAP:
        raise SystemExit(f"unknown action op: {op!r}")
    cls = _ACTION_MAP[op]
    kwargs = {k: v for k, v in step.items() if k != "op"}
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as exc:
        raise SystemExit(f"invalid arguments for op={op!r}: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
