# computer-use-agent

A minimal, well-tested reference implementation of an **AI computer-use
agent**: the kind of system that can *see* your desktop, *think* about
what to do next, and *act* by moving the mouse and typing.

It's intentionally small (~1k lines) so you can read the whole thing in
an afternoon, but it's wired up the way a real agent should be:

```
┌────────────┐   screenshot    ┌──────────┐   actions    ┌────────────┐
│  Screen    │ ───────────────▶│  Planner │─────────────▶│   Mouse    │
│  Capture   │                 │          │              │  (human-   │
│  (mss)     │     detections  │  (rules  │              │   like)    │
└────────────┘ ◀────────────── │   or LLM)│              └────────────┘
       │        (Vision: OpenCV│          │                    │
       │         + OCR)        └──────────┘                    │
       │                             │                         │
       ▼                             ▼                         ▼
                    ┌─────────────────────────────┐
                    │          Session            │
                    │  screenshots/*.png          │
                    │  events.jsonl (perceive /   │
                    │   plan / act / result log)  │
                    └─────────────────────────────┘
```

## Why this exists

The ideas behind modern computer-use agents (Anthropic's Computer Use,
Self-Operating Computer, etc.) aren't magic — they combine a handful of
well-understood pieces:

1. **Computer vision on screenshots** (template matching, OCR)
2. **Mouse/keyboard automation** (PyAutoGUI) with human-like motion
3. **A planner** that turns observations into the next action
4. **Memory** of what was seen and done so behaviour can be replayed

This repo is a clean, dependency-light reference for each piece with
unit tests so you can modify any layer without breaking the others.

## Install

```bash
cd computer_use_agent
pip install -e '.[dev]'

# Optional: install tesseract so OCR actually works.
sudo apt-get install -y tesseract-ocr        # Debian/Ubuntu
brew install tesseract                       # macOS
```

## Quickstart

```bash
# 1. Save the current screen to disk.
computer-use-agent screenshot --out /tmp/screen.png

# 2. Find where the word "Submit" is on screen (needs tesseract).
computer-use-agent find-text "Submit"

# 3. Drive the mouse/keyboard from a scripted plan.
cat > /tmp/plan.json <<'JSON'
[
  {"op": "move_to", "x": 500, "y": 400, "duration": 0.7},
  {"op": "click",   "x": 500, "y": 400},
  {"op": "type",    "text": "hello from an AI"},
  {"op": "wait",    "seconds": 0.5},
  {"op": "done"}
]
JSON
computer-use-agent run-script /tmp/plan.json --goal "demo" --dry-run
```

`--dry-run` replaces the real mouse with a fake backend so you can
inspect what would have happened without touching your desktop.

## Python API

```python
import random
from computer_use_agent import (
    Agent, AgentConfig,
    ScriptedPlanner, CallablePlanner,
    Click, Done, MoveTo, Type, Wait,
    Session, Vision,
)
from computer_use_agent.mouse import Mouse, FakeBackend
from computer_use_agent.screen import MSSScreen

planner = ScriptedPlanner(actions=[
    MoveTo(600, 400, duration=0.5),
    Click(600, 400),
    Type("hello, world"),
    Wait(0.5),
    Done(),
])

agent = Agent(
    goal="say hello",
    planner=planner,
    screen=MSSScreen(),
    mouse=Mouse(),
    session=Session(root="sessions"),
    config=AgentConfig(max_steps=10),
)
results = agent.run()
```

### Plug in your own planner (e.g. an LLM)

```python
def my_policy(obs):
    # obs.metadata["image"] is the current screenshot as a BGR ndarray.
    # Call any model you like here, then return one of the Action dataclasses.
    if obs.step == 1:
        return MoveTo(100, 100)
    return Done()

agent = Agent(..., planner=CallablePlanner(my_policy))
```

### Vision helpers

```python
from computer_use_agent import Vision
import cv2

screen = cv2.imread("screenshot.png")
button = cv2.imread("submit_button.png")

vision = Vision()
det = vision.find_template(screen, button, threshold=0.85)
if det:
    print("Click at", det.center, "confidence", det.confidence)

det = vision.find_text(screen, "Submit")
```

## What's inside

| Module                                                    | Responsibility                                   |
| ---                                                       | ---                                              |
| [`screen.py`](src/computer_use_agent/screen.py)           | Screen capture via `mss`, plus a `StaticScreen` backend for tests |
| [`vision.py`](src/computer_use_agent/vision.py)           | Multi-scale template matching + OCR (`pytesseract`) |
| [`mouse.py`](src/computer_use_agent/mouse.py)             | Human-like cubic-Bezier cursor paths, jittered click positions, fake backend for tests |
| [`actions.py`](src/computer_use_agent/actions.py)         | Immutable dataclasses describing every action the agent can take |
| [`planner.py`](src/computer_use_agent/planner.py)         | `ScriptedPlanner` and `CallablePlanner` (plug in an LLM, a DSL, whatever) |
| [`memory.py`](src/computer_use_agent/memory.py)           | Session directories with screenshots + JSONL event log |
| [`agent.py`](src/computer_use_agent/agent.py)             | The perceive–plan–act loop that glues it all together |
| [`cli.py`](src/computer_use_agent/cli.py)                 | `computer-use-agent {screenshot,find-text,run-script}` |

## Design notes

### Human-like mouse motion

A straight-line, constant-speed mouse path is trivially detectable by
anti-bot systems (and looks obviously robotic). `mouse.bezier_path`
samples a **cubic Bezier curve** with randomized control points offset
perpendicular to the travel direction by a configurable `curvature`
fraction. Per-step sleeps come from `step_durations`, which emits a
jittered schedule that sums to the requested total duration. The result
is a smooth, slightly-curved path whose timing naturally eases in and
out, just like a human wrist. Jitter is also applied to the final click
position so repeated clicks don't land on the exact same pixel.

### Perceive-plan-act, not prompt-and-pray

The agent loop (`agent.py`) is deliberately *separate* from the
planning logic. Every tick it:

1. Grabs a fresh screenshot and persists it.
2. Calls the planner with an `Observation` (step counter + screenshot
   + a free-form `metadata` dict where you can stash detections, OCR
   text, page hashes, …).
3. Dispatches the returned `Action` to the right subsystem.
4. Writes `observation`, `plan`, `action`, and `result` events to
   `events.jsonl`.

Because the planner is just a callable, you can swap in a rule engine,
a Markov chain, a prompt to an LLM, or a human-in-the-loop CLI prompt
without touching `Agent` at all.

### Memory that you can actually inspect

Each run writes to `sessions/<timestamp>/`:

```
sessions/2026-04-17T18-42-13/
├── events.jsonl          # one JSON event per line (note, observation, plan, action, result)
└── screenshots/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

This makes it trivial to replay a run, diff two runs, or ship a failing
session to the author when something goes wrong.

## Testing

```bash
python -m pytest      # ~50 unit tests
python -m ruff check src tests
```

All tests run headlessly — the `StaticScreen` and `FakeBackend`
implementations let us exercise the full agent loop without touching a
real display, so the test suite works in CI.

## Roadmap / ideas

- [ ] LLM-backed planner that consumes a screenshot + goal
- [ ] UI element detector (YOLO / SAM) as an alternative to OCR+templates
- [ ] Keyboard shortcut abstraction (hotkeys beyond `type_text`)
- [ ] Replay viewer (HTML timeline of a session)

## Safety

The `pyautogui` backend enables `FAILSAFE=True` — slam the cursor into
the top-left corner of the screen to abort. For additional safety,
always develop against `Mouse(backend=FakeBackend())` or the `--dry-run`
CLI flag until you're happy with the plan.

## License

MIT
