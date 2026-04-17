"""Tests for human-like mouse motion."""

from __future__ import annotations

import math
import random

import pytest

from computer_use_agent.mouse import (
    FakeBackend,
    Mouse,
    MouseConfig,
    bezier_path,
    step_durations,
)


class TestBezierPath:
    def test_endpoints_match(self):
        path = bezier_path((0, 0), (100, 50), steps=20, rng=random.Random(1))
        assert path[0] == (0, 0)
        # smoothstep with t=1 gives exactly the end point.
        assert math.isclose(path[-1][0], 100, abs_tol=1e-6)
        assert math.isclose(path[-1][1], 50, abs_tol=1e-6)

    def test_path_is_not_a_straight_line(self):
        rng = random.Random(42)
        path = bezier_path((0, 0), (200, 0), steps=40, curvature=0.5, rng=rng)
        # Straight line would have y == 0 everywhere.
        max_dev = max(abs(p[1]) for p in path)
        assert max_dev > 1.0

    def test_zero_distance_returns_trivial_path(self):
        path = bezier_path((5, 5), (5, 5), steps=10)
        assert path[0] == path[-1] == (5, 5)

    def test_invalid_steps_raises(self):
        with pytest.raises(ValueError):
            bezier_path((0, 0), (1, 1), steps=1)

    def test_different_seeds_produce_different_paths(self):
        p1 = bezier_path((0, 0), (100, 100), steps=10, rng=random.Random(1))
        p2 = bezier_path((0, 0), (100, 100), steps=10, rng=random.Random(2))
        assert p1 != p2


class TestStepDurations:
    def test_sums_to_total(self):
        durs = step_durations(1.0, 20, rng=random.Random(0))
        assert math.isclose(sum(durs), 1.0, rel_tol=1e-9)
        assert len(durs) == 20
        assert all(d >= 0 for d in durs)

    def test_zero_total(self):
        durs = step_durations(0.0, 5)
        assert all(d == 0 for d in durs)

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            step_durations(1.0, 0)
        with pytest.raises(ValueError):
            step_durations(-1.0, 5)


class TestMouse:
    def test_move_to_generates_many_intermediate_points(self):
        backend = FakeBackend(start=(0, 0))
        mouse = Mouse(backend=backend, rng=random.Random(0), sleep=lambda _s: None)
        mouse.move_to(300, 200)
        moves = [evt for evt in backend.events if evt[0] == "move"]
        # Plenty of intermediate steps, not just a single jump.
        assert len(moves) > 10
        assert moves[-1][1] == (300, 200)

    def test_move_to_same_point_is_a_single_move(self):
        backend = FakeBackend(start=(50, 60))
        mouse = Mouse(backend=backend, rng=random.Random(0), sleep=lambda _s: None)
        mouse.move_to(50, 60)
        moves = [evt for evt in backend.events if evt[0] == "move"]
        assert moves == [("move", (50, 60))]

    def test_click_moves_and_clicks(self):
        backend = FakeBackend(start=(0, 0))
        mouse = Mouse(
            backend=backend,
            rng=random.Random(3),
            sleep=lambda _s: None,
            config=MouseConfig(click_jitter_px=0),
        )
        mouse.click(100, 100)
        move_events = [e for e in backend.events if e[0] == "move"]
        click_events = [e for e in backend.events if e[0] == "click"]
        assert move_events[-1] == ("move", (100, 100))
        assert click_events == [("click", ("left",))]

    def test_click_without_coords_only_clicks(self):
        backend = FakeBackend(start=(10, 10))
        mouse = Mouse(backend=backend, sleep=lambda _s: None)
        mouse.click()
        move_events = [e for e in backend.events if e[0] == "move"]
        click_events = [e for e in backend.events if e[0] == "click"]
        assert move_events == []
        assert click_events == [("click", ("left",))]

    def test_double_click_uses_backend(self):
        backend = FakeBackend()
        mouse = Mouse(backend=backend, sleep=lambda _s: None)
        mouse.double_click(5, 5)
        assert ("double_click", ("left",)) in backend.events

    def test_type_text_forwards_to_backend(self):
        backend = FakeBackend()
        mouse = Mouse(backend=backend, sleep=lambda _s: None)
        mouse.type_text("hi", interval=0.05)
        assert ("type", ("hi", 0.05)) in backend.events

    def test_click_jitter_stays_close(self):
        backend = FakeBackend()
        mouse = Mouse(
            backend=backend,
            rng=random.Random(7),
            sleep=lambda _s: None,
            config=MouseConfig(click_jitter_px=3),
        )
        mouse.click(200, 150)
        final = [e for e in backend.events if e[0] == "move"][-1][1]
        assert abs(final[0] - 200) <= 3
        assert abs(final[1] - 150) <= 3
