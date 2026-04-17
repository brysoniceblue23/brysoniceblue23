"""CLI smoke tests."""

from __future__ import annotations

import json

import pytest

from computer_use_agent.actions import Click, Done, MoveTo, Type, Wait
from computer_use_agent.cli import _action_from_dict, _parse_region


def test_parse_region_roundtrip():
    r = _parse_region("10, 20,30,40")
    assert r is not None
    assert (r.x, r.y, r.width, r.height) == (10, 20, 30, 40)


def test_parse_region_none():
    assert _parse_region(None) is None
    assert _parse_region("") is None


def test_parse_region_invalid():
    with pytest.raises(SystemExit):
        _parse_region("bogus")


def test_action_from_dict_known_ops():
    assert isinstance(_action_from_dict({"op": "move_to", "x": 1, "y": 2}), MoveTo)
    assert isinstance(_action_from_dict({"op": "click"}), Click)
    assert isinstance(_action_from_dict({"op": "type", "text": "hi"}), Type)
    assert isinstance(_action_from_dict({"op": "wait", "seconds": 0.1}), Wait)
    assert isinstance(_action_from_dict({"op": "done"}), Done)


def test_action_from_dict_unknown_op():
    with pytest.raises(SystemExit):
        _action_from_dict({"op": "nope"})


def test_action_from_dict_invalid_kwargs():
    with pytest.raises(SystemExit):
        _action_from_dict({"op": "move_to", "zzz": 1})


def test_json_script_shape():
    # Sanity: the examples in the module docstring remain valid.
    script = json.loads(
        """
        [
          {"op": "move_to", "x": 100, "y": 100, "duration": 0.3},
          {"op": "click",   "x": 100, "y": 100},
          {"op": "type",    "text": "hi"},
          {"op": "wait",    "seconds": 0.1},
          {"op": "done"}
        ]
        """
    )
    parsed = [_action_from_dict(s) for s in script]
    assert [type(a).__name__ for a in parsed] == [
        "MoveTo",
        "Click",
        "Type",
        "Wait",
        "Done",
    ]
