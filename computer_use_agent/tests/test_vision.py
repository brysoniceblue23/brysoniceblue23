"""Tests for the computer-vision layer."""

from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest

from computer_use_agent.vision import (
    Detection,
    Vision,
    find_text,
    match_template,
    match_template_all,
)


def _patterned_tile(size: int = 40) -> np.ndarray:
    """A non-uniform tile so TM_CCOEFF_NORMED has a meaningful variance."""
    tile = np.zeros((size, size, 3), dtype=np.uint8)
    # Outer white border, dark interior, diagonal stripe -- unique pattern.
    tile[:, :] = (255, 255, 255)
    tile[3 : size - 3, 3 : size - 3] = (20, 20, 20)
    for i in range(size):
        if 0 <= i < size:
            tile[i, i] = (255, 0, 0)
    tile[size // 2, :] = (0, 255, 0)
    return tile


def _scene_with_square(square_xy: tuple[int, int], size: int = 40) -> np.ndarray:
    img = np.full((300, 400, 3), 20, dtype=np.uint8)
    x, y = square_xy
    img[y : y + size, x : x + size] = _patterned_tile(size)
    return img


def _template(size: int = 40) -> np.ndarray:
    return _patterned_tile(size)


class TestMatchTemplate:
    def test_finds_single_square(self):
        img = _scene_with_square((120, 80))
        det = match_template(img, _template(), label="square")
        assert det is not None
        assert det.label == "square"
        assert det.confidence > 0.9
        # Centre should land inside the square.
        cx, cy = det.center
        assert 115 < cx < 165
        assert 75 < cy < 125

    def test_returns_none_when_absent(self):
        img = np.full((200, 200, 3), 20, dtype=np.uint8)
        det = match_template(img, _template(), threshold=0.95)
        assert det is None

    def test_multi_scale_matches_resized_template(self):
        # Place an exact 2x-upscaled copy of the template into the scene so
        # the multi-scale search has a scale=2.0 match to find.
        tpl = _template(size=40)
        upscaled = cv2.resize(tpl, (80, 80), interpolation=cv2.INTER_NEAREST)
        img = np.full((300, 400, 3), 20, dtype=np.uint8)
        img[50:130, 50:130] = upscaled
        det = match_template(img, tpl, threshold=0.7, scales=(1.0, 1.5, 2.0))
        assert det is not None
        # The 2.0x scaled template should be the one that wins.
        assert det.width >= 70

    def test_rejects_oversize_template(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        tpl = np.zeros((50, 50, 3), dtype=np.uint8)
        assert match_template(img, tpl) is None


class TestMatchTemplateAll:
    def test_finds_multiple_non_overlapping(self):
        img = np.full((300, 400, 3), 20, dtype=np.uint8)
        tile = _patterned_tile(40)
        for (x, y) in [(30, 30), (200, 30), (30, 200)]:
            img[y : y + 40, x : x + 40] = tile
        dets = match_template_all(img, _template(), threshold=0.9)
        assert len(dets) == 3
        centres = sorted(d.center for d in dets)
        assert centres == sorted([(50, 50), (220, 50), (50, 220)])

    def test_non_max_suppression_dedupes(self):
        img = _scene_with_square((100, 100))
        dets = match_template_all(img, _template(), threshold=0.9)
        assert len(dets) == 1


class TestDetection:
    def test_center_and_bbox(self):
        d = Detection(x=10, y=20, width=40, height=60, confidence=0.9)
        assert d.center == (30, 50)
        assert d.bbox == (10, 20, 40, 60)


class _FakeTesseract(types.ModuleType):
    """Stand-in for the ``pytesseract`` module used to test ``find_text``."""

    class Output:  # noqa: D401
        DICT = "dict"

    def __init__(self, data: dict):
        super().__init__("pytesseract")
        self._data = data

    def image_to_data(self, image, output_type=None):
        return self._data


@pytest.fixture
def install_fake_tesseract(monkeypatch):
    def _install(data: dict) -> _FakeTesseract:
        fake = _FakeTesseract(data)
        monkeypatch.setitem(sys.modules, "pytesseract", fake)
        return fake

    return _install


class TestFindText:
    def test_finds_matching_string(self, install_fake_tesseract):
        install_fake_tesseract(
            {
                "text": ["", "Hello", "Submit", "World"],
                "conf": ["-1", "90", "85", "40"],
                "left": [0, 10, 120, 300],
                "top": [0, 40, 60, 80],
                "width": [0, 50, 70, 55],
                "height": [0, 20, 22, 20],
            }
        )
        img = np.zeros((200, 500, 3), dtype=np.uint8)
        det = find_text(img, "submit")
        assert det is not None
        assert det.label == "Submit"
        assert det.center == (155, 71)

    def test_respects_case_sensitive_flag(self, install_fake_tesseract):
        install_fake_tesseract(
            {
                "text": ["submit"],
                "conf": ["95"],
                "left": [0],
                "top": [0],
                "width": [10],
                "height": [10],
            }
        )
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert find_text(img, "Submit", case_sensitive=True) is None
        assert find_text(img, "submit", case_sensitive=True) is not None

    def test_returns_none_below_confidence(self, install_fake_tesseract):
        install_fake_tesseract(
            {
                "text": ["Submit"],
                "conf": ["10"],
                "left": [0],
                "top": [0],
                "width": [10],
                "height": [10],
            }
        )
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        assert find_text(img, "Submit", min_confidence=50) is None

    def test_returns_none_when_tesseract_missing(self, monkeypatch):
        # Force the import inside ``find_text`` to fail.
        monkeypatch.setitem(sys.modules, "pytesseract", None)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert find_text(img, "hi") is None


def test_vision_facade_delegates(install_fake_tesseract):
    v = Vision()
    img = _scene_with_square((100, 100))
    det = v.find_template(img, _template())
    assert det is not None

    install_fake_tesseract(
        {
            "text": ["OK"],
            "conf": ["80"],
            "left": [5],
            "top": [7],
            "width": [20],
            "height": [10],
        }
    )
    text_det = v.find_text(img, "ok")
    assert text_det is not None
    assert text_det.center == (15, 12)


def test_template_match_works_on_png_roundtrip(tmp_path):
    img = _scene_with_square((50, 50))
    tpl_path = tmp_path / "tpl.png"
    cv2.imwrite(str(tpl_path), _template())
    loaded = cv2.imread(str(tpl_path))
    assert loaded is not None
    det = match_template(img, loaded)
    assert det is not None
