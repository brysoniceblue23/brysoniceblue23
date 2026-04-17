"""Computer vision utilities: template matching and OCR.

This module gives the agent its *eyes*. Two complementary strategies are
provided:

* **Template matching** -- find a known icon/button anywhere on the
  screen using normalized cross-correlation. Supports multi-scale search
  so a template captured at 1x still matches on HiDPI displays.
* **OCR via ``pytesseract``** -- locate a specific piece of text on
  screen (e.g. "Submit") and return its bounding box. Falls back
  gracefully when tesseract is not installed.

Both strategies return a :class:`Detection` with a bounding box,
confidence score and the pixel the agent should click (the centre of
the box by default).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single hit returned by the vision system."""

    x: int
    y: int
    width: int
    height: int
    confidence: float
    label: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def match_template(
    haystack: np.ndarray,
    needle: np.ndarray,
    *,
    threshold: float = 0.85,
    scales: Sequence[float] = (1.0, 0.75, 1.25, 0.5, 1.5),
    label: str = "",
) -> Optional[Detection]:
    """Locate ``needle`` inside ``haystack`` using multi-scale NCC.

    Returns the best detection whose score exceeds ``threshold`` or
    ``None`` if nothing matches. The needle is resampled at each scale
    in ``scales`` to give some tolerance to DPI differences.
    """

    if haystack.size == 0 or needle.size == 0:
        return None

    hay = _to_gray(haystack)
    ndl = _to_gray(needle)
    best: Optional[Detection] = None

    for scale in scales:
        if scale <= 0:
            continue
        if scale == 1.0:
            tpl = ndl
        else:
            new_w = max(1, int(round(ndl.shape[1] * scale)))
            new_h = max(1, int(round(ndl.shape[0] * scale)))
            tpl = cv2.resize(ndl, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if tpl.shape[0] > hay.shape[0] or tpl.shape[1] > hay.shape[1]:
            continue

        result = cv2.matchTemplate(hay, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < threshold:
            continue
        if best is None or max_val > best.confidence:
            best = Detection(
                x=int(max_loc[0]),
                y=int(max_loc[1]),
                width=int(tpl.shape[1]),
                height=int(tpl.shape[0]),
                confidence=float(max_val),
                label=label,
            )

    return best


def match_template_all(
    haystack: np.ndarray,
    needle: np.ndarray,
    *,
    threshold: float = 0.85,
    nms_iou: float = 0.3,
    label: str = "",
) -> List[Detection]:
    """Return every non-overlapping occurrence of ``needle`` in ``haystack``."""

    if haystack.size == 0 or needle.size == 0:
        return []

    hay = _to_gray(haystack)
    ndl = _to_gray(needle)
    if ndl.shape[0] > hay.shape[0] or ndl.shape[1] > hay.shape[1]:
        return []

    result = cv2.matchTemplate(hay, ndl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(result >= threshold)
    raw: List[Detection] = [
        Detection(
            x=int(x),
            y=int(y),
            width=int(ndl.shape[1]),
            height=int(ndl.shape[0]),
            confidence=float(result[y, x]),
            label=label,
        )
        for x, y in zip(xs, ys)
    ]
    return _non_max_suppression(raw, iou_threshold=nms_iou)


def _non_max_suppression(detections: List[Detection], *, iou_threshold: float) -> List[Detection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for det in detections:
        if all(_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


def _iou(a: Detection, b: Detection) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    inter_w = max(0, min(ax2, bx2) - max(a.x, b.x))
    inter_h = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union else 0.0


def find_text(
    image: np.ndarray,
    query: str,
    *,
    min_confidence: float = 50.0,
    case_sensitive: bool = False,
) -> Optional[Detection]:
    """Find ``query`` inside ``image`` using pytesseract OCR.

    Returns ``None`` if tesseract is not installed on the host, the text
    cannot be found, or no candidate exceeds ``min_confidence`` (a 0-100
    score straight from tesseract).
    """

    try:
        import pytesseract
    except ImportError:  # pragma: no cover - import is tested via monkeypatching
        logger.warning("pytesseract is not installed; find_text() is a no-op")
        return None

    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("pytesseract failed: %s", exc)
        return None

    needle = query if case_sensitive else query.lower()
    best: Optional[Detection] = None
    for i, raw_text in enumerate(data.get("text", [])):
        text = raw_text.strip()
        if not text:
            continue
        compare = text if case_sensitive else text.lower()
        if needle not in compare:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_confidence:
            continue
        det = Detection(
            x=int(data["left"][i]),
            y=int(data["top"][i]),
            width=int(data["width"][i]),
            height=int(data["height"][i]),
            confidence=conf / 100.0,
            label=text,
        )
        if best is None or det.confidence > best.confidence:
            best = det
    return best


class Vision:
    """Convenience facade combining template matching and OCR.

    >>> v = Vision()
    >>> det = v.find_template(screen_bgr, button_png)
    >>> det = v.find_text(screen_bgr, "Submit")
    """

    def find_template(
        self,
        image: np.ndarray,
        template: np.ndarray,
        *,
        threshold: float = 0.85,
        label: str = "",
    ) -> Optional[Detection]:
        return match_template(image, template, threshold=threshold, label=label)

    def find_all_templates(
        self,
        image: np.ndarray,
        template: np.ndarray,
        *,
        threshold: float = 0.85,
        label: str = "",
    ) -> List[Detection]:
        return match_template_all(image, template, threshold=threshold, label=label)

    def find_text(
        self,
        image: np.ndarray,
        query: str,
        *,
        min_confidence: float = 50.0,
        case_sensitive: bool = False,
    ) -> Optional[Detection]:
        return find_text(
            image,
            query,
            min_confidence=min_confidence,
            case_sensitive=case_sensitive,
        )
