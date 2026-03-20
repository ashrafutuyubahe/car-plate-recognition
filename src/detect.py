"""
detect.py  –  Step 1: License-Plate Detection

Uses a classical OpenCV contour pipeline (no deep learning):
  grayscale → bilateral blur → Canny edges → morphological close → contour search
Candidate rectangles are filtered by area and aspect ratio so that only
plate-shaped regions survive.
"""

import cv2
import numpy as np

# ── Tunable detection thresholds ──────────────────────────────────────────────
MIN_PLATE_AREA   = 1000    # ignore contours smaller than this (px²)
ASPECT_RATIO_MIN = 2.0     # plates are always wider than tall
ASPECT_RATIO_MAX = 6.5
MAX_CANDIDATES   = 5       # keep at most this many candidates per frame
# Morphology kernel – merges nearby character edges into one plate blob
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))


def _sort_corners(pts):
    """Order four corner points: top-left, top-right, bottom-right, bottom-left.
    Required for a consistent perspective transform later."""
    pts  = np.array(pts, dtype="float32")
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],       # top-left  (smallest x+y)
        pts[np.argmin(diff)],    # top-right (smallest x-y)
        pts[np.argmax(s)],       # bottom-right (largest x+y)
        pts[np.argmax(diff)],    # bottom-left  (largest x-y)
    ], dtype="float32")


def detect_plate(frame):
    """Scan *frame* for rectangular plate-like contours.

    Returns
    -------
    candidates : list[np.ndarray]
        Each element is a (4, 2) float32 array of ordered corner points.
        Sorted largest-first.  May be empty.
    debug_frame : np.ndarray
        Copy of *frame* with green boxes drawn around every candidate.
    """
    debug_frame = frame.copy()

    # 1. Convert to grayscale and smooth to reduce texture noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # 2. Edge detection
    edges = cv2.Canny(gray, 30, 200)

    # 3. Close small gaps so individual characters merge into one plate blob
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    # 4. Find external contours, keep only the 20 largest
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    # 5. Filter by geometry: must look like a plate (wide rectangle)
    scored = []  # (area, 4-point box)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (_cx, _cy), (w, h), _angle = rect
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        area   = w * h
        if ASPECT_RATIO_MIN <= aspect <= ASPECT_RATIO_MAX and area >= MIN_PLATE_AREA:
            box = cv2.boxPoints(rect).astype(np.int32)
            scored.append((area, box))

    # 6. Keep the N best candidates, largest first
    scored.sort(key=lambda pair: pair[0], reverse=True)
    scored = scored[:MAX_CANDIDATES]

    ordered_candidates = []
    for _area, quad in scored:
        cv2.drawContours(debug_frame, [quad], -1, (0, 255, 0), 2)
        ordered_candidates.append(_sort_corners(quad))

    return ordered_candidates, debug_frame