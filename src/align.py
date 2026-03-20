"""
align.py  –  Step 2: Perspective Correction

Given four corner points from the detector, warp the image region into a
flat, top-down view and resize it to a fixed 400×100 strip that the OCR
engine can work with consistently.
"""

import cv2
import numpy as np

# Target size for the rectified plate image (width × height).
PLATE_WIDTH  = 400
PLATE_HEIGHT = 100


def _warp_quadrilateral(image, corners):
    """Perspective-warp the quadrilateral defined by *corners* into a rectangle.

    Parameters
    ----------
    image   : source BGR frame
    corners : (4, 2) float32 – ordered TL, TR, BR, BL

    Returns
    -------
    np.ndarray or None  – warped image, or None if the quad is degenerate.
    """
    corners = np.array(corners, dtype="float32")
    tl, tr, br, bl = corners

    # Compute output dimensions from the longest pair of opposite edges
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w <= 0 or h <= 0:
        return None

    destination = np.array([
        [0,     0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0,     h - 1],
    ], dtype="float32")

    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (w, h))


def align_plate(frame, plate_corners):
    """Correct the perspective of the plate region and resize.

    Returns a 400×100 BGR image, or *None* if alignment fails.
    """
    if plate_corners is None:
        return None

    warped = _warp_quadrilateral(frame, plate_corners)
    if warped is None:
        return None

    return cv2.resize(warped, (PLATE_WIDTH, PLATE_HEIGHT))