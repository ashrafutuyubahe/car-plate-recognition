"""
detect.py — Step 1: Plate Detection
Detects license plate candidate regions from a live camera frame using
contour analysis and geometric filtering (no deep learning required).

Detection logic:
  1. Convert frame to grayscale
  2. Reduce noise with Gaussian blur
  3. Find edges with Canny
  4. Extract contours
  5. Filter by minimum area and aspect ratio (wider than tall)
  6. Return plausible plate candidates as rotated rectangles

Run standalone to validate detection: python src/detect.py
"""

import cv2
import numpy as np

# ----- Tunable constants -----
MIN_AREA = 600          # Minimum contour area in pixels to be a plate candidate
AR_MIN   = 2.0          # Minimum aspect ratio (width/height) of a plate
AR_MAX   = 8.0          # Maximum aspect ratio


def find_plate_candidates(frame):
    """
    Analyse a BGR frame and return a list of cv2.minAreaRect tuples
    that are plausible license plate regions.
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue

        ar = max(w, h) / max(1.0, min(w, h))
        if AR_MIN <= ar <= AR_MAX:
            candidates.append(rect)

    return candidates


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis        = frame.copy()
        candidates = find_plate_candidates(frame)

        if candidates:
            msg   = f"Plate detected  ({len(candidates)} candidate(s))"
            color = (0, 255, 0)
            for rect in candidates:
                box = cv2.boxPoints(rect).astype(int)
                cv2.polylines(vis, [box], True, (0, 255, 0), 2)
        else:
            msg   = "Searching for plate..."
            color = (0, 200, 255)

        cv2.putText(vis, msg,               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Step 1 — Plate Detection", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
