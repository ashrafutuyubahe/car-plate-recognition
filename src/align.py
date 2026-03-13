"""
align.py — Step 2: Plate Alignment (Perspective Rectification)
After a plate candidate is found, this module warps it into a fixed
450 × 140 pixel canonical image so OCR receives clean, upright text.

Alignment logic:
  1. Get four corner points of the rotated bounding box
  2. Order them: top-left, top-right, bottom-right, bottom-left
  3. Compute a perspective transform to a fixed-size destination rectangle
  4. Warp the frame region into the canonical output size

Run standalone to validate alignment: python src/align.py
"""

import cv2
import numpy as np

from detect import find_plate_candidates

# ----- Output plate size -----
W_OUT, H_OUT = 450, 140


def order_points(pts):
    """
    Order four corner points as: top-left, top-right, bottom-right, bottom-left.
    This ordering is required for a consistent perspective transform.
    """
    pts    = np.array(pts, dtype=np.float32)
    s      = pts.sum(axis=1)
    diff   = np.diff(pts, axis=1)

    top_left     = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right    = pts[np.argmin(diff)]
    bottom_left  = pts[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )


def warp_plate(frame, rect):
    """
    Perspective-warp the plate region described by `rect` (a cv2.minAreaRect
    result) into a canonical W_OUT × H_OUT image.
    """
    box = cv2.boxPoints(rect)
    src = order_points(box)

    dst = np.array([
        [0,         0        ],
        [W_OUT - 1, 0        ],
        [W_OUT - 1, H_OUT - 1],
        [0,         H_OUT - 1],
    ], dtype=np.float32)

    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (W_OUT, H_OUT))
    return warped


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
        best_plate = None

        if candidates:
            # Pick the largest candidate by area
            rect = max(candidates, key=lambda r: r[1][0] * r[1][1])
            box  = cv2.boxPoints(rect).astype(int)

            # Draw bounding box and corner circles
            cv2.polylines(vis, [box], True, (255, 0, 0), 2)
            for (x, y) in box:
                cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

            best_plate = warp_plate(frame, rect)
            msg   = "Plate aligned"
            color = (0, 255, 0)
        else:
            msg   = "Detecting plate..."
            color = (0, 200, 255)

        cv2.putText(vis, msg,               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Step 2 — Alignment Test", vis)

        if best_plate is not None:
            cv2.imshow("Aligned Plate (450×140)", best_plate)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
