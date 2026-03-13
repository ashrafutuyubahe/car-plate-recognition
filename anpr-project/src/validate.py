"""
validate.py — Step 4: Regex Validation of OCR Output
Validates the raw OCR text against the Rwandan number plate pattern:
  3 uppercase letters  +  3 digits  +  1 uppercase letter
  Example: RAA123A, RAB456C

Why validation is needed:
  - OCR may confuse B/8, O/0, S/5
  - National emblems or border elements may be mis-read as characters
  - A regex filter discards all outputs that do not match the expected format

Run standalone to validate + show stages: python src/validate.py
"""

import re

import cv2
import numpy as np
import pytesseract

from detect import find_plate_candidates
from align  import warp_plate
from ocr    import read_plate_text

# Rwandan number plate pattern: AAA999A
PLATE_RE = re.compile(r"[A-Z]{3}[0-9]{3}[A-Z]")


def extract_valid_plate(raw_text):
    """
    Search `raw_text` for a substring matching the plate regex.
    Returns the matched plate string, or None if no valid plate found.
    """
    text = raw_text.upper().replace(" ", "")
    m    = PLATE_RE.search(text)
    return m.group(0) if m else None


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
        plate_img  = None
        thresh     = None

        if candidates:
            rect      = max(candidates, key=lambda r: r[1][0] * r[1][1])
            box       = cv2.boxPoints(rect).astype(int)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)

            plate_img          = warp_plate(frame, rect)
            raw_text, thresh   = read_plate_text(plate_img)
            valid_plate        = extract_valid_plate(raw_text)

            x = max(0, min(int(np.max(box[:, 0])) - 300, vis.shape[1] - 200))
            y = max(0, min(int(np.max(box[:, 1])) + 25,  vis.shape[0] - 10))

            if valid_plate:
                cv2.putText(
                    vis, f"VALID: {valid_plate}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                msg   = f"Valid plate: {valid_plate}"
                color = (0, 255, 0)
            else:
                if raw_text:
                    cv2.putText(
                        vis, f"OCR: {raw_text}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )
                msg   = "No valid plate pattern"
                color = (0, 165, 255)
        else:
            msg   = "Searching for plate..."
            color = (0, 200, 255)

        cv2.putText(vis, msg,               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Step 4 — Validation Stage", vis)

        if plate_img is not None:
            cv2.imshow("Aligned Plate", plate_img)
        if thresh is not None:
            cv2.imshow("Thresholded Plate", thresh)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
