"""
ocr.py — Step 3: OCR (Optical Character Recognition)
Preprocesses the aligned plate image and passes it to Tesseract to extract
the alphanumeric characters from the plate.

OCR preprocessing pipeline:
  1. Grayscale conversion
  2. Mild Gaussian blur to reduce noise
  3. Otsu's binarisation (auto-threshold)
  4. Pass to pytesseract with whitelist: A-Z and 0-9 only

Tesseract config:
  --psm 8  → treat the image as a single word
  --oem 3  → use default (LSTM + legacy) engine

Run standalone to test OCR: python src/ocr.py
"""

import cv2
import numpy as np
import pytesseract

from detect import find_plate_candidates
from align  import warp_plate


def read_plate_text(plate_img):
    """
    Preprocess `plate_img` (a 450×140 BGR image) and return
    (raw_text, threshold_image).
    raw_text has spaces stripped and is all uppercase.
    """
    gray   = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        thresh,
        config=(
            "--psm 8 --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    )
    return text.strip().replace(" ", ""), thresh


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

            plate_img              = warp_plate(frame, rect)
            plate_text, thresh     = read_plate_text(plate_img)

            if plate_text:
                # Annotate text near the bounding box
                x = min(int(np.max(box[:, 0])), vis.shape[1] - 200)
                y = min(int(np.max(box[:, 1])) + 25, vis.shape[0] - 10)
                cv2.putText(
                    vis, plate_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

            msg   = f"OCR: {plate_text if plate_text else '(no text)'}"
            color = (0, 255, 0)
        else:
            msg   = "Searching for plate..."
            color = (0, 200, 255)

        cv2.putText(vis, msg,               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, "Press q to quit", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Step 3 — OCR Test", vis)

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
