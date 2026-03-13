"""
main.py — Full Live ANPR Pipeline
Integrates all five stages into a single continuous loop:

  Camera → Detection → Alignment → OCR → Validation → Temporal Confirmation → CSV Log

Pipeline overview:
  1. Capture a live frame from the webcam
  2. Detect plate-like regions via contour + aspect-ratio filtering
  3. Align (perspective-warp) the best candidate to 450×140 px
  4. Preprocess and run Tesseract OCR on the aligned plate
  5. Validate the raw OCR result against the plate regex
  6. Collect valid readings in a rolling buffer of BUFFER_SIZE frames
  7. Confirm the plate by majority vote over the buffer
  8. Write the confirmed plate + timestamp to data/plates.csv
     (with a COOLDOWN-second duplicate suppression guard)

Usage:
  python src/main.py

Press q to quit.
"""

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_AREA    = 600            # Min contour area to be a plate candidate (px²)
AR_MIN      = 2.0            # Min aspect ratio (width/height)
AR_MAX      = 8.0            # Max aspect ratio
W_OUT       = 450            # Output plate width after alignment (px)
H_OUT       = 140            # Output plate height after alignment (px)
PLATE_RE    = re.compile(r"[A-Z]{3}[0-9]{3}[A-Z]")  # Rwandan plate pattern
BUFFER_SIZE = 5              # Frames to accumulate before majority vote
COOLDOWN    = 10             # Seconds before the same plate can be saved again

# CSV log location: project_root/data/plates.csv
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
CSV_FILE  = _ROOT_DIR / "data" / "plates.csv"
CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

# Initialise CSV with header if it does not exist yet
if not CSV_FILE.exists():
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Plate Number", "Timestamp"])


# ── Detection ──────────────────────────────────────────────────────────────────
def find_plate_candidates(frame):
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)
    edges    = cv2.Canny(blur, 100, 200)
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


# ── Alignment ─────────────────────────────────────────────────────────────────
def order_points(pts):
    pts         = np.array(pts, dtype=np.float32)
    s           = pts.sum(axis=1)
    diff        = np.diff(pts, axis=1)
    top_left    = pts[np.argmin(s)]
    bottom_right= pts[np.argmax(s)]
    top_right   = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    return np.array(
        [top_left, top_right, bottom_right, bottom_left], dtype=np.float32
    )


def warp_plate(frame, rect):
    box = cv2.boxPoints(rect)
    src = order_points(box)
    dst = np.array(
        [[0, 0], [W_OUT - 1, 0], [W_OUT - 1, H_OUT - 1], [0, H_OUT - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (W_OUT, H_OUT))


# ── OCR ───────────────────────────────────────────────────────────────────────
def read_plate_text(plate_img):
    gray   = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    raw = pytesseract.image_to_string(
        thresh,
        config=(
            "--psm 8 --oem 3 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        ),
    )
    return raw.upper().replace(" ", "").replace("-", "")


# ── Validation ────────────────────────────────────────────────────────────────
def extract_valid_plate(text):
    m = PLATE_RE.search(text)
    return m.group(0) if m else None


# ── Temporal confirmation ─────────────────────────────────────────────────────
def majority_vote(buffer):
    if not buffer:
        return None
    return Counter(buffer).most_common(1)[0][0]


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Webcam not available. Check USB connection or camera index."
        )

    plate_buffer    = []
    last_saved_plate= None
    last_saved_time = 0.0

    print("ANPR pipeline running. Press q in any window to quit.")
    print(f"CSV log: {CSV_FILE}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis        = frame.copy()
        candidates = find_plate_candidates(frame)

        if candidates:
            # Work with the largest candidate
            rect     = max(candidates, key=lambda r: r[1][0] * r[1][1])
            box      = cv2.boxPoints(rect).astype(int)
            cv2.polylines(vis, [box], True, (0, 255, 0), 2)

            plate_img  = warp_plate(frame, rect)
            raw_text   = read_plate_text(plate_img)
            valid_plate= extract_valid_plate(raw_text)

            if valid_plate:
                plate_buffer.append(valid_plate)
                if len(plate_buffer) > BUFFER_SIZE:
                    plate_buffer.pop(0)

                confirmed = majority_vote(plate_buffer)

                # Annotate confirmed plate on video
                x = max(0, min(int(np.max(box[:, 0])) - 300, vis.shape[1] - 220))
                y = max(15, min(int(np.max(box[:, 1])) + 28, vis.shape[0] - 10))
                cv2.putText(
                    vis, f"CONFIRMED: {confirmed}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )

                # Save to CSV with cooldown + duplicate guard
                now = time.time()
                if (
                    confirmed
                    and confirmed != last_saved_plate
                    and (now - last_saved_time) > COOLDOWN
                ):
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([confirmed, ts])
                    print(f"[SAVED] {confirmed}  @ {ts}")
                    last_saved_plate = confirmed
                    last_saved_time  = now

            elif raw_text:
                # Show raw (invalid) OCR text in orange for debugging
                x = max(0, min(int(np.max(box[:, 0])) - 300, vis.shape[1] - 220))
                y = max(15, min(int(np.max(box[:, 1])) + 28, vis.shape[0] - 10))
                cv2.putText(
                    vis, f"OCR: {raw_text}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                )

        # HUD overlay
        cv2.putText(
            vis, "ANPR — Live Pipeline",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 150, 0), 2,
        )
        cv2.putText(
            vis, "Press q to quit",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        cv2.imshow("ANPR — Number Plate Recognition", vis)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")


if __name__ == "__main__":
    main()
