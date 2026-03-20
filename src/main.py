"""
main.py  –  ANPR Pipeline Entry Point

Captures live video and runs the full pipeline on every frame:

    Camera → Detect → Align → OCR → Validate → Temporal vote → Store

Press  q  to quit  |  s  to save debug screenshots manually.
"""

import os
import cv2

from camera   import open_camera
from detect   import detect_plate
from align    import align_plate
from ocr      import read_plate_text
from validate import is_valid_plate, extract_plate
from temporal import TemporalConfirm
from storage  import PlateStorage

# Directories that are created on demand
_SCREENSHOTS_DIR = "screenshots"
_CAPTURES_DIR    = "data/captures"


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_debug_screenshots(frame, aligned, ocr_img):
    """Dump the three main visual stages to disk."""
    os.makedirs(_SCREENSHOTS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(_SCREENSHOTS_DIR, "detection.png"), frame)
    if aligned is not None:
        cv2.imwrite(os.path.join(_SCREENSHOTS_DIR, "alignment.png"), aligned)
    if ocr_img is not None:
        cv2.imwrite(os.path.join(_SCREENSHOTS_DIR, "ocr.png"), ocr_img)


def _overlay_text(img, text, origin, colour, scale=0.8):
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    cap      = open_camera()
    voter    = TemporalConfirm(window_size=10, threshold=3)
    storage  = PlateStorage("data/plates.csv")

    print("[main] Pipeline running.  q = quit  |  s = screenshot")

    prev_aligned = None
    prev_ocr     = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[main] Camera read failed – exiting.")
            break

        # ── Step 1  Detection ──────────────────────────────────────────
        candidates, debug_frame = detect_plate(frame)

        aligned_plate = None
        ocr_img       = None
        plate_text    = ""
        valid         = False
        confirmed     = None
        need_autosave = False

        # ── Steps 2-6  (run on each candidate until one is confirmed) ─
        for corners in candidates:
            # Step 2  Alignment
            warped = align_plate(frame, corners)
            if warped is None:
                continue

            # Step 3  OCR
            text, binary = read_plate_text(warped)
            aligned_plate = warped
            ocr_img       = binary
            plate_text    = text

            # Step 4  Validation
            if not is_valid_plate(text):
                continue
            valid      = True
            plate_text = extract_plate(text)

            # Step 5  Temporal confirmation
            confirmed = voter.update(plate_text)

            # Step 6  Storage
            if confirmed:
                os.makedirs(_CAPTURES_DIR, exist_ok=True)
                img_path = os.path.join(_CAPTURES_DIR, f"{confirmed}.png")

                if storage.save_plate(confirmed, img_path):
                    cv2.imwrite(img_path, aligned_plate)
                    print(f"[SAVED] {confirmed}  →  {img_path}")
                    need_autosave = True

            break     # stop after first valid candidate

        # ── HUD overlay ───────────────────────────────────────────────
        display = debug_frame.copy()
        _overlay_text(display,
                      f"OCR: {plate_text or 'N/A'}  |  valid: {valid}",
                      (20, 40), (0, 255, 255))
        if confirmed:
            _overlay_text(display, f"CONFIRMED: {confirmed}",
                          (20, 80), (0, 255, 0), scale=0.9)

        cv2.imshow("Detection", display)

        if aligned_plate is not None:
            cv2.imshow("Aligned Plate", aligned_plate)
            prev_aligned = aligned_plate
        if ocr_img is not None:
            cv2.imshow("OCR Input", ocr_img)
            prev_ocr = ocr_img

        if need_autosave:
            _save_debug_screenshots(display, aligned_plate, ocr_img)
            print("[main] Auto-saved debug screenshots.")

        # ── keyboard ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            _save_debug_screenshots(display, prev_aligned, prev_ocr)
            print("[main] Manual screenshots saved.")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()