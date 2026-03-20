"""
ocr.py  –  Step 3: Optical Character Recognition

Preprocesses the aligned plate image (grayscale → blur → Otsu threshold)
then feeds it to Tesseract in single-line mode with a strict A-Z / 0-9
whitelist.  The raw Tesseract output is sanitised before it is returned.
"""

import os
import re
import cv2
import pytesseract

# ── Tesseract binary path (only needed on Windows) ───────────────────────────
_TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(_TESSERACT_WIN):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN

# Tesseract config:  LSTM engine, treat image as a single text line,
# allow only capital letters and digits.
_TESS_CONFIG = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)


def _binarise_plate(plate_bgr):
    """Convert a colour plate crop to a clean binary image for OCR."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)    # edge-preserving denoise
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _sanitise(raw_text: str) -> str:
    """Strip whitespace / punctuation and upper-case everything."""
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper().strip())


def read_plate_text(plate_img):
    """Run Tesseract on *plate_img* and return ``(clean_text, binary_img)``.

    *clean_text* contains only uppercase letters and digits (may be empty if
    nothing was recognised).
    """
    binary = _binarise_plate(plate_img)
    raw    = pytesseract.image_to_string(binary, config=_TESS_CONFIG)
    return _sanitise(raw), binary