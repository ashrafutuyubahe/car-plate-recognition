"""
validate.py  –  Plate-Text Validation

Checks OCR output against known Rwandan (and common regional) plate formats
and extracts the best matching substring.  Falls back to a minimum-length
rule so short OCR noise is still rejected.
"""

import re

# Accepted plate patterns (unanchored – OCR may have leading/trailing junk)
_PLATE_PATTERNS = [
    re.compile(r"[A-Z]{3}[0-9]{3}[A-Z]?"),   # RAB123  or RAB123A
    re.compile(r"[A-Z]{2}[0-9]{3}[A-Z]{2}"),  # CD123AB
]

# If no pattern matches, we still accept text that has at least this many
# alphanumeric characters (catches unusual / foreign plates).
_MIN_PLAUSIBLE_LENGTH = 5


def extract_plate(text: str) -> str:
    """Return the plate substring that matches a known pattern, or "" ."""
    if not text:
        return ""
    for pat in _PLATE_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(0)
    return ""


def is_valid_plate(text: str) -> bool:
    """True when *text* matches at least one known plate format."""
    if not text or len(text) < _MIN_PLAUSIBLE_LENGTH:
        return False
    return any(pat.search(text) for pat in _PLATE_PATTERNS)