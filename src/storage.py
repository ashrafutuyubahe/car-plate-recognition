"""
storage.py  –  CSV Logging for Confirmed Plates

Every confirmed plate is appended to a CSV file with three columns:
    plate_number, timestamp, image_path

A short cooldown prevents the same plate from being logged repeatedly
when it stays in front of the camera.
"""

import csv
import os
from datetime import datetime

_CSV_HEADER   = ["plate_number", "timestamp", "image_path"]
_COOLDOWN_SEC = 15        # ignore the same plate within this window


class PlateStorage:
    """Append-only CSV store with per-plate cooldown."""

    def __init__(self, csv_path: str = "data/plates.csv"):
        self._path = csv_path
        self._cooldowns: dict[str, datetime] = {}

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Write header only for a brand-new (or empty) file
        if not os.path.isfile(self._path) or os.path.getsize(self._path) == 0:
            with open(self._path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(_CSV_HEADER)

    def save_plate(self, plate_text: str, image_path: str = "") -> bool:
        """Append a row and return *True*, or *False* if still in cooldown."""
        now = datetime.now()

        last = self._cooldowns.get(plate_text)
        if last and (now - last).total_seconds() < _COOLDOWN_SEC:
            return False

        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(self._path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([plate_text, timestamp, image_path])

        self._cooldowns[plate_text] = now
        return True