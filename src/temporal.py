"""
temporal.py  –  Temporal Majority-Vote Confirmation

OCR on a single frame is unreliable.  This module collects the last N
readings and only "confirms" a plate once the same text appears at
least *threshold* times – a simple majority vote that filters noise.
"""

from collections import Counter, deque


class TemporalConfirm:
    """Sliding-window majority voter for plate strings."""

    def __init__(self, window_size: int = 10, threshold: int = 3):
        self._window   = deque(maxlen=window_size)
        self._threshold = threshold
        self._last_confirmed: str | None = None

    # ── public API ────────────────────────────────────────────────────────
    def update(self, plate_text: str) -> str | None:
        """Push a new reading and return the plate string if it just got
        confirmed, otherwise *None*.

        A plate is confirmed when:
        1. it is the most frequent entry in the window,
        2. its count ≥ threshold, **and**
        3. it is not the same plate we last confirmed (prevents duplicates).
        """
        if not plate_text:
            return None

        self._window.append(plate_text)

        best, count = Counter(self._window).most_common(1)[0]
        if count >= self._threshold and best != self._last_confirmed:
            self._last_confirmed = best
            return best

        return None

    def reset(self) -> None:
        """Clear all history (e.g. when the camera scene changes)."""
        self._window.clear()
        self._last_confirmed = None