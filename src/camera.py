"""
camera.py  –  Webcam Initialisation

Opens a video-capture device at the requested resolution and verifies
that the device is actually available before returning.
"""

import cv2


def open_camera(index: int = 0, width: int = 1280, height: int = 720):
    """Return an opened ``cv2.VideoCapture`` or raise ``RuntimeError``."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera (index={index}).")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera] Opened camera {index}  –  {actual_w}×{actual_h}")
    return cap