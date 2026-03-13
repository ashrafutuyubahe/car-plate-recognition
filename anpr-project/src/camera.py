"""
camera.py — Step 0: Camera Validation
Open the webcam, display live frames, and exit cleanly on 'q'.
Run this first to confirm your camera works before running the pipeline.
"""

import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not opened. Check permissions or camera index.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Camera Test  |  Press q to quit", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
