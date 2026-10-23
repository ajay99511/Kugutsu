"""Hand-tracking input drivers (webcam MediaPipe now, ESP32 IMU later)."""


class HandTracker:
    """Placeholder interface; implement with MediaPipe in webcam_mediapipe.py."""

    def read(self):
        raise NotImplementedError

    def close(self):
        pass
