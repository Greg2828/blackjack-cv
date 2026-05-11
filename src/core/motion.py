import time
import cv2
import numpy as np


class MotionDetector:
    """Detecta movimiento por diferencia de frames y señala cuándo la escena
    se estabiliza — útil para saber cuándo el jugador terminó de colocar cartas.

    Returns (motion_detected, just_stabilized) en cada llamada a update().
    just_stabilized es True durante exactamente un frame: el primero estable
    tras un período de movimiento.
    """

    def __init__(self, threshold: int = 30, stability_seconds: float = 2.0,
                 min_area: int = 1500):
        self.threshold = threshold
        self.stability_seconds = stability_seconds
        self.min_area = min_area

        self._prev_gray: np.ndarray | None = None
        self._last_motion_time = time.monotonic()
        self._was_stable = False

    def update(self, frame: np.ndarray) -> tuple[bool, bool]:
        """Procesa un frame nuevo.

        Returns:
            motion_detected  — True si hay movimiento este frame
            just_stabilized  — True en el primer frame estable después de movimiento
        """
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0
        )

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, False

        diff = cv2.absdiff(self._prev_gray, gray)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        motion = int(mask.sum()) > self.min_area

        self._prev_gray = gray

        if motion:
            self._last_motion_time = time.monotonic()

        elapsed = time.monotonic() - self._last_motion_time
        is_stable = elapsed >= self.stability_seconds
        just_stabilized = is_stable and not self._was_stable
        self._was_stable = is_stable

        return motion, just_stabilized

    def reset(self) -> None:
        self._prev_gray = None
        self._last_motion_time = time.monotonic()
        self._was_stable = False
