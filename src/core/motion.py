"""
ARCHIVO: src/core/motion.py
PROPÓSITO: Detecta si hay movimiento en la escena comparando frames consecutivos.
           Señala cuándo la escena se ha ESTABILIZADO (dejó de haber movimiento).

           En main.py usamos esto para no activar YOLO en cada frame,
           sino solo cuando el jugador ya puso las cartas y se quedó quieto.
           Esto es importante porque:
           - YOLO es lento (~100ms por frame): no podemos correrlo 30 veces/segundo
           - Si hay movimiento, la imagen sale borrosa → detección mala

CÓMO SE CONECTA:
  - main.py crea un MotionDetector y llama a update() con cada frame nuevo
  - config.py provee MOTION_THRESHOLD y STABILITY_SECONDS
"""

import time   # para medir cuántos segundos han pasado
import cv2    # OpenCV: procesamiento de imagen
import numpy as np  # arrays de números


class MotionDetector:
    """Detecta movimiento por diferencia de frames y señala cuándo la escena
    se estabiliza — útil para saber cuándo el jugador terminó de colocar cartas.

    Returns (motion_detected, just_stabilized) en cada llamada a update().
    just_stabilized es True durante exactamente un frame: el primero estable
    tras un período de movimiento.
    """

    def __init__(self, threshold: int = 30, stability_seconds: float = 2.0,
                 min_area: int = 1500):
        """Constructor.

        Parámetros:
          threshold: diferencia de brillo (0-255) entre frames para detectar movimiento.
                     30 = cualquier cambio de más de 30 unidades de gris se considera movimiento.
                     Más alto = menos sensible (ignora movimientos pequeños).
          stability_seconds: cuántos segundos sin movimiento para declarar escena estable.
          min_area: área mínima en píxeles² de la zona de movimiento para considerarlo real.
                    Filtra sombras y reflejos muy pequeños.
        """
        self.threshold         = threshold
        self.stability_seconds = stability_seconds
        self.min_area          = min_area

        # Frame anterior convertido a escala de grises.
        # Lo guardamos para compararlo con el siguiente frame.
        # None al inicio porque todavía no hemos procesado ningún frame.
        self._prev_gray: np.ndarray | None = None

        # Marca de tiempo del último momento en que detectamos movimiento.
        # time.monotonic() devuelve el tiempo transcurrido en segundos desde
        # que arrancó el sistema (no es la hora del reloj, es un contador que nunca retrocede).
        self._last_motion_time = time.monotonic()

        # Recuerda si el estado anterior era "estable" o "en movimiento".
        # Necesitamos esto para detectar la transición movimiento→estable (just_stabilized).
        self._was_stable = False

    def update(self, frame: np.ndarray) -> tuple[bool, bool]:
        """Procesa un frame nuevo y devuelve el estado de movimiento.

        Parámetro:
          frame: imagen BGR actual de la cámara

        Devuelve: (motion_detected, just_stabilized)
          motion_detected  — True si hay movimiento en este frame
          just_stabilized  — True SOLO en el primer frame estable después de movimiento
                             (es True durante exactamente 1 frame, luego vuelve a False)
        """

        # Convertimos el frame a escala de grises (1 canal de brillo en vez de 3 colores).
        # La detección de movimiento no necesita color, solo variaciones de brillo.
        # GaussianBlur suaviza la imagen para eliminar el ruido del sensor de la cámara.
        # (21, 21) es el tamaño del filtro de suavizado; debe ser impar.
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0
        )

        # Primer frame: no hay frame anterior para comparar.
        # Guardamos este como el "frame anterior" y declaramos que no hay movimiento.
        if self._prev_gray is None:
            self._prev_gray = gray
            return False, False

        # Calculamos la diferencia absoluta píxel a píxel entre el frame actual y el anterior.
        # Si un píxel tenía brillo 100 antes y ahora tiene 140, la diferencia es 40.
        # Si tenía 100 y sigue teniendo 100, la diferencia es 0 (sin movimiento en ese píxel).
        diff = cv2.absdiff(self._prev_gray, gray)

        # Umbralización: convertimos las diferencias en un mapa binario blanco/negro.
        # Diferencia > threshold → píxel BLANCO (movimiento)
        # Diferencia ≤ threshold → píxel NEGRO (sin movimiento)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Sumamos todos los píxeles blancos (cada uno vale 255).
        # Si la suma supera min_area * 255, hay suficiente "área de movimiento" para ser real.
        motion = int(mask.sum()) > self.min_area

        # Actualizamos el frame anterior para la próxima comparación.
        self._prev_gray = gray

        # Si hay movimiento, actualizamos el tiempo del último movimiento detectado.
        if motion:
            self._last_motion_time = time.monotonic()

        # Calculamos cuántos segundos han pasado desde el último movimiento.
        elapsed = time.monotonic() - self._last_motion_time

        # is_stable = True si han pasado más de stability_seconds sin movimiento.
        is_stable = elapsed >= self.stability_seconds

        # just_stabilized = True SOLO en el frame en que la escena pasa de
        # "en movimiento" (was_stable=False) a "estable" (is_stable=True).
        # En todos los demás frames es False.
        just_stabilized = is_stable and not self._was_stable

        # Guardamos el estado actual para la próxima llamada.
        self._was_stable = is_stable

        return motion, just_stabilized

    def reset(self) -> None:
        """Reinicia el detector como si acabara de arrancar.
        Se llama en main.py cuando el jugador empieza una nueva mano (tecla 'n' o 'r').
        Forzamos que just_stabilized dispare en el próximo frame estable."""
        self._prev_gray        = None
        self._last_motion_time = time.monotonic()
        self._was_stable       = False
