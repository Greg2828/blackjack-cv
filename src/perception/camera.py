"""
ARCHIVO: src/perception/camera.py
PROPÓSITO: Abstrae el acceso a la cámara.
           "Abstraer" significa que el resto del código no necesita saber
           cómo funciona la Pi Camera internamente.
           Solo llama a cam.read() y recibe un frame (imagen).

           También gestiona automáticamente el cierre de la cámara
           cuando termina el programa (usando el patrón "context manager").

CÓMO SE CONECTA:
  - main.py abre la cámara con: with Camera(...) as cam:
  - test_camera.py usa Camera para verificar que la Pi Camera funciona
  - capture_dataset.py usa Camera para tomar fotos para el dataset

NOTA: Usa picamera2 (la librería oficial de Raspberry Pi para Pi 5 + Bookworm).
      cv2.VideoCapture no funciona con la Pi Camera en Pi 5.
"""

import cv2
from picamera2 import Picamera2


class Camera:
    """Wrapper sobre Picamera2 para la Pi Camera Module en Raspberry Pi 5."""

    def __init__(self, source: int = 0, width: int = 1280, height: int = 720):
        """Abre la cámara y configura su resolución.

        Parámetros:
          source: índice de la cámara (0 = primera Pi Camera). Reservado para
                  compatibilidad futura; actualmente se usa siempre la cámara 0.
          width, height: resolución en píxeles.
        """
        # Picamera2() inicia la conexión con la Pi Camera a través de libcamera.
        # Es la librería oficial para Pi 5 con Raspberry Pi OS Bookworm.
        self._cam = Picamera2(camera_num=source)

        # RGB888: picamera2 devuelve los canales en orden R-G-B.
        # OpenCV trabaja en B-G-R, así que convertimos en read().
        cfg = self._cam.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self._cam.configure(cfg)
        self._cam.start()

        # Activamos los modos automáticos del IMX708:
        #   AfMode=2  → autofoco continuo (el sensor recalcula el foco constantemente)
        #   AfRange=0 → rango normal (0=normal, 1=macro, 2=full)
        #   AfSpeed=1 → respuesta rápida al cambiar la distancia
        #   AeEnable  → exposición automática (ajusta brillo según la luz)
        #   AwbEnable → balance de blancos automático (colores correctos bajo cualquier luz)
        self._cam.set_controls({
            "AfMode": 2,
            "AfRange": 0,
            "AfSpeed": 1,
            "AeEnable": True,
            "AwbEnable": True,
        })

    def read(self):
        """Lee y devuelve el siguiente frame (imagen) de la cámara.

        Devuelve: un array numpy de shape (height, width, 3) en formato BGR.
                  BGR = Blue, Green, Red (el orden de colores que usa OpenCV).
        """
        # picamera2 en Pi 5 devuelve BGR aunque el formato sea "RGB888".
        return self._cam.capture_array("main")

    def release(self) -> None:
        """Detiene la cámara y libera sus recursos del sistema.
        Importante: si no liberas la cámara, puede quedar bloqueada para otros programas."""
        self._cam.stop()
        self._cam.close()

    def __enter__(self):
        """Se llama al entrar al bloque 'with'. Devuelve self para poder usar 'as cam'."""
        return self

    def __exit__(self, *_):
        """Se llama al salir del bloque 'with' (con o sin error).
        Garantiza que la cámara siempre se cierre limpiamente."""
        self.release()
