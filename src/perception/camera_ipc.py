"""
ARCHIVO: src/perception/camera_ipc.py
PROPÓSITO: Alternativa a Camera que ejecuta picamera2 en un SUBPROCESO Python
           separado, comunicándose con el proceso principal vía memoria
           compartida (shared memory).

POR QUÉ EXISTE (ver README §13 — bug activo):
  picamera2 (GLib) + torch + cv2.imshow (Qt5) en el MISMO proceso provocan
  que la ventana de OpenCV salga en blanco — los event loops de GLib y Qt5
  se pisan. Si aislamos picamera2 en otro proceso, el proceso principal
  sólo tiene torch + cv2 (combinación que SÍ funciona, intento 5 del README).

ARQUITECTURA:
  ┌──────────────────────────┐         ┌──────────────────────────┐
  │ Proceso principal        │         │ Subproceso (spawn)       │
  │  - torch / YOLO          │  ◄────  │  - picamera2             │
  │  - cv2.imshow            │  shm    │  - escribe frames        │
  │  - lee frames de shm     │         │  - NO carga torch        │
  └──────────────────────────┘         └──────────────────────────┘

USO (drop-in replacement de Camera):
    from src.perception.camera_ipc import CameraIPC as Camera
    with Camera(width=1280, height=720) as cam:
        frame = cam.read()  # numpy uint8 BGR (H, W, 3)
"""
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


def _camera_worker(shm_name, width, height, ready_event, stop_event, lock):
    """Bucle del SUBPROCESO. Solo importa picamera2 + numpy.

    La importación de picamera2 está aquí DENTRO (no en el módulo) a propósito:
    así el proceso principal nunca carga picamera2 ni inicia su event loop de GLib.
    """
    # picamera2 sólo se importa en el subproceso → GLib vive aquí, no en el padre.
    from picamera2 import Picamera2

    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    cam.configure(cfg)
    cam.start()
    cam.set_controls({
        "AfMode": 2, "AfRange": 0, "AfSpeed": 1,
        "AeEnable": True, "AwbEnable": True,
    })

    # Abrimos el segmento de memoria compartida que ya creó el padre y lo
    # vemos como un array numpy normal. Cualquier escritura en `buf` es
    # visible inmediatamente en el proceso padre (es la misma RAM).
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

    try:
        while not stop_event.is_set():
            frame = cam.capture_array("main")  # numpy BGR (picamera2 en Pi 5)
            # Lock para evitar que el padre lea un frame a medio escribir
            # (tearing). El bloqueo es muy corto (sólo el memcpy).
            with lock:
                buf[:] = frame
            # Avisamos al padre de que hay un frame nuevo disponible.
            ready_event.set()
    finally:
        cam.stop()
        cam.close()
        shm.close()


class CameraIPC:
    """Cámara con misma interfaz que src.perception.camera.Camera, pero corre
    picamera2 en un subproceso aparte.

    Diseñado como reemplazo directo en scripts donde coexisten picamera2 +
    torch + cv2.imshow (ver README §13).
    """

    def __init__(self, source: int = 0, width: int = 1280, height: int = 720):
        # 'spawn' = Python NUEVO desde cero (no hereda imports del padre).
        # Esto garantiza que el subproceso no tiene torch cargado.
        ctx = mp.get_context('spawn')

        # Reservamos un segmento de memoria compartida del tamaño de un frame.
        # 1280×720×3 bytes ≈ 2.76 MB. El SO le da un nombre único accesible
        # desde cualquier proceso que lo abra por nombre.
        nbytes = width * height * 3
        self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self._buf = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=self._shm.buf
        )

        # Primitivas de sincronización entre los dos procesos:
        self._ready = ctx.Event()   # el worker la "set"-ea cuando hay frame nuevo
        self._stop = ctx.Event()    # el padre la "set"-ea para pedir cierre
        self._lock = ctx.Lock()     # evita lecturas a medio escribir

        self._proc = ctx.Process(
            target=_camera_worker,
            args=(self._shm.name, width, height,
                  self._ready, self._stop, self._lock),
            daemon=True,  # muere automáticamente si el padre se cierra
        )
        self._proc.start()

        # Bloqueamos hasta tener el primer frame (o timeout si la cámara
        # no responde — p.ej. no está conectada).
        if not self._ready.wait(timeout=15):
            self.release()
            raise RuntimeError(
                "El subproceso de la cámara no produjo ningún frame en 15s. "
                "Comprueba que la Pi Camera está conectada y que "
                "picamera2 funciona (prueba: python scripts/test_camera.py)."
            )

    def read(self):
        """Devuelve el frame más reciente (numpy uint8 BGR, shape (H,W,3)).

        Bloquea hasta 1 s esperando un frame nuevo. Si no llega, devuelve
        el último que hubiera (el buffer siempre contiene algo tras __init__).
        """
        self._ready.wait(timeout=1.0)
        self._ready.clear()
        # Copiamos bajo lock para que el worker no sobrescriba mientras leemos.
        # .copy() es importante: si devolvemos self._buf directamente, el
        # consumidor vería el siguiente frame en cuanto el worker escribiera.
        with self._lock:
            return self._buf.copy()

    def release(self) -> None:
        """Detiene el subproceso y libera la memoria compartida."""
        self._stop.set()
        if self._proc.is_alive():
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                # Cinturón + tirantes: si no respondió al stop_event, lo matamos.
                self._proc.terminate()
                self._proc.join(timeout=2)
        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
