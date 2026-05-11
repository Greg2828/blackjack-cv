import cv2


class Camera:
    """Wrapper sobre VideoCapture para la Pi Camera o cualquier fuente de vídeo."""

    def __init__(self, source: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara (source={source})")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("No se pudo leer frame de la cámara")
        return frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
