"""
Verifica que la Pi Camera funciona y muestra las zonas de detección.

Uso:
    python scripts/test_camera.py

Controles:
    s  → guardar foto en data/
    q  → salir
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.perception.camera import Camera
import config

_ZONE_COLORS = {
    'DEALER': (0,  100, 255),
    'PLAYER': (0,  220,   0),
    'BETS':   (255, 100,  0),
}


def _draw_zones(frame: cv2.typing.MatLike) -> None:
    h, w = frame.shape[:2]
    zones = [
        ('DEALER', config.ZONE_DEALER,  _ZONE_COLORS['DEALER']),
        ('PLAYER', config.ZONE_PLAYER,  _ZONE_COLORS['PLAYER']),
        ('BETS',   config.ZONE_BETTING, _ZONE_COLORS['BETS']),
    ]
    for label, zone, color in zones:
        y1 = int(h * zone['y_min'])
        y2 = int(h * zone['y_max'])
        cv2.rectangle(frame, (0, y1), (w - 1, y2), color, 2)
        cv2.putText(frame, label, (12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main() -> None:
    print("Abriendo cámara...")
    print("  s = guardar foto   |   q = salir")

    Path("data").mkdir(exist_ok=True)

    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        prev  = time.monotonic()
        count = 0
        fps   = 0.0

        while True:
            frame = cam.read()
            count += 1

            elapsed = time.monotonic() - prev
            if elapsed >= 0.5:
                fps   = count / elapsed
                count = 0
                prev  = time.monotonic()

            h, w = frame.shape[:2]
            _draw_zones(frame)

            cv2.putText(frame, f"{w}x{h}  {fps:.1f} fps",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "s=foto  q=salir",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Test Camara", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = f"data/test_{int(time.time())}.jpg"
                cv2.imwrite(path, frame)
                print(f"  Foto guardada: {path}")

    cv2.destroyAllWindows()
    print("Cámara cerrada.")


if __name__ == '__main__':
    main()
