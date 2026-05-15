"""
ARCHIVO: scripts/test_detector.py
PROPÓSITO: Verifica que el modelo YOLOv8 detecta las cartas correctamente.

Uso:
    python scripts/test_detector.py

Controles:
    ESPACIO  → detectar cartas en el frame actual
    s        → guardar captura con detecciones
    q        → salir
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

import config
from src.perception.camera import Camera

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _box_color(conf):
    if conf >= 0.80: return (0, 210, 0)
    if conf >= 0.50: return (0, 165, 255)
    return (0, 0, 220)


def _draw_zones(frame):
    h, w = frame.shape[:2]
    for label, zone, color in [
        ('DEALER',  config.ZONE_DEALER,  (0, 100, 255)),
        ('PLAYER',  config.ZONE_PLAYER,  (0, 210,   0)),
        ('BETTING', config.ZONE_BETTING, (255, 100,  0)),
    ]:
        y1 = int(h * zone['y_min'])
        cv2.line(frame, (0, y1), (w, y1), color, 1)
        cv2.putText(frame, label, (8, y1 + 22), _FONT, 0.65, color, 2)


def _run_yolo(frame, model):
    h, w = frame.shape[:2]
    out  = frame.copy()
    _draw_zones(out)

    t0      = time.monotonic()
    results = model(frame, verbose=False)[0]
    ms      = (time.monotonic() - t0) * 1000

    d_max = int(h * config.ZONE_DEALER['y_max'])
    p_min = int(h * config.ZONE_PLAYER['y_min'])
    p_max = int(h * config.ZONE_PLAYER['y_max'])
    dealer_cards, player_cards = [], []

    for box in results.boxes:
        cls_idx = int(box.cls[0])
        conf    = float(box.conf[0])
        rank    = config.CARD_CLASSES[cls_idx]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        yc    = (y1 + y2) / 2
        color = _box_color(conf)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{rank} {conf*100:.0f}%"
        (lw, lh), _ = cv2.getTextSize(label, _FONT, 0.75, 2)
        ly = max(y1 - 8, 20)
        cv2.rectangle(out, (x1, ly - lh - 4), (x1 + lw + 6, ly + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ly), _FONT, 0.75, (0, 0, 0), 2)

        if yc < d_max:
            dealer_cards.append(f"{rank}({conf*100:.0f}%)")
        elif p_min <= yc < p_max:
            player_cards.append(f"{rank}({conf*100:.0f}%)")

    summary = (f"DEALER: {', '.join(dealer_cards) or '--'}   "
               f"PLAYER: {', '.join(player_cards) or '--'}   [{ms:.0f}ms]")
    cv2.rectangle(out, (0, h - 38), (w, h), (20, 20, 20), -1)
    cv2.putText(out, summary, (10, h - 12), _FONT, 0.58, (200, 200, 200), 1)
    cv2.rectangle(out, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(out, "Cualquier tecla = feed en vivo   s = guardar   q = salir",
                (10, 24), _FONT, 0.58, (160, 160, 160), 1)

    print(f"  {ms:.0f}ms | DEALER: {dealer_cards or ['--']} | PLAYER: {player_cards or ['--']}")
    return out


def main():
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        print(f"ERROR: modelo no encontrado en {model_path}")
        sys.exit(1)

    Path("data").mkdir(exist_ok=True)

    # Abrir cámara primero para que GLib/picamera2 se inicialice antes de torch
    print("Abriendo cámara...")
    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        for _ in range(20):
            cam.read()

        print("Cargando modelo YOLO (puede tardar 1-2 min en Pi 5)...")
        model = YOLO(str(model_path))
        print("Listo. Pulsa ESPACIO para detectar. Q para salir.")

        frozen    = None
        win_title = "Test Detector — YOLOv8"

        while True:
            if frozen is None:
                frame   = cam.read()
                display = frame.copy()
                _draw_zones(display)
                h, w = display.shape[:2]
                cv2.rectangle(display, (0, h - 38), (w, h), (20, 20, 20), -1)
                cv2.putText(display, "ESPACIO = detectar   s = guardar   q = salir",
                            (10, h - 12), _FONT, 0.6, (160, 160, 160), 1)
                cv2.imshow(win_title, display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    frozen = _run_yolo(frame, model)
                elif key == ord('s'):
                    path = f"data/detector_test_{int(time.time())}.jpg"
                    cv2.imwrite(path, display)
                    print(f"  Guardado: {path}")
            else:
                cv2.imshow(win_title, frozen)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    path = f"data/detector_test_{int(time.time())}.jpg"
                    cv2.imwrite(path, frozen)
                    print(f"  Guardado: {path}")
                elif key != 255:
                    frozen = None

    cv2.destroyAllWindows()
    print("Cerrado.")


if __name__ == '__main__':
    main()
