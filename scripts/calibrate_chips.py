"""
Calibración de fichas para detección por color (HSV).

Muestra el feed de la cámara (o una imagen). Haz clic en el centro de una
ficha para muestrear su color en una región de 15×15 px. Luego asigna esa
muestra a chip_1, chip_2 o chip_3 con las teclas 1 / 2 / 3.
Al terminar, presiona 'p' para imprimir el bloque listo para pegar en config.py.

Uso:
    python scripts/calibrate_chips.py                    # usa la cámara
    python scripts/calibrate_chips.py --image foto.jpg   # usa una imagen
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

WINDOW = "Calibración de fichas"
SAMPLE_HALF = 7          # radio de la región de muestreo (15×15 px)
HUE_TOL    = 12          # tolerancia en H
SAT_TOL    = 50          # tolerancia en S
VAL_TOL    = 50          # tolerancia en V

_mouse_pos = (0, 0)
_last_sample: dict | None = None          # {'lower': [...], 'upper': [...], 'mean': [...]}
_assigned: dict[str, dict | None] = {
    'chip_1': None,
    'chip_2': None,
    'chip_3': None,
}


def _on_mouse(event, x, y, flags, param):
    global _mouse_pos, _last_sample
    _mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param['frame']
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x1 = max(0, x - SAMPLE_HALF)
        x2 = min(frame.shape[1], x + SAMPLE_HALF + 1)
        y1 = max(0, y - SAMPLE_HALF)
        y2 = min(frame.shape[0], y + SAMPLE_HALF + 1)
        roi = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(float)
        mean = roi.mean(axis=0)
        std  = roi.std(axis=0)
        lower = [
            int(max(0,   mean[0] - HUE_TOL)),
            int(max(0,   mean[1] - SAT_TOL)),
            int(max(0,   mean[2] - VAL_TOL)),
        ]
        upper = [
            int(min(179, mean[0] + HUE_TOL)),
            int(min(255, mean[1] + SAT_TOL)),
            int(min(255, mean[2] + VAL_TOL)),
        ]
        _last_sample = {'lower': lower, 'upper': upper, 'mean': mean.tolist()}
        print(f"  Muestra → HSV medio: H={mean[0]:.0f} S={mean[1]:.0f} V={mean[2]:.0f} "
              f"| lower={lower} upper={upper}")


def _overlay(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    x, y = _mouse_pos
    hsv_px = cv2.cvtColor(frame[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0, 0]

    lines = [
        f"HSV bajo cursor: H={hsv_px[0]} S={hsv_px[1]} V={hsv_px[2]}",
        "Clic: muestrear  |  1/2/3: asignar a chip  |  p: imprimir  |  q: salir",
    ]
    for i, (chip, data) in enumerate(_assigned.items()):
        status = f"{data['lower']} → {data['upper']}" if data else "sin calibrar"
        lines.append(f"  {chip} (${config.CHIP_VALUES[chip]}): {status}")

    if _last_sample:
        lines.append(f"Última muestra: {_last_sample['lower']} → {_last_sample['upper']}")

    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

    cv2.circle(out, _mouse_pos, SAMPLE_HALF, (0, 255, 0), 1)
    return out


def _print_config() -> None:
    print("\n─── Pega esto en config.py ───────────────────────")
    print("CHIP_HSV_RANGES = {")
    for chip, data in _assigned.items():
        if data:
            print(f"    '{chip}': {{'lower': {data['lower']}, 'upper': {data['upper']}}},")
        else:
            print(f"    '{chip}': {{'lower': None, 'upper': None}},")
    print("}")
    print("──────────────────────────────────────────────────\n")


def main() -> None:
    global _last_sample

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help="Ruta a una imagen en lugar de la cámara")
    args = parser.parse_args()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 900, 560)

    if args.image:
        static_frame = cv2.imread(args.image)
        if static_frame is None:
            print(f"No se pudo abrir la imagen: {args.image}")
            sys.exit(1)
        get_frame = lambda: static_frame.copy()  # noqa: E731
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            print("No se pudo abrir la cámara. Usa --image para modo estático.")
            sys.exit(1)
        get_frame = lambda: cap.read()[1]  # noqa: E731

    frame_ref = {'frame': get_frame()}
    cv2.setMouseCallback(WINDOW, _on_mouse, frame_ref)

    print("Instrucciones:")
    print("  Haz CLIC en el centro de una ficha para muestrear su color.")
    print("  Presiona 1 / 2 / 3 para asignar la muestra a chip_1 / chip_2 / chip_3.")
    print("  Presiona 'p' para imprimir el bloque de config.")
    print("  Presiona 'q' para salir.\n")

    while True:
        frame = get_frame()
        if frame is None:
            break
        frame_ref['frame'] = frame
        cv2.imshow(WINDOW, _overlay(frame))

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2'), ord('3')) and _last_sample:
            chip = f"chip_{chr(key)}"
            _assigned[chip] = {k: v for k, v in _last_sample.items() if k != 'mean'}
            print(f"  Asignado {_last_sample['lower']} → {_last_sample['upper']} a {chip}")
        elif key == ord('p'):
            _print_config()

    cv2.destroyAllWindows()
    if not args.image:
        cap.release()

    if any(v is not None for v in _assigned.values()):
        _print_config()


if __name__ == '__main__':
    main()
