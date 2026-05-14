"""
ARCHIVO: scripts/calibrate_chips.py
PROPÓSITO: Herramienta interactiva para calibrar los rangos de color HSV de tus fichas.

           El detector de fichas (chip_detector.py) usa rangos de color en espacio HSV
           para identificar cada tipo de ficha. Estos rangos son específicos de TUS fichas
           bajo TU iluminación, así que hay que medirlos con este script.

           Proceso:
           1. Ejecuta este script (con cámara o con una foto de tus fichas)
           2. Haz CLIC en el centro de una ficha para muestrear su color
           3. Pulsa 1, 2 o 3 para asignar la muestra a chip_1, chip_2 o chip_3
           4. Pulsa 'p' para imprimir el bloque listo para pegar en config.py
           5. Copia ese bloque en config.py (reemplaza los None)

Uso:
    python scripts/calibrate_chips.py                    # usa la cámara
    python scripts/calibrate_chips.py --image foto.jpg   # usa una imagen estática
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

WINDOW      = "Calibración de fichas"  # nombre de la ventana
SAMPLE_HALF = 7    # la región de muestreo es un cuadrado de (2×7+1)×(2×7+1) = 15×15 px
                   # muestrea 225 píxeles alrededor del punto donde haces clic

# Tolerancias para los rangos HSV:
# Al hacer clic, calculamos la media HSV de la región → añadimos/restamos estas tolerancias
# para crear el rango [lower, upper] que acepta colores "similares".
HUE_TOL = 12   # Hue: ±12 de rango (el tono del color, ej: rojo, azul, verde)
SAT_TOL = 50   # Saturation: ±50 (qué tan puro vs. apagado es el color)
VAL_TOL = 50   # Value: ±50 (qué tan brillante vs. oscuro)

# Variables globales compartidas entre la función del ratón y el bucle principal.
# (En Python, las variables globales en scripts simples como este son aceptables).
_mouse_pos   = (0, 0)    # posición actual del cursor sobre la imagen
_last_sample: dict | None = None   # última muestra tomada con clic
_assigned: dict[str, dict | None] = {
    'chip_1': None,   # None = sin calibrar
    'chip_2': None,
    'chip_3': None,
}


def _on_mouse(event, x, y, flags, param):
    """Función callback que OpenCV llama cada vez que hay un evento del ratón.

    Parámetros:
      event: tipo de evento (clic, movimiento, etc.)
      x, y: coordenadas del cursor en píxeles
      flags: teclas modificadoras (Shift, Ctrl...) — no usamos
      param: datos extra que pasamos al registrar el callback
             (en nuestro caso, un dict con el frame actual)
    """
    global _mouse_pos, _last_sample

    # Actualizamos la posición del cursor para el overlay.
    _mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # El usuario hizo CLIC IZQUIERDO → muestreamos el color en esa zona.
        frame = param['frame']

        # Convertimos el frame de BGR a HSV para muestrear el color.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Definimos la región de muestreo: 15×15 píxeles centrados en (x, y).
        # max/min aseguran que no nos salgamos de los bordes del frame.
        x1 = max(0, x - SAMPLE_HALF)
        x2 = min(frame.shape[1], x + SAMPLE_HALF + 1)
        y1 = max(0, y - SAMPLE_HALF)
        y2 = min(frame.shape[0], y + SAMPLE_HALF + 1)

        # roi = Region Of Interest = el recorte de 15×15 píxeles.
        # .reshape(-1, 3) convierte el array 2D de píxeles a una lista de colores [H, S, V].
        # .astype(float) convierte a float para calcular media y desviación estándar.
        roi  = hsv[y1:y2, x1:x2].reshape(-1, 3).astype(float)
        mean = roi.mean(axis=0)    # media de H, S, V en la región
        std  = roi.std(axis=0)     # desviación estándar (no usada en los rangos, pero informativa)

        # Calculamos el rango [lower, upper] con las tolerancias.
        # int(max(0, ...)) y int(min(179/255, ...)) aseguran que los valores sean válidos.
        lower = [
            int(max(0,   mean[0] - HUE_TOL)),
            int(max(0,   mean[1] - SAT_TOL)),
            int(max(0,   mean[2] - VAL_TOL)),
        ]
        upper = [
            int(min(179, mean[0] + HUE_TOL)),   # H va de 0 a 179 en OpenCV (no 0-360)
            int(min(255, mean[1] + SAT_TOL)),   # S va de 0 a 255
            int(min(255, mean[2] + VAL_TOL)),   # V va de 0 a 255
        ]

        _last_sample = {'lower': lower, 'upper': upper, 'mean': mean.tolist()}
        print(f"  Muestra → HSV medio: H={mean[0]:.0f} S={mean[1]:.0f} V={mean[2]:.0f} "
              f"| lower={lower} upper={upper}")


def _overlay(frame: np.ndarray) -> np.ndarray:
    """Dibuja información útil sobre el frame:
      - Valores HSV del píxel bajo el cursor
      - Estado de calibración de cada ficha
      - Instrucciones de uso
      - Círculo de la zona de muestreo alrededor del cursor
    """
    out = frame.copy()
    x, y = _mouse_pos

    # Obtenemos los valores HSV del píxel exactamente bajo el cursor.
    # [y:y+1, x:x+1] = recorte de 1×1 píxel → convertimos a HSV → tomamos el primer (único) píxel.
    hsv_px = cv2.cvtColor(frame[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)[0, 0]

    # Construimos las líneas de texto para el overlay.
    lines = [
        f"HSV bajo cursor: H={hsv_px[0]} S={hsv_px[1]} V={hsv_px[2]}",
        "Clic: muestrear  |  1/2/3: asignar a chip  |  p: imprimir  |  q: salir",
    ]

    # Estado de calibración de cada tipo de ficha.
    for i, (chip, data) in enumerate(_assigned.items()):
        status = f"{data['lower']} → {data['upper']}" if data else "sin calibrar"
        lines.append(f"  {chip} (${config.CHIP_VALUES[chip]}): {status}")

    # Última muestra tomada.
    if _last_sample:
        lines.append(f"Última muestra: {_last_sample['lower']} → {_last_sample['upper']}")

    # Dibujamos cada línea de texto.
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

    # Círculo verde alrededor del cursor del tamaño de la zona de muestreo.
    # Ayuda al usuario a saber exactamente qué área está muestreando.
    cv2.circle(out, _mouse_pos, SAMPLE_HALF, (0, 255, 0), 1)
    return out


def _print_config() -> None:
    """Imprime en la terminal el bloque de código Python listo para
    copiar y pegar en config.py, reemplazando el CHIP_HSV_RANGES actual."""
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
    """Bucle principal de calibración."""
    global _last_sample

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help="Ruta a una imagen en lugar de la cámara")
    args = parser.parse_args()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 900, 560)

    if args.image:
        # Modo imagen estática: cargamos la foto una sola vez.
        static_frame = cv2.imread(args.image)
        if static_frame is None:
            print(f"No se pudo abrir la imagen: {args.image}")
            sys.exit(1)
        # lambda: función anónima de una línea.
        # get_frame() siempre devuelve una copia de la imagen estática.
        get_frame = lambda: static_frame.copy()
    else:
        # Modo cámara: capturamos frames en tiempo real.
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            print("No se pudo abrir la cámara. Usa --image para modo estático.")
            sys.exit(1)
        get_frame = lambda: cap.read()[1]   # cap.read() devuelve (ok, frame), tomamos frame

    # frame_ref es un diccionario mutable compartido con el callback del ratón.
    # Usamos un dict en vez de una variable normal porque los callbacks de OpenCV
    # no pueden modificar variables de la función exterior directamente.
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
        frame_ref['frame'] = frame   # actualizamos el frame en el dict compartido
        cv2.imshow(WINDOW, _overlay(frame))

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2'), ord('3')) and _last_sample:
            # Asignamos la última muestra al chip correspondiente.
            chip = f"chip_{chr(key)}"   # chr(ord('1')) = '1' → 'chip_1'
            _assigned[chip] = {k: v for k, v in _last_sample.items() if k != 'mean'}
            print(f"  Asignado {_last_sample['lower']} → {_last_sample['upper']} a {chip}")
        elif key == ord('p'):
            _print_config()

    cv2.destroyAllWindows()
    if not args.image:
        cap.release()

    # Al salir, si hay alguna ficha calibrada, imprimimos el config para que no se pierda.
    if any(v is not None for v in _assigned.values()):
        _print_config()


if __name__ == '__main__':
    main()
