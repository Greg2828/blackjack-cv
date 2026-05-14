"""
ARCHIVO: scripts/test_camera.py
PROPÓSITO: Verifica que la Pi Camera funciona correctamente y
           muestra las 3 zonas de detección (dealer/player/betting)
           superpuestas sobre el feed en vivo.

           Úsalo ANTES de empezar a usar el sistema para:
           1. Verificar que la cámara se abre sin errores
           2. Comprobar que las zonas están bien posicionadas sobre la mesa
           3. Ajustar ZONE_DEALER/ZONE_PLAYER/ZONE_BETTING en config.py si hace falta

Uso:
    python scripts/test_camera.py

Controles:
    s  → guardar foto en data/test_TIMESTAMP.jpg
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

# Colores para cada zona (formato BGR).
_ZONE_COLORS = {
    'DEALER': (0,  100, 255),   # naranja = zona del crupier (parte superior)
    'PLAYER': (0,  220,   0),   # verde   = zona del jugador (parte media)
    'BETS':   (255, 100,  0),   # azul    = zona de apuestas (parte inferior)
}


def _draw_zones(frame: cv2.typing.MatLike) -> None:
    """Dibuja los rectángulos de las 3 zonas sobre el frame.
    Los rectángulos muestran visualmente dónde busca el sistema las cartas.

    Si las zonas no corresponden a la posición real de las cartas en tu mesa,
    ajusta los valores en config.py (ZONE_DEALER, ZONE_PLAYER, ZONE_BETTING).
    """
    h, w = frame.shape[:2]

    # Lista de zonas a dibujar: (nombre, configuración desde config.py, color)
    zones = [
        ('DEALER', config.ZONE_DEALER,  _ZONE_COLORS['DEALER']),
        ('PLAYER', config.ZONE_PLAYER,  _ZONE_COLORS['PLAYER']),
        ('BETS',   config.ZONE_BETTING, _ZONE_COLORS['BETS']),
    ]

    for label, zone, color in zones:
        # Convertimos las fracciones de config.py a píxeles absolutos.
        y1 = int(h * zone['y_min'])
        y2 = int(h * zone['y_max'])

        # Dibujamos el rectángulo que abarca toda la anchura del frame.
        # (0, y1) = esquina superior izquierda, (w-1, y2) = esquina inferior derecha.
        # 2 = grosor del borde en píxeles.
        cv2.rectangle(frame, (0, y1), (w - 1, y2), color, 2)

        # Etiqueta del nombre de la zona en la esquina superior izquierda del rectángulo.
        cv2.putText(frame, label, (12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def main() -> None:
    """Bucle principal de prueba de cámara."""
    print("Abriendo cámara...")
    print("  s = guardar foto   |   q = salir")

    # Creamos la carpeta data/ si no existe (para guardar las fotos de prueba).
    Path("data").mkdir(exist_ok=True)

    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        prev  = time.monotonic()   # marca de tiempo para calcular FPS
        count = 0                   # contador de frames desde la última actualización de FPS
        fps   = 0.0                 # FPS actual (se actualiza cada 0.5 segundos)

        while True:
            frame = cam.read()
            count += 1

            # Actualizamos el contador de FPS cada 0.5 segundos.
            # FPS = frames_procesados / segundos_transcurridos
            elapsed = time.monotonic() - prev
            if elapsed >= 0.5:
                fps   = count / elapsed
                count = 0
                prev  = time.monotonic()

            h, w = frame.shape[:2]

            # Dibujamos las zonas sobre el frame.
            _draw_zones(frame)

            # Mostramos resolución y FPS en la esquina superior izquierda.
            cv2.putText(frame, f"{w}x{h}  {fps:.1f} fps",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Instrucciones en la parte inferior.
            cv2.putText(frame, "s=foto  q=salir",
                        (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Test Camara", frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Guardamos la foto con el timestamp Unix como nombre único.
                # int(time.time()) = segundos desde 1970 → siempre único.
                path = f"data/test_{int(time.time())}.jpg"
                cv2.imwrite(path, frame)
                print(f"  Foto guardada: {path}")

    cv2.destroyAllWindows()
    print("Cámara cerrada.")


if __name__ == '__main__':
    main()
