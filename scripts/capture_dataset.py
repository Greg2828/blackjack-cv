"""
ARCHIVO: scripts/capture_dataset.py
PROPÓSITO: Toma fotos de cartas reales con la Pi Camera para construir
           el dataset de entrenamiento del modelo YOLO.

           El proceso es sencillo:
           1. Pon una carta delante de la cámara
           2. Selecciona qué carta es pulsando la tecla correspondiente
           3. Pulsa ESPACIO para guardar la foto
           4. Repite hasta tener ≥100 fotos por cada una de las 14 clases

           Las fotos se guardan en: data/raw_images/<RANGO>/
           Ejemplo: data/raw_images/A/, data/raw_images/K/, data/raw_images/BACK/

Uso:
    python scripts/capture_dataset.py

Controles:
    A           → seleccionar As
    2-9         → seleccionar número
    0           → seleccionar 10
    J / Q / K   → seleccionar figura
    B           → seleccionar BACK (dorso)
    ESPACIO     → capturar imagen
    D           → borrar última imagen de la clase actual
    q           → salir
"""
import os
import sys
import cv2
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

# Añadimos la raíz del proyecto al path para poder importar config y src.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.perception.camera import Camera

# Mapeo de código ASCII de cada tecla al rango de carta correspondiente.
# ord('a') devuelve el código ASCII del carácter 'a' = 97.
# Esto permite hacer: if key in _RANK_KEYS: rank = _RANK_KEYS[key]
_RANK_KEYS: dict[int, str] = {
    ord('a'): 'A',
    ord('2'): '2', ord('3'): '3', ord('4'): '4', ord('5'): '5',
    ord('6'): '6', ord('7'): '7', ord('8'): '8', ord('9'): '9',
    ord('0'): '10',   # la tecla '0' selecciona el '10' (no hay tecla '10')
    ord('j'): 'J', ord('w'): 'Q', ord('k'): 'K',
    ord('b'): 'BACK',
}

# Fuentes y colores para el overlay visual en la ventana de captura.
_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_DFONT = cv2.FONT_HERSHEY_DUPLEX
_GREEN = (0, 220, 0)     # texto de la clase seleccionada
_WHITE = (255, 255, 255)
_GRAY  = (160, 160, 160)
_RED   = (0, 0, 220)


def _overlay(frame, current_rank: str | None, counts: dict[str, int]) -> None:
    """Dibuja información útil sobre el frame de la cámara:
      - Barra superior: clase seleccionada y número de fotos ya capturadas
      - Cruz de alineación en el centro: guía para centrar la carta
      - Barra inferior: instrucciones de teclado
      - Panel lateral: contador de fotos por clase

    NOTA: este overlay se dibuja sobre una COPIA del frame.
          La foto que se guarda es el frame LIMPIO (sin overlay).
    """
    h, w = frame.shape[:2]   # alto y ancho del frame en píxeles

    # Barra superior oscura (fondo para el texto de la clase).
    # -1 como último parámetro = rectángulo relleno
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 25), -1)

    if current_rank:
        # Mostramos la clase actual en verde grande.
        cv2.putText(frame, f"Clase: {current_rank}", (20, 42),
                    _DFONT, 1.3, _GREEN, 2)
        # Y cuántas fotos llevamos de esa clase.
        cv2.putText(frame, f"Fotos: {counts[current_rank]}",
                    (300, 42), _FONT, 1.0, _WHITE, 2)
    else:
        # Instrucción si aún no se ha seleccionado ninguna clase.
        cv2.putText(frame, "Selecciona una clase (A 2-9 0=10 J Q K B)",
                    (20, 38), _FONT, 0.8, _GRAY, 1)

    # Cruz de alineación en el centro del frame.
    # Ayuda a centrar y alinear la carta en el encuadre.
    cx, cy = w // 2, h // 2   # centro del frame
    cv2.line(frame, (cx - 80, cy), (cx + 80, cy), _GREEN, 1)    # línea horizontal
    cv2.line(frame, (cx, cy - 110), (cx, cy + 110), _GREEN, 1)  # línea vertical
    # Rectángulo guía del tamaño aproximado de una carta.
    cv2.rectangle(frame, (cx - 55, cy - 78), (cx + 55, cy + 78), _GREEN, 1)

    # Barra inferior con instrucciones.
    cv2.rectangle(frame, (0, h - 40), (w, h), (25, 25, 25), -1)
    cv2.putText(frame,
                "A/2-9/0/J/W=Q/K/B = clase   ESPACIO = capturar   D = borrar   ESC = salir",
                (10, h - 12), _FONT, 0.55, _GRAY, 1)

    # Panel lateral derecho: contador de fotos por clase.
    # Muestra cuántas fotos tiene cada clase para saber cuáles faltan.
    x_cnt = w - 160
    # Fondo del panel.
    cv2.rectangle(frame, (x_cnt - 8, 60), (w, 60 + len(config.CARD_CLASSES) * 22 + 8),
                  (25, 25, 25), -1)
    for i, rank in enumerate(config.CARD_CLASSES):
        # Resaltamos en verde la clase actualmente seleccionada.
        color = _GREEN if rank == current_rank else _GRAY
        cv2.putText(frame, f"{rank:>4}: {counts[rank]}",   # :>4 = alinea a la derecha en 4 caracteres
                    (x_cnt, 78 + i * 22), _FONT, 0.55, color, 1)


def _flash(cam: Camera) -> None:
    """Muestra 3 frames más brillantes para simular el destello de una cámara.
    Da retroalimentación visual de que la foto fue guardada correctamente."""
    for _ in range(3):
        f = cam.read()
        # alpha=2.0 duplica el brillo, beta=0 no añade un offset constante.
        bright = cv2.convertScaleAbs(f, alpha=2.0, beta=0)
        cv2.imshow("Captura Dataset", bright)
        cv2.waitKey(40)   # espera 40ms entre flashes (total ~120ms)


def main() -> None:
    """Bucle principal de captura de fotos."""

    # Creamos las carpetas de destino si no existen.
    # Una carpeta por clase: data/raw_images/A/, data/raw_images/2/, ..., data/raw_images/BACK/
    base = Path("data/raw_images")
    for rank in config.CARD_CLASSES:
        (base / rank).mkdir(parents=True, exist_ok=True)

    # Contamos cuántas fotos ya existen de cada clase
    # (útil si estamos continuando una sesión anterior).
    # glob("*.jpg") devuelve todos los archivos .jpg en la carpeta.
    counts = {r: len(list((base / r).glob("*.jpg"))) for r in config.CARD_CLASSES}

    current_rank: str | None = None   # clase actualmente seleccionada (ninguna al inicio)
    total_captured = 0                 # contador de fotos tomadas en esta sesión

    print("Captura de dataset iniciada.")
    print(f"Imágenes guardadas en: {base.resolve()}")

    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        while True:
            # Leemos el frame limpio (sin overlay) → este es el que se guardará.
            clean   = cam.read()

            # Hacemos una copia para dibujar el overlay encima.
            # .copy() es necesario para no modificar el frame original.
            display = clean.copy()
            _overlay(display, current_rank, counts)

            cv2.imshow("Captura Dataset", display)

            # Esperamos 30ms una tecla. Si no se pulsa nada, key = -1.
            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # Escape
                break  # salir del bucle

            elif key in _RANK_KEYS:
                # El usuario seleccionó una clase.
                current_rank = _RANK_KEYS[key]
                print(f"Clase: {current_rank}  ({counts[current_rank]} fotos)")

            elif key == ord(' ') and current_rank:
                # ESPACIO: guardar la foto actual con un nombre único.
                n    = counts[current_rank]
                # Nombre: RANGO_0000.jpg, RANGO_0001.jpg, etc.
                # :04d formatea el número con al menos 4 dígitos (ej: 0023)
                path = base / current_rank / f"{current_rank}_{n:04d}.jpg"
                cv2.imwrite(str(path), clean)   # guardamos el frame LIMPIO (sin overlay)
                counts[current_rank] += 1
                total_captured += 1
                print(f"  [{total_captured}] Guardada: {path.name}")
                _flash(cam)   # destello de confirmación

            elif key == ord('d') and current_rank:
                # D: borrar la última foto de la clase actual (para deshacer errores).
                images = sorted((base / current_rank).glob("*.jpg"))
                if images:
                    images[-1].unlink()   # .unlink() elimina el archivo del disco
                    counts[current_rank] -= 1
                    print(f"  Borrada: {images[-1].name}")
                else:
                    print("  No hay imágenes que borrar.")

    cv2.destroyAllWindows()

    # Resumen final: cuántas fotos tomamos y cómo van las clases.
    print(f"\n{'─'*35}")
    print(f"  Total capturado: {total_captured} imágenes nuevas")
    print(f"{'─'*35}")
    total_all = sum(counts.values())
    for rank in config.CARD_CLASSES:
        if counts[rank] > 0:
            # Barra de progreso visual: █ por cada foto (máximo 30 caracteres)
            bar = "█" * min(counts[rank], 30)
            print(f"  {rank:>4}: {counts[rank]:>3}  {bar}")
    print(f"{'─'*35}")
    print(f"  TOTAL : {total_all}")
    print(f"\nSiguiente paso:")
    print(f"  Etiqueta las imágenes con LabelImg o Roboflow, luego entrena.")


if __name__ == '__main__':
    main()
