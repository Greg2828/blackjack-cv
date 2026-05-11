"""
Captura imágenes de cartas desde la Pi Camera para construir el dataset
de entrenamiento YOLOv8.

Coloca una carta frente a la cámara, selecciona su rango con el teclado
y pulsa ESPACIO para guardar la foto. Repite para cada carta.

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
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.perception.camera import Camera

_RANK_KEYS: dict[int, str] = {
    ord('a'): 'A',
    ord('2'): '2', ord('3'): '3', ord('4'): '4', ord('5'): '5',
    ord('6'): '6', ord('7'): '7', ord('8'): '8', ord('9'): '9',
    ord('0'): '10',
    ord('j'): 'J', ord('q'): 'Q', ord('k'): 'K',
    ord('b'): 'BACK',
}

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_DFONT = cv2.FONT_HERSHEY_DUPLEX
_GREEN = (0, 220, 0)
_WHITE = (255, 255, 255)
_GRAY  = (160, 160, 160)
_RED   = (0, 0, 220)


def _overlay(frame, current_rank: str | None, counts: dict[str, int]) -> None:
    h, w = frame.shape[:2]

    # Barra superior
    cv2.rectangle(frame, (0, 0), (w, 60), (25, 25, 25), -1)

    if current_rank:
        cv2.putText(frame, f"Clase: {current_rank}", (20, 42),
                    _DFONT, 1.3, _GREEN, 2)
        cv2.putText(frame, f"Fotos: {counts[current_rank]}",
                    (300, 42), _FONT, 1.0, _WHITE, 2)
    else:
        cv2.putText(frame, "Selecciona una clase (A 2-9 0=10 J Q K B)",
                    (20, 38), _FONT, 0.8, _GRAY, 1)

    # Cruz de alineación
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 80, cy), (cx + 80, cy), _GREEN, 1)
    cv2.line(frame, (cx, cy - 110), (cx, cy + 110), _GREEN, 1)
    cv2.rectangle(frame, (cx - 55, cy - 78), (cx + 55, cy + 78), _GREEN, 1)

    # Barra inferior con instrucciones
    cv2.rectangle(frame, (0, h - 40), (w, h), (25, 25, 25), -1)
    cv2.putText(frame,
                "A/2-9/0/J/Q/K/B = clase   ESPACIO = capturar   D = borrar ultima   q = salir",
                (10, h - 12), _FONT, 0.55, _GRAY, 1)

    # Minicontador lateral: cuántas fotos hay de cada clase
    x_cnt = w - 160
    cv2.rectangle(frame, (x_cnt - 8, 60), (w, 60 + len(config.CARD_CLASSES) * 22 + 8),
                  (25, 25, 25), -1)
    for i, rank in enumerate(config.CARD_CLASSES):
        color = _GREEN if rank == current_rank else _GRAY
        cv2.putText(frame, f"{rank:>4}: {counts[rank]}",
                    (x_cnt, 78 + i * 22), _FONT, 0.55, color, 1)


def _flash(cam: Camera) -> None:
    """Destello visual de confirmación de captura."""
    for _ in range(3):
        blank = cv2.imread.__func__ if False else None
        f = cam.read()
        bright = cv2.convertScaleAbs(f, alpha=2.0, beta=0)
        cv2.imshow("Captura Dataset", bright)
        cv2.waitKey(40)


def main() -> None:
    base = Path("data/raw_images")
    for rank in config.CARD_CLASSES:
        (base / rank).mkdir(parents=True, exist_ok=True)

    counts = {r: len(list((base / r).glob("*.jpg"))) for r in config.CARD_CLASSES}
    current_rank: str | None = None
    total_captured = 0

    print("Captura de dataset iniciada.")
    print(f"Imágenes guardadas en: {base.resolve()}")

    with Camera(width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT) as cam:
        while True:
            frame = cam.read()
            _overlay(frame, current_rank, counts)
            cv2.imshow("Captura Dataset", frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break

            elif key in _RANK_KEYS:
                current_rank = _RANK_KEYS[key]
                print(f"Clase: {current_rank}  ({counts[current_rank]} fotos)")

            elif key == ord(' ') and current_rank:
                n    = counts[current_rank]
                path = base / current_rank / f"{current_rank}_{n:04d}.jpg"
                cv2.imwrite(str(path), cam.read())
                counts[current_rank] += 1
                total_captured += 1
                print(f"  [{total_captured}] Guardada: {path.name}")
                _flash(cam)

            elif key == ord('d') and current_rank:
                images = sorted((base / current_rank).glob("*.jpg"))
                if images:
                    images[-1].unlink()
                    counts[current_rank] -= 1
                    print(f"  Borrada: {images[-1].name}")
                else:
                    print("  No hay imágenes que borrar.")

    cv2.destroyAllWindows()

    print(f"\n{'─'*35}")
    print(f"  Total capturado: {total_captured} imágenes nuevas")
    print(f"{'─'*35}")
    total_all = sum(counts.values())
    for rank in config.CARD_CLASSES:
        if counts[rank] > 0:
            bar = "█" * min(counts[rank], 30)
            print(f"  {rank:>4}: {counts[rank]:>3}  {bar}")
    print(f"{'─'*35}")
    print(f"  TOTAL : {total_all}")
    print(f"\nSiguiente paso:")
    print(f"  Etiqueta las imágenes con LabelImg o Roboflow, luego entrena.")


if __name__ == '__main__':
    main()
