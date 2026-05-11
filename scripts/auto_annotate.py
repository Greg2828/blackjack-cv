"""
Genera etiquetas YOLO automáticamente a partir de las fotos capturadas con
capture_dataset.py y estructura el dataset listo para entrenar en Google Colab.

Para cada imagen en data/raw_images/RANK/ detecta la carta por análisis de
contorno (Canny + Otsu) y escribe el .txt de anotación YOLO correspondiente.
Al final divide en train/val y genera data/labeled/dataset.yaml.

Uso:
    python scripts/auto_annotate.py                 # anota y estructura
    python scripts/auto_annotate.py --preview       # muestra cada bbox en pantalla
    python scripts/auto_annotate.py --val 0.2       # fracción de validación (defecto 0.2)
    python scripts/auto_annotate.py --min-photos 80 # avisa si una clase tiene menos fotos
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

import sys
import argparse
import random
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CARD_CLASSES = config.CARD_CLASSES
CLASS_IDX    = {r: i for i, r in enumerate(CARD_CLASSES)}

RAW_DIR     = Path("data/raw_images")
LABELED_DIR = Path("data/labeled")

# Ratio ancho/alto de una carta estándar (63.5 mm / 88.9 mm)
_CARD_RATIO      = 63.5 / 88.9   # ≈ 0.714
_RATIO_TOLERANCE = 0.28           # acepta 0.43 – 1.00 (cubre rotaciones leves)
_BBOX_PADDING    = 0.04           # margen extra alrededor del contorno (fracción)
_MIN_AREA_FRAC   = 0.025          # la carta debe ocupar al menos el 2.5 % del frame


def _detect_card(img: np.ndarray) -> tuple[float, float, float, float] | None:
    """Detecta la bounding box de la carta y devuelve (xc, yc, w, h) normalizados
    en formato YOLO. Prueba tres estrategias en orden; devuelve None si falla todo.

    Estrategia 1 — Canny: mejor para cartas blancas sobre fondos oscuros.
    Estrategia 2 — Otsu normal: fondo oscuro, carta clara.
    Estrategia 3 — Otsu inverso: fondo claro, carta oscura.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    min_area = _MIN_AREA_FRAC * w * h

    def _best_bbox(mask: np.ndarray) -> tuple | None:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        cnt = max(valid, key=cv2.contourArea)
        return cv2.boundingRect(cnt)   # (x, y, bw, bh)

    def _ratio_ok(bw: int, bh: int) -> bool:
        if bh == 0:
            return False
        r = bw / bh
        return abs(r - _CARD_RATIO) < _RATIO_TOLERANCE

    def _to_yolo(x: int, y: int, bw: int, bh: int) -> tuple:
        pad_x = bw * _BBOX_PADDING
        pad_y = bh * _BBOX_PADDING
        x1 = max(0.0, x - pad_x)
        y1 = max(0.0, y - pad_y)
        x2 = min(float(w), x + bw + pad_x)
        y2 = min(float(h), y + bh + pad_y)
        return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h

    # Estrategia 1: Canny
    edges = cv2.Canny(blur, 25, 90)
    bbox = _best_bbox(edges)
    if bbox and _ratio_ok(bbox[2], bbox[3]):
        return _to_yolo(*bbox)

    # Estrategia 2: Otsu — carta clara sobre fondo oscuro
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bbox = _best_bbox(thresh)
    if bbox and _ratio_ok(bbox[2], bbox[3]):
        return _to_yolo(*bbox)

    # Estrategia 3: Otsu inverso — carta oscura sobre fondo claro
    _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bbox = _best_bbox(thresh_inv)
    if bbox and _ratio_ok(bbox[2], bbox[3]):
        return _to_yolo(*bbox)

    # Fallback sin exigir ratio: al menos devuelve algo
    bbox = _best_bbox(edges) or _best_bbox(thresh) or _best_bbox(thresh_inv)
    if bbox:
        return _to_yolo(*bbox)

    return None


def _draw_bbox(img: np.ndarray, xc: float, yc: float, bw: float, bh: float,
               label: str) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
    cv2.putText(out, label, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
    return out


def _run_preview(img: np.ndarray, label: str, ok: bool) -> str:
    """Muestra la imagen con bbox. Devuelve 'next', 'delete' o 'quit'."""
    h, w = img.shape[:2]
    info = "DETECCION FALLIDA" if not ok else f"Clase: {label}"
    color = (0, 0, 220) if not ok else (0, 220, 0)
    disp = img.copy()
    cv2.putText(disp, info, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(disp, "ESPACIO=ok  D=borrar foto  Q=salir preview",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Preview anotaciones", disp)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        return 'quit'
    if key == ord('d'):
        return 'delete'
    return 'next'


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-anotación del dataset de cartas")
    parser.add_argument('--preview',    action='store_true',
                        help="Muestra cada anotación en pantalla para revisarla")
    parser.add_argument('--val',        type=float, default=0.20,
                        help="Fracción de imágenes para validación (defecto 0.20)")
    parser.add_argument('--min-photos', type=int,   default=80,
                        help="Avisa si una clase tiene menos de N fotos (defecto 80)")
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print(f"No se encontró {RAW_DIR}.")
        print("Captura fotos primero con:  python scripts/capture_dataset.py")
        sys.exit(1)

    # Reconstruir data/labeled/ desde cero
    if LABELED_DIR.exists():
        shutil.rmtree(LABELED_DIR)
    for split in ('train', 'val'):
        (LABELED_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (LABELED_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    annotated: dict[str, list[tuple]] = {r: [] for r in CARD_CLASSES}
    failed:    dict[str, list[Path]]  = {r: [] for r in CARD_CLASSES}
    stop_preview = False

    print(f"Leyendo fotos de: {RAW_DIR.resolve()}")
    print()

    for rank in CARD_CLASSES:
        class_dir = RAW_DIR / rank
        if not class_dir.exists():
            continue
        images = sorted(class_dir.glob("*.jpg"))
        if not images:
            continue

        ok_count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                failed[rank].append(img_path)
                continue

            result = _detect_card(img)

            if args.preview and not stop_preview:
                if result:
                    disp = _draw_bbox(img, *result, rank)
                else:
                    disp = img
                action = _run_preview(disp, rank, result is not None)
                if action == 'quit':
                    stop_preview = True
                elif action == 'delete':
                    img_path.unlink()
                    print(f"  Borrada: {img_path.name}")
                    failed[rank].append(img_path)
                    continue

            if result is None:
                failed[rank].append(img_path)
            else:
                xc, yc, bw, bh = result
                yolo_line = f"{CLASS_IDX[rank]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                annotated[rank].append((img_path, yolo_line))
                ok_count += 1

        n_fail = len(images) - ok_count
        status = f"OK={ok_count}"
        if n_fail:
            status += f"  fallidas={n_fail}"
        print(f"  {rank:>4}: {status}")

    if args.preview:
        cv2.destroyAllWindows()

    print()
    print("Dividiendo train / val...")

    total_train = total_val = 0
    for rank, pairs in annotated.items():
        if not pairs:
            continue
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * args.val))
        splits = {'val': pairs[:n_val], 'train': pairs[n_val:]}

        for split, items in splits.items():
            for img_path, yolo_line in items:
                dst_img = LABELED_DIR / 'images' / split / img_path.name
                dst_lbl = LABELED_DIR / 'labels' / split / (img_path.stem + '.txt')
                shutil.copy(img_path, dst_img)
                dst_lbl.write_text(yolo_line + '\n')

        total_train += len(splits['train'])
        total_val   += len(splits['val'])

    # dataset.yaml
    yaml_data = {
        'path':  str(LABELED_DIR.resolve()),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    len(CARD_CLASSES),
        'names': CARD_CLASSES,
    }
    yaml_path = LABELED_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    # ── Resumen ────────────────────────────────────────────────────────────────
    total_ok  = sum(len(v) for v in annotated.values())
    total_bad = sum(len(v) for v in failed.values())

    print()
    print("─" * 52)
    print(f"  Anotadas correctamente : {total_ok}")
    print(f"  Fallidas (excluidas)   : {total_bad}")
    print(f"  Train                  : {total_train}")
    print(f"  Val                    : {total_val}")
    print(f"  Dataset                : {yaml_path}")
    print("─" * 52)
    print()

    warnings = []
    for rank in CARD_CLASSES:
        n = len(annotated[rank])
        if n < args.min_photos:
            warnings.append(
                f"  [AVISO] {rank}: {n} fotos anotadas (mínimo recomendado: {args.min_photos})"
            )
        if failed[rank]:
            fnames = [p.name for p in failed[rank][:4]]
            extra  = len(failed[rank]) - len(fnames)
            extra_str = f" (+{extra} más)" if extra > 0 else ""
            warnings.append(
                f"  [AVISO] {rank}: {len(failed[rank])} fotos con detección fallida"
                f" → {', '.join(fnames)}{extra_str}"
            )

    if warnings:
        for w in warnings:
            print(w)
        print()
        print("  Tip: usa --preview para revisar visualmente las fotos fallidas.")
    else:
        print("  Sin advertencias. Dataset listo.")

    print()
    print("Siguiente paso — comprime y sube a Colab:")
    print("  cd data && zip -r labeled.zip labeled/")
    print("  Sube labeled.zip a Google Drive y abre notebooks/colab_train.ipynb")


if __name__ == '__main__':
    main()
