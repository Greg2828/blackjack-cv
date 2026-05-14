"""
ARCHIVO: scripts/auto_annotate.py
PROPÓSITO: Genera etiquetas YOLO automáticamente a partir de las fotos capturadas
           con capture_dataset.py, sin necesidad de etiquetar manualmente.

           YOLO necesita para cada foto un archivo .txt con el formato:
             class_id x_center y_center width height
           donde todos los valores (excepto class_id) son fracciones entre 0 y 1.

           Este script detecta la carta en cada foto usando análisis de contornos
           (bordes) y genera el archivo .txt correspondiente.
           Al final, organiza todo en train/val y crea dataset.yaml para Colab.

Uso:
    python scripts/auto_annotate.py                 # anota y estructura
    python scripts/auto_annotate.py --preview       # muestra cada bbox en pantalla
    python scripts/auto_annotate.py --val 0.2       # fracción de validación (defecto 0.2)
    python scripts/auto_annotate.py --min-photos 80 # avisa si una clase tiene menos fotos

FLUJO:
  data/raw_images/A/*.jpg  →  detectar carta  →  data/labeled/images/train/*.jpg
                                               →  data/labeled/labels/train/*.txt
                                               →  data/labeled/dataset.yaml
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

import sys
import argparse
import random
import shutil   # para copiar archivos
import cv2
import numpy as np
import yaml     # para generar el archivo dataset.yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

CARD_CLASSES = config.CARD_CLASSES
# Diccionario inverso: rango → índice de clase (ej: 'A' → 0, '2' → 1, ...)
# YOLO necesita el índice numérico, no el nombre.
CLASS_IDX    = {r: i for i, r in enumerate(CARD_CLASSES)}

RAW_DIR     = Path("data/raw_images")  # carpeta con las fotos originales sin etiquetar
LABELED_DIR = Path("data/labeled")    # carpeta de destino con el dataset estructurado para YOLO

# Proporción estándar ancho/alto de una carta de juego:
# 63.5mm de ancho / 88.9mm de alto ≈ 0.714
# Usamos esto para verificar que el contorno detectado tiene forma de carta.
_CARD_RATIO      = 63.5 / 88.9    # ≈ 0.714
_RATIO_TOLERANCE = 0.28            # margen de error: acepta ratios en [0.43, 1.00]
                                   # (cubre inclinaciones de hasta ~30°)

# Margen extra alrededor del contorno para el bounding box.
# 0.04 = 4% del ancho/alto del contorno como padding.
_BBOX_PADDING    = 0.04

# La carta debe ocupar al menos 2.5% del frame para ser detectada.
# Evita confundir pequeños reflejos o ruido con cartas.
_MIN_AREA_FRAC   = 0.025


def _detect_card(img: np.ndarray) -> tuple[float, float, float, float] | None:
    """Detecta la bounding box de la carta en la imagen y devuelve sus coordenadas
    en formato YOLO normalizado: (x_centro, y_centro, ancho, alto) con valores 0-1.

    Usa 3 estrategias de detección en orden de preferencia:
      1. Canny: detecta bordes → bueno para cartas blancas sobre fondo oscuro
      2. Otsu normal: umbral automático → fondo oscuro, carta clara
      3. Otsu inverso: umbral invertido → fondo claro, carta oscura

    Devuelve None si ninguna estrategia detecta una carta válida.
    """
    h, w = img.shape[:2]
    # Convertimos a escala de grises (1 canal) para el análisis de contornos.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Suavizamos para reducir el ruido antes de detectar bordes.
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Kernel rectangular 9×9 para operaciones morfológicas (cierre de huecos).
    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # Área mínima en píxeles² (2.5% del frame total)
    min_area = _MIN_AREA_FRAC * w * h

    def _best_bbox(mask: np.ndarray) -> tuple | None:
        """Dada una máscara binaria, encuentra el contorno más grande y devuelve su bbox.
        Devuelve None si no hay contornos suficientemente grandes."""
        # MORPH_CLOSE llena huecos dentro de la carta (sombras, reflejos en el borde).
        # iterations=2 aplica la operación 2 veces para ser más agresivo.
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filtramos contornos demasiado pequeños.
        valid = [c for c in cnts if cv2.contourArea(c) > min_area]
        if not valid:
            return None
        # Tomamos el contorno más grande (que debería ser la carta).
        cnt = max(valid, key=cv2.contourArea)
        return cv2.boundingRect(cnt)   # devuelve (x, y, ancho, alto) en píxeles

    def _ratio_ok(bw: int, bh: int) -> bool:
        """Verifica que el bounding box tiene una proporción similar a una carta estándar.
        Un bounding box muy cuadrado o muy alargado probablemente no es una carta."""
        if bh == 0:
            return False
        r = bw / bh
        return abs(r - _CARD_RATIO) < _RATIO_TOLERANCE

    def _to_yolo(x: int, y: int, bw: int, bh: int) -> tuple:
        """Convierte coordenadas absolutas en píxeles a formato YOLO normalizado.
        Añade un pequeño padding para no cortar los bordes de la carta.

        YOLO usa: x_centro, y_centro, ancho, alto — todos como fracción de 0.0 a 1.0
        """
        pad_x = bw * _BBOX_PADDING   # padding horizontal
        pad_y = bh * _BBOX_PADDING   # padding vertical
        # Expandimos el bounding box, pero sin salir de los bordes del frame.
        x1 = max(0.0,    x - pad_x)
        y1 = max(0.0,    y - pad_y)
        x2 = min(float(w), x + bw + pad_x)
        y2 = min(float(h), y + bh + pad_y)
        # Calculamos el centro y el tamaño normalizados (dividimos por w/h del frame).
        return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h

    # ── Estrategia 1: Canny (detección de bordes) ─────────────────────────────
    # Canny detecta cambios bruscos de brillo → ideal para bordes de cartas.
    # 25 y 90 son los umbrales de histéresis (bordes débiles y fuertes).
    edges = cv2.Canny(blur, 25, 90)
    bbox  = _best_bbox(edges)
    if bbox and _ratio_ok(bbox[2], bbox[3]):
        return _to_yolo(*bbox)

    # ── Estrategia 2: Otsu — carta clara sobre fondo oscuro ───────────────────
    # Otsu calcula automáticamente el mejor umbral para separar fondo de primer plano.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bbox = _best_bbox(thresh)
    if bbox and _ratio_ok(bbox[2], bbox[3]):
        return _to_yolo(*bbox)

    # ── Estrategia 3: Otsu inverso — carta oscura sobre fondo claro ───────────
    _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bbox = _best_bbox(thresh_inv)
    if bbox and _ratio_ok(bbox[2], bbox[3]):
        return _to_yolo(*bbox)

    # ── Fallback: intentamos sin exigir el ratio de aspecto ──────────────────
    # Si ninguna estrategia encuentra una carta con proporción correcta,
    # devolvemos el mejor contorno que encontremos (puede no ser ideal, pero es algo).
    bbox = _best_bbox(edges) or _best_bbox(thresh) or _best_bbox(thresh_inv)
    if bbox:
        return _to_yolo(*bbox)

    return None  # completamente no se detectó nada


def _draw_bbox(img: np.ndarray, xc: float, yc: float, bw: float, bh: float,
               label: str) -> np.ndarray:
    """Dibuja el bounding box sobre la imagen para la vista previa.
    Convierte las coordenadas YOLO (0-1) de vuelta a píxeles para dibujar."""
    h, w = img.shape[:2]
    # Convertimos de coordenadas normalizadas YOLO a píxeles absolutos.
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
    """Muestra la imagen con la bbox en pantalla para revisión manual.
    Espera que el usuario pulse una tecla para continuar.

    Devuelve: 'next' (continuar), 'delete' (borrar esta foto), 'quit' (salir del preview).
    """
    h, w = img.shape[:2]
    info  = "DETECCION FALLIDA" if not ok else f"Clase: {label}"
    color = (0, 0, 220) if not ok else (0, 220, 0)   # rojo si falló, verde si OK
    disp  = img.copy()
    cv2.putText(disp, info, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(disp, "ESPACIO=ok  D=borrar foto  Q=salir preview",
                (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Preview anotaciones", disp)
    key = cv2.waitKey(0) & 0xFF   # waitKey(0) espera indefinidamente hasta una tecla
    if key == ord('q'):
        return 'quit'
    if key == ord('d'):
        return 'delete'
    return 'next'


def main() -> None:
    """Proceso principal de auto-anotación."""
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

    # ── Reconstruir el directorio labeled/ desde cero ─────────────────────────
    # shutil.rmtree() borra toda la carpeta y su contenido.
    # Lo hacemos para que no queden restos de anotaciones anteriores.
    if LABELED_DIR.exists():
        shutil.rmtree(LABELED_DIR)

    # Creamos la estructura de carpetas que espera YOLO:
    #   data/labeled/images/train/    ← imágenes de entrenamiento
    #   data/labeled/images/val/      ← imágenes de validación
    #   data/labeled/labels/train/    ← archivos .txt de entrenamiento
    #   data/labeled/labels/val/      ← archivos .txt de validación
    for split in ('train', 'val'):
        (LABELED_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (LABELED_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)   # generador reproducible para el train/val split

    # Diccionarios para acumular resultados.
    annotated: dict[str, list[tuple]] = {r: [] for r in CARD_CLASSES}   # (path, yolo_line)
    failed:    dict[str, list[Path]]  = {r: [] for r in CARD_CLASSES}   # fotos fallidas

    stop_preview = False   # si el usuario pulsa 'q' en preview, paramos la revisión visual

    print(f"Leyendo fotos de: {RAW_DIR.resolve()}")
    print()

    # ── Procesamos cada clase ─────────────────────────────────────────────────
    for rank in CARD_CLASSES:
        class_dir = RAW_DIR / rank
        if not class_dir.exists():
            continue   # esta clase no tiene carpeta, la saltamos
        images = sorted(class_dir.glob("*.jpg"))
        if not images:
            continue   # carpeta vacía, sin fotos

        ok_count = 0
        for img_path in images:
            img = cv2.imread(str(img_path))   # leemos la imagen
            if img is None:
                failed[rank].append(img_path)
                continue

            # Detectamos la carta en la imagen.
            result = _detect_card(img)

            # Si --preview está activado y aún no lo cerramos, mostramos la bbox.
            if args.preview and not stop_preview:
                if result:
                    disp = _draw_bbox(img, *result, rank)
                else:
                    disp = img  # si no detectó nada, mostramos la imagen sin bbox
                action = _run_preview(disp, rank, result is not None)
                if action == 'quit':
                    stop_preview = True   # paramos el preview pero seguimos procesando
                elif action == 'delete':
                    img_path.unlink()    # borramos la foto del disco
                    print(f"  Borrada: {img_path.name}")
                    failed[rank].append(img_path)
                    continue

            if result is None:
                failed[rank].append(img_path)
            else:
                xc, yc, bw, bh = result
                # Formato de línea YOLO: "class_id xc yc w h" con 6 decimales.
                yolo_line = f"{CLASS_IDX[rank]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                annotated[rank].append((img_path, yolo_line))
                ok_count += 1

        # Resumen por clase.
        n_fail = len(images) - ok_count
        status = f"OK={ok_count}"
        if n_fail:
            status += f"  fallidas={n_fail}"
        print(f"  {rank:>4}: {status}")

    if args.preview:
        cv2.destroyAllWindows()

    # ── Dividimos en train y val ──────────────────────────────────────────────
    print()
    print("Dividiendo train / val...")

    total_train = total_val = 0
    for rank, pairs in annotated.items():
        if not pairs:
            continue
        rng.shuffle(pairs)   # mezclamos para que train y val sean representativos
        # 20% (por defecto) para validación, mínimo 1 imagen.
        n_val = max(1, int(len(pairs) * args.val))
        splits = {'val': pairs[:n_val], 'train': pairs[n_val:]}

        for split, items in splits.items():
            for img_path, yolo_line in items:
                # Copiamos la imagen al directorio correspondiente.
                dst_img = LABELED_DIR / 'images' / split / img_path.name
                # Creamos el archivo .txt de etiqueta con el mismo nombre base.
                dst_lbl = LABELED_DIR / 'labels' / split / (img_path.stem + '.txt')
                shutil.copy(img_path, dst_img)         # copia la imagen
                dst_lbl.write_text(yolo_line + '\n')   # escribe el .txt de etiqueta

        total_train += len(splits['train'])
        total_val   += len(splits['val'])

    # ── Generamos dataset.yaml ────────────────────────────────────────────────
    # YOLO necesita este archivo para saber dónde están los datos y cuántas clases hay.
    yaml_data = {
        'path':  str(LABELED_DIR.resolve()),  # ruta absoluta al dataset
        'train': 'images/train',              # ruta relativa a las imágenes de train
        'val':   'images/val',
        'nc':    len(CARD_CLASSES),           # número de clases = 14
        'names': CARD_CLASSES,                # nombres de las clases
    }
    yaml_path = LABELED_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    # ── Resumen final ─────────────────────────────────────────────────────────
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

    # Avisos si alguna clase tiene pocas fotos o muchas fallidas.
    warnings = []
    for rank in CARD_CLASSES:
        n = len(annotated[rank])
        if n < args.min_photos:
            warnings.append(
                f"  [AVISO] {rank}: {n} fotos anotadas (mínimo recomendado: {args.min_photos})"
            )
        if failed[rank]:
            fnames    = [p.name for p in failed[rank][:4]]
            extra     = len(failed[rank]) - len(fnames)
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
