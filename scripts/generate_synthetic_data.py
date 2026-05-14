"""
ARCHIVO: scripts/generate_synthetic_data.py
PROPÓSITO: Genera imágenes de cartas FALSAS (sintéticas) dibujadas por código
           para pre-entrenar el modelo YOLO antes de tener fotos reales.

           El proceso:
           1. Genera un fondo de tapete verde con ruido y viñeta (realista)
           2. Dibuja 1-4 cartas sobre el fondo con rotación y escala aleatorias
           3. Genera el archivo .txt de etiquetas YOLO para cada imagen
           4. Organiza todo en train/val + dataset.yaml

           Las imágenes sintéticas no son tan buenas como las reales,
           pero permiten que el modelo aprenda la forma básica de las cartas
           antes de que tengas el hardware disponible.

Uso:
    python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic
    python scripts/generate_synthetic_data.py --n 500  --out data/synthetic --seed 42
"""
import sys
import argparse
import random
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Pillow: generación de imágenes
import config

# ── Dimensiones de la carta y la escena ──────────────────────────────────────

CARD_W, CARD_H = 80, 112    # tamaño base de la carta en píxeles (proporciones reales)
CORNER_R = 6                 # radio de las esquinas redondeadas en píxeles
SCENE_W, SCENE_H = config.FRAME_WIDTH, config.FRAME_HEIGHT  # tamaño del fondo (1280×720)

# ── Colores ──────────────────────────────────────────────────────────────────

# Rangos de las cartas "rojas" (aunque aquí dibujamos en negro, esto es para futura expansión).
_RED_RANKS = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}

# Colores posibles para el dorso de las cartas (BACK): azul, rojo, verde, morado oscuro.
_BACK_COLORS = [
    (0, 0, 150), (150, 0, 0), (0, 100, 150), (100, 0, 100),
]

# Intentamos cargar una fuente TTF del sistema para dibujar los números.
# Si no existe (en algunos sistemas mínimos), usamos la fuente por defecto de Pillow.
try:
    _FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    _FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
except OSError:
    _FONT_LG = ImageFont.load_default()
    _FONT_SM = ImageFont.load_default()


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline) -> None:
    """Dibuja un rectángulo con esquinas redondeadas usando Pillow.
    Las cartas reales tienen esquinas redondeadas, así que las imitamos.

    Pillow no tiene una función nativa para esto, así que lo construimos:
    - 2 rectángulos superpuestos para el cuerpo central
    - 4 elipses en las esquinas
    - Arcos y líneas para el borde (outline)
    """
    x0, y0, x1, y1 = xy
    r = radius
    # Los dos rectángulos forman una cruz que cubre el interior de la carta.
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)     # rectángulo horizontal
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)     # rectángulo vertical
    # Las 4 elipses redondean cada esquina.
    draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)   # esquina sup. izq.
    draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)   # esquina sup. der.
    draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)   # esquina inf. izq.
    draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)   # esquina inf. der.
    # Borde (outline): arcos en las esquinas + líneas en los lados.
    draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=2)
    draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 0,   fill=outline, width=2)
    draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90,  180, fill=outline, width=2)
    draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0,   90,  fill=outline, width=2)
    draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=2)   # borde superior
    draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=2)   # borde inferior
    draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=2)   # borde izquierdo
    draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=2)   # borde derecho


def _make_card(rank: str, rng: random.Random) -> Image.Image:
    """Genera una imagen RGBA de una carta individual.

    RGBA = Red, Green, Blue, Alpha (transparencia).
    El canal Alpha (A) permite pegar la carta sobre el fondo con bordes transparentes.

    Parámetros:
      rank: el rango de la carta ('A', '7', 'K', 'BACK'...)
      rng: generador de números aleatorios (para el color del dorso)

    Devuelve: imagen Pillow en modo RGBA de tamaño CARD_W × CARD_H.
    """

    # Creamos una imagen transparente del tamaño de la carta.
    # (0, 0, 0, 0) = negro totalmente transparente.
    img  = Image.new('RGBA', (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if rank == 'BACK':
        # Dorso de carta: color sólido + rayas diagonales más claras.
        color = rng.choice(_BACK_COLORS)
        _rounded_rect(draw, [0, 0, CARD_W - 1, CARD_H - 1],
                      CORNER_R, fill=color, outline=(200, 200, 200))
        # Rayas diagonales más claras para dar textura al dorso.
        for i in range(-CARD_H, CARD_W + CARD_H, 14):
            draw.line([(i, 0), (i + CARD_H, CARD_H)],
                      fill=(min(color[0]+60, 255), min(color[1]+60, 255), min(color[2]+60, 255)),
                      width=5)
    else:
        # Carta normal: fondo blanco con el rango en 3 posiciones.
        # Las figuras (A, J, Q, K) se dibujan en rojo oscuro; el resto en negro.
        text_color = (180, 0, 0) if rank in ('A', 'J', 'Q', 'K') else (20, 20, 20)
        _rounded_rect(draw, [0, 0, CARD_W - 1, CARD_H - 1],
                      CORNER_R, fill=(252, 252, 252), outline=(40, 40, 40))

        # Esquina superior izquierda: rango en letra pequeña.
        draw.text((6, 4), rank, font=_FONT_SM, fill=text_color)

        # Centro: rango en letra grande.
        draw.text((CARD_W // 2 - 14, CARD_H // 2 - 14), rank, font=_FONT_LG, fill=text_color)

        # Esquina inferior derecha: rango en letra pequeña rotado 180°.
        # Primero dibujamos en la esquina sup. izq. de una imagen temporal
        # y luego la rotamos 180° para pegarla en la posición correcta.
        r_img  = Image.new('RGBA', (CARD_W, CARD_H), (0, 0, 0, 0))
        r_draw = ImageDraw.Draw(r_img)
        r_draw.text((6, 4), rank, font=_FONT_SM, fill=text_color)
        r_img = r_img.rotate(180)   # rotamos la imagen entera 180°
        # alpha_composite pega r_img sobre img respetando la transparencia.
        img = Image.alpha_composite(img, r_img)

    return img


def _felt_background(rng: random.Random) -> np.ndarray:
    """Genera un fondo de tapete de casino verde con ruido y viñeta suave.

    La viñeta (bordes oscurecidos) imita la iluminación real de una cámara.
    El ruido simula la textura del tapete.

    Devuelve: array numpy H×W×3 en formato RGB (uint8).
    """
    # Color base del tapete: verde con brillo variable.
    base_g = rng.randint(80, 120)          # brillo del verde (entre 80 y 120)
    base   = np.array([0, base_g, 0], dtype=np.float32)
    bg     = np.full((SCENE_H, SCENE_W, 3), base, dtype=np.float32)

    # Ruido gaussiano: añade textura aleatoria para imitar la tela del tapete.
    # rng.gauss(0, 8) = media 0, desviación estándar 8 → variaciones pequeñas.
    noise = rng.gauss(0, 8)
    bg   += np.random.normal(noise, 8, bg.shape)

    # Viñeta: oscurece los bordes del frame.
    # ogrid crea arrays de coordenadas Y y X.
    cx, cy = SCENE_W / 2, SCENE_H / 2
    Y, X   = np.ogrid[:SCENE_H, :SCENE_W]
    # dist = distancia normalizada desde el centro (0 en el centro, ~1.4 en las esquinas).
    dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Factor de viñeta: 1.0 en el centro, 0.7 en los bordes.
    # [..newaxis] añade una dimensión para que se pueda multiplicar con el array 3D del bg.
    vignette = np.clip(1 - 0.25 * dist, 0.7, 1.0)[..., np.newaxis]
    bg *= vignette

    # Aseguramos valores en [0, 255] y convertimos a uint8 (enteros de 8 bits).
    return np.clip(bg, 0, 255).astype(np.uint8)


def _paste_card(bg: np.ndarray, card_rgba: Image.Image,
                cx: int, cy: int) -> tuple[int, int, int, int]:
    """Pega la carta sobre el fondo en la posición (cx, cy) con transparencia.

    Parámetros:
      bg: el fondo (array numpy, se modifica en el lugar)
      card_rgba: imagen Pillow de la carta con canal alfa
      cx, cy: coordenadas del centro de la carta en el fondo

    Devuelve: bounding box (x1, y1, x2, y2) en píxeles (para la etiqueta YOLO).
              (0,0,0,0) si la carta está completamente fuera del frame.
    """
    w, h   = card_rgba.size
    x0, y0 = cx - w // 2, cy - h // 2   # esquina superior izquierda de la carta
    x1, y1 = x0 + w, y0 + h             # esquina inferior derecha

    # Recortamos a los límites del frame para no escribir fuera del array.
    sx0 = max(0, x0);      sy0 = max(0, y0)
    sx1 = min(SCENE_W, x1); sy1 = min(SCENE_H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return (0, 0, 0, 0)   # la carta está completamente fuera del frame

    # Calculamos qué parte de la carta debemos pegar (la que cabe en el frame).
    cx0 = sx0 - x0;  cy0 = sy0 - y0
    cx1 = cx0 + (sx1 - sx0); cy1 = cy0 + (sy1 - sy0)

    # Recortamos la carta al área visible y convertimos el fondo a Pillow para pegar.
    card_crop = card_rgba.crop((cx0, cy0, cx1, cy1))
    bg_crop   = Image.fromarray(bg[sy0:sy1, sx0:sx1])
    # .paste con mask usa el canal alfa: los píxeles transparentes de la carta no tapan el fondo.
    bg_crop.paste(card_crop, mask=card_crop.split()[3])
    # Convertimos de vuelta a numpy y actualizamos el fondo.
    bg[sy0:sy1, sx0:sx1] = np.array(bg_crop)

    return (sx0, sy0, sx1, sy1)   # bounding box del área realmente pegada


def generate_scene(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, str, tuple]]]:
    """Genera una escena completa con fondo de tapete y 1-4 cartas aleatorias.

    Devuelve:
      image: array numpy H×W×3 con la escena completa
      annotations: lista de (class_id, rank, (x1,y1,x2,y2)) para cada carta
    """
    bg      = _felt_background(rng)                                    # fondo verde
    n_cards = rng.randint(1, 4)                                       # 1 a 4 cartas
    ranks   = rng.choices(config.CARD_CLASSES, k=n_cards)            # rangos aleatorios
    annotations = []

    for rank in ranks:
        card_pil = _make_card(rank, rng)

        # Escala aleatoria: la carta puede ser un 80-160% de su tamaño original.
        scale = rng.uniform(0.8, 1.6)
        new_w = int(CARD_W * scale)
        new_h = int(CARD_H * scale)
        card_pil = card_pil.resize((new_w, new_h), Image.LANCZOS)

        # Rotación aleatoria: ±35 grados.
        # expand=True amplía el bounding box para no cortar las esquinas rotadas.
        angle    = rng.uniform(-35, 35)
        card_rot = card_pil.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Posición aleatoria: dejamos margen para que la carta no se salga del frame.
        cx = rng.randint(card_rot.width // 2 + 10,  SCENE_W - card_rot.width // 2 - 10)
        cy = rng.randint(card_rot.height // 2 + 10, SCENE_H - card_rot.height // 2 - 10)

        # Pegamos la carta sobre el fondo y obtenemos su bbox.
        bbox = _paste_card(bg, card_rot, cx, cy)
        if bbox != (0, 0, 0, 0):
            cls_id = config.CARD_CLASSES.index(rank)   # índice numérico de la clase
            annotations.append((cls_id, rank, bbox))

    return bg, annotations


def bbox_to_yolo(bbox: tuple[int, int, int, int]) -> str:
    """Convierte un bounding box en píxeles al formato YOLO normalizado.

    YOLO espera: x_centro, y_centro, ancho, alto — todos divididos entre las dimensiones del frame.

    Ejemplo: bbox (100, 50, 200, 150) en un frame 1280×720
      x_centro = (100+200)/2 / 1280 = 0.117
      y_centro = (50+150)/2  / 720  = 0.139
      ancho    = (200-100)   / 1280 = 0.078
      alto     = (150-50)    / 720  = 0.139
    """
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / SCENE_W
    cy = ((y1 + y2) / 2) / SCENE_H
    w  = (x2 - x1) / SCENE_W
    h  = (y2 - y1) / SCENE_H
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    """Genera el dataset sintético completo."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',     type=int,   default=2000, help='Número de imágenes')
    parser.add_argument('--out',   type=str,   default='data/synthetic',
                        help='Carpeta de salida')
    parser.add_argument('--seed',  type=int,   default=None,
                        help='Semilla aleatoria (None = aleatoriedad real)')
    parser.add_argument('--split', type=float, default=0.85,
                        help='Fracción para train (resto val), defecto 0.85')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)   # semilla para el ruido de numpy también

    # Creamos la estructura de carpetas del dataset.
    out       = Path(args.out)
    train_img = out / 'images' / 'train'; train_img.mkdir(parents=True, exist_ok=True)
    train_lbl = out / 'labels' / 'train'; train_lbl.mkdir(parents=True, exist_ok=True)
    val_img   = out / 'images' / 'val';   val_img.mkdir(parents=True, exist_ok=True)
    val_lbl   = out / 'labels' / 'val';   val_lbl.mkdir(parents=True, exist_ok=True)

    # Generamos el dataset.yaml para que YOLO sepa dónde están los datos.
    yaml_path = out / 'dataset.yaml'
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(config.CARD_CLASSES)}\n"
        f"names: {config.CARD_CLASSES}\n"
    )

    n_train = int(args.n * args.split)   # número de imágenes para entrenamiento
    counts  = {r: 0 for r in config.CARD_CLASSES}  # contador por clase (para el resumen)

    for i in range(args.n):
        # Las primeras n_train imágenes van a train, el resto a val.
        is_train = i < n_train
        img_dir  = train_img if is_train else val_img
        lbl_dir  = train_lbl if is_train else val_lbl

        # Generamos la escena con sus anotaciones.
        image, annotations = generate_scene(rng)
        fname = f"{i:05d}"   # nombre del archivo con 5 dígitos: 00000, 00001...

        # Guardamos la imagen como JPEG con calidad 92 (buen balance tamaño/calidad).
        Image.fromarray(image).save(img_dir / f"{fname}.jpg", quality=92)

        # Guardamos las anotaciones YOLO: una línea por carta en la imagen.
        with open(lbl_dir / f"{fname}.txt", 'w') as f:
            for cls_id, rank, bbox in annotations:
                f.write(f"{cls_id} {bbox_to_yolo(bbox)}\n")
                counts[rank] += 1

        # Progreso: mostramos cada 200 imágenes.
        if (i + 1) % 200 == 0 or i == args.n - 1:
            print(f"  {i+1}/{args.n} imágenes generadas...")

    # Resumen final.
    print(f"\nDataset en: {out.resolve()}")
    print(f"  Train: {n_train}  |  Val: {args.n - n_train}")
    print(f"  dataset.yaml: {yaml_path}")
    print("\nDistribución de clases generadas:")
    for rank, cnt in counts.items():
        print(f"  {rank:>4}: {cnt}")
    print("\nPara entrenar en Colab:")
    print("  from ultralytics import YOLO")
    print(f"  model = YOLO('yolov8n.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=50, imgsz=640)")


if __name__ == '__main__':
    main()
