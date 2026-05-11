"""
Genera imágenes sintéticas de cartas para pre-entrenar YOLOv8.

Las imágenes muestran 1-4 cartas sobre un fondo verde, con rotación,
escala variable e iluminación ruidosa. Las anotaciones se generan
automáticamente en formato YOLO.

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
from PIL import Image, ImageDraw, ImageFont

import config

# ── Dimensiones de la carta (px) ────────────────────────────────────────────
CARD_W, CARD_H = 80, 112
CORNER_R = 6          # radio de esquinas redondeadas
SCENE_W, SCENE_H = config.FRAME_WIDTH, config.FRAME_HEIGHT

# ── Colores ──────────────────────────────────────────────────────────────────
_RED_RANKS = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}
_BACK_COLORS = [
    (0, 0, 150), (150, 0, 0), (0, 100, 150), (100, 0, 100),
]

try:
    _FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    _FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
except OSError:
    _FONT_LG = ImageFont.load_default()
    _FONT_SM = ImageFont.load_default()


def _rounded_rect(draw: ImageDraw.ImageDraw, xy, radius: int, fill, outline) -> None:
    x0, y0, x1, y1 = xy
    r = radius
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)
    draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)
    draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)
    draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)
    # Outline
    draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=2)
    draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 0,   fill=outline, width=2)
    draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90,  180, fill=outline, width=2)
    draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0,   90,  fill=outline, width=2)
    draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=2)
    draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=2)
    draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=2)
    draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=2)


def _make_card(rank: str, rng: random.Random) -> Image.Image:
    img = Image.new('RGBA', (CARD_W, CARD_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if rank == 'BACK':
        color = rng.choice(_BACK_COLORS)
        _rounded_rect(draw, [0, 0, CARD_W - 1, CARD_H - 1],
                      CORNER_R, fill=color, outline=(200, 200, 200))
        # Diagonal stripes
        for i in range(-CARD_H, CARD_W + CARD_H, 14):
            draw.line([(i, 0), (i + CARD_H, CARD_H)],
                      fill=(min(color[0]+60, 255), min(color[1]+60, 255), min(color[2]+60, 255)),
                      width=5)
    else:
        text_color = (180, 0, 0) if rank in ('A', 'J', 'Q', 'K') else (20, 20, 20)
        _rounded_rect(draw, [0, 0, CARD_W - 1, CARD_H - 1],
                      CORNER_R, fill=(252, 252, 252), outline=(40, 40, 40))

        # Esquina superior izquierda
        draw.text((6, 4), rank, font=_FONT_SM, fill=text_color)
        # Centro
        draw.text((CARD_W // 2 - 14, CARD_H // 2 - 14), rank, font=_FONT_LG, fill=text_color)
        # Esquina inferior derecha (invertida)
        r_img = Image.new('RGBA', (CARD_W, CARD_H), (0, 0, 0, 0))
        r_draw = ImageDraw.Draw(r_img)
        r_draw.text((6, 4), rank, font=_FONT_SM, fill=text_color)
        r_img = r_img.rotate(180)
        img = Image.alpha_composite(img, r_img)

    return img


def _felt_background(rng: random.Random) -> np.ndarray:
    """Fondo de tapete verde con ruido y gradiente suave."""
    base_g = rng.randint(80, 120)
    base   = np.array([0, base_g, 0], dtype=np.float32)
    bg     = np.full((SCENE_H, SCENE_W, 3), base, dtype=np.float32)

    # Ruido de textura
    noise = rng.gauss(0, 8)
    bg   += np.random.normal(noise, 8, bg.shape)

    # Viñeta suave
    cx, cy = SCENE_W / 2, SCENE_H / 2
    Y, X   = np.ogrid[:SCENE_H, :SCENE_W]
    dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1 - 0.25 * dist, 0.7, 1.0)[..., np.newaxis]
    bg *= vignette

    return np.clip(bg, 0, 255).astype(np.uint8)


def _paste_card(bg: np.ndarray, card_rgba: Image.Image,
                cx: int, cy: int) -> tuple[int, int, int, int]:
    """Pega la carta centrada en (cx, cy). Devuelve bbox (x1,y1,x2,y2) en píxeles."""
    w, h   = card_rgba.size
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h

    # Recortar al borde del frame
    sx0 = max(0, x0);   sy0 = max(0, y0)
    sx1 = min(SCENE_W, x1); sy1 = min(SCENE_H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return (0, 0, 0, 0)

    cx0 = sx0 - x0;  cy0 = sy0 - y0
    cx1 = cx0 + (sx1 - sx0); cy1 = cy0 + (sy1 - sy0)

    card_crop = card_rgba.crop((cx0, cy0, cx1, cy1))
    bg_crop   = Image.fromarray(bg[sy0:sy1, sx0:sx1])
    bg_crop.paste(card_crop, mask=card_crop.split()[3])
    bg[sy0:sy1, sx0:sx1] = np.array(bg_crop)

    return (sx0, sy0, sx1, sy1)


def generate_scene(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, str, tuple]]]:
    """
    Returns:
        image (H×W×3 uint8)
        annotations: list of (class_id, rank, (x1,y1,x2,y2))
    """
    bg = _felt_background(rng)
    n_cards = rng.randint(1, 4)
    ranks   = rng.choices(config.CARD_CLASSES, k=n_cards)
    annotations = []

    for rank in ranks:
        card_pil = _make_card(rank, rng)

        # Escala aleatoria
        scale = rng.uniform(0.8, 1.6)
        new_w = int(CARD_W * scale)
        new_h = int(CARD_H * scale)
        card_pil = card_pil.resize((new_w, new_h), Image.LANCZOS)

        # Rotación
        angle = rng.uniform(-35, 35)
        card_rot = card_pil.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Posición
        cx = rng.randint(card_rot.width // 2 + 10, SCENE_W - card_rot.width // 2 - 10)
        cy = rng.randint(card_rot.height // 2 + 10, SCENE_H - card_rot.height // 2 - 10)

        bbox = _paste_card(bg, card_rot, cx, cy)
        if bbox != (0, 0, 0, 0):
            cls_id = config.CARD_CLASSES.index(rank)
            annotations.append((cls_id, rank, bbox))

    return bg, annotations


def bbox_to_yolo(bbox: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / SCENE_W
    cy = ((y1 + y2) / 2) / SCENE_H
    w  = (x2 - x1) / SCENE_W
    h  = (y2 - y1) / SCENE_H
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',    type=int,   default=2000, help='Número de imágenes')
    parser.add_argument('--out',  type=str,   default='data/synthetic')
    parser.add_argument('--seed', type=int,   default=None)
    parser.add_argument('--split', type=float, default=0.85,
                        help='Fracción para train (resto val)')
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    train_img = out / 'images' / 'train'; train_img.mkdir(parents=True, exist_ok=True)
    train_lbl = out / 'labels' / 'train'; train_lbl.mkdir(parents=True, exist_ok=True)
    val_img   = out / 'images' / 'val';   val_img.mkdir(parents=True, exist_ok=True)
    val_lbl   = out / 'labels' / 'val';   val_lbl.mkdir(parents=True, exist_ok=True)

    # dataset.yaml
    yaml_path = out / 'dataset.yaml'
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(config.CARD_CLASSES)}\n"
        f"names: {config.CARD_CLASSES}\n"
    )

    n_train = int(args.n * args.split)
    counts  = {r: 0 for r in config.CARD_CLASSES}

    for i in range(args.n):
        is_train = i < n_train
        img_dir  = train_img if is_train else val_img
        lbl_dir  = train_lbl if is_train else val_lbl

        image, annotations = generate_scene(rng)
        fname = f"{i:05d}"

        Image.fromarray(image).save(img_dir / f"{fname}.jpg", quality=92)

        with open(lbl_dir / f"{fname}.txt", 'w') as f:
            for cls_id, rank, bbox in annotations:
                f.write(f"{cls_id} {bbox_to_yolo(bbox)}\n")
                counts[rank] += 1

        if (i + 1) % 200 == 0 or i == args.n - 1:
            print(f"  {i+1}/{args.n} imágenes generadas...")

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
