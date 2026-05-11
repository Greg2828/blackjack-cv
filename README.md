# Blackjack CV

Sistema de visión artificial sobre Raspberry Pi 5 que observa una mesa real de blackjack en tiempo real, recomienda la jugada óptima según basic strategy y registra cada partida para análisis estadístico posterior.

**Hardware**: Raspberry Pi 5 + Pi Camera  
**Stack**: Python · YOLOv8 · OpenCV · NumPy · Pandas

## Cómo funciona

1. La cámara graba la mesa en tiempo real
2. YOLOv8 detecta las cartas del jugador y del crupier
3. El motor de basic strategy calcula la mejor acción (HIT / STAND / DOUBLE / SPLIT / SURRENDER)
4. La recomendación se muestra en el monitor conectado a la Pi
5. Al finalizar la mano se guarda todo en CSV para análisis posterior

## Setup

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

**Modo simulación** (sin cámara, para probar ya):
```bash
python scripts/simulate.py
python scripts/simulate.py --bankroll 200
```

**Modo real** (requiere modelo entrenado en `models/`):
```bash
python main.py
```

Controles en modo real: `n` nueva mano · `r` reset · `+/-` apuesta · `q` salir

## Entrenamiento del modelo

Genera datos sintéticos y entrena en Colab:
```bash
python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic
```
Luego sube `data/synthetic/` a Google Drive y ejecuta en Colab:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/synthetic/dataset.yaml', epochs=50, imgsz=640)
```
Copia el modelo resultante a `models/yolov8n_blackjack.pt`.

## Tests

```bash
pytest tests/ -v
```

## Estructura

```
blackjack-cv/
├── main.py                  # bucle principal (cámara → detección → estrategia → display)
├── config.py                # constantes: zonas, fichas, umbrales, paths
├── scripts/
│   ├── simulate.py          # modo manual para testear sin cámara
│   └── generate_synthetic_data.py  # genera dataset para YOLOv8
├── src/
│   ├── game/                # Card, Hand, Deck, GameState, Phase, Action, Outcome
│   ├── decision/            # basic strategy (HIT/STAND/DOUBLE/SPLIT/SURRENDER)
│   ├── perception/          # cámara y detección YOLOv8
│   ├── ui/                  # display OpenCV con recomendación
│   ├── analysis/            # logging CSV por mano
│   └── core/                # detector de movimiento
├── tests/                   # 37 tests unitarios
├── models/                  # modelos .pt (excluidos de git)
└── data/                    # imágenes y logs (excluidos de git)
```
