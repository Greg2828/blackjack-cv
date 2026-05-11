# Blackjack CV

Sistema de visión artificial sobre **Raspberry Pi 5** que observa una mesa real de blackjack en tiempo real, recomienda la jugada óptima según *basic strategy* y registra cada partida para análisis estadístico posterior.

**Este README es el documento de referencia del proyecto.** Está escrito para que cualquier persona o IA pueda entender el estado exacto del proyecto sin necesidad de leer el código.

---

## Índice

1. [Qué hace el sistema](#qué-hace-el-sistema)
2. [Hardware y stack tecnológico](#hardware-y-stack-tecnológico)
3. [Estado actual del proyecto](#estado-actual-del-proyecto)
4. [Estructura completa de archivos](#estructura-completa-de-archivos)
5. [Módulos src/ — detalle](#módulos-src--detalle)
6. [Scripts — detalle](#scripts--detalle)
7. [Tests](#tests)
8. [Notebooks](#notebooks)
9. [Cómo ejecutar cada parte](#cómo-ejecutar-cada-parte)
10. [Flujo de datos](#flujo-de-datos)
11. [Pendiente / próximos pasos](#pendiente--próximos-pasos)

---

## Qué hace el sistema

1. Una cámara en la Raspberry Pi graba la mesa de blackjack en tiempo real.
2. Cuando el jugador coloca cartas y deja de moverse, el detector de movimiento activa el análisis.
3. YOLOv8 identifica las cartas visibles en la mesa y determina a quién pertenece cada una (jugador o crupier).
4. El motor de *basic strategy* calcula la acción matemáticamente óptima para la combinación actual.
5. La recomendación se muestra en grande en un monitor conectado a la Pi: **HIT / STAND / DOUBLE / SPLIT / SURRENDER**.
6. La ventana también muestra una fila de la tabla de estrategia con los 10 posibles upcards del crupier coloreados, para que el jugador entienda el razonamiento.
7. Al terminar cada mano, el sistema guarda en un CSV: las cartas que salieron, la secuencia de recomendaciones, lo que el jugador hizo, el resultado y el estado del bankroll.
8. Un notebook de Jupyter analiza ese historial con gráficos (bankroll, tasa de victoria, EV por carta del crupier, adherencia a la estrategia).

---

## Hardware y stack tecnológico

| Elemento | Detalle |
|---|---|
| Hardware | Raspberry Pi 5, Pi Camera |
| OS | Raspberry Pi OS Bookworm (64-bit) |
| Lenguaje | Python 3.13 |
| Detección | YOLOv8 nano (Ultralytics) |
| Visión | OpenCV 4.x |
| Análisis | Pandas, Matplotlib, Seaborn |
| Tests | pytest |
| Entorno | venv con `--system-site-packages` |

---

## Estado actual del proyecto

### ✅ Completado y funcionando

| Componente | Estado | Verificado |
|---|---|---|
| Lógica del juego (Card, Hand, GameState) | ✅ Completo | 37/37 tests pasan |
| Motor de *basic strategy* | ✅ Completo | HIT/STAND/DOUBLE/SPLIT/SURRENDER |
| Simulador sin cámara (`simulate.py`) | ✅ Funciona | Probado manualmente |
| Ventana OpenCV con recomendación | ✅ Funciona | Muestra estrategia + tabla visual |
| Logger CSV por mano | ✅ Funciona | 302 manos guardadas |
| Módulo de estadísticas (`stats.py`) | ✅ Completo | EV, win rate, gráficos |
| Notebook de análisis | ✅ Completo | Con datos de muestra |
| Generador de datos sintéticos | ✅ Funciona | Imágenes + anotaciones YOLO |
| Detector de movimiento | ✅ Completo | Frame differencing, 2s estabilidad |
| Bucle principal (`main.py`) | ✅ Estructura completa | Arranca; sin detección hasta tener modelo |
| Notebook de entrenamiento Colab | ✅ Completo | Listo para usar |

### ⚠️ Pendiente (bloqueado por hardware/modelo)

| Componente | Estado | Bloqueado por |
|---|---|---|
| Detector de cartas YOLO | ⚠️ Stub | Modelo `.pt` no entrenado aún |
| Test de cámara real | ⚠️ Sin probar | Cámara no conectada aún |
| Sistema completo en tiempo real | ⚠️ Sin probar | Necesita modelo entrenado |

### ❌ No implementado

| Componente | Notas |
|---|---|
| Manejo completo de split | Split registra acción pero no divide la mano en dos manos independientes. Funciona como hit simplificado. |
| Detección de fichas (apuesta) | El sistema no detecta el valor de la apuesta visualmente. Se introduce manualmente. |

---

## Estructura completa de archivos

```
blackjack-cv/
│
├── main.py                          # Bucle principal del sistema en tiempo real
├── config.py                        # Todas las constantes del sistema
├── requirements.txt                 # Dependencias Python
├── .gitignore                       # Excluye venv/, data/, models/*.pt, __pycache__/
│
├── src/                             # Código fuente principal
│   ├── game/                        # Lógica pura del blackjack
│   │   ├── card.py                  # Clase Card
│   │   ├── hand.py                  # Clase Hand
│   │   ├── deck.py                  # Clase Deck (solo para simulaciones/tests)
│   │   └── state.py                 # GameState, Phase, Action, Outcome
│   ├── decision/
│   │   └── strategy.py              # Motor de basic strategy completo
│   ├── perception/
│   │   ├── camera.py                # Wrapper Pi Camera / VideoCapture
│   │   └── detector.py              # Detector YOLOv8 (stub hasta tener modelo)
│   ├── ui/
│   │   └── display.py               # Ventana OpenCV: recomendación + tabla estrategia
│   ├── analysis/
│   │   ├── logger.py                # Escribe CSV por mano
│   │   └── stats.py                 # Funciones de análisis y gráficos
│   └── core/
│       └── motion.py                # Detector de movimiento por frame differencing
│
├── scripts/
│   ├── simulate.py                  # Simulador manual (sin cámara) — FUNCIONA HOY
│   ├── test_camera.py               # Verifica Pi Camera y muestra zonas — necesita cámara
│   ├── generate_synthetic_data.py   # Genera dataset sintético para entrenar YOLO
│   ├── gen_sample_data.py           # Simula partidas y llena el CSV de log
│   └── capture_dataset.py           # Captura imágenes reales con la cámara — necesita cámara
│
├── notebooks/
│   ├── analysis.ipynb               # Análisis estadístico del historial de partidas
│   └── colab_train.ipynb            # Entrenamiento YOLOv8 en Google Colab
│
├── tests/
│   ├── test_card.py                 # 6 tests — Card values, is_ace, is_back
│   ├── test_hand.py                 # 10 tests — totals, soft/hard, bust, blackjack, pair
│   ├── test_deck.py                 # 3 tests — 52 cards, draw, seed reproducibility
│   └── test_strategy.py             # 18 tests — pairs, soft, hard, surrender, fallbacks
│
├── models/                          # Vacío — aquí va el modelo entrenado yolov8n_blackjack.pt
└── data/
    ├── raw_images/                  # Fotos reales de cartas (excluido de git)
    ├── labeled/                     # Imágenes etiquetadas para YOLO (excluido de git)
    ├── synthetic/                   # Dataset sintético generado (excluido de git)
    └── games_log.csv                # Historial de partidas (excluido de git)
```

---

## Módulos src/ — detalle

### `src/game/`

El único módulo que no sabe nada de cámaras ni pantallas. Contiene las reglas puras del juego.

**`card.py`** — Clase `Card(rank)` donde rank es `'A'`, `'2'`..`'10'`, `'J'`, `'Q'`, `'K'` o `'BACK'` (carta tapada).
- `card.value` → valor numérico (A=11, figuras=10, BACK lanza ValueError)
- `card.is_ace`, `card.is_back` → propiedades booleanas

**`hand.py`** — Clase `Hand(cards=[])` que representa una mano.
- `hand.total()` → suma óptima: ases bajan de 11 a 1 si superan 21
- `hand.is_soft()` → True si hay un as contando como 11
- `hand.is_bust()` → total > 21
- `hand.is_blackjack()` → exactamente 2 cartas visibles que suman 21
- `hand.is_pair()` → 2 cartas con el mismo rango
- `hand.visible_cards` → filtra cartas BACK
- `hand.has_hidden` → True si hay alguna carta BACK

**`deck.py`** — Clase `Deck(n_decks=1, seed=None)`. Solo se usa en tests y en `gen_sample_data.py`. En el juego real las cartas vienen de la cámara.

**`state.py`** — Contiene:
- `Phase` (enum): `WAITING_BET`, `BET_PLACED`, `PLAYER_TURN`, `DEALER_TURN`, `RESOLVED`
- `Action` (enum): `HIT`, `STAND`, `DOUBLE`, `SPLIT`, `SURRENDER`
- `Outcome` (enum): `WIN`, `LOSE`, `PUSH`, `BLACKJACK`
- `GameState` (dataclass): `player_hand`, `dealer_hand`, `bet`, `bankroll`, `phase`, `doubled`, `surrendered`
- `GameState.resolve()` → devuelve `(Outcome, delta_bankroll)`. Maneja surrender (devuelve -bet/2), blackjack (paga 1.5x), doubled (dobla la apuesta).

---

### `src/decision/strategy.py`

Motor de *basic strategy* estándar (6 barajas, crupier planta en soft 17, late surrender).

Tres tablas internas:
- `_PAIR` — 10 filas (2,2 hasta A,A) × 10 columnas (upcard 2..A)
- `_SOFT` — 8 filas (soft 13..20) × 10 columnas
- `_HARD` — 10 filas (hard 8..17) × 10 columnas

Funciones públicas:
- `recommend(player_hand, dealer_upcard, *, can_split, can_double, can_surrender) → Action`
  Devuelve la acción óptima. Los flags controlan qué acciones están disponibles (solo en la primera decisión).
- `full_row(player_hand, *, can_split, can_double, can_surrender) → list[Action]`
  Devuelve 10 acciones, una por cada posible upcard del crupier (2..A). Usado para la tabla visual en pantalla.

Casos especiales verificados por tests:
- Hard 16 vs 9, 10, A → SURRENDER
- Hard 16 vs 2-6 → STAND
- Soft 18 vs 3-6 → DOUBLE; vs 7-8 → STAND; vs 9,10,A → HIT
- Par de ases → siempre SPLIT
- Par de 8s → siempre SPLIT
- Par de 5s → nunca SPLIT, tratar como hard 10 (DOUBLE vs 2-9)
- Si `can_surrender=False`: soft 16 vs 10 hace fallback a HIT

---

### `src/perception/`

**`camera.py`** — Clase `Camera(source=0, width=1280, height=720)`.
- Wrapper sobre `cv2.VideoCapture`.
- Context manager: `with Camera() as cam: frame = cam.read()`
- Lanza `RuntimeError` si la cámara no se puede abrir.

**`detector.py`** — Clase `CardDetector(model_path, card_classes, zone_dealer, zone_player)`.
- Si el modelo `.pt` no existe, `detector.ready = False` y `detect()` devuelve listas vacías sin error.
- Si el modelo existe, usa YOLOv8 para detectar cartas en el frame.
- Separa detecciones por zona Y: `ZONE_DEALER` (0-40% altura) y `ZONE_PLAYER` (40-75% altura).
- Devuelve `(player_cards: list[Card], dealer_cards: list[Card])`.
- **Estado actual: `detector.ready = False`** — el modelo `models/yolov8n_blackjack.pt` no existe todavía.

---

### `src/ui/display.py`

Clase `Display(width=900, height=430)` — ventana OpenCV para el monitor de la Pi.

**`display.show(state, recommendation, strategy_row, dealer_upcard_rank)`**
- Zona superior: cartas del jugador y del crupier con totales.
- Centro: acción recomendada en grande y coloreada (verde=HIT, rojo=STAND, naranja=DOUBLE, morado=SPLIT, amarillo=SURRENDER).
- Zona inferior: fila de la tabla de estrategia — 10 celdas coloreadas (una por cada upcard del crupier), con la celda actual resaltada en blanco.
- Pie: bankroll y apuesta actual.

**`display.show_outcome(outcome, delta)`**
- Pantalla de resultado: muestra WIN/LOSE/PUSH/BLACKJACK con el delta económico.

Colores de acciones: HIT=(0,210,0) verde, STAND=(0,0,210) rojo, DOUBLE=(0,165,255) naranja, SPLIT=(255,0,255) morado, SURRENDER=(0,220,220) amarillo.

---

### `src/analysis/`

**`logger.py`** — Clase `HandLogger(path='data/games_log.csv')`.
- Crea el CSV con cabecera si no existe.
- `logger.log(state, recommended, taken, outcome, delta)` escribe una fila con:
  `timestamp, player_cards, dealer_upcard, dealer_final, actions_recommended, actions_taken, bet, outcome, delta, bankroll`

**`stats.py`** — Funciones de análisis sobre el DataFrame del CSV:
- `load_log(path)` → DataFrame con columna `session` inferida (gaps > 10min = nueva sesión)
- `summary(df)` → dict con: manos, win_rate, blackjack_rate, push_rate, loss_rate, ev_por_mano, ev_pct_apuesta, delta_total, sesiones
- `print_summary(df)` → imprime resumen formateado en terminal
- `plot_bankroll(df)` → gráfico de línea con área verde/roja según si está por encima o debajo del inicio
- `plot_outcomes(df)` → gráfico de barras de resultados (ganas/pierdes/empate/blackjack)
- `plot_delta_by_upcard(df)` → EV medio por carta visible del crupier (barras verdes/rojas)
- `plot_action_distribution(df)` → distribución de acciones recomendadas
- `plot_adherence_by_session(df)` → % de manos donde la acción tomada coincide con la recomendada
- `plot_player_total_distribution(df)` → distribución de totales iniciales del jugador (2 primeras cartas)

---

### `src/core/motion.py`

Clase `MotionDetector(threshold=30, stability_seconds=2.0, min_area=1500)`.
- Compara frames consecutivos (Gaussian blur + absdiff + threshold).
- `motion.update(frame)` → devuelve `(motion_detected: bool, just_stabilized: bool)`.
- `just_stabilized` es True **exactamente en un frame**: el primero estable tras un período de movimiento.
- `main.py` solo lanza la detección YOLO cuando `just_stabilized=True`, evitando procesar frames mientras el jugador mueve las cartas.
- `motion.reset()` → reinicia el estado para la siguiente mano.

---

## Scripts — detalle

### `scripts/simulate.py` ✅ FUNCIONA SIN CÁMARA

Simulador interactivo completo. Introduce cartas manualmente por teclado. Abre la ventana OpenCV real con la recomendación y la tabla de estrategia. Guarda cada mano en el CSV.

```bash
python scripts/simulate.py
python scripts/simulate.py --bankroll 500
```

Flujo de una mano:
1. Introduce la apuesta (enter = 10)
2. Introduce la carta visible del crupier (ej: `7`)
3. Introduce tus cartas (ej: `A 8`)
4. El sistema muestra la recomendación y la tabla visual
5. Introduce tu acción: `h`=hit, `s`=stand, `d`=double, `sp`=split, `su`=surrender (enter acepta la recomendada)
6. Si hiciste hit/double, introduce la nueva carta
7. Introduce la carta tapada del crupier y las que saque
8. El sistema muestra el resultado y lo guarda

---

### `scripts/gen_sample_data.py` ✅ FUNCIONA

Simula partidas automáticas siguiendo *basic strategy* al 100%. Puebla el CSV de log sin necesidad de jugar manualmente.

```bash
python scripts/gen_sample_data.py
python scripts/gen_sample_data.py --hands 500 --bankroll 200
```

Con 300 manos: win rate ~43.7%, EV ~-0.41% (correcto: house edge teórico con basic strategy perfecta es ~0.5%).

---

### `scripts/generate_synthetic_data.py` ✅ FUNCIONA

Genera imágenes sintéticas de cartas sobre fondo verde con anotaciones YOLO automáticas. Necesario para entrenar el modelo.

```bash
python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic
python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic --seed 42
```

Produce:
- `data/synthetic/images/train/` y `/val/` — imágenes JPG 1280×720
- `data/synthetic/labels/train/` y `/val/` — anotaciones en formato YOLO
- `data/synthetic/dataset.yaml` — configuración lista para YOLOv8

Cada imagen contiene 1-4 cartas con rotación aleatoria (-35°/+35°), escala variable (0.8x-1.6x), ruido de iluminación y viñeta.

---

### `scripts/test_camera.py` ⚠️ NECESITA CÁMARA

Verifica que la Pi Camera funciona correctamente y muestra las zonas de detección.

```bash
python scripts/test_camera.py
```

Muestra el feed en vivo con tres rectángulos superpuestos:
- Azul: zona DEALER (0-40% de la altura)
- Verde: zona PLAYER (40-75%)
- Naranja: zona BETS (75-100%)

Controles: `s` guarda foto, `q` sale.

---

### `scripts/capture_dataset.py` ⚠️ NECESITA CÁMARA

Herramienta para fotografiar cartas reales y construir el dataset de fine-tuning.

```bash
python scripts/capture_dataset.py
```

Controles: teclas `A`, `2`-`9`, `0`(=10), `J`, `Q`, `K`, `B`(=BACK) para seleccionar clase. `ESPACIO` captura. Las fotos se guardan en `data/raw_images/{rank}/`.

---

## Tests

**37 tests, todos pasan.** Ejecutar con:

```bash
python -m pytest tests/ -v
```

| Archivo | Tests | Qué verifica |
|---|---|---|
| `test_card.py` | 6 | Valores numéricos, is_ace, is_back, ValueError en BACK.value |
| `test_hand.py` | 10 | Totales soft/hard, bust, blackjack natural, pair, carta tapada ignorada |
| `test_deck.py` | 3 | 52 cartas por baraja, draw() reduce el count, seed reproducible |
| `test_strategy.py` | 18 | Pares (A,A; 8,8; 10,10; 5,5), soft manos (18 vs varios upcards), hard manos (11, 16, 15, 17+, 12), surrender, fallback sin can_surrender |

---

## Notebooks

### `notebooks/analysis.ipynb` ✅ FUNCIONA

Análisis estadístico del historial. Requiere datos en `data/games_log.csv`.

Si el CSV está vacío, generarlo primero:
```bash
python scripts/gen_sample_data.py --hands 300
```

Luego abrir:
```bash
jupyter notebook notebooks/analysis.ipynb
```

Contiene 7 secciones: carga de datos, resumen numérico, evolución del bankroll, distribución de resultados, EV por upcard del crupier, distribución de acciones, adherencia a la estrategia.

---

### `notebooks/colab_train.ipynb` ✅ LISTO PARA USAR

Guía paso a paso para entrenar YOLOv8 en Google Colab (GPU gratuita T4).

**Flujo:**
1. Generar datos sintéticos en la Pi: `python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic`
2. Comprimir: `cd data && zip -r synthetic.zip synthetic/`
3. Subir `synthetic.zip` a Google Drive
4. Abrir el notebook en Colab y ejecutar celdas en orden
5. El notebook monta Drive, extrae el dataset, entrena, evalúa y guarda el modelo en Drive
6. Copiar `yolov8n_blackjack.pt` a `models/` en la Pi

El notebook incluye: verificación de GPU, corrección automática de rutas del dataset.yaml, entrenamiento con early stopping (patience=15), métricas finales (mAP50, precision, recall), visualización de detecciones de ejemplo.

---

## Cómo ejecutar cada parte

### Hoy mismo (sin cámara ni modelo)

```bash
cd ~/projects/blackjack-cv
source venv/bin/activate

# Tests
python -m pytest tests/ -v

# Simulador interactivo (abre ventana OpenCV)
python scripts/simulate.py

# Generar datos de partidas simuladas
python scripts/gen_sample_data.py --hands 300

# Estadísticas en terminal
python -c "from src.analysis.stats import load_log, print_summary; print_summary(load_log('data/games_log.csv'))"

# Notebook de análisis
jupyter notebook notebooks/analysis.ipynb

# Generar dataset para entrenar YOLO
python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic
```

### Cuando la cámara esté conectada

```bash
# Verificar que la cámara funciona y las zonas están bien posicionadas
python scripts/test_camera.py

# Capturar imágenes reales para fine-tuning
python scripts/capture_dataset.py
```

### Cuando el modelo esté entrenado

```bash
# Copiar modelo a models/yolov8n_blackjack.pt
# Luego arrancar el sistema completo
python main.py
```

---

## Flujo de datos

```
[Pi Camera]
    │
    ▼
[motion.py]  ←── detecta cuando la escena se estabiliza (2 segundos sin movimiento)
    │  just_stabilized=True
    ▼
[detector.py] ──── YOLOv8 ────► lista de Card por zona (PLAYER / DEALER)
    │
    ▼
[state.py]  ←── construye Hand para jugador y crupier, infiere Phase
    │
    ▼
[strategy.py] ──► recommend() + full_row() ──► Action recomendada + fila de tabla
    │
    ├──► [display.py] ──► ventana OpenCV (recomendación + tabla visual)
    │
    └──► [logger.py] ──► data/games_log.csv (al finalizar la mano)
                              │
                              ▼
                         [stats.py + analysis.ipynb]
                         análisis post-sesión
```

---

## `config.py` — constantes del sistema

| Constante | Valor | Descripción |
|---|---|---|
| `CARD_CLASSES` | lista de 14 strings | A, 2..10, J, Q, K, BACK |
| `FRAME_WIDTH / HEIGHT` | 1280 / 720 | Resolución de la cámara |
| `ZONE_DEALER` | y: 0.00–0.40 | Zona de las cartas del crupier |
| `ZONE_PLAYER` | y: 0.40–0.75 | Zona de las cartas del jugador |
| `ZONE_BETTING` | y: 0.75–1.00 | Zona de fichas/apuesta |
| `MOTION_THRESHOLD` | 30 | Sensibilidad del detector de movimiento |
| `STABILITY_SECONDS` | 2.0 | Segundos de quietud para activar detección |
| `DEALER_STAND_ON_SOFT_17` | True | Regla del crupier |
| `BLACKJACK_PAYOUT` | 1.5 | Pago del blackjack natural |
| `STARTING_BANKROLL` | 100.0 | Bankroll inicial por defecto |
| `MODEL_PATH` | models/yolov8n_blackjack.pt | Ruta del modelo YOLOv8 |
| `LOG_FILE` | data/games_log.csv | Ruta del historial de partidas |

---

## Pendiente / próximos pasos

### Inmediato (no requiere hardware)
- [ ] Entrenar el modelo YOLOv8 en Google Colab con `notebooks/colab_train.ipynb`
- [ ] Generar 2000 imágenes sintéticas y subir a Drive

### Requiere Pi Camera conectada
- [ ] Ejecutar `scripts/test_camera.py` y ajustar las zonas en `config.py` si es necesario
- [ ] Capturar imágenes reales con `scripts/capture_dataset.py`
- [ ] Fine-tune del modelo con imágenes reales de la mesa

### Mejoras de código identificadas
- [ ] Split completo: cuando el jugador hace split, gestionar dos manos independientes (actualmente se simplifica como HIT)
- [ ] Detección automática de apuesta: detectar fichas por color HSV (zona BETTING en config ya definida, pero `CHIP_HSV_RANGES` están en None)
- [ ] Modo multi-mano: soporte para más de un jugador frente al crupier

---

## Setup inicial

```bash
git clone https://github.com/Greg2828/blackjack-cv.git
cd blackjack-cv
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v   # debe mostrar 37 passed
```
