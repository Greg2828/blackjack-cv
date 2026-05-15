# Blackjack CV

Sistema de visión artificial sobre **Raspberry Pi 5** que observa una mesa real de blackjack, recomienda la jugada óptima según *basic strategy* y registra cada partida para análisis estadístico.

**Este README es el documento de referencia del proyecto.** Está escrito para que cualquier persona o IA pueda entender el estado exacto del proyecto sin necesidad de leer el código.

---

## Índice

1. [Qué hace el sistema](#1-qué-hace-el-sistema)
2. [Hardware y stack tecnológico](#2-hardware-y-stack-tecnológico)
3. [Estado actual del proyecto](#3-estado-actual-del-proyecto)
4. [Estructura completa de archivos](#4-estructura-completa-de-archivos)
5. [Pipeline completo: de fotos a modelo funcionando](#5-pipeline-completo-de-fotos-a-modelo-funcionando)
6. [Módulos src/ — detalle técnico](#6-módulos-src--detalle-técnico)
7. [Scripts — detalle](#7-scripts--detalle)
8. [Tests](#8-tests)
9. [Notebooks](#9-notebooks)
10. [Cómo ejecutar cada parte](#10-cómo-ejecutar-cada-parte)
11. [Flujo de datos en tiempo real](#11-flujo-de-datos-en-tiempo-real)
12. [config.py — referencia de constantes](#12-configpy--referencia-de-constantes)
13. [Bug activo — cv2.imshow no muestra imagen con picamera2 + torch](#13-bug-activo--cv2imshow-no-muestra-imagen-con-picamera2--torch-en-el-mismo-proceso)
14. [Próximos pasos](#14-próximos-pasos)

---

## 1. Qué hace el sistema

1. Una Pi Camera graba la mesa en tiempo real.
2. Un detector de movimiento espera a que la escena se estabilice (2 s sin movimiento) antes de analizar — evita procesar frames borrosos mientras el jugador coloca cartas.
3. YOLOv8 identifica cada carta visible y determina si pertenece al jugador o al crupier según su posición en el frame.
4. El motor de *basic strategy* calcula la acción matemáticamente óptima para la combinación actual.
5. La recomendación aparece en grande en un monitor conectado a la Pi: **HIT / STAND / DOUBLE / SPLIT / SURRENDER**, con la fila completa de la tabla de estrategia (los 10 posibles upcards del crupier) para que el jugador entienda el razonamiento.
6. Si las fichas están calibradas por HSV, el sistema detecta automáticamente el valor de la apuesta en la zona de betting.
7. Al terminar cada mano, guarda en CSV: cartas, recomendaciones, acción tomada, resultado y bankroll.
8. Un notebook de Jupyter analiza ese historial: curva de bankroll, EV por upcard del crupier, adherencia a la estrategia, tasa de victoria.

---

## 2. Hardware y stack tecnológico

| Elemento | Detalle |
|---|---|
| Hardware | Raspberry Pi 5, Pi Camera Module 3 (IMX708, 12.3 MP) |
| OS | Raspberry Pi OS Bookworm (64-bit) |
| Lenguaje | Python 3.13 |
| Acceso a cámara | **picamera2** (obligatorio en Pi 5 + Bookworm — `cv2.VideoCapture` no funciona con libcamera) |
| Detección de cartas | YOLOv8 nano (Ultralytics) — inferencia en Pi, entrenamiento en Colab |
| Visión | OpenCV 4.x |
| Análisis | Pandas, Matplotlib, Seaborn |
| Tests | pytest |
| Entorno virtual | `python3 -m venv venv --system-site-packages` (picamera2 preinstalado en Bookworm) |

### Nota sobre la Pi Camera Module 3 (IMX708)

En Pi 5 con Bookworm, la cámara usa el stack **libcamera**. `cv2.VideoCapture` abre el dispositivo pero no puede leer frames. La solución es `picamera2`.

Además, `picamera2` devuelve datos en orden **BGR** aunque el formato configurado sea `RGB888` — verificado empíricamente. No se aplica ninguna conversión de color.

La documentación completa del sensor y su configuración está en `docs/camera.md`.

---

## 3. Estado actual del proyecto

### ✅ Completado y funcionando

| Componente | Tests | Notas |
|---|---|---|
| Lógica del juego (Card, Hand, Deck, GameState) | 19/19 | Reglas completas |
| Motor de *basic strategy* | 18/18 | Pairs, soft, hard, surrender, fallbacks |
| `GameState.resolve()` y `resolve_hand()` | 18/18 | Incluye lógica de split (21≠blackjack tras split) |
| Simulador sin cámara (`simulate.py`) | — | Split real: dos manos independientes, regla de As-split |
| Ventana OpenCV (`display.py`) | — | Recomendación + tabla visual de estrategia |
| Logger CSV por mano | — | 302 manos de muestra guardadas |
| Análisis estadístico (`stats.py` + notebook) | — | EV, win rate, adherencia, gráficos |
| Detector de movimiento (`motion.py`) | — | Frame differencing, estabilización configurable |
| Bucle principal (`main.py`) | — | Estructura completa |
| Generador datos sintéticos | — | Cartas sobre fondo verde con anotaciones YOLO |
| Captura de fotos reales (`capture_dataset.py`) | — | Guarda frames limpios (sin overlay) |
| Auto-anotación (`auto_annotate.py`) | — | Genera etiquetas YOLO por contorno, sin herramientas externas |
| Detector de fichas (`chip_detector.py`) | — | HSV con filtro circular; degradación elegante sin calibrar |
| Calibración de fichas (`calibrate_chips.py`) | — | Herramienta de muestreo HSV con ratón |
| Notebook de entrenamiento Colab | — | Soporta fotos reales y datos sintéticos |
| **Configuración Pi Camera Module 3** | — | picamera2 + autofoco continuo + AWB + AE |
| **Dataset capturado** | — | 1450 fotos reales, 14 clases, baraja de corazones |
| **Auto-anotación ejecutada** | — | 1164 train / 286 val, 0 fallos |
| **Modelo YOLOv8n entrenado** | — | **mAP50 = 0.960** · mAP50-95 = 0.949 · 3.9ms/imagen |

**Total: 55 tests, todos pasan.**

### ⚠️ Pendiente

| Componente | Qué falta |
|---|---|
| Fichas calibradas | Ejecutar `calibrate_chips.py` con la cámara apuntando a la mesa |
| Prueba en producción | Ejecutar `main.py` con modelo y cámara sobre la mesa real |
| Fine-tuning (opcional) | Añadir fotos del 2 y el 5 (recall más bajo), fotos sobre tapiz real |

---

## 4. Estructura completa de archivos

```
blackjack-cv/
├── main.py                          # Bucle principal en tiempo real
├── config.py                        # Todas las constantes del sistema
├── requirements.txt
│
├── docs/
│   └── camera.md                    # Especificaciones IMX708 y decisiones de configuración
│
├── src/
│   ├── game/
│   │   ├── card.py                  # Clase Card (rank, value, is_ace, is_back)
│   │   ├── hand.py                  # Clase Hand (total, soft/hard, bust, blackjack, pair)
│   │   ├── deck.py                  # Clase Deck (solo para simulaciones y tests)
│   │   └── state.py                 # GameState, Phase, Action, Outcome, resolve(), resolve_hand()
│   ├── decision/
│   │   └── strategy.py              # Basic strategy completa: recommend() y full_row()
│   ├── perception/
│   │   ├── camera.py                # Wrapper picamera2 (Pi 5 + Bookworm)
│   │   ├── detector.py              # Detector YOLOv8 (stub gracioso sin modelo)
│   │   └── chip_detector.py         # Detección de fichas por HSV
│   ├── ui/
│   │   └── display.py               # Ventana OpenCV: recomendación + tabla estrategia
│   ├── analysis/
│   │   ├── logger.py                # Escribe CSV por mano
│   │   └── stats.py                 # Funciones de análisis y gráficos
│   └── core/
│       └── motion.py                # Detector de movimiento por frame differencing
│
├── scripts/
│   ├── simulate.py                  # Simulador manual completo (sin cámara) — FUNCIONA HOY
│   ├── gen_sample_data.py           # Simula partidas y llena el CSV de log
│   ├── generate_synthetic_data.py   # Genera dataset sintético para pre-entrenar YOLO
│   ├── capture_dataset.py           # Captura fotos reales de cartas — necesita cámara
│   ├── auto_annotate.py             # Auto-genera etiquetas YOLO desde raw_images/
│   ├── calibrate_chips.py           # Calibración HSV de fichas con ratón
│   └── test_camera.py               # Verifica Pi Camera y muestra zonas — necesita cámara
│
├── notebooks/
│   ├── analysis.ipynb                              # Análisis estadístico del historial de partidas
│   ├── colab_train.ipynb                           # Entrenamiento YOLOv8 en Google Colab
│   ├── colab_trainPrimerEntrenamientoEjecutado.ipynb  # Notebook ejecutado — primer entrenamiento real
│   └── dia_2_dataset_y_entrenamiento.ipynb         # Documentación sesión 2: cámara, dataset y resultados
│
├── tests/
│   ├── test_card.py                 # 6 tests
│   ├── test_hand.py                 # 10 tests
│   ├── test_deck.py                 # 3 tests
│   ├── test_strategy.py             # 18 tests
│   └── test_state.py                # 18 tests
│
├── models/
│   └── yolov8n_blackjack.pt         # Modelo entrenado — mAP50=0.960 (14 clases, 1450 fotos)
│
└── data/
    ├── raw_images/                  # Fotos brutas capturadas (excluido de git)
    │   ├── A/ … BACK/               # 14 carpetas, ~103 fotos cada una
    ├── labeled/                     # Dataset anotado (excluido de git)
    │   ├── images/train/ y val/     # 1164 train / 286 val
    │   ├── labels/train/ y val/
    │   └── dataset.yaml
    └── games_log.csv                # Historial de partidas (excluido de git)
```

---

## 5. Pipeline completo: de fotos a modelo funcionando

### ¿Qué es YOLO y por qué lo necesitamos?

YOLO (You Only Look Once) es un algoritmo de detección de objetos. Dado un frame de vídeo, devuelve una lista de objetos detectados con su clase (ej. "A", "K", "BACK"), su posición en el frame (bounding box) y una puntuación de confianza.

Para que YOLO funcione con nuestras cartas hay que entrenarlo: mostrarle miles de fotos de cartas con su posición marcada exactamente. Sin ese entrenamiento, el modelo no sabe nada sobre cartas de blackjack.

El Raspberry Pi 5 no tiene GPU y no puede entrenar el modelo eficientemente. Por eso entrenamos en **Google Colab**, que ofrece GPUs en la nube. Una vez entrenado, el modelo resultante (un archivo `.pt` de ~6 MB) se copia a la Pi, donde solo hace *inferencia* — eso sí es rápido incluso sin GPU (≈3.9ms por imagen en T4, velocidad estimada ≈15–20ms en Pi 5).

---

### Paso 1 — Capturar fotos (`capture_dataset.py`)

```bash
python scripts/capture_dataset.py
```

**Las 14 clases y sus teclas:**

| Tecla | Clase | Tecla | Clase |
|-------|-------|-------|-------|
| `A` | As | `J` | Jota |
| `2`–`9` | Números | `W` | Reina (no `Q` — ver nota) |
| `0` | Diez | `K` | Rey |
| `B` | BACK (dorso) | `ESC` | Salir |

> **Nota:** La tecla `Q` está reservada para salir en otros scripts. En `capture_dataset.py` la Reina se selecciona con `W` para evitar el conflicto.

`ESPACIO` = capturar foto | `D` = borrar última

**Objetivo:** ≥100 fotos por clase × 14 clases = ≥1400 fotos. Variar ángulo (0°–40°), posición, distancia, rotación y oclusión parcial. Incluir fotos con dos cartas solapadas — reproduce la situación real de la mesa.

**Fondo:** usar tela oscura sobre el tapiz real. El tapiz verde con patrones confunde la auto-anotación basada en contornos.

---

### Paso 2 — Auto-anotar (`auto_annotate.py`)

YOLO necesita para cada imagen un archivo `.txt` con la bounding box en formato normalizado:

```
class_id  x_centro  y_centro  ancho  alto
```

`auto_annotate.py` genera estos archivos automáticamente por análisis de contornos (Canny + Otsu), válida la proporción ancho/alto contra la de una carta estándar (63.5mm / 88.9mm ≈ 0.714) y divide en train/val.

```bash
python scripts/auto_annotate.py --preview   # revisión visual con bboxes
python scripts/auto_annotate.py             # ejecución directa
```

En `--preview`: `ESPACIO` = ok | `D` = borrar foto | `Q` = salir del preview.

---

### Paso 3 — Comprimir y subir a Google Drive

```bash
cd ~/projects/blackjack-cv/data
zip -r labeled.zip labeled/
```

Sube `labeled.zip` a cualquier carpeta de tu Google Drive. El notebook la buscará automáticamente de forma recursiva.

---

### Paso 4 — Entrenar en Google Colab (`colab_train.ipynb`)

1. Ve a colab.research.google.com
2. Archivo → Abrir notebook → Google Drive → `blackjack-cv/notebooks/colab_train.ipynb`
3. Entorno de ejecución → Cambiar tipo → GPU (T4 o superior) → Guardar
4. Ejecutar celdas en orden

**Resultado del primer entrenamiento (2026-05-15):**

| Métrica | Valor |
|---------|-------|
| Dataset | 1450 fotos reales, 1 baraja (corazones) |
| GPU | Tesla T4 (Google Colab Pro) |
| Épocas | 100 (sin early stopping) |
| mAP50 | **0.960** (mejor época: 82) |
| mAP50-95 | 0.949 |
| Precision | 0.942 |
| Recall | 0.898 |
| Velocidad | 3.9 ms/imagen (inferencia en T4) |

**Rendimiento por clase:**

| Clase | mAP50 | Nota |
|-------|-------|------|
| BACK | 0.995 | Casi perfecto — el dorso es visualmente único |
| J / K | 0.993 | Excelente |
| 7 / 9 | 0.988 / 0.983 | Muy bueno |
| A / 10 | 0.953 / 0.960 | Bueno |
| 2 / 5 | 0.925 | Recall más bajo (0.761 / 0.745) — candidatos a más fotos |

Ver análisis detallado en `notebooks/dia_2_dataset_y_entrenamiento.ipynb`.

---

### Paso 5 — Copiar el modelo a la Pi

```bash
# El modelo ya está en models/yolov8n_blackjack.pt
# Verificar zonas con la cámara conectada
python scripts/test_camera.py

# Calibrar fichas (opcional)
python scripts/calibrate_chips.py

# Arrancar el sistema
python main.py
```

---

### Calibrar las fichas (apuesta automática)

```bash
python scripts/calibrate_chips.py
```

Mueve el ratón sobre una ficha para ver su HSV. Clic para muestrear. `1`/`2`/`3` asigna la muestra. `p` imprime el bloque para `config.py`. `q` sale. Ver `CHIP_HSV_RANGES` en `config.py`.

---

## 6. Módulos src/ — detalle técnico

### `src/game/`

Lógica pura del juego — no sabe nada de cámaras ni pantallas.

**`card.py`** — `Card(rank)` donde rank es `'A'`, `'2'`..`'10'`, `'J'`, `'Q'`, `'K'` o `'BACK'`.
- `card.value` → A=11, figuras=10, números=su valor; `BACK` lanza ValueError
- `card.is_ace`, `card.is_back`

**`hand.py`** — `Hand(cards=[])`.
- `total()` → suma óptima (ases bajan de 11 a 1 si superan 21)
- `is_soft()` → True si hay un as contando como 11
- `is_bust()`, `is_blackjack()` (exactamente 2 cartas visibles sumando 21), `is_pair()`
- `visible_cards` → filtra BACK; `has_hidden` → hay algún BACK

**`deck.py`** — `Deck(n_decks=1, seed=None)`. Solo para simulaciones y tests.

**`state.py`**
- `Phase`: `WAITING_BET`, `BET_PLACED`, `PLAYER_TURN`, `DEALER_TURN`, `RESOLVED`
- `Action`: `HIT`, `STAND`, `DOUBLE`, `SPLIT`, `SURRENDER`
- `Outcome`: `WIN`, `LOSE`, `PUSH`, `BLACKJACK`
- `GameState.resolve()` → `(Outcome, delta)`. Maneja surrender (-bet/2), blackjack (1.5x), double (2x bet).
- `GameState.resolve_hand(hand, doubled, is_split)` → mismo cálculo para mano de split. Con `is_split=True`, el 21 de 2 cartas no paga como blackjack natural.

---

### `src/decision/strategy.py`

Motor de *basic strategy* estándar (6 barajas, crupier planta en soft 17, late surrender).

- `recommend(player_hand, dealer_upcard, *, can_split, can_double, can_surrender) → Action`
- `full_row(player_hand, ...) → list[Action]` — 10 acciones (una por upcard), para la tabla visual

---

### `src/perception/`

**`camera.py`** — `Camera(source=0, width=1280, height=720)`. Wrapper sobre **picamera2**.
- Usa `create_video_configuration(format="RGB888")` + `capture_array("main")`
- El frame raw ya está en BGR — no se aplica conversión de color
- Activa autofoco continuo (AfMode=2), AE y AWB al arrancar
- Ver `docs/camera.md` para la documentación completa del sensor IMX708

**`detector.py`** — `CardDetector(model_path, card_classes, zone_dealer, zone_player)`.
- Si el `.pt` no existe: `detector.ready = False`, `detect()` devuelve listas vacías sin error.
- Si existe: detecta con YOLOv8 y separa cartas por zona Y (DEALER: 0–40%, PLAYER: 40–75%).

**`chip_detector.py`** — `ChipDetector(chip_values, chip_hsv_ranges, min_area)`.
- `calibrated` → True si al menos un chip tiene rangos HSV definidos.
- `detect(frame, zone_y_min, zone_y_max)` → suma del valor de fichas detectadas. Devuelve 0.0 sin error si no está calibrado.

---

### `src/ui/display.py`

`Display(width=900, height=430)` — ventana OpenCV para el monitor.

`show(state, recommendation, strategy_row, dealer_upcard_rank)`:
- Zona superior: cartas de jugador y crupier con totales.
- Centro: acción recomendada en grande y coloreada.
- Zona inferior: fila de estrategia — 10 celdas coloreadas, celda activa resaltada.
- Pie: bankroll y apuesta.

Colores: HIT=verde, STAND=rojo, DOUBLE=naranja, SPLIT=morado, SURRENDER=amarillo.

---

### `src/analysis/`

**`logger.py`** — `HandLogger`. Escribe CSV con: timestamp, player_cards, dealer_upcard, dealer_final, actions_recommended, actions_taken, bet, outcome, delta, bankroll.

**`stats.py`** — Funciones sobre el DataFrame del CSV:
- `load_log(path)` → DataFrame con columna `session` (gaps > 10 min = nueva sesión)
- `summary(df)` → dict con win_rate, EV, delta_total, sesiones
- `plot_bankroll`, `plot_outcomes`, `plot_delta_by_upcard`, `plot_action_distribution`, `plot_adherence_by_session`, `plot_player_total_distribution`

---

### `src/core/motion.py`

`MotionDetector(threshold=30, stability_seconds=2.0)`.
- `update(frame)` → `(motion_detected, just_stabilized)`
- `just_stabilized` es True **solo en un frame**: el primero estable tras movimiento. Es la señal que activa la detección YOLO.

---

## 7. Scripts — detalle

### `simulate.py` — Simulador manual ✅ FUNCIONA SIN CÁMARA

```bash
python scripts/simulate.py
python scripts/simulate.py --bankroll 500
```

Introduce cartas manualmente por teclado. Flujo: apuesta → carta visible crupier → tus cartas → recomendación → acción → resultado → CSV.

Split real implementado: dos manos independientes. As-split: una carta por mano, sin poder pedir más.

---

### `capture_dataset.py` — Captura de fotos ⚠️ NECESITA CÁMARA

```bash
python scripts/capture_dataset.py
```

Feed en vivo con cruz guía y contador por clase. Frame guardado en disco es limpio (sin overlay). Ver teclas en [Paso 1](#paso-1--capturar-fotos-capture_datasetpy).

---

### `auto_annotate.py` — Auto-anotación ✅

```bash
python scripts/auto_annotate.py
python scripts/auto_annotate.py --preview
python scripts/auto_annotate.py --min-photos 100
```

Lee `data/raw_images/`, detecta contornos, escribe `.txt` YOLO, divide train/val (80/20 estratificado) y genera `dataset.yaml`.

---

### `calibrate_chips.py` — Calibración de fichas ✅

```bash
python scripts/calibrate_chips.py
```

Ratón sobre ficha → ver HSV. Clic → muestrear. `1`/`2`/`3` → asignar chip. `p` → imprimir bloque config. `q` → salir.

---

### `test_camera.py` — Verificación de cámara ⚠️ NECESITA CÁMARA

```bash
python scripts/test_camera.py
```

Muestra feed con las 3 zonas superpuestas. Verificar que las cartas caen en la zona correcta. `s` guarda foto, `q` sale.

---

### `gen_sample_data.py` ✅

```bash
python scripts/gen_sample_data.py --hands 300 --bankroll 100
```

Simula partidas siguiendo basic strategy. Puebla el CSV sin jugar manualmente.

---

### `generate_synthetic_data.py` ✅

```bash
python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic
```

Genera imágenes sintéticas de cartas con anotaciones YOLO. Útil para pre-entrenar antes de tener fotos reales.

---

## 8. Tests

**55 tests, todos pasan.**

```bash
python -m pytest tests/ -v
```

| Archivo | Tests | Qué verifica |
|---|---|---|
| `test_card.py` | 6 | Valores numéricos, is_ace, is_back, ValueError en BACK.value |
| `test_hand.py` | 10 | Totales soft/hard, bust, blackjack natural, pair, carta tapada ignorada |
| `test_deck.py` | 3 | 52 cartas, draw() reduce count, seed reproducible |
| `test_strategy.py` | 18 | Pares (A,A; 8,8; 10,10; 5,5), soft 18, hard (11, 16, 15, 17+, 12), surrender, fallback |
| `test_state.py` | 18 | resolve(): bust, dealer bust, push, blackjack, surrender, double. resolve_hand(): split |

---

## 9. Notebooks

### `analysis.ipynb` ✅
Análisis del historial de partidas. Requiere `data/games_log.csv`. 7 secciones: resumen numérico, bankroll, resultados, EV por upcard, acciones, adherencia.

### `colab_train.ipynb` ✅
Entrenamiento YOLOv8 en Google Colab. Soporta `labeled.zip` (fotos reales) y `synthetic.zip`.

### `colab_trainPrimerEntrenamientoEjecutado.ipynb` ✅
Notebook ejecutado del primer entrenamiento real (2026-05-15). Contiene el log completo de las 100 épocas y los resultados de evaluación.

### `dia_2_dataset_y_entrenamiento.ipynb` ✅
Documentación de la sesión 2. Incluye: configuración de la cámara (problemas y soluciones), diseño del dataset, curva de entrenamiento completa con análisis por fases, resultados por clase con gráficos. Documento de referencia para el proyecto académico.

---

## 10. Cómo ejecutar cada parte

### Sin cámara ni modelo (funciona hoy)

```bash
cd ~/projects/blackjack-cv
source venv/bin/activate

python -m pytest tests/ -v
python scripts/simulate.py
python scripts/gen_sample_data.py --hands 300
jupyter notebook notebooks/analysis.ipynb
```

### Pipeline de entrenamiento (ya completado)

```bash
python scripts/capture_dataset.py    # ≥100 fotos por clase
python scripts/auto_annotate.py --preview
cd data && zip -r labeled.zip labeled/
# → subir a Drive → colab_train.ipynb → descargar best.pt → models/
```

### Con modelo entrenado (estado actual)

```bash
python scripts/test_camera.py       # verificar zonas
python scripts/calibrate_chips.py   # calibrar fichas
python main.py                      # sistema completo
```

---

## 11. Flujo de datos en tiempo real

```
[Pi Camera IMX708]
    │  frame 1280×720 @ ~30 fps  (picamera2, BGR sin conversión)
    ▼
[motion.py]  ──── detecta movimiento (Gaussian blur + absdiff)
    │  just_stabilized=True  (exactamente un frame, tras 2 s quieto)
    ▼
[detector.py] ──── YOLOv8n ──── lista de Card por zona Y del frame
    │  mAP50=0.960             DEALER: y 0–40%   PLAYER: y 40–75%
    ▼
[chip_detector.py] ──── HSV en zona BETTING (y 75–100%)
    │                   → state.bet actualizado si calibrado
    ▼
[state.py]  ──── construye Hand jugador + crupier, infiere Phase
    │
    ▼
[strategy.py] ──── recommend() + full_row() ──── Action + fila de tabla
    │
    ├──► [display.py] ──── ventana OpenCV
    │                      recomendación en grande + tabla visual 10 celdas
    │
    └──► [logger.py] ──── data/games_log.csv  (al finalizar la mano)
                               │
                               ▼
                    [stats.py + analysis.ipynb]
                    análisis post-sesión
```

---

## 12. `config.py` — referencia de constantes

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `CARD_CLASSES` | `['A','2'...'K','BACK']` | 14 clases de detección |
| `FRAME_WIDTH / HEIGHT` | 1280 / 720 | Resolución de la cámara |
| `ZONE_DEALER` | y: 0.00–0.40 | Zona de cartas del crupier |
| `ZONE_PLAYER` | y: 0.40–0.75 | Zona de cartas del jugador |
| `ZONE_BETTING` | y: 0.75–1.00 | Zona de fichas/apuesta |
| `CHIP_VALUES` | chip_1=1, chip_2=5, chip_3=25 | Valor monetario por tipo de ficha |
| `CHIP_HSV_RANGES` | todos `None` | Sin calibrar — calibrar con `calibrate_chips.py` |
| `CHIP_MIN_AREA` | 500 | Área mínima de contorno para reconocer una ficha (px²) |
| `MOTION_THRESHOLD` | 30 | Sensibilidad del detector de movimiento |
| `STABILITY_SECONDS` | 2.0 | Segundos quieto para activar detección YOLO |
| `DEALER_STAND_ON_SOFT_17` | True | El crupier planta en soft 17 |
| `BLACKJACK_PAYOUT` | 1.5 | Pago del blackjack natural (3:2) |
| `STARTING_BANKROLL` | 100.0 | Bankroll inicial por defecto |
| `MODEL_PATH` | `models/yolov8n_blackjack.pt` | Ruta del modelo entrenado |
| `LOG_FILE` | `data/games_log.csv` | Historial de partidas |

---

## 13. Bug activo — cv2.imshow no muestra imagen con picamera2 + torch en el mismo proceso

### Resumen del problema

`scripts/test_detector.py` abre la cámara vía picamera2, carga YOLOv8 (torch), y llama a `cv2.imshow()` para mostrar el feed en vivo. La **detección YOLO funciona perfectamente** (detecta cartas con >90% de confianza), pero **la ventana de OpenCV no muestra imagen** — aparece en blanco o no aparece en absoluto.

---

### Entorno exacto

| Elemento | Valor |
|---|---|
| Hardware | Raspberry Pi 5 (BCM2712, aarch64) |
| OS | Raspberry Pi OS Bookworm 64-bit |
| Compositor de escritorio | **Wayland** (labwc). `XDG_SESSION_TYPE=wayland`, `WAYLAND_DISPLAY=wayland-0`, `DISPLAY=:0` (XWayland activo) |
| Python | 3.13.5 (sistema: `/usr/bin/python3.13`) |
| Entorno virtual | `venv/` creado con `--system-site-packages` |
| OpenCV en venv | **4.10.0** — sistema (`/usr/lib/python3/dist-packages/cv2.cpython-313-aarch64-linux-gnu.so`), compilado con Qt5 |
| OpenCV GUI backend | Qt5 15.15.15 (según `cv2.getBuildInformation()`) |
| torch | 2.11.0+cu130 |
| ultralytics | 8.4.48 |
| picamera2 | 0.3.36 |
| numpy | 2.2.4 |

---

### Qué funciona y qué no

| Test | Resultado |
|---|---|
| `cv2.imshow` con numpy puro (sin cámara, sin torch) | ✅ Muestra ventana correctamente |
| `cv2.imshow` con numpy + `import torch` (sin cámara) | ✅ Muestra ventana correctamente |
| `cv2.imshow` con picamera2 abierta + torch cargado | ❌ Ventana blanca o no aparece |
| YOLO detectando cartas (mismo proceso) | ✅ Detecta correctamente (K 90%, 10 95%, etc.) |
| `capture_dataset.py` (picamera2 + cv2, sin torch) | ✅ Funciona — el usuario capturó 1450 fotos |

**Conclusión:** el problema ocurre únicamente cuando **picamera2 + torch coexisten en el mismo proceso** con `cv2.imshow`. Por separado, cada uno funciona.

---

### Síntomas exactos observados

**Intento 1 — `QT_QPA_PLATFORM=xcb`, pip `opencv-python` 4.13 en venv:**
```
Listo. Pulsa ESPACIO para detectar. Q para salir.
QFontDatabase: Cannot find font directory .../venv/lib/python3.13/site-packages/cv2/qt/fonts.
[× 5 veces]
```
Resultado: no aparece ninguna ventana.

**Intento 2 — misma config tras copiar fuentes DejaVu al directorio de Qt:**
```
(python:12066): GLib-GObject-CRITICAL **: g_object_unref: assertion 'G_IS_OBJECT (object)' failed
```
Resultado: aparece una ventana **blanca** (el frame no se renderiza). YOLO detecta cartas correctamente en terminal.

**Intento 3 — `QT_QPA_PLATFORM=wayland` explícito:**
Resultado: sin ventana visible.

**Intento 4 — sin `QT_QPA_PLATFORM` (auto-detección):**
Resultado: ventana blanca, misma situación que intento 2.

**Intento 5 — pip `opencv-python` desinstalado, usando OpenCV 4.10 del sistema (mismo que `capture_dataset.py`):**
- `cv2.imshow` con tensor + torch solo → ✅ ventana verde "FUNCIONA?" visible por el usuario.
- `cv2.imshow` con picamera2 + torch → ❌ ventana blanca, error GLib-GObject.

**Intento 6 — cámara primero, YOLO después (orden invertido):**
Resultado: ventana blanca idéntica.

---

### Error GLib-GObject observado

```
(python:12066): GLib-GObject-CRITICAL **: 02:23:24.380:
g_object_unref: assertion 'G_IS_OBJECT (object)' failed
```

Este error aparece cuando coexisten picamera2 (que usa GLib internamente para el event loop de libcamera) y el OpenCV Qt5 (que usa su propio event loop). Indica un conflicto de gestión de objetos GLib entre las dos bibliotecas.

---

### Archivos relevantes

**`src/perception/camera.py`** — wrapper picamera2:
```python
from picamera2 import Picamera2
import cv2

class Camera:
    def __init__(self, source=0, width=1280, height=720):
        self._cam = Picamera2(camera_num=source)
        cfg = self._cam.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self._cam.configure(cfg)
        self._cam.start()
        self._cam.set_controls({
            "AfMode": 2, "AfRange": 0, "AfSpeed": 1,
            "AeEnable": True, "AwbEnable": True,
        })

    def read(self):
        # picamera2 en Pi 5 devuelve BGR pese al nombre "RGB888" — sin conversión
        return self._cam.capture_array("main")

    def release(self): self._cam.stop(); self._cam.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.release()
```

**`scripts/test_detector.py`** — script con el bug (extracto principal):
```python
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
import cv2
import numpy as np
from ultralytics import YOLO
from src.perception.camera import Camera

def main():
    with Camera(width=1280, height=720) as cam:
        for _ in range(20): cam.read()           # warmup
        model = YOLO("models/yolov8n_blackjack.pt")

        while True:
            frame = cam.read()                   # numpy uint8 BGR 1280×720
            display = frame.copy()
            cv2.imshow("Test Detector", display) # ← ventana blanca / no aparece
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                results = model(frame, verbose=False)[0]  # ← ESTO SÍ FUNCIONA
                # imprime detecciones en terminal correctamente
```

---

### Hipótesis de la causa raíz

picamera2 utiliza **GLib/GObject** internamente (libcamera usa el event loop de GLib). Al importar y arrancar picamera2, se inicializa un contexto GLib. OpenCV con Qt5 también intenta gestionar eventos mediante su propio bucle. Cuando torch/ultralytics también está presente, alguno de los tres inicializa algo que **invalida el surface/contexto de renderizado de Qt**, resultando en una ventana sin contenido (blanca).

El error `g_object_unref: assertion 'G_IS_OBJECT (object)' failed` sugiere que un objeto GLib está siendo liberado por dos sistemas distintos (double-free o uso tras liberación).

---

### Lo que se necesita

Una solución que permita en el **mismo proceso Python 3.13**:
1. Leer frames de picamera2 (IMX708, Pi 5)
2. Procesar con YOLOv8 / torch
3. Mostrar el frame en una ventana visible en el escritorio Wayland de la Pi

Soluciones aceptables:
- Configuración de entorno / env vars que resuelva el conflicto Qt-GLib
- Uso de otra API de display (picamera2 QtGlPreview, SDL2, tkinter, etc.) en lugar de `cv2.imshow`
- Arquitectura alternativa (subproceso para cámara, IPC para frames, etc.)
- Cualquier otra solución probada en Pi 5 + Bookworm + Wayland

---

## 14. Próximos pasos

### Inmediato
- [ ] Calibrar fichas con `scripts/calibrate_chips.py`
- [ ] Probar `main.py` en sesión real sobre la mesa
- [ ] Ajustar `ZONE_DEALER` / `ZONE_PLAYER` en `config.py` si alguna carta se asigna a la zona incorrecta

### Mejoras del modelo
- [ ] Fine-tuning: añadir más fotos del **2** (recall 0.761) y del **5** (recall 0.745)
- [ ] Capturar fotos directamente sobre el tapiz real con distintas condiciones de luz
- [ ] Re-entrenar desde `yolov8n_blackjack.pt` (fine-tuning, no desde cero)

---

## Setup inicial

```bash
git clone <repo-url>
cd blackjack-cv
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v   # debe mostrar: 55 passed
```
