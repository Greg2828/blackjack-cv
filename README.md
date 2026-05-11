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
13. [Próximos pasos](#13-próximos-pasos)

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
| Hardware | Raspberry Pi 5, Pi Camera |
| OS | Raspberry Pi OS Bookworm (64-bit) |
| Lenguaje | Python 3.13 |
| Detección de cartas | YOLOv8 nano (Ultralytics) — inferencia solo, entrenamiento en Colab |
| Visión | OpenCV 4.x |
| Análisis | Pandas, Matplotlib, Seaborn |
| Tests | pytest |
| Entorno virtual | `python3 -m venv venv --system-site-packages` (picamera2 preinstalado en Bookworm) |

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
| Bucle principal (`main.py`) | — | Estructura completa; sin detección hasta tener modelo |
| Generador datos sintéticos | — | Cartas sobre fondo verde con anotaciones YOLO |
| **Captura de fotos reales** (`capture_dataset.py`) | — | Guarda frames limpios (sin overlay) |
| **Auto-anotación** (`auto_annotate.py`) | — | Genera etiquetas YOLO por contorno, sin herramientas externas |
| **Detector de fichas** (`chip_detector.py`) | — | HSV con filtro circular; degradación elegante sin calibrar |
| **Calibración de fichas** (`calibrate_chips.py`) | — | Herramienta de muestreo HSV con ratón |
| Notebook de entrenamiento Colab | — | Soporta fotos reales y datos sintéticos |

**Total: 55 tests, todos pasan.**

### ⚠️ Pendiente (bloqueado por hardware/datos)

| Componente | Qué falta |
|---|---|
| Modelo YOLOv8 entrenado | Hacer fotos → `auto_annotate.py` → entrenar en Colab |
| Fichas calibradas | Ejecutar `calibrate_chips.py` con la cámara apuntando a la mesa |
| Prueba con cámara real | Conectar Pi Camera y ejecutar `test_camera.py` |

---

## 4. Estructura completa de archivos

```
blackjack-cv/
├── main.py                          # Bucle principal en tiempo real
├── config.py                        # Todas las constantes del sistema
├── requirements.txt
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
│   │   ├── camera.py                # Wrapper Pi Camera / VideoCapture
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
│   ├── analysis.ipynb               # Análisis estadístico del historial de partidas
│   └── colab_train.ipynb            # Entrenamiento YOLOv8 en Google Colab (GPU gratuita)
│
├── tests/
│   ├── test_card.py                 # 6 tests
│   ├── test_hand.py                 # 10 tests
│   ├── test_deck.py                 # 3 tests
│   ├── test_strategy.py             # 18 tests
│   └── test_state.py                # 18 tests — resolve() y resolve_hand() (split)
│
├── models/                          # Vacío — aquí va yolov8n_blackjack.pt tras entrenar
└── data/
    ├── raw_images/                  # Fotos brutas capturadas (excluido de git)
    │   ├── A/                       # Una carpeta por clase
    │   ├── 2/ … K/
    │   └── BACK/
    ├── labeled/                     # Dataset anotado listo para Colab (excluido de git)
    │   ├── images/train/ y val/
    │   ├── labels/train/ y val/
    │   └── dataset.yaml
    ├── synthetic/                   # Dataset sintético generado (excluido de git)
    └── games_log.csv                # Historial de partidas (excluido de git)
```

---

## 5. Pipeline completo: de fotos a modelo funcionando

Esta sección explica paso a paso cómo pasar de no tener modelo a tener el sistema funcionando en tiempo real. Si nunca has entrenado un modelo de visión artificial, léela entera — cada concepto está explicado.

### ¿Qué es YOLO y por qué lo necesitamos?

YOLO (You Only Look Once) es un algoritmo de detección de objetos. Dado un frame de vídeo, devuelve una lista de objetos detectados, cada uno con: su clase (ej. "A", "K", "BACK"), su posición en el frame (bounding box), y una puntuación de confianza.

Para que YOLO funcione con nuestras cartas, hay que entrenarlo: mostrarle miles de fotos de cartas con su posición marcada exactamente, para que aprenda qué aspecto tiene cada rango. Sin ese entrenamiento, el modelo no sabe nada sobre cartas de blackjack.

El Raspberry Pi 5 no tiene GPU y no puede entrenar el modelo (tardaría días). Por eso entrenamos en **Google Colab**, que ofrece GPUs gratuitas. Una vez entrenado, el modelo resultante (un archivo `.pt` de ~6 MB) se copia a la Pi, donde solo hace *inferencia* (predecir) — eso sí es rápido incluso sin GPU.

---

### Paso 1 — Capturar fotos (`capture_dataset.py`)

**¿Por qué fotos reales y no sintéticas?**

Las imágenes sintéticas generadas por `generate_synthetic_data.py` sirven para pre-entrenar rápido, pero son dibujos vectoriales sobre fondo verde. Un modelo entrenado solo con datos sintéticos puede fallar con tus cartas físicas reales bajo la iluminación de tu mesa. Las fotos reales le muestran al modelo exactamente lo que verá en producción.

**¿Cuántas fotos?**

- **Mínimo funcional**: 80 fotos por clase × 14 clases = 1.120 fotos
- **Recomendado para alta precisión**: 150 fotos por clase × 14 clases = 2.100 fotos
- Las 14 clases son: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, BACK

**Qué hacer durante la captura:**

```bash
python scripts/capture_dataset.py
```

- La pantalla muestra el feed de la cámara con una **cruz y un rectángulo guía** verde en el centro.
- Teclas `A`, `2`–`9`, `0` (=10), `J`, `Q`, `K`, `B` (=BACK) seleccionan la clase activa.
- `ESPACIO` guarda la foto. `D` borra la última.
- El panel lateral derecho muestra cuántas fotos tienes de cada clase.
- El script guarda frames **limpios** (sin texto ni rectángulos superpuestos) en `data/raw_images/RANK/`.

**Consejos para fotos de calidad:**

| Qué variar | Por qué |
|---|---|
| Iluminación (luz natural, artificial, sombras) | El modelo verá la mesa en condiciones reales variadas |
| Ángulo leve de la carta (0°–20°) | Las cartas raramente están perfectamente horizontales |
| Posición en el frame (no siempre centrada) | En juego real las cartas están en distintas zonas |
| Distancia a la cámara (ligeramente más cerca/lejos) | Cubre variación de escala |
| Fondos: el tapete real, la mano del crupier cerca | El modelo aprende a ignorar el contexto |
| **No variar**: el rango de la carta — una foto = una clase | La etiqueta tiene que ser correcta |

---

### Paso 2 — Auto-anotar (`auto_annotate.py`)

**¿Qué es una anotación YOLO?**

YOLO no solo necesita saber *qué* carta sale en cada foto — también necesita saber *dónde* está exactamente, con un rectángulo de localización (bounding box). Para cada imagen, el formato YOLO es un archivo `.txt` con una línea por objeto:

```
clase  x_centro  y_centro  ancho  alto
```

Todos los valores son decimales entre 0 y 1 (normalizados al tamaño del frame). Por ejemplo, una carta en el centro de la imagen con un cuarto del tamaño del frame sería:

```
0  0.500000  0.500000  0.250000  0.357000
```

Donde `0` es el índice de la clase "A" en `CARD_CLASSES`. Sin estos archivos `.txt`, YOLO no puede aprender.

**Lo que hace `auto_annotate.py`:**

Como cada foto tiene una sola carta sobre un fondo relativamente uniforme, el script puede detectar el borde de la carta automáticamente mediante análisis de contornos (sin necesitar herramientas externas como LabelImg o Roboflow). Prueba tres estrategias en cascada:

1. **Canny** — detecta bordes por gradiente de intensidad. Funciona bien con cartas blancas sobre tapete oscuro.
2. **Otsu normal** — umbralización automática del histograma. Carta clara sobre fondo oscuro.
3. **Otsu inverso** — lo contrario. Carta oscura sobre fondo claro.

Para cada contorno encontrado, verifica que el ratio ancho/alto sea compatible con una carta (63.5 mm / 88.9 mm ≈ 0.71). Si ninguna estrategia produce un contorno razonable, la foto se marca como fallida.

```bash
# Anotar todas las fotos de raw_images/ y generar data/labeled/
python scripts/auto_annotate.py

# Modo visual: muestra cada foto con su bounding box para que puedas verificar
# ESPACIO = ok    D = borrar esta foto    Q = salir del preview
python scripts/auto_annotate.py --preview

# Avisar si una clase tiene menos de 100 fotos válidas
python scripts/auto_annotate.py --min-photos 100
```

**Salida:**

```
data/labeled/
├── images/
│   ├── train/   ← 80% de las fotos de cada clase (mezcladas aleatoriamente)
│   └── val/     ← 20% de las fotos de cada clase
├── labels/
│   ├── train/   ← un .txt por imagen con la bounding box YOLO
│   └── val/
└── dataset.yaml ← configuración completa lista para el notebook de Colab
```

La división train/val es **estratificada por clase**: el 20% de validación se toma de forma proporcional de cada clase, no aleatoriamente del total. Esto garantiza que haya ejemplos de cada carta tanto en train como en val.

**¿Qué hacer si hay muchas fotos fallidas?**

Ejecuta con `--preview`, revisa las fotos marcadas como fallidas y borra las que tengan mala iluminación, la carta fuera de cuadro o el fondo demasiado similar a la carta. Luego vuelve a ejecutar sin `--preview`.

---

### Paso 3 — Comprimir y subir a Google Drive

```bash
cd ~/projects/blackjack-cv/data
zip -r labeled.zip labeled/
```

Sube `labeled.zip` a cualquier carpeta de tu Google Drive (no importa cuál — el notebook la buscará automáticamente).

---

### Paso 4 — Entrenar en Google Colab (`colab_train.ipynb`)

**¿Por qué Colab?**

Google Colab ofrece GPUs gratuitas (normalmente NVIDIA T4). Con esa GPU, entrenar 100 épocas sobre ~2.000 fotos tarda unos **15–25 minutos**. En el Raspberry Pi 5 (sin GPU) tardaría entre 8 y 15 horas.

**Abrir el notebook:**

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Archivo → Abrir notebook → Google Drive → navega hasta `blackjack-cv/notebooks/colab_train.ipynb`
3. En el menú superior: Entorno de ejecución → Cambiar tipo → GPU T4 → Guardar
4. Ejecuta las celdas en orden (Shift+Enter o el botón ▶ de cada celda)

**Qué hacen las celdas:**

| Celda | Qué hace |
|---|---|
| 1. Verificar GPU | Comprueba que tienes GPU activa (si no, el entrenamiento tardará horas) |
| 2. Instalar dependencias | `pip install ultralytics` — instala YOLOv8 |
| 3. Montar Drive | Conecta tu Google Drive al entorno de Colab |
| 4. Localizar dataset | Busca `labeled.zip` (fotos reales) o `synthetic.zip` como fallback |
| 5. Extraer y preparar | Descomprime el zip y corrige las rutas del `dataset.yaml` para Colab |
| 6. Entrenar | Ejecuta `model.train()` — aquí ocurre el aprendizaje real |
| 7. Evaluar | Calcula mAP50, precision y recall sobre el conjunto de validación |
| 8. Ver ejemplos | Muestra 4 imágenes de validación con las detecciones superpuestas |
| 9. Guardar en Drive | Copia el mejor modelo (`best.pt`) a `Drive/blackjack-cv/models/` |

**¿Qué ocurre durante el entrenamiento?**

En cada *época*, el modelo ve todas las imágenes de entrenamiento, calcula cuánto se equivoca (loss) y ajusta sus pesos internos para equivocarse menos. Después de cada época, se evalúa en el conjunto de validación (fotos que nunca ha visto) para medir si realmente está aprendiendo o solo memorizando.

El entrenamiento para automáticamente (`patience=20`) si el mAP en validación no mejora en 20 épocas consecutivas.

**¿Qué es mAP50 y cuándo es suficientemente bueno?**

mAP50 (mean Average Precision at IoU ≥ 0.5) es la métrica principal de calidad. Un valor de 0.50 significa "detecta la mitad de las cartas de forma aceptable". Un valor de 1.00 sería perfección.

| mAP50 | Interpretación |
|---|---|
| < 0.70 | El modelo no está aprendiendo bien — revisar fotos y anotaciones |
| 0.70 – 0.85 | Funciona pero cometerá errores notables en el juego real |
| 0.85 – 0.93 | Bueno. Usado con cautela (umbral de confianza alto) funciona bien |
| > 0.93 | Excelente. Objetivo con fotos reales bien anotadas |

Si el mAP es bajo, las causas más comunes son: pocas fotos por clase, fotos con mala iluminación, anotaciones incorrectas, o demasiada similitud entre el fondo y las cartas.

---

### Paso 5 — Copiar el modelo a la Pi y arrancar

```bash
# Opción A — desde el navegador:
# Descarga best.pt desde Drive/blackjack-cv/models/
# Cópialo a la Pi con USB o scp

# Opción B — directamente en la Pi:
pip install gdown
gdown 'URL_DEL_ARCHIVO' -O ~/projects/blackjack-cv/models/yolov8n_blackjack.pt

# Verificar zonas con la cámara conectada
python scripts/test_camera.py

# Arrancar el sistema
python main.py
```

---

### Calibrar las fichas (apuesta automática)

Una vez el sistema detecta cartas correctamente, puedes calibrar la detección de fichas para que lea la apuesta automáticamente:

```bash
python scripts/calibrate_chips.py
```

- La ventana muestra el feed de la cámara con el valor HSV del píxel bajo el cursor.
- Haz clic en el centro de una ficha para muestrear su color (región 15×15 px).
- Presiona `1`, `2` o `3` para asignar la muestra a `chip_1`, `chip_2` o `chip_3`.
- Presiona `p` para imprimir el bloque de configuración.
- Copia el bloque resultante en `config.py` bajo `CHIP_HSV_RANGES`.

Una vez calibrado, `main.py` detectará el total de fichas en la zona de betting al estabilizarse la escena y actualizará la apuesta automáticamente.

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
- `GameState.resolve_hand(hand, doubled, is_split)` → mismo cálculo pero para una mano de split. Con `is_split=True`, el 21 de 2 cartas no paga como blackjack natural (regla estándar de casino).

---

### `src/decision/strategy.py`

Motor de *basic strategy* estándar (6 barajas, crupier planta en soft 17, late surrender).

Tablas internas:
- `_PAIR` — 10 filas (2,2 hasta A,A) × 10 columnas (upcard 2..A)
- `_SOFT` — 8 filas (soft 13..20) × 10 columnas
- `_HARD` — 10 filas (hard 8..17) × 10 columnas

Funciones públicas:
- `recommend(player_hand, dealer_upcard, *, can_split, can_double, can_surrender) → Action`
- `full_row(player_hand, ...) → list[Action]` — 10 acciones (una por upcard), para la tabla visual

---

### `src/perception/`

**`camera.py`** — `Camera(source=0, width=1280, height=720)`. Context manager sobre `cv2.VideoCapture`.

**`detector.py`** — `CardDetector(model_path, card_classes, zone_dealer, zone_player)`.
- Si el `.pt` no existe: `detector.ready = False`, `detect()` devuelve listas vacías sin error.
- Si existe: detecta con YOLOv8 y separa cartas por zona Y (DEALER: 0–40%, PLAYER: 40–75%).

**`chip_detector.py`** — `ChipDetector(chip_values, chip_hsv_ranges, min_area)`.
- `calibrated` → True si al menos un chip tiene rangos HSV definidos (no None).
- `detect(frame, zone_y_min, zone_y_max)` → suma del valor de las fichas detectadas. Usa detección de contornos circulares en espacio HSV. Devuelve 0.0 sin error si no está calibrado.

---

### `src/ui/display.py`

`Display(width=900, height=430)` — ventana OpenCV para el monitor.

`show(state, recommendation, strategy_row, dealer_upcard_rank)`:
- Zona superior: cartas de jugador y crupier con totales.
- Centro: acción recomendada en grande y coloreada.
- Zona inferior: fila de estrategia — 10 celdas coloreadas, celda activa resaltada en blanco.
- Pie: bankroll y apuesta.

Colores: HIT=verde, STAND=rojo, DOUBLE=naranja, SPLIT=morado, SURRENDER=amarillo.

`show_outcome(outcome, delta)` — pantalla de resultado con delta económico.

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

Introduce cartas manualmente por teclado. Flujo de una mano:
1. Apuesta (enter = 10)
2. Carta visible del crupier (ej: `7`)
3. Tus cartas (ej: `A 8`)
4. El sistema muestra recomendación y tabla visual en ventana OpenCV
5. Acción: `h` hit, `s` stand, `d` double, `sp` split, `su` surrender (enter acepta la recomendada)
6. Si HIT o DOUBLE: introduce la nueva carta
7. Carta tapada del crupier y las que saque
8. Resultado mostrado en pantalla y guardado en CSV

**Split real implementado**: al elegir `sp`, el sistema separa las dos cartas y juega cada sub-mano de forma independiente. Si es un split de Ases, se reparte una carta por mano y el jugador no puede pedir más (regla estándar de casino). Cada sub-mano se resuelve y registra por separado.

---

### `gen_sample_data.py` — Generador de partidas simuladas ✅

```bash
python scripts/gen_sample_data.py
python scripts/gen_sample_data.py --hands 500 --bankroll 200
```

Simula partidas siguiendo *basic strategy* al 100%. Puebla el CSV sin jugar manualmente. Con 300 manos: win rate ~43.7%, EV ~-0.41% (house edge teórico con estrategia perfecta ≈ 0.5%).

---

### `generate_synthetic_data.py` — Dataset sintético ✅

```bash
python scripts/generate_synthetic_data.py --n 2000 --out data/synthetic
```

Genera imágenes de cartas vectoriales sobre fondo verde con rotación, escala variable e iluminación ruidosa. Útil para un primer entrenamiento rápido. Las fotos reales darán más precisión.

---

### `capture_dataset.py` — Captura de fotos reales ⚠️ NECESITA CÁMARA

```bash
python scripts/capture_dataset.py
```

Muestra el feed con overlay (cruz guía, contadores por clase, instrucciones). El frame que se guarda en disco es **limpio** — sin texto ni rectángulos superpuestos, porque el modelo no debe ver esos artefactos durante el entrenamiento.

Controles: `A` `2`–`9` `0`(=10) `J` `Q` `K` `B` = seleccionar clase | `ESPACIO` = capturar | `D` = borrar última | `Q` = salir.

---

### `auto_annotate.py` — Auto-anotación ✅

```bash
python scripts/auto_annotate.py
python scripts/auto_annotate.py --preview       # revisión visual
python scripts/auto_annotate.py --min-photos 100
```

Lee `data/raw_images/`, detecta la carta en cada imagen por análisis de contorno, escribe `.txt` YOLO, divide train/val (80/20 estratificado) y genera `data/labeled/dataset.yaml`. Avisa de clases con pocas fotos y fotos con detección fallida. Ver [Paso 2](#paso-2--auto-anotar-auto_annotatepy) para detalles.

---

### `calibrate_chips.py` — Calibración de fichas ✅

```bash
python scripts/calibrate_chips.py
python scripts/calibrate_chips.py --image foto_mesa.jpg
```

Mueve el ratón sobre una ficha para ver su HSV en tiempo real. Clic para muestrear. `1`/`2`/`3` asigna la muestra a chip_1/chip_2/chip_3. `p` imprime el bloque para copiar en `config.py`. `q` sale.

---

### `test_camera.py` — Verificación de cámara ⚠️ NECESITA CÁMARA

```bash
python scripts/test_camera.py
```

Muestra el feed con las tres zonas superpuestas: azul (DEALER), verde (PLAYER), naranja (BETTING). Verificar que las cartas del jugador caen en la zona verde y las del crupier en la azul. Ajustar `config.py` si no. `s` guarda foto, `q` sale.

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
| `test_strategy.py` | 18 | Pares (A,A; 8,8; 10,10; 5,5), soft 18 vs varios upcards, hard (11, 16, 15, 17+, 12), surrender, fallback sin can_surrender |
| `test_state.py` | 18 | `resolve()`: bust, dealer bust, player/dealer higher, push, blackjack, surrender, double. `resolve_hand()`: 21 tras split ≠ blackjack, doubled, dealer blackjack |

---

## 9. Notebooks

### `analysis.ipynb` ✅

Análisis del historial de partidas. Requiere `data/games_log.csv`.

```bash
python scripts/gen_sample_data.py --hands 300   # si el CSV está vacío
jupyter notebook notebooks/analysis.ipynb
```

7 secciones: carga de datos, resumen numérico (EV, win rate), evolución del bankroll, distribución de resultados, EV por upcard del crupier, distribución de acciones, adherencia a la estrategia por sesión.

---

### `colab_train.ipynb` ✅

Entrenamiento YOLOv8 en Google Colab. Soporta fotos reales (`labeled.zip`) y datos sintéticos (`synthetic.zip`). Ver [Paso 4](#paso-4--entrenar-en-google-colab-colab_trainipynb) para el flujo completo.

---

## 10. Cómo ejecutar cada parte

### Sin cámara ni modelo (funciona hoy)

```bash
cd ~/projects/blackjack-cv
source venv/bin/activate

python -m pytest tests/ -v                          # 55 tests
python scripts/simulate.py                          # simulador interactivo
python scripts/gen_sample_data.py --hands 300       # generar datos CSV
python -c "from src.analysis.stats import load_log, print_summary; \
           print_summary(load_log('data/games_log.csv'))"
jupyter notebook notebooks/analysis.ipynb
```

### Pipeline de entrenamiento con fotos reales

```bash
python scripts/capture_dataset.py    # ≥100 fotos por clase con cámara
python scripts/auto_annotate.py --preview   # verificar anotaciones
python scripts/auto_annotate.py             # generar data/labeled/
cd data && zip -r labeled.zip labeled/
# → subir labeled.zip a Drive → colab_train.ipynb en Colab
# → descargar best.pt → copiar a models/yolov8n_blackjack.pt
```

### Con cámara y modelo entrenados

```bash
python scripts/test_camera.py       # verificar zonas
python scripts/calibrate_chips.py   # calibrar fichas (opcional)
python main.py                      # sistema completo
```

---

## 11. Flujo de datos en tiempo real

```
[Pi Camera]
    │  frame 1280×720 @ ~30 fps
    ▼
[motion.py]  ──── detecta movimiento (Gaussian blur + absdiff)
    │  just_stabilized=True  (exactamente un frame, tras 2 s quieto)
    ▼
[detector.py] ──── YOLOv8 ──── lista de Card por zona Y del frame
    │                          DEALER: y 0–40%   PLAYER: y 40–75%
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
| `CHIP_HSV_RANGES` | todos `None` | Calibrar con `calibrate_chips.py` |
| `CHIP_MIN_AREA` | 500 | Área mínima de contorno para reconocer una ficha (px²) |
| `MOTION_THRESHOLD` | 30 | Sensibilidad del detector de movimiento |
| `STABILITY_SECONDS` | 2.0 | Segundos quieto para activar detección YOLO |
| `DEALER_STAND_ON_SOFT_17` | True | El crupier planta en soft 17 |
| `BLACKJACK_PAYOUT` | 1.5 | Pago del blackjack natural (3:2) |
| `STARTING_BANKROLL` | 100.0 | Bankroll inicial por defecto |
| `MODEL_PATH` | `models/yolov8n_blackjack.pt` | Ruta del modelo entrenado |
| `LOG_FILE` | `data/games_log.csv` | Historial de partidas |

---

## 13. Próximos pasos

### Inmediato — construir el modelo
- [ ] Conectar Pi Camera y verificar zonas con `scripts/test_camera.py`
- [ ] Capturar ≥100 fotos por clase (14 clases) con `scripts/capture_dataset.py`
- [ ] Anotar con `scripts/auto_annotate.py --preview` y revisar calidad
- [ ] Entrenar en Google Colab con `notebooks/colab_train.ipynb`
- [ ] Objetivo de calidad: **mAP50 > 0.93** en el conjunto de validación
- [ ] Si mAP es bajo: añadir más fotos variando iluminación y ángulo

### Tras el modelo
- [ ] Calibrar fichas con `scripts/calibrate_chips.py`
- [ ] Probar `main.py` en sesión real y ajustar `ZONE_DEALER` / `ZONE_PLAYER` en `config.py` si alguna carta se asigna a la zona incorrecta
- [ ] Si el modelo comete errores en producción: capturar los frames problemáticos, añadirlos al dataset y re-entrenar

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
