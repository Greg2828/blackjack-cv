"""
ARCHIVO: config.py
PROPÓSITO: Es el "panel de control" central de todo el proyecto.
           Aquí viven todos los valores que puedes necesitar cambiar
           según tu setup físico (cámara, fichas, mesa).
           El resto de archivos importan desde aquí en vez de tener
           números "hardcodeados" (escritos a mano dentro del código).

CÓMO SE CONECTA: Casi todos los demás archivos hacen:
    import config
    y luego usan  config.FRAME_WIDTH,  config.LOG_FILE, etc.
"""

# =============================================================================
# SECCIÓN 1: CARTAS
# =============================================================================

# Lista con los 14 rangos posibles que el modelo YOLO va a reconocer.
# Son 13 valores reales de carta (A=As, 2-10, J=Jota, Q=Reina, K=Rey)
# más "BACK" que significa el DORSO de una carta tapada (la carta secreta del crupier).
# El orden importa: el índice de esta lista (0, 1, 2...) es el número de "clase"
# que usa YOLO internamente. Es decir, YOLO ve "clase 0" y nosotros sabemos que es 'A'.
CARD_CLASSES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                'J', 'Q', 'K', 'BACK']

# =============================================================================
# SECCIÓN 2: FICHAS
# =============================================================================

# Cuánto vale en euros (o la moneda que uses) cada tipo de ficha.
# chip_1, chip_2, chip_3 son nombres internos; tú decides qué color físico
# corresponde a cada uno después de calibrar con scripts/calibrate_chips.py.
CHIP_VALUES = {
    'chip_1': 1,    # ficha pequeña, p.ej. color blanco = 1€
    'chip_2': 5,    # ficha mediana, p.ej. color rojo = 5€
    'chip_3': 25,   # ficha grande, p.ej. color verde = 25€
}

# Rangos de color en espacio HSV para detectar cada tipo de ficha.
# HSV = Hue (tono de color 0-179), Saturation (saturación 0-255), Value (brillo 0-255).
# 'lower' es el límite inferior del rango y 'upper' es el superior.
# Actualmente son None porque todavía no has calibrado las fichas reales.
# Cuando uses scripts/calibrate_chips.py, ese script te dará los valores exactos
# y te los mostrará listos para pegar aquí.
CHIP_HSV_RANGES = {
    'chip_1': {'lower': None, 'upper': None},  # sin calibrar aún
    'chip_2': {'lower': None, 'upper': None},  # sin calibrar aún
    'chip_3': {'lower': None, 'upper': None},  # sin calibrar aún
}

# Área mínima en píxeles² que debe tener un objeto redondo para considerarse ficha.
# Si la cámara está muy lejos, las fichas se ven pequeñas → bajar este número.
# Si hay mucho ruido (botones, monedas), subirlo filtra los objetos pequeños.
CHIP_MIN_AREA = 500

# =============================================================================
# SECCIÓN 3: ZONAS DEL FRAME (la imagen de la cámara)
# =============================================================================

# Resolución de la imagen que pedimos a la cámara (ancho × alto en píxeles).
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# El frame se divide en 3 franjas horizontales según su posición vertical (Y).
# Los valores son FRACCIONES de la altura total (0.0 = arriba, 1.0 = abajo).
# Ejemplo: si el frame mide 720px de alto,
#   ZONE_DEALER va de y=0 a y=0.40×720=288px  → parte superior = crupier
#   ZONE_PLAYER va de y=288 a y=0.75×720=540px → parte media   = jugador
#   ZONE_BETTING va de y=540 a y=720px          → parte inferior = apuestas
# Esto permite que detector.py sepa si una carta pertenece al crupier o al jugador.
ZONE_DEALER  = {'y_min': 0.00, 'y_max': 0.40}   # franja superior: cartas del crupier
ZONE_PLAYER  = {'y_min': 0.40, 'y_max': 0.75}   # franja media: cartas del jugador
ZONE_BETTING = {'y_min': 0.75, 'y_max': 1.00}   # franja inferior: zona de apuestas

# =============================================================================
# SECCIÓN 4: DETECTOR DE MOVIMIENTO
# =============================================================================

# Diferencia mínima de brillo entre dos frames consecutivos para considerar
# que "algo se movió". Rango 0-255. Más alto = menos sensible al movimiento.
MOTION_THRESHOLD = 30

# Segundos que la escena debe estar completamente quieta antes de activar
# la detección de cartas. Evita procesar imágenes borrosas mientras el jugador
# aún está colocando las cartas sobre la mesa.
STABILITY_SECONDS = 2.0

# =============================================================================
# SECCIÓN 5: REGLAS DEL JUEGO
# =============================================================================

# True = el crupier se planta en Soft 17 (As + 6).
# Soft 17 significa que el crupier tiene As + cartas que suman 6
# (el As cuenta como 11 dando 17 total, pero podría contar como 1).
# Esta es la regla estándar en la mayoría de casinos europeos.
DEALER_STAND_ON_SOFT_17 = True

# Multiplicador de pago para blackjack natural (As + carta de 10 puntos con las 2 primeras).
# 1.5 significa que si apostaste 10€ y sacas blackjack, ganas 15€ (total recibes 25€).
BLACKJACK_PAYOUT = 1.5

# =============================================================================
# SECCIÓN 6: BANCA
# =============================================================================

# Dinero inicial con el que empieza el jugador al arrancar el sistema.
STARTING_BANKROLL = 100.0

# =============================================================================
# SECCIÓN 7: RUTAS DE ARCHIVOS
# =============================================================================

# Archivo CSV donde se guardan todas las manos jugadas para análisis posterior.
# CSV = archivo de texto donde cada línea es una fila y los valores van separados por comas.
LOG_FILE = 'data/games_log.csv'

# Ruta al modelo entrenado de YOLO para detectar cartas.
# Este archivo (best.pt) lo obtendrás después de entrenar en Google Colab
# y copiarlo aquí. Hasta entonces el sistema funciona sin él (modo simulación).
MODEL_PATH = 'models/yolov8n_blackjack.pt'
