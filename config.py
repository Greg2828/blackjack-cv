"""Configuración central del sistema. Aquí viven las constantes
que se ajustan al setup físico (zonas, valores de fichas, umbrales)."""

# === Cartas ===
# 14 clases: 13 valores + dorso
CARD_CLASSES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                'J', 'Q', 'K', 'BACK']

# === Fichas ===
# Asignar el color real de tus fichas a cada chip_n el día 4
# tras calibrar los rangos HSV con fotos reales.
CHIP_VALUES = {
    'chip_1': 1,
    'chip_2': 5,
    'chip_3': 25,
}
CHIP_HSV_RANGES = {
    'chip_1': {'lower': None, 'upper': None},
    'chip_2': {'lower': None, 'upper': None},
    'chip_3': {'lower': None, 'upper': None},
}
# Área mínima de contorno para reconocer un blob como ficha (px²).
# Ajustar según distancia de la cámara a la mesa.
CHIP_MIN_AREA = 500

# === Zonas del frame (fracciones de altura) ===
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ZONE_DEALER  = {'y_min': 0.00, 'y_max': 0.40}
ZONE_PLAYER  = {'y_min': 0.40, 'y_max': 0.75}
ZONE_BETTING = {'y_min': 0.75, 'y_max': 1.00}

# === Motion detector ===
MOTION_THRESHOLD = 30
STABILITY_SECONDS = 2.0

# === Reglas del juego ===
DEALER_STAND_ON_SOFT_17 = True
BLACKJACK_PAYOUT = 1.5

# === Banca ===
STARTING_BANKROLL = 100.0

# === Paths ===
LOG_FILE = 'data/games_log.csv'
MODEL_PATH = 'models/yolov8n_blackjack.pt'
