"""
ARCHIVO: src/perception/detector.py
PROPÓSITO: Detecta cartas en un frame de cámara usando el modelo YOLOv8.
           YOLO (You Only Look Once) es una red neuronal que analiza
           una imagen y devuelve: qué objetos hay y dónde están.

           Después de detectar, clasifica cada carta como del jugador
           o del crupier según su posición vertical en el frame
           (zona superior = crupier, zona media = jugador).

CÓMO SE CONECTA:
  - main.py crea un CardDetector al arrancar y llama a detect() cada vez
    que la escena se estabiliza (el motion detector da luz verde)
  - config.py provee MODEL_PATH, CARD_CLASSES, ZONE_DEALER, ZONE_PLAYER

FLUJO:
  frame → YOLOv8 → lista de bounding boxes → zona Y → player_cards / dealer_cards
"""

from pathlib import Path   # Path: manejo de rutas de archivos de forma independiente del OS
import numpy as np         # numpy: arrays de números (los frames son arrays)
from ..game.card import Card  # Card: la clase que representa una carta


class CardDetector:
    """Detecta cartas en un frame usando YOLOv8.

    Si el modelo no existe aún, detect() devuelve listas vacías sin error,
    lo que permite arrancar el sistema antes de tener el modelo entrenado.
    """

    def __init__(self, model_path: str, card_classes: list[str],
                 zone_dealer: dict, zone_player: dict):
        """Constructor: carga el modelo YOLO si existe.

        Parámetros:
          model_path: ruta al archivo 'best.pt' del modelo entrenado
          card_classes: lista de rangos ['A','2',...,'K','BACK'] (de config.py)
          zone_dealer: diccionario con 'y_min' y 'y_max' (fracciones) de la zona del crupier
          zone_player: igual pero para la zona del jugador
        """
        self.card_classes = card_classes  # guardamos la lista de clases para traducir índices
        self.zone_dealer  = zone_dealer
        self.zone_player  = zone_player
        self.model        = None  # el modelo empieza como None hasta que lo cargamos

        # Path(model_path).exists() devuelve True si el archivo existe en disco.
        # Solo cargamos el modelo si el archivo está disponible.
        if Path(model_path).exists():
            # Importamos Ultralytics YOLO aquí (dentro del if) porque es una librería
            # grande que tarda en cargar. Si el modelo no existe, no tiene sentido cargarla.
            from ultralytics import YOLO
            self.model = YOLO(model_path)  # carga el modelo en memoria (puede tardar unos segundos)

    @property
    def ready(self) -> bool:
        """True si el modelo está cargado y listo para detectar.
        False si 'best.pt' no existe todavía (antes de entrenar en Colab)."""
        return self.model is not None

    def detect(self, frame: np.ndarray) -> tuple[list[Card], list[Card]]:
        """Analiza el frame y devuelve las cartas detectadas separadas por zona.

        Parámetro:
          frame: imagen BGR de la cámara (array numpy H×W×3)

        Devuelve: (player_cards, dealer_cards)
          Ambas son listas de objetos Card, posiblemente vacías.
          Si el modelo no está listo, devuelve ([], []) sin error.
        """

        # Si el modelo no está cargado, no podemos detectar nada.
        # Devolvemos listas vacías → el sistema puede correr en modo "sin cámara".
        if not self.ready:
            return [], []

        # Calculamos los límites en píxeles de cada zona.
        # frame.shape devuelve (altura, anchura, canales_de_color).
        # frame.shape[0] es la altura en píxeles.
        h = frame.shape[0]

        # Zona del crupier: de y=0 a y=40% de la altura
        d_min = int(h * self.zone_dealer['y_min'])  # ej: int(720 * 0.00) = 0
        d_max = int(h * self.zone_dealer['y_max'])  # ej: int(720 * 0.40) = 288

        # Zona del jugador: de y=40% a y=75% de la altura
        p_min = int(h * self.zone_player['y_min'])  # ej: int(720 * 0.40) = 288
        p_max = int(h * self.zone_player['y_max'])  # ej: int(720 * 0.75) = 540

        # Ejecutamos el modelo YOLO sobre el frame completo.
        # verbose=False suprime los mensajes de progreso que YOLO imprime por defecto.
        # [0] toma el primer (y único) resultado (procesamos 1 imagen a la vez).
        results = self.model(frame, verbose=False)[0]

        player_cards: list[Card] = []
        dealer_cards: list[Card] = []

        # Iteramos por cada "box" (rectángulo delimitador) que YOLO detectó.
        for box in results.boxes:
            # box.cls[0] es el índice de clase (número) → lo convertimos a entero
            cls_idx = int(box.cls[0])

            # Traducimos el índice numérico al rango de carta usando la lista de clases.
            # Ejemplo: cls_idx=0 → card_classes[0] = 'A' (As)
            rank = self.card_classes[cls_idx]

            # Calculamos el centro vertical (Y) del bounding box.
            # box.xyxy[0] contiene [x_izquierda, y_arriba, x_derecha, y_abajo].
            # El centro Y es el promedio de y_arriba e y_abajo.
            y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

            # Creamos la carta con el rango detectado.
            card = Card(rank)

            # Clasificamos la carta según en qué zona vertical se encuentra.
            if d_min <= y_center < d_max:
                # El centro de la carta está en la zona superior → es del crupier.
                dealer_cards.append(card)
            elif p_min <= y_center < p_max:
                # El centro está en la zona media → es del jugador.
                player_cards.append(card)
            # Si la carta está en la zona de apuestas (abajo), la ignoramos.

        return player_cards, dealer_cards
