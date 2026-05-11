from pathlib import Path
import numpy as np
from ..game.card import Card


class CardDetector:
    """Detecta cartas en un frame usando YOLOv8.

    Si el modelo no existe aún, detect() devuelve listas vacías sin error,
    lo que permite arrancar el sistema antes de tener el modelo entrenado.
    """

    def __init__(self, model_path: str, card_classes: list[str],
                 zone_dealer: dict, zone_player: dict):
        self.card_classes = card_classes
        self.zone_dealer = zone_dealer
        self.zone_player = zone_player
        self.model = None

        if Path(model_path).exists():
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def detect(self, frame: np.ndarray) -> tuple[list[Card], list[Card]]:
        """Devuelve (player_cards, dealer_cards) detectadas en el frame."""
        if not self.ready:
            return [], []

        h = frame.shape[0]
        d_min = int(h * self.zone_dealer['y_min'])
        d_max = int(h * self.zone_dealer['y_max'])
        p_min = int(h * self.zone_player['y_min'])
        p_max = int(h * self.zone_player['y_max'])

        results = self.model(frame, verbose=False)[0]
        player_cards: list[Card] = []
        dealer_cards: list[Card] = []

        for box in results.boxes:
            cls_idx = int(box.cls[0])
            rank = self.card_classes[cls_idx]
            y_center = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            card = Card(rank)
            if d_min <= y_center < d_max:
                dealer_cards.append(card)
            elif p_min <= y_center < p_max:
                player_cards.append(card)

        return player_cards, dealer_cards
