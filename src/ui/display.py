import cv2
import numpy as np
from ..game.state import Action, GameState, Outcome

_COLORS: dict[Action, tuple[int, int, int]] = {
    Action.HIT:       (0,   210,   0),
    Action.STAND:     (0,    0,  210),
    Action.DOUBLE:    (0,   165, 255),
    Action.SPLIT:     (255,   0, 255),
    Action.SURRENDER: (0,   220, 220),
}

_LABELS: dict[Action, str] = {
    Action.HIT:       "HIT",
    Action.STAND:     "STAND",
    Action.DOUBLE:    "DOUBLE",
    Action.SPLIT:     "SPLIT",
    Action.SURRENDER: "SURRENDER",
}

_OUTCOME_COLORS: dict[Outcome, tuple[int, int, int]] = {
    Outcome.WIN:       (0,  210,   0),
    Outcome.BLACKJACK: (0,  210,   0),
    Outcome.LOSE:      (0,    0, 210),
    Outcome.PUSH:      (200, 200,   0),
}

_WHITE  = (255, 255, 255)
_GRAY   = (160, 160, 160)
_BG     = (25,  25,  25)
_FONT   = cv2.FONT_HERSHEY_DUPLEX
_PLAIN  = cv2.FONT_HERSHEY_SIMPLEX


class Display:
    WINDOW = "Blackjack CV"

    def __init__(self, width: int = 900, height: int = 320):
        self.w = width
        self.h = height
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, width, height)

    def show(self, state: GameState, recommendation: Action | None = None) -> None:
        canvas = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)

        player_total = state.player_hand.total() if state.player_hand.visible_cards else 0
        dealer_str   = str(state.dealer_hand.visible_cards[0]) if state.dealer_hand.visible_cards else "?"

        player_cards = ' '.join(str(c) for c in state.player_hand.cards) or "---"
        dealer_cards = ' '.join(str(c) for c in state.dealer_hand.cards) or "---"

        cv2.putText(canvas, f"Jugador: {player_cards}  ({player_total})",
                    (30, 55),  _PLAIN, 0.9, _WHITE, 2)
        cv2.putText(canvas, f"Crupier: {dealer_cards}",
                    (30, 100), _PLAIN, 0.9, _GRAY, 2)

        if recommendation is not None:
            color = _COLORS[recommendation]
            label = _LABELS[recommendation]
            text_size = cv2.getTextSize(label, _FONT, 2.8, 4)[0]
            x = (self.w - text_size[0]) // 2
            cv2.putText(canvas, label, (x, 210), _FONT, 2.8, color, 4)
        else:
            cv2.putText(canvas, "Esperando cartas...",
                        (30, 190), _PLAIN, 1.0, _GRAY, 2)

        cv2.putText(canvas,
                    f"Banca: ${state.bankroll:.0f}   Apuesta: ${state.bet:.0f}",
                    (30, self.h - 18), _PLAIN, 0.65, _GRAY, 1)

        cv2.imshow(self.WINDOW, canvas)
        cv2.waitKey(1)

    def show_outcome(self, outcome: Outcome, delta: float) -> None:
        canvas = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)
        color = _OUTCOME_COLORS[outcome]
        label = outcome.value.upper()
        sign  = "+" if delta >= 0 else ""

        text_size = cv2.getTextSize(label, _FONT, 2.5, 4)[0]
        x = (self.w - text_size[0]) // 2
        cv2.putText(canvas, label,                (x, 160), _FONT, 2.5, color, 4)
        cv2.putText(canvas, f"{sign}${delta:.2f}", (x, 230), _PLAIN, 1.4, color, 2)

        cv2.imshow(self.WINDOW, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyAllWindows()
