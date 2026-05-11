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
_ABBREV: dict[Action, str] = {
    Action.HIT:       "H",
    Action.STAND:     "S",
    Action.DOUBLE:    "D",
    Action.SPLIT:     "SP",
    Action.SURRENDER: "SU",
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

_WHITE = (255, 255, 255)
_GRAY  = (160, 160, 160)
_BG    = (25,  25,  25)
_FONT  = cv2.FONT_HERSHEY_DUPLEX
_PLAIN = cv2.FONT_HERSHEY_SIMPLEX

_UPCARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
_CELL_W  = 78
_CELL_H  = 44
_LABEL_W = 95
_TABLE_Y = 268   # y donde empieza la sección de la tabla


def _hand_label(state: GameState) -> str:
    h = state.player_hand
    if not h.visible_cards:
        return ""
    if h.is_pair() and len(h.cards) == 2:
        r = h.cards[0].rank
        return f"Par {r}"
    if h.is_soft():
        return f"Soft {h.total()}"
    return f"Hard {h.total()}"


class Display:
    WINDOW = "Blackjack CV"

    def __init__(self, width: int = 900, height: int = 430):
        self.w = width
        self.h = height
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, width, height)

    def show(
        self,
        state: GameState,
        recommendation: Action | None = None,
        strategy_row: list[Action] | None = None,
        dealer_upcard_rank: str | None = None,
    ) -> None:
        canvas = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)

        player_total = state.player_hand.total() if state.player_hand.visible_cards else 0
        player_cards = ' '.join(str(c) for c in state.player_hand.cards) or "---"
        dealer_cards = ' '.join(str(c) for c in state.dealer_hand.cards) or "---"

        # ── Cartas ─────────────────────────────────────────────────────
        cv2.putText(canvas, f"Jugador: {player_cards}  ({player_total})",
                    (30, 50), _PLAIN, 0.9, _WHITE, 2)
        cv2.putText(canvas, f"Crupier: {dealer_cards}",
                    (30, 90), _PLAIN, 0.9, _GRAY, 2)
        cv2.line(canvas, (20, 108), (self.w - 20, 108), (60, 60, 60), 1)

        # ── Recomendación ──────────────────────────────────────────────
        if recommendation is not None:
            color = _COLORS[recommendation]
            label = _LABELS[recommendation]
            sz = cv2.getTextSize(label, _FONT, 2.5, 4)[0]
            x  = (self.w - sz[0]) // 2
            cv2.putText(canvas, label, (x, 200), _FONT, 2.5, color, 4)
        else:
            cv2.putText(canvas, "Esperando cartas...",
                        (30, 185), _PLAIN, 1.0, _GRAY, 2)

        # ── Tabla de estrategia (fila de la mano actual) ───────────────
        if strategy_row and len(strategy_row) == 10:
            self._draw_strategy_row(
                canvas, strategy_row, dealer_upcard_rank, _hand_label(state)
            )

        # ── Banca ──────────────────────────────────────────────────────
        cv2.putText(
            canvas,
            f"Banca: ${state.bankroll:.0f}   Apuesta: ${state.bet:.0f}",
            (30, self.h - 12), _PLAIN, 0.65, _GRAY, 1,
        )

        cv2.imshow(self.WINDOW, canvas)
        cv2.waitKey(1)

    def _draw_strategy_row(
        self,
        canvas: np.ndarray,
        row: list[Action],
        active_rank: str | None,
        hand_label: str,
    ) -> None:
        ty = _TABLE_Y
        cv2.line(canvas, (20, ty - 8), (self.w - 20, ty - 8), (60, 60, 60), 1)

        # Etiqueta de la mano (Hard 16, Soft 18, Par 8…)
        cv2.putText(canvas, hand_label, (8, ty + 20), _PLAIN, 0.65, _GRAY, 1)

        # Cabecera: upcards del crupier
        for i, rank in enumerate(_UPCARD_RANKS):
            x = _LABEL_W + i * _CELL_W
            color = _WHITE if rank == active_rank else _GRAY
            sz = cv2.getTextSize(rank, _PLAIN, 0.6, 1)[0]
            cv2.putText(canvas, rank,
                        (x + (_CELL_W - sz[0]) // 2, ty + 18),
                        _PLAIN, 0.6, color, 1)

        # Celdas de acción
        cell_y = ty + 24
        for i, (rank, action) in enumerate(zip(_UPCARD_RANKS, row)):
            x  = _LABEL_W + i * _CELL_W
            x1, y1 = x + 2,         cell_y
            x2, y2 = x + _CELL_W - 2, cell_y + _CELL_H

            # Fondo: color de la acción, muy atenuado
            base  = _COLORS[action]
            fill  = tuple(int(c * 0.30) for c in base)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)

            # Borde: blanco y grueso si es la carta activa del crupier
            is_active    = (rank == active_rank)
            border_color = _WHITE          if is_active else (70, 70, 70)
            border_w     = 2               if is_active else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, border_w)

            # Texto de la acción
            abbrev = _ABBREV[action]
            sz     = cv2.getTextSize(abbrev, _PLAIN, 0.7, 2)[0]
            tx     = x1 + ((_CELL_W - 4) - sz[0]) // 2
            ty_txt = y1 + (_CELL_H + sz[1]) // 2
            cv2.putText(canvas, abbrev, (tx, ty_txt),
                        _PLAIN, 0.7, _COLORS[action], 2)

    def show_outcome(self, outcome: Outcome, delta: float) -> None:
        canvas = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)
        color  = _OUTCOME_COLORS[outcome]
        label  = outcome.value.upper()
        sign   = "+" if delta >= 0 else ""

        sz = cv2.getTextSize(label, _FONT, 2.5, 4)[0]
        x  = (self.w - sz[0]) // 2
        cv2.putText(canvas, label,                (x, 180), _FONT, 2.5, color, 4)
        cv2.putText(canvas, f"{sign}${delta:.2f}", (x, 260), _PLAIN, 1.4, color, 2)

        cv2.imshow(self.WINDOW, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyAllWindows()
