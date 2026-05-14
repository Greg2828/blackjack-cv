"""
ARCHIVO: src/ui/display.py
PROPÓSITO: Dibuja la interfaz visual del sistema en una ventana separada.
           Todo lo que el jugador ve en pantalla (cartas, recomendación,
           tabla de estrategia, banca) se genera en este archivo.

           La ventana que se abre se llama "Blackjack CV" y se actualiza
           ~30 veces por segundo mientras el sistema está corriendo.

           NOTA: esta ventana es SEPARADA de la ventana "Camara" que muestra
           el feed en vivo. Son dos ventanas distintas.

CÓMO SE CONECTA:
  - main.py y simulate.py crean un Display y llaman a show() cada frame
  - recibe el GameState para saber qué mostrar
  - recibe la recommendation (de strategy.py) para mostrar HIT/STAND/etc.
"""

import cv2            # OpenCV: para dibujar texto, rectángulos y mostrar ventanas
import numpy as np    # para crear el "lienzo" (canvas) de la interfaz
from ..game.state import Action, GameState, Outcome

# =============================================================================
# COLORES (en formato BGR: Blue, Green, Red — OpenCV usa este orden, no RGB)
# =============================================================================

# Color de fondo de cada tipo de acción (usados en los textos grandes y celdas).
# Ejemplo: Action.HIT → verde (0, 210, 0), Action.STAND → azul (0, 0, 210)
_COLORS: dict[Action, tuple[int, int, int]] = {
    Action.HIT:       (0,   210,   0),   # verde brillante
    Action.STAND:     (0,    0,  210),   # rojo (recuerda: BGR, no RGB)
    Action.DOUBLE:    (0,   165, 255),   # naranja
    Action.SPLIT:     (255,   0, 255),   # magenta
    Action.SURRENDER: (0,   220, 220),   # amarillo claro
}

# Abreviaturas de 1-2 letras para las celdas pequeñas de la tabla de estrategia.
_ABBREV: dict[Action, str] = {
    Action.HIT:       "H",
    Action.STAND:     "S",
    Action.DOUBLE:    "D",
    Action.SPLIT:     "SP",
    Action.SURRENDER: "SU",
}

# Nombres completos para el texto grande de recomendación en el centro.
_LABELS: dict[Action, str] = {
    Action.HIT:       "HIT",
    Action.STAND:     "STAND",
    Action.DOUBLE:    "DOUBLE",
    Action.SPLIT:     "SPLIT",
    Action.SURRENDER: "SURRENDER",
}

# Colores para mostrar el resultado final de la mano.
_OUTCOME_COLORS: dict[Outcome, tuple[int, int, int]] = {
    Outcome.WIN:       (0,  210,   0),   # verde → ganaste
    Outcome.BLACKJACK: (0,  210,   0),   # verde → blackjack (también ganaste)
    Outcome.LOSE:      (0,    0, 210),   # rojo  → perdiste
    Outcome.PUSH:      (200, 200,   0),  # amarillo → empate
}

# Colores básicos reutilizados en toda la interfaz.
_WHITE = (255, 255, 255)  # texto principal
_GRAY  = (160, 160, 160)  # texto secundario / cartas del crupier
_BG    = (25,  25,  25)   # fondo oscuro casi negro

# Tipos de fuente de OpenCV (tipografías disponibles).
_FONT  = cv2.FONT_HERSHEY_DUPLEX   # más elegante, usada para la recomendación grande
_PLAIN = cv2.FONT_HERSHEY_SIMPLEX  # más simple, usada para textos pequeños

# =============================================================================
# CONFIGURACIÓN DE LA TABLA DE ESTRATEGIA (dimensiones en píxeles)
# =============================================================================

_UPCARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']  # 10 columnas
_CELL_W  = 78    # ancho de cada celda de la tabla en píxeles
_CELL_H  = 44    # alto de cada celda
_LABEL_W = 95    # ancho del área de la etiqueta izquierda ("Hard 16", etc.)
_TABLE_Y = 268   # posición Y (vertical) donde empieza la sección de la tabla


def _hand_label(state: GameState) -> str:
    """Genera la etiqueta descriptiva de la mano del jugador.
    Ejemplos: 'Par 8', 'Soft 18', 'Hard 16'.
    Se muestra a la izquierda de la tabla de estrategia."""
    h = state.player_hand
    if not h.visible_cards:
        return ""  # sin cartas visibles, no hay nada que mostrar
    if h.is_pair() and len(h.cards) == 2:
        r = h.cards[0].rank
        return f"Par {r}"   # ej: "Par 8" si tienes dos ochos
    if h.is_soft():
        return f"Soft {h.total()}"   # ej: "Soft 18" si tienes As+7
    return f"Hard {h.total()}"       # ej: "Hard 16" si tienes 9+7


class Display:
    WINDOW = "Blackjack CV"  # nombre de la ventana (título de la barra superior)

    def __init__(self, width: int = 900, height: int = 430):
        """Crea la ventana de la interfaz.

        Parámetros:
          width, height: dimensiones de la ventana en píxeles.
        """
        self.w = width
        self.h = height

        # Crea la ventana con nombre WINDOW.
        # cv2.WINDOW_NORMAL permite que el usuario redimensione la ventana manualmente.
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, width, height)

    def show(
        self,
        state: GameState,
        recommendation: Action | None = None,    # None si todavía no hay recomendación
        strategy_row: list[Action] | None = None, # None si no hay tabla que mostrar
        dealer_upcard_rank: str | None = None,   # para resaltar la columna activa de la tabla
    ) -> None:
        """Dibuja y muestra el frame principal de la interfaz.

        Dibuja (de arriba a abajo):
          1. Cartas del jugador y del crupier con sus totales
          2. Recomendación grande en el centro ("HIT", "STAND", etc.)
          3. Tabla de estrategia (fila de la mano actual) si está disponible
          4. Banca y apuesta en el borde inferior
        """

        # Creamos el "canvas" (lienzo) con el color de fondo.
        # np.full((alto, ancho, 3), color, dtype) crea una imagen de un solo color.
        canvas = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)

        # Preparamos los textos de las cartas.
        player_total = state.player_hand.total() if state.player_hand.visible_cards else 0
        player_cards = ' '.join(str(c) for c in state.player_hand.cards) or "---"
        dealer_cards = ' '.join(str(c) for c in state.dealer_hand.cards) or "---"

        # ── 1. CARTAS ─────────────────────────────────────────────────────────
        # cv2.putText(imagen, texto, posición, fuente, escala, color, grosor)
        # posición = (x, y) donde y es desde la parte superior
        cv2.putText(canvas, f"Jugador: {player_cards}  ({player_total})",
                    (30, 50), _PLAIN, 0.9, _WHITE, 2)
        cv2.putText(canvas, f"Crupier: {dealer_cards}",
                    (30, 90), _PLAIN, 0.9, _GRAY, 2)

        # Línea horizontal separadora (decorativa).
        # cv2.line(imagen, punto_inicio, punto_fin, color, grosor)
        cv2.line(canvas, (20, 108), (self.w - 20, 108), (60, 60, 60), 1)

        # ── 2. RECOMENDACIÓN ──────────────────────────────────────────────────
        if recommendation is not None:
            color = _COLORS[recommendation]
            label = _LABELS[recommendation]

            # Calculamos el ancho del texto para centrarlo horizontalmente.
            # getTextSize devuelve ((ancho, alto), baseline).
            sz = cv2.getTextSize(label, _FONT, 2.5, 4)[0]
            x  = (self.w - sz[0]) // 2  # posición X centrada

            cv2.putText(canvas, label, (x, 200), _FONT, 2.5, color, 4)
        else:
            # Sin recomendación todavía (esperando que se detecten las cartas).
            cv2.putText(canvas, "Esperando cartas...",
                        (30, 185), _PLAIN, 1.0, _GRAY, 2)

        # ── 3. TABLA DE ESTRATEGIA ────────────────────────────────────────────
        # Solo se dibuja si tenemos la fila completa de la estrategia (10 acciones).
        if strategy_row and len(strategy_row) == 10:
            self._draw_strategy_row(
                canvas, strategy_row, dealer_upcard_rank, _hand_label(state)
            )

        # ── 4. BANCA Y APUESTA ────────────────────────────────────────────────
        cv2.putText(
            canvas,
            f"Banca: ${state.bankroll:.0f}   Apuesta: ${state.bet:.0f}",
            (30, self.h - 12), _PLAIN, 0.65, _GRAY, 1,
        )

        # Mostramos el canvas en la ventana y esperamos 1ms para refrescar la GUI.
        cv2.imshow(self.WINDOW, canvas)
        cv2.waitKey(1)

    def _draw_strategy_row(
        self,
        canvas: np.ndarray,
        row: list[Action],
        active_rank: str | None,
        hand_label: str,
    ) -> None:
        """Dibuja la fila de la tabla de estrategia para la mano actual.

        Parámetros:
          canvas: el lienzo sobre el que dibujamos (se modifica en el lugar)
          row: lista de 10 Action (una por carta del crupier: 2,3,4,5,6,7,8,9,10,A)
          active_rank: rango de la carta visible del crupier (ej: '7') → se resalta su columna
          hand_label: texto descriptivo de la mano (ej: 'Hard 16')
        """
        ty = _TABLE_Y  # coordenada Y donde empieza la tabla

        # Línea separadora encima de la tabla.
        cv2.line(canvas, (20, ty - 8), (self.w - 20, ty - 8), (60, 60, 60), 1)

        # Etiqueta de la mano (a la izquierda de la tabla): "Hard 16", "Soft 18", etc.
        cv2.putText(canvas, hand_label, (8, ty + 20), _PLAIN, 0.65, _GRAY, 1)

        # ── Cabecera: rangos del crupier (2, 3, 4... 10, A) ──────────────────
        for i, rank in enumerate(_UPCARD_RANKS):
            x     = _LABEL_W + i * _CELL_W  # posición X de esta columna
            # El rango activo (carta del crupier) se muestra en blanco; el resto en gris.
            color = _WHITE if rank == active_rank else _GRAY

            # Centramos el texto dentro de la celda.
            sz = cv2.getTextSize(rank, _PLAIN, 0.6, 1)[0]
            cv2.putText(canvas, rank,
                        (x + (_CELL_W - sz[0]) // 2, ty + 18),
                        _PLAIN, 0.6, color, 1)

        # ── Celdas de acción ─────────────────────────────────────────────────
        cell_y = ty + 24  # posición Y de las celdas (justo debajo de los encabezados)

        for i, (rank, action) in enumerate(zip(_UPCARD_RANKS, row)):
            x = _LABEL_W + i * _CELL_W  # posición X de esta celda

            # Coordenadas del rectángulo de la celda.
            x1, y1 = x + 2,             cell_y
            x2, y2 = x + _CELL_W - 2,   cell_y + _CELL_H

            # Fondo de la celda: color de la acción muy atenuado (30% del color original).
            # int(c * 0.30) reduce cada componente de color al 30%.
            base = _COLORS[action]
            fill = tuple(int(c * 0.30) for c in base)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)  # -1 = relleno

            # Borde de la celda: más grueso y blanco para la columna activa del crupier.
            is_active    = (rank == active_rank)
            border_color = _WHITE if is_active else (70, 70, 70)
            border_w     = 2     if is_active else 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, border_w)

            # Texto de la acción: H, S, D, SP o SU centrado dentro de la celda.
            abbrev = _ABBREV[action]
            sz     = cv2.getTextSize(abbrev, _PLAIN, 0.7, 2)[0]
            tx     = x1 + ((_CELL_W - 4) - sz[0]) // 2     # centro horizontal
            ty_txt = y1 + (_CELL_H + sz[1]) // 2            # centro vertical
            cv2.putText(canvas, abbrev, (tx, ty_txt),
                        _PLAIN, 0.7, _COLORS[action], 2)

    def show_outcome(self, outcome: Outcome, delta: float) -> None:
        """Muestra el resultado final de la mano (ganas/pierdes/empate/blackjack)
        con el cambio de bankroll en grande.

        Reemplaza completamente la interfaz normal durante unos instantes
        hasta que el jugador presione 'n' para empezar una nueva mano.

        Parámetros:
          outcome: el resultado (WIN, LOSE, PUSH, BLACKJACK)
          delta: cuánto cambió la banca (+10.0, -5.0, etc.)
        """
        # Nuevo canvas en blanco (mismo fondo oscuro).
        canvas = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)
        color  = _OUTCOME_COLORS[outcome]
        label  = outcome.value.upper()  # ej: Outcome.WIN.value = "ganas" → "GANAS"

        # Signo + explícito para deltas positivos (ganancias).
        sign = "+" if delta >= 0 else ""

        # Texto grande del resultado (centrado horizontalmente).
        sz = cv2.getTextSize(label, _FONT, 2.5, 4)[0]
        x  = (self.w - sz[0]) // 2
        cv2.putText(canvas, label,                 (x, 180), _FONT, 2.5, color, 4)

        # Texto del cambio de banca debajo del resultado.
        # :.2f formatea el número con 2 decimales (ej: 15.00)
        cv2.putText(canvas, f"{sign}${delta:.2f}", (x, 260), _PLAIN, 1.4, color, 2)

        cv2.imshow(self.WINDOW, canvas)
        cv2.waitKey(1)

    def close(self) -> None:
        """Cierra todas las ventanas de OpenCV.
        Se llama al final de main() cuando el usuario presiona 'q'."""
        cv2.destroyAllWindows()
