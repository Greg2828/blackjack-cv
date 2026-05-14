"""
ARCHIVO: src/game/state.py
PROPÓSITO: Define el "estado" completo de una mano en curso.
           Un "estado" es una fotografía de todo lo que está pasando
           en un momento dado: qué cartas tiene cada jugador, cuánto
           se apostó, en qué fase está la partida, y cuál fue el resultado.

           También contiene la lógica para resolver quién ganó al final.

CÓMO SE CONECTA:
  - main.py y simulate.py crean un GameState al inicio de cada mano
    y lo van actualizando durante toda la partida
  - strategy.py recibe player_hand (que vive dentro de GameState)
  - display.py recibe el GameState completo para mostrar la información en pantalla
  - logger.py recibe el GameState para guardarlo en el CSV
"""

from dataclasses import dataclass, field
from enum import Enum    # Enum: permite crear conjuntos de opciones nombradas
from .hand import Hand   # importamos Hand del archivo hermano hand.py


# =============================================================================
# ENUMERACIONES (conjuntos de opciones fijas)
# =============================================================================
# Un Enum es como una lista de opciones donde cada opción tiene un nombre
# y un valor. Usar Enums evita escribir cadenas de texto sueltas ('player_turn',
# 'resolved'...) que son fáciles de escribir mal.

class Phase(Enum):
    """Las 5 fases posibles de una mano, en orden cronológico.
    La fase actual se guarda en GameState.phase y se usa en main.py
    para decidir qué mostrar en pantalla y qué detectar."""
    WAITING_BET  = "esperando apuesta"  # antes de que el jugador coloque fichas
    BET_PLACED   = "apuesta colocada"   # fichas en la mesa, esperando reparto
    PLAYER_TURN  = "turno del jugador"  # el jugador decide: hit, stand, double...
    DEALER_TURN  = "turno del crupier"  # el crupier voltea su carta y pide más si necesita
    RESOLVED     = "mano resuelta"      # la mano terminó, ya se sabe quién ganó


class Action(Enum):
    """Las 5 acciones que puede hacer el jugador en su turno.
    Se usan en strategy.py (para recomendar) y en simulate.py (para ejecutar)."""
    HIT       = "hit"        # pedir una carta más
    STAND     = "stand"      # quedarse con las cartas actuales
    DOUBLE    = "double"     # doblar la apuesta y recibir exactamente 1 carta más
    SPLIT     = "split"      # dividir un par en 2 manos independientes
    SURRENDER = "surrender"  # rendirse: recuperar la mitad de la apuesta


class Outcome(Enum):
    """Los 4 posibles resultados de una mano."""
    WIN       = "ganas"      # el jugador ganó normalmente
    LOSE      = "pierdes"    # el jugador perdió
    PUSH      = "empate"     # empate: devuelven la apuesta sin ganancias ni pérdidas
    BLACKJACK = "blackjack"  # blackjack natural: paga 3:2 (más que una victoria normal)


# =============================================================================
# ESTADO DEL JUEGO
# =============================================================================

@dataclass
class GameState:
    """Estado completo de una mano en curso.
    Agrupa toda la información que necesitamos para saber exactamente
    qué está pasando en la partida en cualquier momento."""

    # Mano actual del jugador (empieza vacía, se llena al repartir).
    player_hand: Hand = field(default_factory=Hand)

    # Mano actual del crupier (empieza vacía, se llena al repartir).
    dealer_hand: Hand = field(default_factory=Hand)

    # Cantidad apostada en esta mano (en euros o la moneda que uses).
    bet: float = 0.0

    # Dinero total disponible del jugador (se actualiza al final de cada mano).
    bankroll: float = 100.0

    # Fase actual de la mano (empieza esperando apuesta).
    phase: Phase = Phase.WAITING_BET

    # True si el jugador eligió DOUBLE (dobló la apuesta).
    # Necesitamos recordarlo para calcular correctamente cuánto se gana o pierde.
    doubled: bool = False

    # True si el jugador eligió SURRENDER (se rindió).
    # En ese caso recupera la mitad de la apuesta y la mano termina inmediatamente.
    surrendered: bool = False

    def resolve_hand(
        self,
        hand: 'Hand',
        doubled: bool = False,
        is_split: bool = False,
    ) -> tuple['Outcome', float]:
        """Resuelve una mano concreta contra el crupier actual.
        Se usa para manos de SPLIT (cada sub-mano se evalúa por separado).

        Parámetros:
          hand: la mano del jugador a evaluar
          doubled: si esa sub-mano fue doblada (afecta la apuesta)
          is_split: si viene de un split, el 21 en 2 cartas NO cuenta como blackjack natural
                    (regla estándar de casino)

        Devuelve: (Outcome, delta)
          Outcome: el resultado (WIN, LOSE, PUSH, BLACKJACK)
          delta: cuánto dinero gana (+) o pierde (-) el jugador
        """
        p = hand              # mano del jugador (alias corto para legibilidad)
        d = self.dealer_hand  # mano del crupier

        # Si doblaste, la apuesta efectiva es el doble.
        bet_amount = self.bet * (2 if doubled else 1)

        # Regla 1: si el jugador se pasa de 21 (bust), pierde aunque el crupier también se pase.
        if p.is_bust():
            return Outcome.LOSE, -bet_amount

        # Verificamos si hay blackjack natural.
        # is_split=True → aunque tengas As+10, NO es blackjack natural (regla de casino).
        p_bj = p.is_blackjack() and not is_split  # blackjack del jugador
        d_bj = d.is_blackjack()                   # blackjack del crupier

        # Regla 2: ambos tienen blackjack → empate (ninguno pierde ni gana).
        if p_bj and d_bj:
            return Outcome.PUSH, 0.0

        # Regla 3: solo el jugador tiene blackjack → paga 3:2 (1.5× la apuesta).
        if p_bj:
            return Outcome.BLACKJACK, self.bet * 1.5

        # Regla 4: solo el crupier tiene blackjack → el jugador pierde su apuesta.
        if d_bj:
            return Outcome.LOSE, -bet_amount

        # A partir de aquí, nadie tiene blackjack. Comparamos totales normalmente.

        # Regla 5: si el crupier se pasa de 21, el jugador gana.
        if d.is_bust():
            return Outcome.WIN, bet_amount

        # Regla 6: el que tiene más puntos gana.
        if p.total() > d.total():
            return Outcome.WIN, bet_amount
        if p.total() < d.total():
            return Outcome.LOSE, -bet_amount

        # Regla 7: si tienen el mismo total, es empate.
        return Outcome.PUSH, 0.0

    def resolve(self) -> tuple['Outcome', float]:
        """Calcula el resultado final de la mano principal (sin split).
        Es igual a resolve_hand() pero usa directamente el estado interno.
        Llamar cuando el crupier haya terminado su turno.

        Devuelve: (Outcome, delta) — igual que resolve_hand()."""

        # Caso especial: el jugador se rindió (surrender) → recupera la mitad.
        # Ejemplo: apostaste 10€ → recuperas 5€ (pierdes 5€).
        if self.surrendered:
            return Outcome.LOSE, -(self.bet / 2)

        p, d = self.player_hand, self.dealer_hand

        # Si doblaste, la apuesta es el doble. Si no, es la apuesta normal.
        bet_amount = self.bet * (2 if self.doubled else 1)

        # A partir de aquí, la lógica es idéntica a resolve_hand().
        # (Ver comentarios arriba para el detalle de cada regla).

        if p.is_bust():
            return Outcome.LOSE, -bet_amount

        # Blackjack: en la mano principal, el doble NO cancela el blackjack natural.
        # (Si alguien dobla con As+10 es una apuesta rarísima, pero técnicamente posible).
        p_bj = p.is_blackjack() and not self.doubled
        d_bj = d.is_blackjack()

        if p_bj and d_bj:
            return Outcome.PUSH, 0.0
        if p_bj:
            return Outcome.BLACKJACK, self.bet * 1.5
        if d_bj:
            return Outcome.LOSE, -bet_amount

        if d.is_bust():
            return Outcome.WIN, bet_amount

        if p.total() > d.total():
            return Outcome.WIN, bet_amount
        if p.total() < d.total():
            return Outcome.LOSE, -bet_amount

        return Outcome.PUSH, 0.0
