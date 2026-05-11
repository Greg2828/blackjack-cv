from dataclasses import dataclass, field
from enum import Enum
from .hand import Hand


class Phase(Enum):
    """Fase actual de la mano, inferida del estado de la mesa."""
    WAITING_BET  = "esperando apuesta"
    BET_PLACED   = "apuesta colocada"
    PLAYER_TURN  = "turno del jugador"
    DEALER_TURN  = "turno del crupier"
    RESOLVED     = "mano resuelta"


class Action(Enum):
    HIT       = "hit"
    STAND     = "stand"
    DOUBLE    = "double"
    SPLIT     = "split"
    SURRENDER = "surrender"


class Outcome(Enum):
    WIN       = "ganas"
    LOSE      = "pierdes"
    PUSH      = "empate"
    BLACKJACK = "blackjack"


@dataclass
class GameState:
    """Estado completo de una mano en curso."""
    player_hand: Hand = field(default_factory=Hand)
    dealer_hand: Hand = field(default_factory=Hand)
    bet: float = 0.0
    bankroll: float = 100.0
    phase: Phase = Phase.WAITING_BET
    doubled: bool = False
    surrendered: bool = False

    def resolve_hand(
        self,
        hand: 'Hand',
        doubled: bool = False,
        is_split: bool = False,
    ) -> tuple[Outcome, float]:
        """Resuelve una mano concreta contra el crupier actual.
        Usar para manos de split: cada mano se evalúa por separado y el 21
        de 2 cartas no paga como blackjack natural (regla estándar de casino)."""
        p = hand
        d = self.dealer_hand
        bet_amount = self.bet * (2 if doubled else 1)

        if p.is_bust():
            return Outcome.LOSE, -bet_amount

        p_bj = p.is_blackjack() and not is_split
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

    def resolve(self) -> tuple[Outcome, float]:
        """Calcula el resultado final y el delta de bankroll.
        Llamar cuando el crupier haya terminado su mano."""
        if self.surrendered:
            return Outcome.LOSE, -(self.bet / 2)

        p, d = self.player_hand, self.dealer_hand
        bet_amount = self.bet * (2 if self.doubled else 1)

        if p.is_bust():
            return Outcome.LOSE, -bet_amount

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
