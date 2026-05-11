from ..game.card import Card
from ..game.hand import Hand
from ..game.state import Action

_DEALER_IDX: dict[str, int] = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
    '7': 5, '8': 6, '9': 7, '10': 8,
    'J': 8, 'Q': 8, 'K': 8, 'A': 9,
}

H  = Action.HIT
S  = Action.STAND
D  = Action.DOUBLE
Sp = Action.SPLIT
Su = Action.SURRENDER

#                     2    3    4    5    6    7    8    9   10    A
_PAIR: dict[str, list[Action]] = {
    '2':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H],
    '3':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H],
    '4':  [ H,   H,   H,  Sp,  Sp,   H,   H,   H,   H,   H],
    '5':  [ D,   D,   D,   D,   D,   D,   D,   D,   H,   H],
    '6':  [Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H,   H],
    '7':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H],
    '8':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp],
    '9':  [Sp,  Sp,  Sp,  Sp,  Sp,   S,  Sp,  Sp,   S,   S],
    '10': [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],
    'A':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp],
}

#                      2    3    4    5    6    7    8    9   10    A
_SOFT: dict[int, list[Action]] = {
    13: [ H,   H,   H,   D,   D,   H,   H,   H,   H,   H],
    14: [ H,   H,   H,   D,   D,   H,   H,   H,   H,   H],
    15: [ H,   H,   D,   D,   D,   H,   H,   H,   H,   H],
    16: [ H,   H,   D,   D,   D,   H,   H,   H,   H,   H],
    17: [ H,   D,   D,   D,   D,   H,   H,   H,   H,   H],
    18: [ S,   D,   D,   D,   D,   S,   S,   H,   H,   H],
    19: [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],
    20: [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],
}

#                      2    3    4    5    6    7    8    9   10    A
_HARD: dict[int, list[Action]] = {
     8: [ H,   H,   H,   H,   H,   H,   H,   H,   H,   H],
     9: [ H,   D,   D,   D,   D,   H,   H,   H,   H,   H],
    10: [ D,   D,   D,   D,   D,   D,   D,   D,   H,   H],
    11: [ D,   D,   D,   D,   D,   D,   D,   D,   D,   H],
    12: [ H,   H,   S,   S,   S,   H,   H,   H,   H,   H],
    13: [ S,   S,   S,   S,   S,   H,   H,   H,   H,   H],
    14: [ S,   S,   S,   S,   S,   H,   H,   H,   H,   H],
    15: [ S,   S,   S,   S,   S,   H,   H,   H,  Su,   H],
    16: [ S,   S,   S,   S,   S,   H,   H,  Su,  Su,  Su],
    17: [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],
}


_UPCARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']


def full_row(
    player_hand: Hand,
    *,
    can_split: bool = True,
    can_double: bool = True,
    can_surrender: bool = True,
) -> list[Action]:
    """Devuelve la fila completa de la estrategia para esta mano.
    El resultado es una lista de 10 acciones, una por cada carta del crupier (2..A).
    Útil para mostrar la tabla visual en pantalla."""
    return [
        recommend(player_hand, Card(r),
                  can_split=can_split,
                  can_double=can_double,
                  can_surrender=can_surrender)
        for r in _UPCARD_RANKS
    ]


def recommend(
    player_hand: Hand,
    dealer_upcard: Card,
    *,
    can_split: bool = True,
    can_double: bool = True,
    can_surrender: bool = True,
) -> Action:
    """Devuelve la acción óptima según basic strategy estándar.

    Las flags controlan qué acciones están disponibles en este momento:
    split/double/surrender solo son válidos en la primera decisión de la mano.
    """
    idx = _DEALER_IDX[dealer_upcard.rank]

    if can_split and player_hand.is_pair():
        rank = player_hand.cards[0].rank
        if rank in ('J', 'Q', 'K'):
            rank = '10'
        action = _PAIR[rank][idx]
        if action == Action.DOUBLE:
            return action if can_double else Action.HIT
        return action

    if player_hand.is_soft():
        total = player_hand.total()
        total = min(max(total, 13), 20)
        action = _SOFT[total][idx]
        if action == Action.DOUBLE and not can_double:
            return Action.STAND if total >= 18 else Action.HIT
        return action

    total = player_hand.total()
    if total >= 17:
        return Action.STAND
    if total <= 8:
        return Action.HIT
    action = _HARD[total][idx]
    if action == Action.DOUBLE and not can_double:
        return Action.HIT
    if action == Action.SURRENDER and not can_surrender:
        return Action.HIT
    return action
