"""
ARCHIVO: src/decision/strategy.py
PROPÓSITO: Implementa la "Basic Strategy" (estrategia básica) del blackjack.
           La basic strategy es una tabla matemáticamente óptima que te dice
           exactamente qué hacer en cada situación posible.
           Esta tabla fue calculada con millones de simulaciones por matemáticos.
           Si la sigues perfectamente, reduces la ventaja del casino al mínimo (~0.5%).

CÓMO SE CONECTA:
  - main.py llama a recommend() cada vez que el jugador necesita decidir
  - main.py llama a full_row() para mostrar la fila completa de la tabla en pantalla
  - simulate.py llama a recommend() para jugar automáticamente
  - tests/test_strategy.py verifica que las tablas son correctas

ESTRUCTURA DE LAS TABLAS:
  Hay 3 tablas según el tipo de mano:
    _PAIR: cuando el jugador tiene un par (dos cartas iguales)
    _SOFT: cuando el jugador tiene un As contado como 11
    _HARD: el resto de casos (mano dura sin as flexible)

  Cada tabla tiene 10 columnas, una para cada carta visible del crupier: 2,3,4,5,6,7,8,9,10,A
"""

# Importamos Card y Hand de sus respectivos archivos en src/game/
from ..game.card import Card    # '..' sube un nivel (de decision/ a src/), luego baja a game/
from ..game.hand import Hand
from ..game.state import Action


# =============================================================================
# ÍNDICE DE COLUMNAS: posición de la carta del crupier en las tablas
# =============================================================================

# Mapeo de rango de carta del crupier → índice de columna (0-9) en las tablas.
# Las figuras J, Q, K valen lo mismo que 10, así que comparten el índice 8.
# El As (A) tiene su propia columna (índice 9) porque cambia mucho la estrategia.
_DEALER_IDX: dict[str, int] = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,  # cartas bajas → el crupier tiene más riesgo de bust
    '7': 5, '8': 6, '9': 7, '10': 8,
    'J': 8, 'Q': 8, 'K': 8,                   # figuras: mismo índice que el 10
    'A': 9,                                    # As del crupier: columna especial
}

# Alias cortos para las acciones (mejora la legibilidad de las tablas).
H  = Action.HIT        # pedir carta
S  = Action.STAND      # plantarse
D  = Action.DOUBLE     # doblar
Sp = Action.SPLIT      # dividir par
Su = Action.SURRENDER  # rendirse

# =============================================================================
# TABLA 1: PARES (_PAIR)
# Cuándo partir en dos manos vs. cuándo jugar normalmente.
# Clave = rango del par (ej: '8' significa par de 8s)
# Valor = lista de 10 acciones, una por carta visible del crupier (2,3,4,5,6,7,8,9,10,A)
# =============================================================================
#                         2    3    4    5    6    7    8    9   10    A
_PAIR: dict[str, list[Action]] = {
    '2':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H],  # par de 2s: partir contra 2-7
    '3':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H],  # par de 3s: igual que 2s
    '4':  [ H,   H,   H,  Sp,  Sp,   H,   H,   H,   H,   H],  # par de 4s: solo partir contra 5-6
    '5':  [ D,   D,   D,   D,   D,   D,   D,   D,   H,   H],  # par de 5s: nunca partir, doblar
    '6':  [Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H,   H],  # par de 6s: partir contra 2-6
    '7':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,   H,   H,   H,   H],  # par de 7s: partir contra 2-7
    '8':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp],  # par de 8s: SIEMPRE partir
    '9':  [Sp,  Sp,  Sp,  Sp,  Sp,   S,  Sp,  Sp,   S,   S],  # par de 9s: plantarse vs 7,10,A
    '10': [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],  # par de 10s: NUNCA partir (ya tienes 20)
    'A':  [Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp,  Sp],  # par de Ases: SIEMPRE partir
}

# =============================================================================
# TABLA 2: MANOS SOFT (_SOFT)
# Cuando tienes un As que cuenta como 11 (soft hand).
# Clave = total de la mano (13-20). Nota: soft 21 = blackjack (ya resuelto).
# =============================================================================
#                          2    3    4    5    6    7    8    9   10    A
_SOFT: dict[int, list[Action]] = {
    13: [ H,   H,   H,   D,   D,   H,   H,   H,   H,   H],  # A+2: doblar solo vs 5-6
    14: [ H,   H,   H,   D,   D,   H,   H,   H,   H,   H],  # A+3: igual
    15: [ H,   H,   D,   D,   D,   H,   H,   H,   H,   H],  # A+4: doblar vs 4-6
    16: [ H,   H,   D,   D,   D,   H,   H,   H,   H,   H],  # A+5: igual
    17: [ H,   D,   D,   D,   D,   H,   H,   H,   H,   H],  # A+6: doblar vs 3-6
    18: [ S,   D,   D,   D,   D,   S,   S,   H,   H,   H],  # A+7: plantarse vs 2,7,8; pedir vs 9,10,A
    19: [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],  # A+8: siempre plantarse (19 es muy bueno)
    20: [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],  # A+9: siempre plantarse (20 es excelente)
}

# =============================================================================
# TABLA 3: MANOS DURAS (_HARD)
# Cuando no tienes As flexible (o el As cuenta como 1 obligatoriamente).
# Clave = total de la mano (8-17).
# Nota: ≤7 siempre HIT, ≥17 siempre STAND (no necesitan tabla, se manejan en código).
# =============================================================================
#                          2    3    4    5    6    7    8    9   10    A
_HARD: dict[int, list[Action]] = {
     8: [ H,   H,   H,   H,   H,   H,   H,   H,   H,   H],  # 8: siempre pedir
     9: [ H,   D,   D,   D,   D,   H,   H,   H,   H,   H],  # 9: doblar solo vs 3-6
    10: [ D,   D,   D,   D,   D,   D,   D,   D,   H,   H],  # 10: doblar vs 2-9
    11: [ D,   D,   D,   D,   D,   D,   D,   D,   D,   H],  # 11: doblar vs 2-10 (casi siempre)
    12: [ H,   H,   S,   S,   S,   H,   H,   H,   H,   H],  # 12: plantarse solo vs 4-6 (crupier débil)
    13: [ S,   S,   S,   S,   S,   H,   H,   H,   H,   H],  # 13: plantarse vs 2-6
    14: [ S,   S,   S,   S,   S,   H,   H,   H,   H,   H],  # 14: igual que 13
    15: [ S,   S,   S,   S,   S,   H,   H,   H,  Su,   H],  # 15: rendirse vs 10
    16: [ S,   S,   S,   S,   S,   H,   H,  Su,  Su,  Su],  # 16: rendirse vs 9,10,A
    17: [ S,   S,   S,   S,   S,   S,   S,   S,   S,   S],  # 17: siempre plantarse
}


# Lista ordenada de los rangos del crupier para mostrar la tabla en pantalla.
# Usada por full_row() para generar las 10 celdas de la tabla visual.
_UPCARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']


def full_row(
    player_hand: Hand,
    *,                     # todo lo que viene después es "keyword-only" (debes escribir el nombre)
    can_split: bool = True,
    can_double: bool = True,
    can_surrender: bool = True,
) -> list[Action]:
    """Devuelve la fila completa de la estrategia para esta mano.
    El resultado es una lista de 10 acciones, una por cada carta del crupier (2..A).
    Útil para mostrar la tabla visual en pantalla (display.py la recibe y pinta las celdas).

    Ejemplo: si tienes Hard 16, devuelve [S, S, S, S, S, H, H, Su, Su, Su]
    (plantarse contra 2-6, pedir contra 7-8, rendirse contra 9/10/A)."""

    # Para cada uno de los 10 rangos del crupier, llamamos a recommend() y guardamos la acción.
    # Card(r) crea una carta temporal solo para consultar la tabla.
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

    Parámetros:
      player_hand: la mano actual del jugador
      dealer_upcard: la carta visible del crupier (la que no está tapada)
      can_split: True solo si es la primera decisión Y la mano es un par
      can_double: True solo si es la primera decisión de la mano
      can_surrender: True solo si es la primera decisión de la mano

    Las flags can_* existen porque SPLIT, DOUBLE y SURRENDER solo están disponibles
    en la primera decisión de cada mano. Después de pedir una carta extra (HIT),
    ya no puedes doblar ni rendirte en la mayoría de casinos.

    Devuelve: la Action más rentable (H, S, D, Sp o Su).
    """

    # Calculamos el índice de columna correspondiente a la carta del crupier.
    # Ejemplo: si el crupier muestra un 7, idx = 5 → columna 6 de la tabla (0-indexado).
    idx = _DEALER_IDX[dealer_upcard.rank]

    # ── PASO 1: ¿Es un par? ───────────────────────────────────────────────────
    # La tabla de pares tiene prioridad máxima cuando can_split=True.
    if can_split and player_hand.is_pair():
        rank = player_hand.cards[0].rank  # rango de cualquiera de las dos cartas
        # Las figuras J, Q, K se tratan igual que el 10 en la tabla de pares.
        if rank in ('J', 'Q', 'K'):
            rank = '10'
        action = _PAIR[rank][idx]  # consultamos la tabla de pares
        # Si la tabla recomienda DOUBLE pero no podemos doblar (ya no es primera decisión),
        # retrocedemos a HIT (siempre es seguro pedir carta).
        if action == Action.DOUBLE:
            return action if can_double else Action.HIT
        return action

    # ── PASO 2: ¿Es una mano soft? ───────────────────────────────────────────
    # La mano tiene un As que cuenta como 11 actualmente.
    if player_hand.is_soft():
        total = player_hand.total()
        # Las tablas de soft van de 13 a 20.
        # min(max(total, 13), 20) asegura que nos quedamos dentro de ese rango.
        # (Soft 21 es blackjack, ya resuelto antes. Soft 12 es As+As, se trata como par.)
        total = min(max(total, 13), 20)
        action = _SOFT[total][idx]
        # Si la tabla recomienda DOUBLE pero no podemos:
        # con soft 18+ es mejor plantarse (18 ya es buena mano)
        # con soft <18 es mejor pedir carta.
        if action == Action.DOUBLE and not can_double:
            return Action.STAND if total >= 18 else Action.HIT
        return action

    # ── PASO 3: Mano dura ────────────────────────────────────────────────────
    # Ni par con split disponible, ni soft: usamos la tabla de manos duras.
    total = player_hand.total()

    # Con 17 o más siempre nos plantamos (regla universal del blackjack).
    if total >= 17:
        return Action.STAND

    # Con 8 o menos siempre pedimos carta (nunca puede ser rentable doblar/rendirse con tan poco).
    if total <= 8:
        return Action.HIT

    # Para totales 9-16, consultamos la tabla dura.
    action = _HARD[total][idx]

    # Si la tabla recomienda DOUBLE pero no podemos → mejor pedir carta.
    if action == Action.DOUBLE and not can_double:
        return Action.HIT

    # Si la tabla recomienda SURRENDER pero no podemos → mejor pedir carta
    # (es la segunda opción menos mala cuando tienes, p.ej., Hard 16 vs As del crupier).
    if action == Action.SURRENDER and not can_surrender:
        return Action.HIT

    return action
