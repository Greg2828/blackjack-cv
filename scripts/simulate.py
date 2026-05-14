"""
ARCHIVO: scripts/simulate.py
PROPÓSITO: Modo simulación — permite jugar blackjack SIN cámara,
           introduciendo las cartas manualmente por teclado.

           Esto es útil para:
           - Probar que toda la lógica del juego funciona antes de tener la cámara lista
           - Practicar la basic strategy con el sistema guiándote
           - Generar datos de entrenamiento manualmente

Uso:
    python scripts/simulate.py
    python scripts/simulate.py --bankroll 200

Durante la partida, escribe los rangos de las cartas:
  A=As, 2-9=numerales, 10=diez, J=Jota, Q=Reina, K=Rey, BACK=dorso tapado
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

import sys
import argparse         # para leer argumentos de la línea de comandos (--bankroll 200)
from pathlib import Path

# Añadimos la carpeta raíz del proyecto al path de Python
# para que los imports de 'src.*' y 'config' funcionen aunque estemos en scripts/.
# __file__ = ruta de este archivo → .parent = carpeta scripts/ → .parent = raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.card import Card
from src.game.hand import Hand
from src.game.state import GameState, Phase, Action, Outcome
from src.decision.strategy import full_row, recommend
from src.analysis.logger import HandLogger
from src.ui.display import Display
import config

# Diccionario que mapea teclas (texto) a acciones.
# Si el jugador escribe 'h', se interpreta como Action.HIT, etc.
_ACTION_MAP = {
    'h':  Action.HIT,
    's':  Action.STAND,
    'd':  Action.DOUBLE,
    'sp': Action.SPLIT,
    'su': Action.SURRENDER,
}

# Diccionario inverso: Action → tecla (para mostrar el atajo como "default").
_KEY_MAP = {v: k for k, v in _ACTION_MAP.items()}

SEPARATOR = "─" * 46  # línea decorativa para separar secciones en el terminal


def _play_sub_hand(
    hand: Hand,
    dealer_up: Card,
    state: GameState,
    display: Display,
    *,
    is_ace_split: bool = False,
) -> tuple[Hand, list[Action], list[Action], bool]:
    """Juega el turno completo de UNA mano de split.

    Cuando el jugador decide SPLIT, sus dos cartas se separan en dos manos independientes.
    Esta función gestiona el turno de UNA de esas dos manos.

    Parámetros:
      hand: la sub-mano a jugar (empieza con 1 carta)
      dealer_up: la carta visible del crupier (para consultar la estrategia)
      state: estado del juego (se modifica para actualizar el display)
      display: la ventana de interfaz
      is_ace_split: True si fue un split de Ases.
                    La regla especial de As-split: solo recibes 1 carta más y te plantas
                    (no puedes pedir más cartas en los As split).

    Devuelve: (hand_final, acciones_recomendadas, acciones_tomadas, doubled)
    """
    recommended: list[Action] = []
    taken:       list[Action] = []
    doubled = False

    # Caso especial: split de Ases → solo 1 carta, stand forzado.
    if is_ace_split:
        new_card = _ask_card("Carta")   # pide la carta que recibe este As
        hand.add(new_card)
        state.player_hand = hand
        display.show(state, None, strategy_row=None, dealer_upcard_rank=dealer_up.rank)
        _print_state(state, None)
        print("  (Stand forzado — regla de As split)")
        return hand, recommended, taken, doubled

    # Turno normal de una sub-mano (igual que una mano normal pero sin split ni surrender).
    first = True   # en la primera decisión podemos doblar, después no
    while True:
        # Calculamos la fila de estrategia para esta mano (can_split=False: no se puede re-split).
        row = full_row(hand, can_split=False, can_double=first, can_surrender=False)
        rec = recommend(hand, dealer_up, can_split=False, can_double=first, can_surrender=False)
        recommended.append(rec)

        # Actualizamos el display con la mano actual.
        state.player_hand = hand
        display.show(state, rec, strategy_row=row, dealer_upcard_rank=dealer_up.rank)
        _print_state(state, rec)

        # Si la mano se pasó de 21 o llegó a 21 exacto, el turno termina automáticamente.
        if hand.is_bust() or hand.total() >= 21:
            break

        # Pedimos al jugador qué acción tomar.
        action = _ask_action(rec)
        taken.append(action)
        first = False  # ya no es la primera decisión

        if action == Action.STAND:
            break  # el jugador se planta: fin del turno de esta mano
        elif action == Action.DOUBLE:
            doubled = True
            hand.add(_ask_card("Nueva carta"))
            state.player_hand = hand
            display.show(state, None)
            _print_state(state, None)
            break  # double solo da 1 carta y termina el turno
        elif action == Action.HIT:
            hand.add(_ask_card("Nueva carta"))  # añade la carta y continúa el bucle
        else:
            pass  # SPLIT y SURRENDER no están disponibles dentro de un split

    return hand, recommended, taken, doubled


def _handle_split(
    state: GameState,
    dealer_up: Card,
    display: Display,
    logger: HandLogger,
    first_rec: Action,
) -> None:
    """Gestiona el flujo completo cuando el jugador elige SPLIT.

    Proceso:
    1. Separa las 2 cartas iguales en 2 manos independientes
    2. Juega cada mano por separado (_play_sub_hand)
    3. El crupier juega su turno UNA sola vez (para ambas manos)
    4. Resuelve y loguea cada mano por separado

    Parámetros:
      first_rec: la primera recomendación (SPLIT), necesaria para el log
    """
    card1 = state.player_hand.cards[0]   # primera carta del par
    card2 = state.player_hand.cards[1]   # segunda carta del par
    is_ace = card1.rank == 'A'           # ¿es un split de Ases?

    sub_results = []  # guardaremos (hand, recommended, taken, doubled) por cada sub-mano

    # Jugamos las 2 sub-manos una por una.
    for i, base_card in enumerate([card1, card2], 1):  # enumerate(..., 1) → 1, 2
        print(f"\n  ─── Mano {i} (split) ───")
        hand = Hand([base_card])   # sub-mano empieza con la carta base
        hand, hrec, htaken, hdoubled = _play_sub_hand(
            hand, dealer_up, state, display, is_ace_split=is_ace
        )
        sub_results.append((hand, hrec, htaken, hdoubled))

    # ── Turno del crupier (una sola vez para ambas manos) ──────────────────────
    # Solo jugamos el turno del crupier si al menos una sub-mano no hizo bust.
    all_bust = all(h.is_bust() for h, *_ in sub_results)
    if not all_bust:
        state.phase = Phase.DEALER_TURN
        print(f"\n  --- Turno del crupier ---")
        hidden = _ask_card("Carta tapada del crupier")   # revelamos la carta tapada
        state.dealer_hand = Hand([dealer_up, hidden])
        print(f"  Crupier: {state.dealer_hand}")

        # El crupier pide cartas hasta llegar a 17 (regla del crupier).
        while True:
            d_total = state.dealer_hand.total()
            # Si el crupier tiene Soft 17 Y la regla es "plantarse en Soft 17":
            stands_soft17 = (
                config.DEALER_STAND_ON_SOFT_17
                and state.dealer_hand.is_soft()
                and d_total == 17
            )
            # El crupier se planta en 17+ (o en Soft 17 si aplica la regla).
            if d_total >= 17 or stands_soft17:
                tag = "BUST" if state.dealer_hand.is_bust() else "STAND"
                print(f"  → Crupier {tag} ({d_total})")
                break
            new_card = _ask_card("Crupier pide carta")
            state.dealer_hand.add(new_card)
            print(f"  Crupier: {state.dealer_hand}")

    # ── Resolución y log de cada sub-mano ─────────────────────────────────────
    state.phase = Phase.RESOLVED
    print()
    for i, (hand, hrec, htaken, hdoubled) in enumerate(sub_results, 1):
        state.player_hand = hand
        # resolve_hand() acepta una mano específica y la evalúa contra el crupier actual.
        outcome, delta = state.resolve_hand(hand, doubled=hdoubled, is_split=True)
        state.bankroll += delta
        sign = "+" if delta >= 0 else ""
        print(f"  Mano {i}: {outcome.value.upper()}  {sign}${delta:.2f}")
        display.show_outcome(outcome, delta)

        # El log de la mano 1 incluye el SPLIT como primera acción recomendada/tomada.
        log_rec  = [first_rec] + hrec  if i == 1 else hrec
        log_take = [Action.SPLIT] + htaken if i == 1 else htaken
        logger.log(state, log_rec, log_take, outcome, delta)

    print(f"\n  Banca: ${state.bankroll:.2f}")
    input("  (enter para continuar) ")


def _parse_hand(raw: str) -> Hand:
    """Convierte texto a una mano.
    Ejemplo: "A 8" → Hand([Card('A'), Card('8')])
    .upper() convierte a mayúsculas por si el usuario escribe 'a 8' en vez de 'A 8'.
    """
    return Hand([Card(r.upper()) for r in raw.strip().split() if r])


def _ask_action(rec: Action) -> Action:
    """Muestra el menú de acciones y espera que el jugador elija una.
    Si el jugador pulsa Enter sin escribir nada, usa la acción recomendada como default.

    Devuelve la Action elegida."""
    keys = "h=hit  s=stand  d=double  sp=split  su=surrender"
    default_key = _KEY_MAP[rec].upper()   # tecla de la acción recomendada en mayúsculas
    raw = input(f"  Acción [{keys}] (enter={default_key}): ").strip().lower()
    # _ACTION_MAP.get(raw, rec): si raw no está en el mapa, usa rec (la recomendación).
    return _ACTION_MAP.get(raw, rec)


def _ask_card(prompt: str) -> Card:
    """Pide al usuario que escriba el rango de una carta.
    Ejemplo: si el usuario escribe "7", devuelve Card('7').
    .split()[0] toma solo la primera palabra (por si escribe "7 de corazones")."""
    raw = input(f"  {prompt}: ").strip().upper().split()[0]
    return Card(raw)


def _print_state(state: GameState, rec: Action | None) -> None:
    """Imprime el estado actual en el terminal (útil para ver sin ventana visual)."""
    p = state.player_hand
    d = state.dealer_hand
    p_str = ' '.join(str(c) for c in p.cards) or "—"
    d_str = ' '.join(str(c) for c in d.cards) or "—"
    soft  = " (soft)" if p.is_soft() else ""
    bust  = "  ¡BUST!" if p.is_bust() else ""
    print(f"\n  Jugador : {p_str}  = {p.total()}{soft}{bust}")
    print(f"  Crupier : {d_str}")
    if rec:
        print(f"  ► {rec.value.upper()}")  # ► es un símbolo de "señal" visual


def play_hand(state: GameState, display: Display, logger: HandLogger) -> None:
    """Juega una mano completa en modo interactivo.

    Flujo:
    1. Pide la apuesta
    2. Pide las cartas del reparto inicial (carta del crupier + cartas del jugador)
    3. Turno del jugador (con recomendaciones de la estrategia)
    4. Turno del crupier (si aplica)
    5. Resolución y log
    """
    recommended: list[Action] = []
    taken:       list[Action] = []

    print(f"\n{SEPARATOR}")
    print(f"  Banca: ${state.bankroll:.2f}")

    # ── Apuesta ────────────────────────────────────────────────────────────────
    raw_bet  = input("  Apuesta (enter=10): ").strip()
    state.bet = float(raw_bet) if raw_bet else 10.0   # default 10 si el usuario da Enter

    # ── Reparto inicial ────────────────────────────────────────────────────────
    dealer_up = _ask_card("Carta visible del crupier")
    # El crupier empieza con su carta visible + una carta tapada (BACK).
    state.dealer_hand = Hand([dealer_up, Card('BACK')])
    state.player_hand = _parse_hand(input("  Tus cartas (ej: A 8): "))
    state.phase = Phase.PLAYER_TURN

    # ── Turno del jugador ──────────────────────────────────────────────────────
    first = True   # controla si podemos doblar/dividir/rendirse (solo en la primera decisión)
    while True:
        # Calculamos las flags de disponibilidad para esta decisión.
        row = full_row(
            state.player_hand,
            can_split=first and state.player_hand.is_pair(),   # solo si es par Y primera decisión
            can_double=first,
            can_surrender=first,
        )
        rec = recommend(
            state.player_hand, dealer_up,
            can_split=first and state.player_hand.is_pair(),
            can_double=first,
            can_surrender=first,
        )
        recommended.append(rec)
        display.show(state, rec, strategy_row=row, dealer_upcard_rank=dealer_up.rank)
        _print_state(state, rec)

        # Si llegamos a 21 o hicimos bust, el turno termina automáticamente.
        if state.player_hand.is_bust() or state.player_hand.total() >= 21:
            break

        action = _ask_action(rec)
        taken.append(action)
        first = False

        if action == Action.STAND:
            break

        elif action == Action.SURRENDER:
            state.surrendered = True
            break   # rendirse termina inmediatamente el turno del jugador

        elif action == Action.DOUBLE:
            state.doubled = True
            state.player_hand.add(_ask_card("Nueva carta"))
            display.show(state, None)
            _print_state(state, None)
            break   # double solo permite 1 carta más

        elif action == Action.SPLIT:
            # El split es complejo: delegamos a _handle_split() y salimos de play_hand.
            _handle_split(state, dealer_up, display, logger, rec)
            return  # el log se hace dentro de _handle_split, salimos directamente

        elif action == Action.HIT:
            state.player_hand.add(_ask_card("Nueva carta"))

    # ── Turno del crupier ──────────────────────────────────────────────────────
    # El crupier solo juega si el jugador no hizo bust ni se rindió.
    if not state.player_hand.is_bust() and not state.surrendered:
        state.phase = Phase.DEALER_TURN
        print(f"\n  --- Turno del crupier ---")
        hidden = _ask_card("Carta tapada del crupier")   # revelamos la carta oculta
        state.dealer_hand = Hand([dealer_up, hidden])
        print(f"  Crupier: {state.dealer_hand}")

        # El crupier pide cartas mientras tenga menos de 17.
        while True:
            d_total = state.dealer_hand.total()
            stands_soft17 = (
                config.DEALER_STAND_ON_SOFT_17
                and state.dealer_hand.is_soft()
                and d_total == 17
            )
            if d_total >= 17 or stands_soft17:
                action_str = "BUST" if state.dealer_hand.is_bust() else "STAND"
                print(f"  → Crupier {action_str} ({d_total})")
                break
            new_card = _ask_card("Crupier pide carta")
            state.dealer_hand.add(new_card)
            print(f"  Crupier: {state.dealer_hand}")

    # ── Resolución ─────────────────────────────────────────────────────────────
    state.phase = Phase.RESOLVED
    outcome, delta = state.resolve()
    state.bankroll += delta
    sign = "+" if delta >= 0 else ""
    print(f"\n  {SEPARATOR}")
    print(f"  {outcome.value.upper()}  {sign}${delta:.2f}   Banca: ${state.bankroll:.2f}")
    display.show_outcome(outcome, delta)

    # Guardamos la mano en el CSV.
    logger.log(state, recommended, taken, outcome, delta)
    input("  (enter para continuar) ")


def main() -> None:
    """Punto de entrada del modo simulación."""
    # argparse: gestiona los argumentos de la línea de comandos.
    # Permite hacer: python scripts/simulate.py --bankroll 500
    parser = argparse.ArgumentParser(description="Blackjack CV — Modo Simulación")
    parser.add_argument("--bankroll", type=float, default=config.STARTING_BANKROLL)
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║     Blackjack CV — Modo Simulación       ║")
    print("║  Rangos válidos: A 2-10 J Q K BACK       ║")
    print("╚══════════════════════════════════════════╝")

    display = Display()
    logger  = HandLogger(config.LOG_FILE)
    state   = GameState(bankroll=args.bankroll)

    # try/except: maneja interrupciones del usuario (Ctrl+C o fin de stdin).
    try:
        while True:
            play_hand(state, display, logger)

            # Reiniciamos las manos (pero conservamos bankroll, apuesta y logger).
            state.player_hand  = Hand()
            state.dealer_hand  = Hand()
            state.doubled      = False
            state.surrendered  = False
            state.phase        = Phase.WAITING_BET

            again = input("\n  ¿Otra mano? (s/n): ").strip().lower()
            if again != 's':
                break
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C (KeyboardInterrupt) o fin de stdin (EOFError en tests/pipes) → salir limpiamente.
        pass

    display.close()
    print(f"\nSesión terminada. Banca final: ${state.bankroll:.2f}")
    print(f"Log guardado en: {config.LOG_FILE}")


if __name__ == '__main__':
    main()
