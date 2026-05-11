"""
Modo simulación — introduce cartas manualmente por teclado.
Permite probar el pipeline completo (estrategia, display, log) sin cámara.

Uso:
    python scripts/simulate.py
    python scripts/simulate.py --bankroll 200
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
os.environ.setdefault('QT_LOGGING_RULES', '*.debug=false;qt.qpa.*=false')

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.card import Card
from src.game.hand import Hand
from src.game.state import GameState, Phase, Action, Outcome
from src.decision.strategy import full_row
from src.decision.strategy import recommend
from src.analysis.logger import HandLogger
from src.ui.display import Display
import config

_ACTION_MAP = {
    'h':  Action.HIT,
    's':  Action.STAND,
    'd':  Action.DOUBLE,
    'sp': Action.SPLIT,
    'su': Action.SURRENDER,
}
_KEY_MAP = {v: k for k, v in _ACTION_MAP.items()}

SEPARATOR = "─" * 46


def _parse_hand(raw: str) -> Hand:
    return Hand([Card(r.upper()) for r in raw.strip().split() if r])


def _ask_action(rec: Action) -> Action:
    keys = "h=hit  s=stand  d=double  sp=split  su=surrender"
    default_key = _KEY_MAP[rec].upper()
    raw = input(f"  Acción [{keys}] (enter={default_key}): ").strip().lower()
    return _ACTION_MAP.get(raw, rec)


def _ask_card(prompt: str) -> Card:
    raw = input(f"  {prompt}: ").strip().upper().split()[0]
    return Card(raw)


def _print_state(state: GameState, rec: Action | None) -> None:
    p = state.player_hand
    d = state.dealer_hand
    p_str = ' '.join(str(c) for c in p.cards) or "—"
    d_str = ' '.join(str(c) for c in d.cards) or "—"
    soft = " (soft)" if p.is_soft() else ""
    bust = "  ¡BUST!" if p.is_bust() else ""
    print(f"\n  Jugador : {p_str}  = {p.total()}{soft}{bust}")
    print(f"  Crupier : {d_str}")
    if rec:
        print(f"  ► {rec.value.upper()}")


def play_hand(state: GameState, display: Display, logger: HandLogger) -> None:
    recommended: list[Action] = []
    taken:       list[Action] = []

    print(f"\n{SEPARATOR}")
    print(f"  Banca: ${state.bankroll:.2f}")

    # --- Apuesta ---
    raw_bet = input("  Apuesta (enter=10): ").strip()
    state.bet = float(raw_bet) if raw_bet else 10.0

    # --- Reparto inicial ---
    dealer_up = _ask_card("Carta visible del crupier")
    state.dealer_hand = Hand([dealer_up, Card('BACK')])
    state.player_hand = _parse_hand(input("  Tus cartas (ej: A 8): "))
    state.phase = Phase.PLAYER_TURN

    # --- Turno del jugador ---
    first = True
    while True:
        row = full_row(
            state.player_hand,
            can_split=first and state.player_hand.is_pair(),
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
        display.show(state, rec,
                     strategy_row=row,
                     dealer_upcard_rank=dealer_up.rank)
        _print_state(state, rec)

        if state.player_hand.is_bust() or state.player_hand.total() >= 21:
            break

        action = _ask_action(rec)
        taken.append(action)
        first = False

        if action == Action.STAND:
            break
        elif action == Action.SURRENDER:
            state.surrendered = True
            break
        elif action == Action.DOUBLE:
            state.doubled = True
            state.player_hand.add(_ask_card("Nueva carta"))
            display.show(state, None)
            _print_state(state, None)
            break
        elif action == Action.SPLIT:
            # Simplified: continue playing as a single hand
            print("  (Split registrado — continúa jugando la mano normalmente)")
        elif action == Action.HIT:
            state.player_hand.add(_ask_card("Nueva carta"))

    # --- Turno del crupier ---
    if not state.player_hand.is_bust() and not state.surrendered:
        state.phase = Phase.DEALER_TURN
        print(f"\n  --- Turno del crupier ---")
        hidden = _ask_card("Carta tapada del crupier")
        state.dealer_hand = Hand([dealer_up, hidden])
        print(f"  Crupier: {state.dealer_hand}")

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

    # --- Resolución ---
    state.phase = Phase.RESOLVED
    outcome, delta = state.resolve()
    state.bankroll += delta
    sign = "+" if delta >= 0 else ""
    print(f"\n  {SEPARATOR}")
    print(f"  {outcome.value.upper()}  {sign}${delta:.2f}   Banca: ${state.bankroll:.2f}")
    display.show_outcome(outcome, delta)

    logger.log(state, recommended, taken, outcome, delta)
    input("  (enter para continuar) ")


def main() -> None:
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

    try:
        while True:
            play_hand(state, display, logger)
            state.player_hand  = Hand()
            state.dealer_hand  = Hand()
            state.doubled      = False
            state.surrendered  = False
            state.phase        = Phase.WAITING_BET
            again = input("\n  ¿Otra mano? (s/n): ").strip().lower()
            if again != 's':
                break
    except (KeyboardInterrupt, EOFError):
        pass

    display.close()
    print(f"\nSesión terminada. Banca final: ${state.bankroll:.2f}")
    print(f"Log guardado en: {config.LOG_FILE}")


if __name__ == '__main__':
    main()
