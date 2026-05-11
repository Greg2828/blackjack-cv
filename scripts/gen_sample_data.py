"""
Simula partidas siguiendo basic strategy al 100% y guarda los resultados
en el CSV de log. Útil para poblar datos antes de jugar con la cámara real.

Uso:
    python scripts/gen_sample_data.py              # 5 sesiones × 50 manos
    python scripts/gen_sample_data.py --hands 200  # 200 manos en una sesión
"""
import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.game.card import Card
from src.game.hand import Hand
from src.game.state import GameState, Action, Outcome
from src.game.deck import Deck
from src.decision.strategy import recommend
from src.analysis.logger import HandLogger
import config


def _simulate_hand(deck: Deck, bankroll: float, bet: float
                   ) -> tuple[GameState, list[Action], list[Action], Outcome, float]:
    state = GameState(bankroll=bankroll, bet=bet)

    p1, d_up, p2, d_hid = deck.draw(), deck.draw(), deck.draw(), deck.draw()
    state.player_hand = Hand([p1, p2])
    state.dealer_hand = Hand([d_up, Card('BACK')])

    recommended: list[Action] = []
    taken:       list[Action] = []

    # Blackjack natural — resolución inmediata
    if state.player_hand.is_blackjack():
        state.dealer_hand = Hand([d_up, d_hid])
        outcome, delta = state.resolve()
        state.bankroll += delta
        return state, recommended, taken, outcome, delta

    # Turno del jugador — siempre sigue la estrategia óptima
    first = True
    while True:
        rec = recommend(
            state.player_hand, d_up,
            can_split=first and state.player_hand.is_pair(),
            can_double=first,
            can_surrender=first,
        )
        recommended.append(rec)
        taken.append(rec)
        first = False

        if rec == Action.STAND:
            break
        if rec == Action.SURRENDER:
            state.surrendered = True
            break
        if rec == Action.DOUBLE:
            state.doubled = True
            state.player_hand.add(deck.draw())
            break
        if rec in (Action.HIT, Action.SPLIT):
            state.player_hand.add(deck.draw())

        if state.player_hand.is_bust() or state.player_hand.total() >= 21:
            break

    # Turno del crupier
    state.dealer_hand = Hand([d_up, d_hid])
    if not state.player_hand.is_bust() and not state.surrendered:
        while state.dealer_hand.total() < 17:
            state.dealer_hand.add(deck.draw())

    outcome, delta = state.resolve()
    state.bankroll += delta
    return state, recommended, taken, outcome, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands',    type=int,   default=250)
    parser.add_argument('--bet',      type=float, default=10.0)
    parser.add_argument('--bankroll', type=float, default=config.STARTING_BANKROLL)
    parser.add_argument('--seed',     type=int,   default=42)
    args = parser.parse_args()

    logger   = HandLogger(config.LOG_FILE)
    deck     = Deck(n_decks=6, seed=args.seed)
    bankroll = args.bankroll
    rng      = random.Random(args.seed)

    wins = pushes = losses = 0

    for i in range(args.hands):
        if len(deck) < 30:
            deck = Deck(n_decks=6, seed=rng.randint(0, 9999))

        bet = args.bet * rng.choice([1, 1, 1, 2])  # variación ligera de apuesta
        state, rec, taken, outcome, delta = _simulate_hand(deck, bankroll, bet)
        bankroll = state.bankroll
        logger.log(state, rec, taken, outcome, delta)

        if outcome in (Outcome.WIN, Outcome.BLACKJACK):
            wins += 1
        elif outcome == Outcome.PUSH:
            pushes += 1
        else:
            losses += 1

    total = wins + pushes + losses
    print(f"{total} manos simuladas → {wins}W / {pushes}P / {losses}L")
    print(f"Win rate : {wins/total*100:.1f}%")
    print(f"Bankroll : ${args.bankroll:.0f} → ${bankroll:.0f}  (delta {bankroll-args.bankroll:+.0f})")
    print(f"Log      : {config.LOG_FILE}")


if __name__ == '__main__':
    main()
