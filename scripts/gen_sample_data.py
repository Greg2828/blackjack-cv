"""
ARCHIVO: scripts/gen_sample_data.py
PROPÓSITO: Simula partidas de blackjack COMPLETAMENTE AUTOMÁTICAS
           siguiendo la basic strategy al 100% y guarda los resultados en el CSV.

           A diferencia de simulate.py (donde tú introduces las cartas),
           aquí el ordenador juega solo: reparte cartas de una baraja virtual,
           ejecuta la basic strategy automáticamente y registra el resultado.

           Esto es útil para:
           - Poblar el CSV con datos antes de jugar con la cámara
           - Verificar que el EV es cercano al esperado teóricamente
           - Probar las funciones de estadísticas (stats.py) con datos reales

Uso:
    python scripts/gen_sample_data.py              # 250 manos
    python scripts/gen_sample_data.py --hands 1000 # 1000 manos
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
    """Simula una mano completa de blackjack de forma automática.

    Proceso:
    1. Reparte 4 cartas: jugador1, crupier-visible, jugador2, crupier-tapada
    2. Si el jugador tiene blackjack natural → resolver directamente
    3. Turno del jugador: ejecutar basic strategy automáticamente
    4. Turno del crupier: pedir cartas hasta llegar a 17+
    5. Resolver y devolver el resultado

    Parámetros:
      deck: la baraja de la que se sacan las cartas
      bankroll: dinero actual del jugador
      bet: apuesta de esta mano

    Devuelve: (estado_final, acciones_recomendadas, acciones_tomadas, resultado, delta)
    """
    state = GameState(bankroll=bankroll, bet=bet)

    # Repartimos las 4 cartas iniciales en el orden real del blackjack:
    # jugador carta 1, crupier carta visible, jugador carta 2, crupier carta tapada.
    p1, d_up, p2, d_hid = deck.draw(), deck.draw(), deck.draw(), deck.draw()
    state.player_hand = Hand([p1, p2])
    # El crupier empieza con su carta visible + BACK (tapada).
    state.dealer_hand = Hand([d_up, Card('BACK')])

    recommended: list[Action] = []
    taken:       list[Action] = []

    # ── Blackjack natural ──────────────────────────────────────────────────────
    # Si el jugador tiene 21 en 2 cartas, la mano se resuelve inmediatamente.
    # El crupier revela su carta tapada para verificar si también tiene blackjack.
    if state.player_hand.is_blackjack():
        state.dealer_hand = Hand([d_up, d_hid])   # revelamos la carta tapada
        outcome, delta = state.resolve()
        state.bankroll += delta
        return state, recommended, taken, outcome, delta

    # ── Turno del jugador ──────────────────────────────────────────────────────
    # La basic strategy decide automáticamente cada acción.
    first = True  # para saber si podemos doblar/dividir/rendirse
    while True:
        # Consultamos la basic strategy.
        rec = recommend(
            state.player_hand, d_up,
            can_split=first and state.player_hand.is_pair(),
            can_double=first,
            can_surrender=first,
        )
        recommended.append(rec)
        taken.append(rec)   # en simulación automática, siempre hacemos exactamente lo recomendado
        first = False

        if rec == Action.STAND:
            break   # nos plantamos

        if rec == Action.SURRENDER:
            state.surrendered = True
            break   # nos rendimos

        if rec == Action.DOUBLE:
            state.doubled = True
            state.player_hand.add(deck.draw())   # recibimos exactamente 1 carta
            break   # el double termina el turno del jugador

        if rec in (Action.HIT, Action.SPLIT):
            # En esta simulación simplificada, el SPLIT se trata como HIT:
            # pedimos una carta más (no dividimos en dos manos independientes).
            # Esto es una simplificación para gen_sample_data — simulate.py hace el split real.
            state.player_hand.add(deck.draw())

        # Terminamos si hicimos bust o llegamos a 21.
        if state.player_hand.is_bust() or state.player_hand.total() >= 21:
            break

    # ── Turno del crupier ──────────────────────────────────────────────────────
    # Revelamos la carta tapada del crupier.
    state.dealer_hand = Hand([d_up, d_hid])

    # El crupier solo juega si el jugador no hizo bust ni se rindió.
    if not state.player_hand.is_bust() and not state.surrendered:
        # El crupier pide cartas mientras tenga menos de 17.
        # (Regla simple: no implementamos Soft 17 aquí para mantenerlo simple).
        while state.dealer_hand.total() < 17:
            state.dealer_hand.add(deck.draw())

    # ── Resolución ─────────────────────────────────────────────────────────────
    outcome, delta = state.resolve()
    state.bankroll += delta
    return state, recommended, taken, outcome, delta


def main() -> None:
    """Ejecuta múltiples manos automáticas y las guarda en el CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands',    type=int,   default=250,
                        help='Número de manos a simular')
    parser.add_argument('--bet',      type=float, default=10.0,
                        help='Apuesta base por mano')
    parser.add_argument('--bankroll', type=float, default=config.STARTING_BANKROLL)
    parser.add_argument('--seed',     type=int,   default=42,
                        help='Semilla para reproducibilidad')
    args = parser.parse_args()

    logger   = HandLogger(config.LOG_FILE)
    # Baraja de 6 mazos (estándar en casinos), con semilla para reproducibilidad.
    deck     = Deck(n_decks=6, seed=args.seed)
    bankroll = args.bankroll
    rng      = random.Random(args.seed)   # generador separado para la variación de apuesta

    wins = pushes = losses = 0

    for i in range(args.hands):
        # Si quedan menos de 30 cartas en la baraja, creamos una nueva.
        # Esto simula el "reshuffling" que hacen los casinos.
        if len(deck) < 30:
            deck = Deck(n_decks=6, seed=rng.randint(0, 9999))

        # Variación ligera de apuesta: 75% de las veces usamos la apuesta base,
        # 25% de las veces usamos el doble. Simula un jugador con ligera variación.
        # rng.choice([1, 1, 1, 2]) → elige uno de estos 4 valores con igual probabilidad.
        bet = args.bet * rng.choice([1, 1, 1, 2])

        state, rec, taken, outcome, delta = _simulate_hand(deck, bankroll, bet)
        bankroll = state.bankroll
        logger.log(state, rec, taken, outcome, delta)

        # Acumulamos estadísticas para el resumen final.
        if outcome in (Outcome.WIN, Outcome.BLACKJACK):
            wins += 1
        elif outcome == Outcome.PUSH:
            pushes += 1
        else:
            losses += 1

    # Resumen en el terminal.
    total = wins + pushes + losses
    print(f"{total} manos simuladas → {wins}W / {pushes}P / {losses}L")
    print(f"Win rate : {wins/total*100:.1f}%")
    print(f"Bankroll : ${args.bankroll:.0f} → ${bankroll:.0f}  (delta {bankroll-args.bankroll:+.0f})")
    print(f"Log      : {config.LOG_FILE}")


if __name__ == '__main__':
    main()
