import csv
import datetime
from pathlib import Path
from ..game.state import GameState, Outcome, Action

_FIELDS = [
    'timestamp', 'player_cards', 'dealer_upcard', 'dealer_final',
    'actions_recommended', 'actions_taken',
    'bet', 'outcome', 'delta', 'bankroll',
]


class HandLogger:
    def __init__(self, path: str = 'data/games_log.csv'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with open(self.path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=_FIELDS).writeheader()

    def log(
        self,
        state: GameState,
        recommended: list[Action],
        taken: list[Action],
        outcome: Outcome,
        delta: float,
    ) -> None:
        dealer_visible = state.dealer_hand.visible_cards
        row = {
            'timestamp':           datetime.datetime.now().isoformat(timespec='seconds'),
            'player_cards':        ' '.join(str(c) for c in state.player_hand.cards),
            'dealer_upcard':       str(dealer_visible[0]) if dealer_visible else '',
            'dealer_final':        ' '.join(str(c) for c in state.dealer_hand.cards),
            'actions_recommended': ' '.join(a.value for a in recommended),
            'actions_taken':       ' '.join(a.value for a in taken),
            'bet':                 state.bet,
            'outcome':             outcome.value,
            'delta':               delta,
            'bankroll':            state.bankroll,
        }
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)
