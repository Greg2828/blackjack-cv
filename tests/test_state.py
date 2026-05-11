import pytest
from src.game.card import Card
from src.game.hand import Hand
from src.game.state import GameState, Outcome


def _state(player: list[str], dealer: list[str], bet: float = 10.0) -> GameState:
    s = GameState(bet=bet, bankroll=100.0)
    s.player_hand = Hand([Card(r) for r in player])
    s.dealer_hand = Hand([Card(r) for r in dealer])
    return s


class TestResolve:
    def test_player_bust(self):
        s = _state(['K', 'Q', '5'], ['8', '7'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.LOSE
        assert delta == -10.0

    def test_dealer_bust(self):
        s = _state(['K', '8'], ['K', 'Q', '5'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.WIN
        assert delta == 10.0

    def test_player_higher(self):
        s = _state(['K', '9'], ['K', '8'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.WIN
        assert delta == 10.0

    def test_dealer_higher(self):
        s = _state(['K', '7'], ['K', '8'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.LOSE
        assert delta == -10.0

    def test_push(self):
        s = _state(['K', '8'], ['Q', '8'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.PUSH
        assert delta == 0.0

    def test_player_blackjack(self):
        s = _state(['A', 'K'], ['Q', '8'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.BLACKJACK
        assert delta == pytest.approx(15.0)

    def test_both_blackjack_push(self):
        s = _state(['A', 'K'], ['A', 'Q'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.PUSH
        assert delta == 0.0

    def test_dealer_blackjack_player_loses(self):
        s = _state(['K', '8'], ['A', 'Q'])
        outcome, delta = s.resolve()
        assert outcome == Outcome.LOSE
        assert delta == -10.0

    def test_surrender(self):
        s = _state(['K', '6'], ['A', '5'])
        s.surrendered = True
        outcome, delta = s.resolve()
        assert outcome == Outcome.LOSE
        assert delta == -5.0

    def test_double_win(self):
        s = _state(['5', '6', 'K'], ['K', '6'])
        s.doubled = True
        outcome, delta = s.resolve()
        assert outcome == Outcome.WIN
        assert delta == 20.0

    def test_double_lose(self):
        s = _state(['5', '6', '3'], ['K', '6'])
        s.doubled = True
        outcome, delta = s.resolve()
        assert outcome == Outcome.LOSE
        assert delta == -20.0


class TestResolveHand:
    def test_split_21_is_not_blackjack(self):
        """Tras un split, A+10 cuenta como 21 normal, no blackjack natural."""
        s = _state(['K', '5'], ['K', '7'])
        hand = Hand([Card('A'), Card('Q')])
        outcome, delta = s.resolve_hand(hand, is_split=True)
        assert outcome == Outcome.WIN
        assert delta == 10.0   # pago 1:1, no 1.5:1

    def test_split_hand_bust(self):
        s = _state(['K', '5'], ['K', '8'])
        hand = Hand([Card('K'), Card('Q'), Card('5')])
        outcome, delta = s.resolve_hand(hand, is_split=True)
        assert outcome == Outcome.LOSE
        assert delta == -10.0

    def test_split_hand_wins(self):
        s = _state(['K', '5'], ['K', '6'])
        hand = Hand([Card('K'), Card('9')])
        outcome, delta = s.resolve_hand(hand, is_split=True)
        assert outcome == Outcome.WIN
        assert delta == 10.0

    def test_split_hand_push(self):
        s = _state(['K', '5'], ['K', '8'])
        hand = Hand([Card('K'), Card('8')])
        outcome, delta = s.resolve_hand(hand, is_split=True)
        assert outcome == Outcome.PUSH
        assert delta == 0.0

    def test_split_hand_doubled(self):
        s = _state(['K', '5'], ['K', '6'])
        hand = Hand([Card('K'), Card('A')])
        outcome, delta = s.resolve_hand(hand, doubled=True, is_split=True)
        assert outcome == Outcome.WIN
        assert delta == 20.0

    def test_split_dealer_blackjack(self):
        """El crupier tiene blackjack: todas las manos de split pierden."""
        s = _state(['8', '8'], ['A', 'K'])
        hand = Hand([Card('8'), Card('K')])
        outcome, delta = s.resolve_hand(hand, is_split=True)
        assert outcome == Outcome.LOSE
        assert delta == -10.0

    def test_no_split_flag_preserves_blackjack(self):
        """Sin is_split=True, A+10 sigue siendo blackjack natural."""
        s = _state(['A', 'K'], ['K', '7'])
        hand = Hand([Card('A'), Card('K')])
        outcome, delta = s.resolve_hand(hand, is_split=False)
        assert outcome == Outcome.BLACKJACK
        assert delta == pytest.approx(15.0)
