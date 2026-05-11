from src.game.card import Card
from src.game.hand import Hand


def test_empty_hand_totals_zero():
    assert Hand().total() == 0


def test_simple_sum():
    h = Hand([Card('7'), Card('5')])
    assert h.total() == 12
    assert not h.is_soft()


def test_ace_as_11_when_safe():
    h = Hand([Card('A'), Card('6')])
    assert h.total() == 17
    assert h.is_soft()


def test_ace_drops_to_1_when_needed():
    h = Hand([Card('A'), Card('6'), Card('10')])
    assert h.total() == 17
    assert not h.is_soft()


def test_two_aces():
    h = Hand([Card('A'), Card('A')])
    assert h.total() == 12
    assert h.is_soft()


def test_blackjack_natural():
    h = Hand([Card('A'), Card('K')])
    assert h.is_blackjack()
    assert h.total() == 21


def test_21_with_three_cards_is_not_blackjack():
    h = Hand([Card('7'), Card('7'), Card('7')])
    assert h.total() == 21
    assert not h.is_blackjack()


def test_bust():
    h = Hand([Card('10'), Card('K'), Card('5')])
    assert h.is_bust()


def test_pair():
    assert Hand([Card('8'), Card('8')]).is_pair()
    assert not Hand([Card('8'), Card('9')]).is_pair()


def test_hidden_card_ignored_in_total():
    h = Hand([Card('10'), Card('BACK')])
    assert h.total() == 10
    assert h.has_hidden
