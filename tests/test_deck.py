from src.game.deck import Deck


def test_deck_has_52_cards():
    assert len(Deck()) == 52


def test_draw_decreases_count():
    d = Deck()
    d.draw()
    assert len(d) == 51


def test_reproducible_with_seed():
    d1 = Deck(seed=42)
    d2 = Deck(seed=42)
    assert [d1.draw() for _ in range(5)] == [d2.draw() for _ in range(5)]
