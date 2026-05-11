import random
from .card import Card

RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


class Deck:
    """Baraja simulada de blackjack. Solo se usa en simulaciones y tests.
    En la mesa real, las cartas vienen de la cámara."""

    def __init__(self, n_decks: int = 1, seed: int | None = None):
        self.rng = random.Random(seed)
        self._cards: list[Card] = []
        for _ in range(n_decks):
            for rank in RANKS:
                for _ in range(4):
                    self._cards.append(Card(rank))
        self.shuffle()

    def shuffle(self) -> None:
        self.rng.shuffle(self._cards)

    def draw(self) -> Card:
        if not self._cards:
            raise IndexError("Baraja vacía")
        return self._cards.pop()

    def __len__(self) -> int:
        return len(self._cards)
