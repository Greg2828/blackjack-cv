from dataclasses import dataclass, field
from .card import Card


@dataclass
class Hand:
    """Una mano de cartas. Maneja soft/hard, bust y blackjack."""
    cards: list[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        self.cards.append(card)

    @property
    def visible_cards(self) -> list[Card]:
        return [c for c in self.cards if not c.is_back]

    @property
    def has_hidden(self) -> bool:
        return any(c.is_back for c in self.cards)

    def total(self) -> int:
        """Suma óptima: ases cuentan como 11 si caben, si no como 1.
        Las cartas tapadas no se cuentan."""
        cards = self.visible_cards
        total = sum(c.value for c in cards)
        n_aces = sum(1 for c in cards if c.is_ace)
        while total > 21 and n_aces > 0:
            total -= 10
            n_aces -= 1
        return total

    def is_soft(self) -> bool:
        """Soft: la mano contiene un As contado como 11 ahora mismo."""
        cards = self.visible_cards
        if not any(c.is_ace for c in cards):
            return False
        hard_total = sum(1 if c.is_ace else c.value for c in cards)
        return self.total() != hard_total

    def is_bust(self) -> bool:
        return self.total() > 21

    def is_blackjack(self) -> bool:
        """21 con las DOS primeras cartas exactas (sin tapadas)."""
        return len(self.visible_cards) == 2 and self.total() == 21

    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    def __str__(self) -> str:
        cards_str = ', '.join(str(c) for c in self.cards)
        return f"[{cards_str}] = {self.total()}"
