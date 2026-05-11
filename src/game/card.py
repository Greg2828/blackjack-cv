from dataclasses import dataclass


@dataclass(frozen=True)
class Card:
    """Una carta del blackjack. Sin palo (irrelevante para este juego).

    El rango 'BACK' representa una carta tapada (el dorso del crupier).
    """
    rank: str  # 'A', '2'..'10', 'J', 'Q', 'K', o 'BACK'

    @property
    def value(self) -> int:
        """Valor numérico para sumar. El As vale 11 aquí; la lógica de
        bajarlo a 1 cuando hace falta vive en Hand.total()."""
        if self.rank == 'BACK':
            raise ValueError("Una carta tapada no tiene valor conocido")
        if self.rank == 'A':
            return 11
        if self.rank in ('J', 'Q', 'K'):
            return 10
        return int(self.rank)

    @property
    def is_ace(self) -> bool:
        return self.rank == 'A'

    @property
    def is_back(self) -> bool:
        return self.rank == 'BACK'

    def __str__(self) -> str:
        return self.rank
