"""
ARCHIVO: src/game/hand.py
PROPÓSITO: Define qué es una MANO de cartas.
           Una mano es una lista de cartas con capacidad para calcular
           su suma total, detectar si es "soft", bust, blackjack o par.
           Toda la lógica matemática del blackjack vive aquí.

CÓMO SE CONECTA:
  - state.py usa Hand para crear player_hand y dealer_hand
  - strategy.py recibe player_hand para decidir qué acción recomendar
  - simulate.py crea manos y las va modificando durante la partida
  - detector.py crea Hand con las cartas detectadas por la cámara
"""

# dataclass: igual que en card.py, genera código automático para la clase.
# field: permite definir valores por defecto complejos (como una lista vacía).
from dataclasses import dataclass, field

# Importamos Card desde el archivo card.py que está en la misma carpeta (game/).
# El punto en '.card' significa "busca card.py en el mismo paquete (carpeta)".
from .card import Card


# @dataclass sin 'frozen=True' → la mano SÍ puede modificarse.
# Tiene sentido: durante una partida añades cartas a la mano con add().
@dataclass
class Hand:
    """Una mano de cartas. Maneja soft/hard, bust y blackjack."""

    # 'cards' es la lista de cartas de esta mano.
    # field(default_factory=list) significa que el valor por defecto es [] (lista vacía).
    # No podemos escribir simplemente 'cards: list[Card] = []' porque en Python
    # las listas como valor por defecto se comparten entre instancias (bug clásico).
    # default_factory crea una lista NUEVA para cada Hand que instancies.
    cards: list[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        """Añade una carta a la mano. Se llama cuando el jugador pide carta (HIT)
        o cuando el crupier pide carta para su turno.
        'None' como tipo de retorno significa que esta función no devuelve nada."""
        self.cards.append(card)  # .append() añade el elemento al final de la lista

    @property
    def visible_cards(self) -> list[Card]:
        """Devuelve solo las cartas visibles (excluye las tapadas/BACK).
        [c for c in self.cards if not c.is_back] es una "list comprehension":
        crea una nueva lista incluyendo solo las cartas donde c.is_back es False.
        Se usa para calcular el total SIN contar la carta tapada del crupier."""
        return [c for c in self.cards if not c.is_back]

    @property
    def has_hidden(self) -> bool:
        """Devuelve True si hay al menos una carta tapada en la mano.
        'any(...)' devuelve True si al menos uno de los elementos evaluados es True.
        Se usa en main.py para determinar si todavía es el turno del jugador
        (el crupier tiene una carta tapada durante el turno del jugador)."""
        return any(c.is_back for c in self.cards)

    def total(self) -> int:
        """Suma óptima: ases cuentan como 11 si caben, si no como 1.
        Las cartas tapadas no se cuentan.

        Algoritmo:
        1. Suma todas las cartas visibles (los ases empiezan contando como 11).
        2. Si la suma pasa de 21 y hay ases, baja un as de 11→1 (resta 10).
        3. Repite hasta que la suma sea ≤21 o no queden ases que bajar."""

        cards = self.visible_cards  # descartamos cartas tapadas

        # Sumamos todos los valores. Recuerda: Card.value ya devuelve 11 para ases.
        total = sum(c.value for c in cards)

        # Contamos cuántos ases hay (porque cada uno puede bajar de 11 a 1).
        n_aces = sum(1 for c in cards if c.is_ace)

        # Mientras nos pasemos de 21 Y todavía tengamos ases "contando como 11",
        # bajamos uno de ellos de 11 a 1 (diferencia = 10).
        # Ejemplo: As + As + 9 = 11+11+9 = 31 → bajamos un As → 1+11+9 = 21 ✓
        while total > 21 and n_aces > 0:
            total  -= 10   # bajamos el as de 11 a 1 (diferencia 10)
            n_aces -= 1    # ya no podemos bajar ese as otra vez

        return total

    def is_soft(self) -> bool:
        """Soft: la mano contiene un As contado como 11 ahora mismo.
        Ejemplo: As + 7 = Soft 18 (el as vale 11).
        Ejemplo: As + 7 + 5 = Hard 13 (el as tuvo que bajar a 1 para no pasarse).

        Técnica: calculamos el total "duro" (todos los ases valen 1)
        y lo comparamos con el total real (total()). Si son distintos
        significa que algún as está contando como 11 → la mano es soft."""

        cards = self.visible_cards
        # Si no hay ningún as en la mano, no puede ser soft.
        if not any(c.is_ace for c in cards):
            return False

        # Total "duro": todos los ases valen 1 (en vez de 11).
        # Para cada carta: si es as usamos 1, si no usamos su valor normal.
        hard_total = sum(1 if c.is_ace else c.value for c in cards)

        # Si total() (inteligente) es distinto del total duro,
        # significa que algún as está contando como 11 → es soft.
        return self.total() != hard_total

    def is_bust(self) -> bool:
        """Bust = la suma pasa de 21 → el jugador pierde automáticamente.
        No hace falta que el crupier juegue su mano si el jugador hizo bust."""
        return self.total() > 21

    def is_blackjack(self) -> bool:
        """21 con las DOS primeras cartas exactas (sin tapadas).
        Blackjack natural = As + carta de 10 puntos (10, J, Q, K) en la apertura.
        No es lo mismo que llegar a 21 con más cartas (eso no paga 3:2)."""
        # Debe tener exactamente 2 cartas visibles (no tapadas) que sumen 21.
        return len(self.visible_cards) == 2 and self.total() == 21

    def is_pair(self) -> bool:
        """Devuelve True si la mano tiene exactamente 2 cartas del mismo rango.
        Solo se puede hacer SPLIT (división) si hay un par, y solo en la primera decisión.
        Ejemplo: 8+8, As+As, K+K son pares válidos para split."""
        # len(self.cards) == 2: exactamente 2 cartas en total (no solo visibles)
        # self.cards[0].rank == self.cards[1].rank: ambas del mismo rango
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    def __str__(self) -> str:
        """Representación en texto de la mano, útil para el modo simulación.
        Ejemplo: str(Hand([Card('A'), Card('8')])) → '[A, 8] = 19'"""
        cards_str = ', '.join(str(c) for c in self.cards)  # une con comas
        return f"[{cards_str}] = {self.total()}"
