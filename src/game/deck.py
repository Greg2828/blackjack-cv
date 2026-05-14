"""
ARCHIVO: src/game/deck.py
PROPÓSITO: Simula una baraja (o varias barajas mezcladas) de blackjack.
           SOLO se usa en simulaciones y tests, NO en el modo con cámara real.
           Con cámara real, las cartas vienen de lo que ve la Pi Camera.

CÓMO SE CONECTA:
  - gen_sample_data.py importa Deck para simular 250+ manos automáticamente
  - tests/test_deck.py prueba que la baraja funciona correctamente
"""

import random  # módulo estándar de Python para números aleatorios

# Importamos Card desde el archivo hermano card.py (misma carpeta game/).
from .card import Card

# Los 13 rangos de una baraja estándar (sin el BACK, que es solo para cartas tapadas).
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']


class Deck:
    """Baraja simulada de blackjack. Solo se usa en simulaciones y tests.
    En la mesa real, las cartas vienen de la cámara."""

    def __init__(self, n_decks: int = 1, seed: int | None = None):
        """Constructor: se llama automáticamente al hacer Deck() o Deck(n_decks=6).

        Parámetros:
          n_decks: cuántas barajas de 52 cartas mezclar (los casinos usan 6-8).
          seed: semilla para el generador de números aleatorios.
                Si usas la misma semilla, siempre obtienes la misma secuencia de cartas.
                Útil para tests reproducibles (los tests siempre salen igual).
                None = aleatoriedad real (diferente cada vez).
        """

        # Creamos un generador de números aleatorios con su propia semilla.
        # Usamos random.Random() en vez del módulo global 'random' para que
        # diferentes barajas no interfieran entre sí.
        self.rng = random.Random(seed)

        # Lista donde guardaremos todas las cartas de la baraja.
        self._cards: list[Card] = []

        # Construimos la(s) baraja(s):
        # Para cada baraja (n_decks veces)...
        for _ in range(n_decks):
            # ...para cada rango (A, 2, 3, ..., K)...
            for rank in RANKS:
                # ...añadimos 4 copias (una por palo: corazones, tréboles, diamantes, picas).
                # El palo no importa en blackjack, así que usamos solo el rango.
                for _ in range(4):
                    self._cards.append(Card(rank))

        # Barajamos automáticamente al crear la baraja.
        self.shuffle()

    def shuffle(self) -> None:
        """Baraja todas las cartas en orden aleatorio.
        self.rng.shuffle() reordena la lista de forma aleatoria usando la semilla definida."""
        self.rng.shuffle(self._cards)

    def draw(self) -> Card:
        """Saca y devuelve la carta del tope de la baraja (la última de la lista).
        La carta "sale" de la baraja → la lista se acorta en 1.
        .pop() extrae y devuelve el último elemento de una lista."""
        if not self._cards:
            # Si la baraja está vacía y alguien intenta sacar, lanzamos error.
            # En gen_sample_data.py verificamos len(deck) < 30 antes de esto.
            raise IndexError("Baraja vacía")
        return self._cards.pop()

    def __len__(self) -> int:
        """Devuelve cuántas cartas quedan en la baraja.
        Esto permite hacer len(deck) == 52 o if len(deck) < 30.
        __len__ es un método especial de Python que se llama automáticamente con len()."""
        return len(self._cards)
