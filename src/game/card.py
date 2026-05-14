"""
ARCHIVO: src/game/card.py
PROPÓSITO: Define qué ES una carta de blackjack.
           Es la pieza más pequeña y fundamental del juego.
           Todos los demás archivos del juego (hand.py, deck.py, state.py)
           trabajan con objetos Card creados aquí.

CÓMO SE CONECTA:
  - hand.py importa Card para construir manos (lista de cartas)
  - deck.py importa Card para crear la baraja
  - detector.py importa Card para crear cartas desde lo que detecta la cámara
  - strategy.py importa Card para consultar la carta visible del crupier
"""

# 'dataclass' es una herramienta de Python que genera automáticamente
# funciones básicas (__init__, __repr__, __eq__) para clases que solo
# sirven para guardar datos. Nos ahorra escribir mucho código repetitivo.
from dataclasses import dataclass


# @dataclass(frozen=True) crea la clase Card como un "dataclass inmutable".
# "frozen=True" significa que una vez creada una carta, NO puedes cambiarla.
# Ejemplo: si creas Card('A'), no puedes después hacer carta.rank = 'K'.
# Esto tiene sentido porque una carta real tampoco cambia de valor por arte de magia.
# Además, las cartas inmutables pueden usarse como claves de diccionarios,
# lo cual es útil en strategy.py.
@dataclass(frozen=True)
class Card:
    """Una carta del blackjack. Sin palo (irrelevante para este juego).

    El rango 'BACK' representa una carta tapada (el dorso del crupier).
    """

    # 'rank' es el único dato que guarda una carta: su valor como texto.
    # Puede ser: 'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', o 'BACK'
    # Nota: no guardamos el palo (corazones, tréboles...) porque en blackjack no importa.
    # str es el "tipo" del dato → significa que rank es un texto (string).
    rank: str  # 'A', '2'..'10', 'J', 'Q', 'K', o 'BACK'

    # @property convierte este método en un "atributo calculado".
    # En vez de llamar carta.value(), lo usas como carta.value (sin paréntesis).
    # Cada vez que accedes a carta.value, Python ejecuta este método automáticamente.
    @property
    def value(self) -> int:
        """Valor numérico para sumar. El As vale 11 aquí; la lógica de
        bajarlo a 1 cuando hace falta vive en Hand.total()."""

        # Una carta tapada (BACK) representa la carta secreta del crupier.
        # No podemos conocer su valor hasta que el crupier la voltee,
        # por eso lanzamos un error si alguien intenta acceder a su valor.
        if self.rank == 'BACK':
            raise ValueError("Una carta tapada no tiene valor conocido")

        # El As siempre parte valiendo 11.
        # Si con 11 la mano se pasa de 21, Hand.total() lo bajará a 1.
        # (Esa lógica está en hand.py, no aquí).
        if self.rank == 'A':
            return 11

        # Las figuras (Jota, Reina, Rey) valen 10 en blackjack.
        if self.rank in ('J', 'Q', 'K'):
            return 10

        # Las cartas numéricas (2 al 10) valen su número.
        # int(self.rank) convierte el texto '7' en el número entero 7.
        return int(self.rank)

    # Propiedad de conveniencia: devuelve True si la carta es un As, False en caso contrario.
    # Se usa en hand.py para contar cuántos ases hay en la mano.
    @property
    def is_ace(self) -> bool:
        return self.rank == 'A'

    # Propiedad de conveniencia: devuelve True si la carta es el dorso (tapada).
    # Se usa en hand.py para filtrar cartas visibles vs. tapadas.
    @property
    def is_back(self) -> bool:
        return self.rank == 'BACK'

    # __str__ define cómo se convierte la carta a texto cuando usas str(carta) o print(carta).
    # Ejemplo: str(Card('A')) → 'A',  str(Card('10')) → '10'
    # Útil para mostrar las cartas en pantalla o en los logs del CSV.
    def __str__(self) -> str:
        return self.rank
