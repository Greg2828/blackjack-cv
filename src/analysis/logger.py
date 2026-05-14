"""
ARCHIVO: src/analysis/logger.py
PROPÓSITO: Guarda cada mano jugada en un archivo CSV para análisis posterior.
           CSV = Comma Separated Values = archivo de texto donde cada línea
           es una fila de datos y los valores van separados por comas.
           Este archivo se puede abrir con Excel, LibreOffice Calc o pandas.

           El CSV se usa luego en stats.py y en el notebook analysis.ipynb
           para generar gráficas de tu progreso y estadísticas.

CÓMO SE CONECTA:
  - main.py y simulate.py crean un HandLogger al arrancar y llaman a log() al final de cada mano
  - stats.py lee el CSV con load_log() para calcular estadísticas
  - config.py provee LOG_FILE = 'data/games_log.csv'

FORMATO DEL CSV:
  Columnas: timestamp, player_cards, dealer_upcard, dealer_final,
            actions_recommended, actions_taken, bet, outcome, delta, bankroll
"""

import csv          # módulo estándar de Python para leer/escribir CSV
import datetime     # para añadir la fecha y hora a cada registro
from pathlib import Path  # manejo de rutas de archivos

# Importaciones de otros módulos del proyecto.
from ..game.state import GameState, Outcome, Action

# Lista con los nombres de todas las columnas del CSV, en el orden correcto.
# Debe coincidir exactamente con las claves del diccionario en log().
_FIELDS = [
    'timestamp',           # fecha y hora de la mano (ej: "2026-05-12T18:30:05")
    'player_cards',        # cartas del jugador (ej: "A 8")
    'dealer_upcard',       # carta visible del crupier (ej: "7")
    'dealer_final',        # mano completa del crupier al final (ej: "7 K")
    'actions_recommended', # acciones que la estrategia recomendó (ej: "stand")
    'actions_taken',       # acciones que el jugador ejecutó (ej: "hit stand")
    'bet',                 # apuesta de esta mano (ej: 10.0)
    'outcome',             # resultado (ej: "ganas", "pierdes", "empate", "blackjack")
    'delta',               # cambio en la banca (ej: 10.0 ó -10.0)
    'bankroll',            # banca total tras esta mano (ej: 110.0)
]


class HandLogger:

    def __init__(self, path: str = 'data/games_log.csv'):
        """Constructor: prepara el archivo CSV para recibir datos.

        Si el archivo no existe, lo crea con los encabezados de columnas.
        Si ya existe (de sesiones anteriores), lo deja como está para poder
        añadir nuevas filas sin borrar el historial.

        Parámetro:
          path: ruta al archivo CSV (de config.LOG_FILE)
        """
        self.path = Path(path)

        # parents=True: crea también las carpetas intermedias si no existen.
        # Ej: si path='data/games_log.csv', crea la carpeta 'data/' si no existe.
        # exist_ok=True: no da error si la carpeta ya existe.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Solo escribimos el encabezado si el archivo es nuevo o está vacío.
        # stat().st_size devuelve el tamaño en bytes; 0 = archivo vacío.
        if not self.path.exists() or self.path.stat().st_size == 0:
            # Abrimos el archivo en modo escritura ('w').
            # newline='' es necesario en Python para que csv maneje los saltos de línea correctamente.
            with open(self.path, 'w', newline='') as f:
                # DictWriter escribe filas a partir de diccionarios.
                # Le pasamos los nombres de las columnas (fieldnames).
                csv.DictWriter(f, fieldnames=_FIELDS).writeheader()
                # .writeheader() escribe la primera fila con los nombres de las columnas.

    def log(
        self,
        state: GameState,
        recommended: list[Action],
        taken: list[Action],
        outcome: Outcome,
        delta: float,
    ) -> None:
        """Añade una fila al CSV con los datos de la mano que acaba de terminar.

        Parámetros:
          state: el estado final de la mano (cartas, apuesta, banca)
          recommended: lista de acciones que la estrategia fue recomendando
          taken: lista de acciones que el jugador realmente ejecutó
          outcome: resultado de la mano (WIN, LOSE, PUSH, BLACKJACK)
          delta: cuánto cambió la banca esta mano (+10.0, -5.0, etc.)
        """

        # Las cartas visibles del crupier (excluyendo la tapada).
        dealer_visible = state.dealer_hand.visible_cards

        # Construimos el diccionario con todos los datos de la mano.
        row = {
            # datetime.now().isoformat() → "2026-05-12T18:30:05" (fecha+hora estándar ISO)
            # timespec='seconds' elimina los microsegundos para que sea más legible.
            'timestamp':           datetime.datetime.now().isoformat(timespec='seconds'),

            # Las cartas como texto separado por espacios: "A 8" ó "7 K 2"
            'player_cards':        ' '.join(str(c) for c in state.player_hand.cards),

            # La primera carta visible del crupier (la que no estaba tapada).
            # El if...else evita error si la lista está vacía.
            'dealer_upcard':       str(dealer_visible[0]) if dealer_visible else '',

            # Todas las cartas del crupier al final (ya destapadas todas).
            'dealer_final':        ' '.join(str(c) for c in state.dealer_hand.cards),

            # Acciones recomendadas separadas por espacios: "stand" ó "hit stand"
            # .value accede al texto de cada Action (ej: Action.HIT.value = "hit")
            'actions_recommended': ' '.join(a.value for a in recommended),

            # Acciones tomadas (pueden diferir de las recomendadas si el jugador las ignoró).
            'actions_taken':       ' '.join(a.value for a in taken),

            'bet':                 state.bet,
            'outcome':             outcome.value,   # ej: "ganas", "pierdes"
            'delta':               delta,
            'bankroll':            state.bankroll,
        }

        # Abrimos el archivo en modo "append" ('a'): añade al final sin borrar nada.
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)
