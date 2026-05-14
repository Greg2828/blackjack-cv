"""
ARCHIVO: src/analysis/stats.py
PROPÓSITO: Funciones para analizar estadísticamente el historial de partidas guardado en el CSV.
           Calcula métricas clave (tasa de victoria, EV) y genera gráficas visuales.

           EV = Expected Value (Valor Esperado) = ganancia o pérdida media por mano.
           En blackjack con basic strategy perfecta, el EV debería ser cercano a -0.5%.

CÓMO SE CONECTA:
  - El notebook notebooks/analysis.ipynb importa estas funciones para generar gráficas
  - Puede usarse directamente desde la terminal para ver estadísticas rápidas
  - Lee el CSV generado por logger.py (data/games_log.csv)

USO TÍPICO:
    from src.analysis.stats import load_log, print_summary, plot_bankroll
    df = load_log()
    print_summary(df)
    plot_bankroll(df)
"""

from __future__ import annotations   # permite tipos como 'pd.DataFrame' antes de definirlos
from pathlib import Path

# Librerías de análisis de datos y visualización.
# pandas: maneja "DataFrames" (tablas de datos), perfecto para el CSV.
# numpy: operaciones matemáticas sobre arrays.
# matplotlib: generación de gráficas.
# seaborn: gráficas estadísticas más elegantes sobre matplotlib.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Estilo visual para todas las gráficas: fondo oscuro con rejilla, colores suaves.
sns.set_theme(style="darkgrid", palette="muted")

# Orden fijo de los resultados para las gráficas de barras (más intuitivo que alfabético).
_OUTCOME_ORDER = ["ganas", "blackjack", "empate", "pierdes"]

# Colores para cada tipo de resultado en las gráficas.
_OUTCOME_COLORS = {
    "ganas":     "#2ecc71",  # verde
    "blackjack": "#f1c40f",  # amarillo dorado
    "empate":    "#95a5a6",  # gris
    "pierdes":   "#e74c3c",  # rojo
}

# Orden fijo de las acciones para la gráfica de distribución de acciones.
_ACTION_ORDER = ["hit", "stand", "double", "split", "surrender"]


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_log(path: str | Path = "data/games_log.csv") -> pd.DataFrame:
    """Lee el CSV y devuelve un DataFrame de pandas con los datos limpios.

    Además añade una columna 'session' que agrupa las manos por sesión:
    si hay más de 10 minutos entre dos manos consecutivas, se considera
    que empezó una nueva sesión.

    Devuelve: pd.DataFrame con todas las manos como filas y las columnas del CSV.
    """

    # pd.read_csv() lee el archivo CSV y crea un DataFrame.
    # parse_dates=["timestamp"] convierte la columna timestamp a objetos de fecha/hora.
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Eliminamos espacios extra en la columna outcome (por si acaso).
    df["outcome"] = df["outcome"].str.strip()

    # Detectamos cambios de sesión: si el tiempo entre manos supera 10 minutos,
    # empezamos una nueva sesión (session + 1).
    # .diff() calcula la diferencia entre filas consecutivas en la columna timestamp.
    # .cumsum() acumula los True como 0, 1, 2... para numerar las sesiones.
    df["session"] = (df["timestamp"].diff() > pd.Timedelta("10min")).cumsum()

    return df


# =============================================================================
# RESUMEN ESTADÍSTICO
# =============================================================================

def summary(df: pd.DataFrame) -> dict:
    """Calcula las métricas más importantes del historial.

    Devuelve un diccionario con:
      manos: número total de manos jugadas
      win_rate: fracción de manos ganadas (0.0 - 1.0)
      blackjack_rate: fracción de blackjacks
      push_rate: fracción de empates
      loss_rate: fracción de derrotas
      ev_por_mano: ganancia/pérdida media por mano (en €)
      ev_pct_apuesta: EV como porcentaje de la apuesta media
      delta_total: ganancia/pérdida total acumulada
      sesiones: número de sesiones distintas
      apuesta_media: apuesta promedio
    """

    # .isin() comprueba si cada valor está en la lista dada → devuelve True/False por fila.
    # .sum() cuenta los True (True = 1, False = 0 en Python).
    wins   = df["outcome"].isin(["ganas", "blackjack"]).sum()
    bj     = (df["outcome"] == "blackjack").sum()
    pushes = (df["outcome"] == "empate").sum()
    losses = (df["outcome"] == "pierdes").sum()

    # EV = media del delta de cada mano.
    ev     = df["delta"].mean()

    # EV como porcentaje de la apuesta media (normalizado).
    # Evitamos dividir por cero con 'if df["bet"].mean()'.
    ev_pct = ev / df["bet"].mean() * 100 if df["bet"].mean() else 0

    return {
        "manos":          len(df),          # len() de un DataFrame = número de filas
        "win_rate":       wins / len(df),
        "blackjack_rate": bj / len(df),
        "push_rate":      pushes / len(df),
        "loss_rate":      losses / len(df),
        "ev_por_mano":    ev,
        "ev_pct_apuesta": ev_pct,
        "delta_total":    df["delta"].sum(),
        "sesiones":       df["session"].nunique(),  # .nunique() = número de valores únicos
        "apuesta_media":  df["bet"].mean(),
    }


def print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen bonito en la terminal con las estadísticas más importantes."""
    s = summary(df)
    print(f"{'─'*38}")
    print(f"  Manos jugadas   : {s['manos']}")
    print(f"  Sesiones        : {s['sesiones']}")
    print(f"  Tasa de victoria: {s['win_rate']*100:.1f}%")   # :.1f = 1 decimal
    print(f"  Blackjacks      : {s['blackjack_rate']*100:.1f}%")
    print(f"  Empates         : {s['push_rate']*100:.1f}%")
    print(f"  EV por mano     : {s['ev_por_mano']:+.2f}€")   # :+ muestra siempre el signo
    print(f"  EV % apuesta    : {s['ev_pct_apuesta']:+.2f}%")
    print(f"  Delta total     : {s['delta_total']:+.0f}€")
    print(f"{'─'*38}")


# =============================================================================
# GRÁFICAS
# =============================================================================

def plot_bankroll(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Gráfica de línea: evolución del bankroll mano a mano.
    Las zonas por encima del capital inicial se colorean en verde,
    las zonas por debajo en rojo. Muestra si vas ganando o perdiendo con el tiempo.

    Parámetro ax: si None, crea una nueva figura. Si se pasa, dibuja en esa subgráfica.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))  # crea figura nueva 10×4 pulgadas

    # Línea principal del bankroll.
    ax.plot(df.index, df["bankroll"], color="#3498db", linewidth=1.5)

    # Línea horizontal punteada del bankroll inicial (referencia).
    ax.axhline(df["bankroll"].iloc[0], color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6, label="Inicial")

    # Zona verde: ganancias (por encima del bankroll inicial).
    ax.fill_between(df.index, df["bankroll"].iloc[0], df["bankroll"],
                    where=df["bankroll"] >= df["bankroll"].iloc[0],
                    alpha=0.15, color="#2ecc71")

    # Zona roja: pérdidas (por debajo del bankroll inicial).
    ax.fill_between(df.index, df["bankroll"].iloc[0], df["bankroll"],
                    where=df["bankroll"] < df["bankroll"].iloc[0],
                    alpha=0.15, color="#e74c3c")

    ax.set_title("Evolución del bankroll")
    ax.set_xlabel("Mano #")
    ax.set_ylabel("Bankroll (€)")
    ax.legend()
    return ax


def plot_outcomes(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Gráfica de barras: cuántas manos terminaron en cada resultado.
    Muestra la distribución de victorias, derrotas, empates y blackjacks."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # Contamos cuántas veces aparece cada resultado.
    # .value_counts() cuenta la frecuencia de cada valor único.
    # .reindex() fuerza el orden definido en _OUTCOME_ORDER, rellenando con 0 si falta alguno.
    counts = (df["outcome"]
              .value_counts()
              .reindex(_OUTCOME_ORDER, fill_value=0))

    colors = [_OUTCOME_COLORS[o] for o in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_title("Distribución de resultados")
    ax.set_xlabel("")
    ax.set_ylabel("Manos")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")

    # Añadimos el número encima de cada barra.
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    return ax


def plot_delta_by_upcard(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Gráfica de barras: ganancia/pérdida media según la carta que mostraba el crupier.
    Muestra cuánto te afecta estadísticamente cada carta del crupier.
    Las cartas del crupier débiles (4, 5, 6) deberían dar EVs más positivos."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Orden estándar de las cartas del crupier.
    order = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]

    # Agrupamos por carta del crupier y calculamos el delta medio de cada grupo.
    grp = (df.groupby("dealer_upcard")["delta"]
             .mean()
             .reindex(order)
             .dropna())  # eliminamos cartas sin datos

    # Color verde si EV positivo, rojo si negativo.
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in grp]
    grp.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.65)
    ax.axhline(0, color="white", linewidth=0.8)  # línea en 0 como referencia
    ax.set_title("EV medio por carta del crupier")
    ax.set_xlabel("Carta visible del crupier")
    ax.set_ylabel("Delta medio (€)")
    ax.set_xticklabels(grp.index, rotation=0)
    return ax


def plot_action_distribution(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Gráfica de barras: cuántas veces se recomendó cada acción.
    Muestra si tu juego sigue el patrón correcto de la basic strategy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    # Necesitamos aplanar la columna "actions_recommended" porque cada celda
    # puede tener múltiples acciones separadas por espacios (ej: "hit stand").
    all_actions: list[str] = []
    for cell in df["actions_recommended"].dropna():
        all_actions.extend(str(cell).split())  # split() divide por espacios

    counts = (pd.Series(all_actions)
                .value_counts()
                .reindex(_ACTION_ORDER, fill_value=0))

    counts.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white", width=0.6)
    ax.set_title("Acciones recomendadas")
    ax.set_xlabel("")
    ax.set_ylabel("Veces recomendada")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    return ax


def plot_adherence_by_session(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Gráfica de barras: % de adherencia a la basic strategy por sesión.
    100% = seguiste perfectamente la estrategia.
    Muestra si tu seguimiento de la estrategia mejora con la práctica."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    def adherence(g: pd.DataFrame) -> float:
        """Calcula la adherencia para un grupo de manos (una sesión).
        Compara acción recomendada vs. acción tomada, posición por posición."""
        total = match = 0
        for _, row in g.iterrows():
            recs   = str(row["actions_recommended"]).split()  # lista de recomendaciones
            takens = str(row["actions_taken"]).split()        # lista de acciones reales
            # zip() empareja recomendación[i] con taken[i] para comparar.
            for r, t in zip(recs, takens):
                total += 1
                match += int(r == t)  # int(True) = 1, int(False) = 0
        return match / total * 100 if total else float("nan")

    # Aplicamos la función adherence a cada grupo (sesión) por separado.
    per_session = df.groupby("session").apply(adherence)
    per_session.plot(kind="bar", ax=ax, color="#9b59b6",
                     edgecolor="white", width=0.6)

    # Línea en 100% como referencia de adherencia perfecta.
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Adherencia a la estrategia por sesión (%)")
    ax.set_xlabel("Sesión")
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    # Formateamos el eje Y como porcentaje.
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    return ax


def plot_player_total_distribution(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Gráfica de barras: distribución de totales iniciales del jugador (primeras 2 cartas).
    Muestra con qué frecuencia recibes cada posible total al inicio de una mano.
    Útil para verificar que el dataset o la baraja está bien distribuida."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    # Necesitamos reconstruir las manos para calcular el total.
    from src.game.card import Card
    from src.game.hand import Hand

    totals = []
    for cards_str in df["player_cards"].dropna():
        ranks = cards_str.split()[:2]   # solo las 2 primeras cartas
        try:
            # Creamos una Hand temporal para calcular el total correctamente
            # (los ases se manejan solos gracias a Hand.total()).
            h = Hand([Card(r) for r in ranks])
            totals.append(h.total())
        except Exception:
            pass  # ignoramos filas con datos corruptos

    # Contamos la frecuencia de cada total posible y ordenamos de menor a mayor.
    s = pd.Series(totals).value_counts().sort_index()
    s.plot(kind="bar", ax=ax, color="#1abc9c", edgecolor="white", width=0.7)
    ax.set_title("Distribución de totales iniciales (2 cartas)")
    ax.set_xlabel("Total del jugador")
    ax.set_ylabel("Frecuencia")
    ax.set_xticklabels(s.index, rotation=0)
    return ax
