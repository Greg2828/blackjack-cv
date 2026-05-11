"""
Funciones de análisis estadístico sobre el log de partidas (CSV).
Pensado para usarse desde el notebook o directamente desde scripts.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")

_OUTCOME_ORDER = ["ganas", "blackjack", "empate", "pierdes"]
_OUTCOME_COLORS = {
    "ganas":     "#2ecc71",
    "blackjack": "#f1c40f",
    "empate":    "#95a5a6",
    "pierdes":   "#e74c3c",
}
_ACTION_ORDER = ["hit", "stand", "double", "split", "surrender"]


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_log(path: str | Path = "data/games_log.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["outcome"] = df["outcome"].str.strip()
    df["session"] = (df["timestamp"].diff() > pd.Timedelta("10min")).cumsum()
    return df


# ── Resumen ───────────────────────────────────────────────────────────────────

def summary(df: pd.DataFrame) -> dict:
    wins     = df["outcome"].isin(["ganas", "blackjack"]).sum()
    bj       = (df["outcome"] == "blackjack").sum()
    pushes   = (df["outcome"] == "empate").sum()
    losses   = (df["outcome"] == "pierdes").sum()
    ev       = df["delta"].mean()
    ev_pct   = ev / df["bet"].mean() * 100 if df["bet"].mean() else 0

    return {
        "manos":          len(df),
        "win_rate":       wins / len(df),
        "blackjack_rate": bj / len(df),
        "push_rate":      pushes / len(df),
        "loss_rate":      losses / len(df),
        "ev_por_mano":    ev,
        "ev_pct_apuesta": ev_pct,
        "delta_total":    df["delta"].sum(),
        "sesiones":       df["session"].nunique(),
        "apuesta_media":  df["bet"].mean(),
    }


def print_summary(df: pd.DataFrame) -> None:
    s = summary(df)
    print(f"{'─'*38}")
    print(f"  Manos jugadas   : {s['manos']}")
    print(f"  Sesiones        : {s['sesiones']}")
    print(f"  Tasa de victoria: {s['win_rate']*100:.1f}%")
    print(f"  Blackjacks      : {s['blackjack_rate']*100:.1f}%")
    print(f"  Empates         : {s['push_rate']*100:.1f}%")
    print(f"  EV por mano     : {s['ev_por_mano']:+.2f}€")
    print(f"  EV % apuesta    : {s['ev_pct_apuesta']:+.2f}%")
    print(f"  Delta total     : {s['delta_total']:+.0f}€")
    print(f"{'─'*38}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bankroll(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["bankroll"], color="#3498db", linewidth=1.5)
    ax.axhline(df["bankroll"].iloc[0], color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6, label="Inicial")
    ax.fill_between(df.index, df["bankroll"].iloc[0], df["bankroll"],
                    where=df["bankroll"] >= df["bankroll"].iloc[0],
                    alpha=0.15, color="#2ecc71")
    ax.fill_between(df.index, df["bankroll"].iloc[0], df["bankroll"],
                    where=df["bankroll"] < df["bankroll"].iloc[0],
                    alpha=0.15, color="#e74c3c")
    ax.set_title("Evolución del bankroll")
    ax.set_xlabel("Mano #")
    ax.set_ylabel("Bankroll (€)")
    ax.legend()
    return ax


def plot_outcomes(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    counts = (df["outcome"]
              .value_counts()
              .reindex(_OUTCOME_ORDER, fill_value=0))
    colors = [_OUTCOME_COLORS[o] for o in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.6)
    ax.set_title("Distribución de resultados")
    ax.set_xlabel("")
    ax.set_ylabel("Manos")
    ax.set_xticklabels(counts.index, rotation=30, ha="right")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    return ax


def plot_delta_by_upcard(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    order = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    grp = (df.groupby("dealer_upcard")["delta"]
             .mean()
             .reindex(order)
             .dropna())

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in grp]
    grp.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.65)
    ax.axhline(0, color="white", linewidth=0.8)
    ax.set_title("EV medio por carta del crupier")
    ax.set_xlabel("Carta visible del crupier")
    ax.set_ylabel("Delta medio (€)")
    ax.set_xticklabels(grp.index, rotation=0)
    return ax


def plot_action_distribution(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    all_actions: list[str] = []
    for cell in df["actions_recommended"].dropna():
        all_actions.extend(str(cell).split())

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
    """Porcentaje de adherencia a la estrategia por sesión."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    def adherence(g: pd.DataFrame) -> float:
        total = match = 0
        for _, row in g.iterrows():
            recs   = str(row["actions_recommended"]).split()
            takens = str(row["actions_taken"]).split()
            for r, t in zip(recs, takens):
                total += 1
                match += int(r == t)
        return match / total * 100 if total else float("nan")

    per_session = df.groupby("session").apply(adherence)
    per_session.plot(kind="bar", ax=ax, color="#9b59b6",
                     edgecolor="white", width=0.6)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Adherencia a la estrategia por sesión (%)")
    ax.set_xlabel("Sesión")
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    return ax


def plot_player_total_distribution(df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    """Distribución de totales iniciales del jugador (primeras 2 cartas)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    from src.game.card import Card
    from src.game.hand import Hand

    totals = []
    for cards_str in df["player_cards"].dropna():
        ranks = cards_str.split()[:2]
        try:
            h = Hand([Card(r) for r in ranks])
            totals.append(h.total())
        except Exception:
            pass

    s = pd.Series(totals).value_counts().sort_index()
    s.plot(kind="bar", ax=ax, color="#1abc9c", edgecolor="white", width=0.7)
    ax.set_title("Distribución de totales iniciales (2 cartas)")
    ax.set_xlabel("Total del jugador")
    ax.set_ylabel("Frecuencia")
    ax.set_xticklabels(s.index, rotation=0)
    return ax
