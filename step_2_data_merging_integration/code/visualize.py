"""
visualize.py — Step 2 merge-health & bias plots.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Three figures saved to ``graphs/``:

  1. match_yield_waterfall.png — records resolved at each alignment stage
     (Exact -> Structural -> Crosswalk -> Fuzzy -> Unmatched), proving efficacy.
  2. missingness_bias_by_size.png — cohort size (``takers``) for matched vs
     unmatched records, testing whether data loss is random or structural.
  3. socioeconomic_representation.png — distribution of successfully integrated
     records across the CBS clusters 1-10 (coverage across the SES spectrum).

Labels are kept in English (matplotlib does not shape RTL Hebrew well).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from io_load import load_config, resolve

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
sns.set_theme(style="whitegrid", context="talk")

_STAGE_ORDER = ["exact", "structural", "crosswalk", "fuzzy", "unmatched"]
_STAGE_COLORS = {
    "exact": "#2a9d8f", "structural": "#457b9d", "crosswalk": "#e9c46a",
    "fuzzy": "#f4a261", "unmatched": "#d1495b",
}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_match_yield_waterfall(merged: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 1 — records resolved per alignment stage + cumulative match %.

    One bar per stage (Exact dominates, so a log y-axis keeps the small
    Crosswalk/Fuzzy passes legible) with an overlaid cumulative-match-% line.
    """
    counts = merged["match_stage"].value_counts()
    total = len(merged)
    vals = [int(counts.get(s, 0)) for s in _STAGE_ORDER]
    colors = [_STAGE_COLORS[s] for s in _STAGE_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(_STAGE_ORDER, vals, color=colors, edgecolor="white", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, total * 2)
    for p, v in zip(bars, vals):
        ax.text(p.get_x() + p.get_width() / 2, v * 1.15,
                f"{v:,}\n({v/total*100:.2f}%)", ha="center", va="bottom", fontsize=11)

    # Cumulative matched % across the matching stages (excludes 'unmatched').
    ax2 = ax.twinx()
    cum, cum_pts = 0, []
    for s in _STAGE_ORDER:
        if s != "unmatched":
            cum += int(counts.get(s, 0))
        cum_pts.append(cum / total * 100)
    ax2.plot(_STAGE_ORDER, cum_pts, color="#222", marker="o", lw=2, zorder=4)
    ax2.set_ylim(90, 100.8)
    ax2.set_ylabel("Cumulative match % (line)")
    # Label the two informative endpoints only, offset to avoid the bar labels.
    ax2.annotate(f"{cum_pts[0]:.2f}%", (0, cum_pts[0]), textcoords="offset points",
                 xytext=(-2, -18), ha="center", fontsize=10, color="#222")
    ax2.annotate(f"{cum_pts[3]:.2f}%", (3, cum_pts[3]), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=10, color="#222")

    matched = sum(int(counts.get(s, 0)) for s in _STAGE_ORDER if s != "unmatched")
    ax.set_title("Plot 1 — Stage-by-Stage Match Yield\n"
                 f"matched {matched:,}/{total:,} = {matched/total*100:.2f}%")
    ax.set_ylabel("Bagrut records per stage (log scale, bars)")
    ax.set_xlabel("Alignment stage")
    return _save(fig, out_dir, "match_yield_waterfall.png")


def plot_missingness_bias_by_size(merged: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 2 — takers distribution for matched vs unmatched records."""
    df = merged[["takers"]].copy()
    df["status"] = np.where(merged["cluster"].notna(), "Matched (has cluster)",
                            "Unmatched")
    med = df.groupby("status")["takers"].median()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    order = ["Matched (has cluster)", "Unmatched"]
    sns.boxplot(data=df, x="status", y="takers", order=order,
                palette={"Matched (has cluster)": "#2a9d8f", "Unmatched": "#d1495b"},
                ax=ax)
    ax.set_yscale("log")
    ax.set_title("Plot 2 — Missingness Bias by School Size\n"
                 "(cohort size of matched vs unmatched records)")
    ax.set_xlabel("")
    ax.set_ylabel("Test-takers per record (log scale)")
    for i, s in enumerate(order):
        if s in med.index:
            ax.annotate(f"median = {med[s]:.0f}", (i, med[s]), ha="center",
                        va="bottom", fontsize=12, fontweight="bold")
    return _save(fig, out_dir, "missingness_bias_by_size.png")


def plot_socioeconomic_representation(merged: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 3 — integrated records across CBS socioeconomic clusters 1-10.

    The CBS extract has no district/מחוז column, so we verify coverage along the
    socioeconomic dimension instead: how the successfully merged Bagrut records
    spread across clusters 1 (lowest) to 10 (highest).
    """
    clusters = merged.loc[merged["cluster"].notna(), "cluster"].astype(int)
    counts = clusters.value_counts().reindex(range(1, 11), fill_value=0)
    pct = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = sns.barplot(x=list(counts.index), y=counts.values, palette="viridis", ax=ax)
    for p, c, pc in zip(bars.patches, counts.values, pct.values):
        ax.annotate(f"{c:,}\n{pc:.1f}%", (p.get_x() + p.get_width() / 2, c),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title("Plot 3 — Socioeconomic Representation of Merged Records\n"
                 "(integrated Bagrut records per CBS cluster)")
    ax.set_xlabel("CBS socioeconomic cluster (1 = lowest, 10 = highest)")
    ax.set_ylabel("Merged Bagrut records")
    ax.margins(y=0.12)
    return _save(fig, out_dir, "socioeconomic_representation.png")


def run(merged: pd.DataFrame, cfg: dict[str, Any] | None = None) -> list[Path]:
    cfg = cfg or load_config()
    out_dir = resolve(cfg["paths"]["out_graphs"])
    return [
        plot_match_yield_waterfall(merged, out_dir),
        plot_missingness_bias_by_size(merged, out_dir),
        plot_socioeconomic_representation(merged, out_dir),
    ]
