"""
visualize.py — Step 3 target-exploration plots.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Three figures saved to ``graphs/`` inspecting the engineered targets against the
main predictor (CBS socioeconomic cluster):

  1. cluster_vs_participation.png — cluster (1-10) vs 5-unit participation
     (Math & English), boxplot per cluster.
  2. cluster_vs_avg_grade.png     — cluster (1-10) vs takers-weighted average
     grade (Math & English), boxplot per cluster.
  3. target_distributions.png     — distributions of all four targets (grades and
     the 0-1 bounded participation rates) to expose skew / bounding.

Labels are kept in English (matplotlib does not shape RTL Hebrew well).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from io_load import load_config, resolve

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
sns.set_theme(style="whitegrid", context="talk")

_SUBJ_PALETTE = {"Math": "#3b6ea5", "English": "#c44e52"}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _long_by_cluster(df: pd.DataFrame, math_col: str, eng_col: str,
                     value_name: str) -> pd.DataFrame:
    """Reshape to long form (cluster, subject, value), dropping NaN cluster/value."""
    d = df[df["cluster"].notna()].copy()
    d["cluster"] = d["cluster"].astype(int)
    long = pd.concat([
        d[["cluster", math_col]].rename(columns={math_col: value_name}).assign(Subject="Math"),
        d[["cluster", eng_col]].rename(columns={eng_col: value_name}).assign(Subject="English"),
    ], ignore_index=True).dropna(subset=[value_name])
    return long


def plot_cluster_vs_participation(df: pd.DataFrame, out_dir: Path) -> Path:
    long = _long_by_cluster(df, "math_5unit_participation",
                            "english_5unit_participation", "participation")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.boxplot(data=long, x="cluster", y="participation", hue="Subject",
                order=range(1, 11), palette=_SUBJ_PALETTE, fliersize=2, ax=ax)
    ax.set_title("Plot 1 — Socioeconomic Cluster vs 5-Unit Participation")
    ax.set_xlabel("CBS socioeconomic cluster (1 = lowest, 10 = highest)")
    ax.set_ylabel("Advanced (5-unit) participation rate")
    ax.legend(title="Subject", loc="upper left")
    return _save(fig, out_dir, "cluster_vs_participation.png")


def plot_cluster_vs_avg_grade(df: pd.DataFrame, out_dir: Path) -> Path:
    long = _long_by_cluster(df, "math_avg_grade", "english_avg_grade", "avg_grade")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.boxplot(data=long, x="cluster", y="avg_grade", hue="Subject",
                order=range(1, 11), palette=_SUBJ_PALETTE, fliersize=2, ax=ax)
    ax.set_title("Plot 2 — Socioeconomic Cluster vs Takers-Weighted Average Grade")
    ax.set_xlabel("CBS socioeconomic cluster (1 = lowest, 10 = highest)")
    ax.set_ylabel("Average Bagrut grade (weighted)")
    ax.legend(title="Subject", loc="lower right")
    return _save(fig, out_dir, "cluster_vs_avg_grade.png")


def plot_target_distributions(df: pd.DataFrame, out_dir: Path) -> Path:
    specs = [
        ("math_avg_grade", "Math avg grade", "#3b6ea5", (40, 100)),
        ("english_avg_grade", "English avg grade", "#c44e52", (40, 100)),
        ("math_5unit_participation", "Math 5-unit participation", "#3b6ea5", (0, 1)),
        ("english_5unit_participation", "English 5-unit participation", "#c44e52", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, label, color, xlim) in zip(axes.ravel(), specs):
        data = df[col].dropna()
        sns.histplot(data, bins=30, kde=True, color=color, ax=ax)
        ax.axvline(data.mean(), color="black", ls="--", lw=1.5,
                   label=f"mean={data.mean():.2f}")
        ax.axvline(data.median(), color="grey", ls=":", lw=1.5,
                   label=f"median={data.median():.2f}")
        ax.set_xlim(*xlim)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel("")
        ax.legend(fontsize=10)
    fig.suptitle("Plot 3 — Engineered Target Distributions "
                 "(grades ~normal; participation 0-1 bounded)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_dir, "target_distributions.png")


def run(school_level: pd.DataFrame, cfg: dict[str, Any] | None = None) -> list[Path]:
    cfg = cfg or load_config()
    out_dir = resolve(cfg["paths"]["out_graphs"])
    return [
        plot_cluster_vs_participation(school_level, out_dir),
        plot_cluster_vs_avg_grade(school_level, out_dir),
        plot_target_distributions(school_level, out_dir),
    ]
