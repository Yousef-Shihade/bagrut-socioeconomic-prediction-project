"""
explain.py — SHAP explainability + leaderboard visualisation.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Produces the core presentation visuals:
  * a SHAP beeswarm per target (which municipal features drive each prediction,
    and in which direction) from the tuned HistGradientBoosting champion;
  * a cross-validated R^2 leaderboard comparing all models across the 4 targets.
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
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")


def shap_beeswarm(model, X: pd.DataFrame, target: str, cfg: dict[str, Any],
                  out_dir: Path) -> Path:
    """TreeExplainer beeswarm for one tuned champion model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, Xs, show=False, plot_type="dot", max_display=12)
    plt.title(f"SHAP — {target}", fontsize=15)
    plt.tight_layout()
    path = out_dir / f"shap_beeswarm_{target}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def shap_importance(model, X: pd.DataFrame, target: str, cfg: dict[str, Any]) -> pd.Series:
    """Mean |SHAP| per feature (used for the README importance table)."""
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X
    sv = shap.TreeExplainer(model).shap_values(Xs)
    return pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns).sort_values(ascending=False)


def plot_leaderboard(leaderboards: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Grouped bar of CV R^2 for every model across the four targets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, lb in leaderboards.items():
        for _, r in lb.iterrows():
            rows.append({"target": target, "model": r["model"], "R2": r["R2"]})
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="model", ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Step 5 — Cross-Validated R² Leaderboard (GroupKFold by school)")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in long["target"].unique()], fontsize=11)
    ax.legend(title="Model", fontsize=10, loc="upper right")
    plt.tight_layout()
    path = out_dir / "models_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
