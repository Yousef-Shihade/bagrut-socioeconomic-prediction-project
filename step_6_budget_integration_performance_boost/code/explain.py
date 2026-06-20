"""
explain.py — Step 6 diagnostics & explainability visuals.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

dataset_understanding/ : budget distribution, budget-vs-cluster, missingness,
                         numeric correlation heatmap.
model_performance/     : before/after R^2 comparison, predicted-vs-actual,
                         SHAP beeswarm of the budget-augmented champion (does the
                         institutional feature displace the municipal cluster?).
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
from sklearn.model_selection import GroupKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="talk")

NAVY, TEAL, CORAL = "#1b2a4a", "#2a9d8f", "#d1495b"


# --------------------------------------------------------------------------- #
# dataset understanding                                                       #
# --------------------------------------------------------------------------- #
BUDGET_FEATS = ["total_budget_per_student", "teaching_budget_per_student"]


def plot_budget_distribution(merged: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    titles = ["total_budget_per_student\n(comprehensive grand-total / student)",
              "teaching_budget_per_student\n(instructional teaching-cost / student)"]
    for ax, col, title in zip(axes, BUDGET_FEATS, titles):
        bps = merged[col].replace(0, np.nan).dropna()
        sns.histplot(bps, bins=50, kde=True, color=TEAL, ax=ax)
        ax.axvline(bps.median(), color=CORAL, lw=2, ls="--",
                   label=f"median = {bps.median():,.0f}")
        ax.set_title(title); ax.set_xlabel("₪ per student"); ax.legend()
    fig.suptitle("Dual-budget per-student distributions (matched schools, nonzero)",
                 fontsize=15)
    plt.tight_layout()
    p = out_dir / "budget_per_student_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def plot_budget_by_cluster(merged: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, col in zip(axes, BUDGET_FEATS):
        d = merged.dropna(subset=[col, "cluster"]).copy()
        d["cluster"] = d["cluster"].astype(int)
        sns.boxplot(data=d, x="cluster", y=col, color=TEAL, showfliers=False, ax=ax)
        r = d[col].corr(d["cluster"])
        ax.set_title(f"{col}\n(corr with cluster = {r:+.2f})")
        ax.set_xlabel("CBS cluster (1 = poorest … 10 = richest)"); ax.set_ylabel("₪ / student")
    fig.suptitle("Budget per student across socioeconomic clusters "
                 "(near-orthogonal to SES → new signal)", fontsize=15)
    plt.tight_layout()
    p = out_dir / "budget_per_student_by_cluster.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def plot_correlation_heatmap(merged: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = BUDGET_FEATS + ["cluster", "log_population",
            "math_avg_grade", "english_avg_grade",
            "math_5unit_participation", "english_5unit_participation"]
    cols = [c for c in cols if c in merged.columns]
    corr = merged[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Numeric correlation — budget, SES & targets")
    plt.tight_layout()
    p = out_dir / "correlation_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def plot_missingness(merged: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["total_budget_per_student", "teaching_budget_per_student",
            "grand_total", "teaching_cost", "students",
            "cluster", "math_avg_grade", "english_avg_grade",
            "math_5unit_participation", "english_5unit_participation"]
    cols = [c for c in cols if c in merged.columns]
    miss = merged[cols].isna().mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=miss.values, y=miss.index, color=NAVY, ax=ax)
    for i, v in enumerate(miss.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=11)
    ax.set_title("Missingness after 3-dataset consolidation")
    ax.set_xlabel("% missing"); ax.set_ylabel("")
    plt.tight_layout()
    p = out_dir / "missingness_check.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# model performance                                                           #
# --------------------------------------------------------------------------- #
def plot_before_after(comparison: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar: tuned-HGB R^2 before vs after budget, per target (same rows)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    long = comparison.melt(id_vars="target", value_vars=["R2_before", "R2_after"],
                           var_name="phase", value_name="R2")
    long["phase"] = long["phase"].map({"R2_before": "Before (SES only)",
                                       "R2_after": "After (+ budget)"})
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=long, x="target", y="R2", hue="phase",
                palette=[NAVY, TEAL], ax=ax)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Tuned HistGradientBoosting — R² before vs after budget\n"
                 "(identical budget-matched rows, GroupKFold by school)")
    ax.set_xlabel(""); ax.set_ylabel("CV R²  (higher = better)")
    ax.set_xticklabels([t.replace("_", "\n") for t in comparison["target"]], fontsize=11)
    ax.legend(title="", fontsize=12, loc="upper right")
    plt.tight_layout()
    p = out_dir / "before_after_r2_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def plot_predicted_vs_actual(model, X, y, groups, target, cfg, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = GroupKFold(n_splits=cfg["modeling"]["cv_splits"])
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y, pred, s=18, alpha=0.5, color=TEAL, edgecolor="none")
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lim, lim, color=CORAL, lw=2, ls="--", label="perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Predicted vs actual — {target}\n(+budget champion, out-of-fold)")
    ax.set_xlabel(f"actual {target}"); ax.set_ylabel("predicted")
    ax.legend()
    plt.tight_layout()
    p = out_dir / f"predicted_vs_actual_{target}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def shap_beeswarm(model, X: pd.DataFrame, target: str, cfg: dict[str, Any],
                  out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(cfg["modeling"]["shap_sample"], len(X))
    Xs = X.sample(n=n, random_state=cfg["seed"]) if len(X) > n else X
    sv = shap.TreeExplainer(model).shap_values(Xs)
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, Xs, show=False, plot_type="dot", max_display=12)
    plt.title(f"SHAP (+budget) — {target}", fontsize=15)
    plt.tight_layout()
    p = out_dir / f"shap_beeswarm_{target}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p
