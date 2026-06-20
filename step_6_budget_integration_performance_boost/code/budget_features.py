"""
budget_features.py — engineer budget_per_student & build the 3-dataset matrix.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Pipeline (DUAL-BUDGET strategy):
  1. Coerce the budget columns to numeric & key on `semel`. The empty Gefen
     column is dropped entirely.
  2. Collapse to one row per school (defensive groupby-sum) and compute TWO
     independent, un-diluted micro-ratios:
        total_budget_per_student    = grand_total   / students   (comprehensive)
        teaching_budget_per_student = teaching_cost  / students   (instructional)
  3. Sanitize division-by-zero / infinities (empty student counts -> NaN).
  4. Left-join onto the Step-4 table strictly on `semel` (static per school,
     broadcast across that school's 2013-2016 rows).
  5. Expose helpers to build the "before" (Step-5) and "after" (+budget) feature
     matrices per target.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def engineer_budget(budget_raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Return (per-school budget table with TWO per-student ratios, diagnostics)."""
    bc = cfg["budget_columns"]
    sem, grand, cost, stud = bc["semel"], bc["grand_total"], bc["teaching_cost"], bc["students"]

    df = budget_raw[[sem, grand, cost, stud]].copy()
    df.columns = ["semel", "grand_total", "teaching_cost", "students"]
    for c in ["semel", "grand_total", "teaching_cost", "students"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["semel"])
    df["semel"] = df["semel"].astype(int)

    # Defensive aggregation (one budget record per school in this extract).
    agg = (df.groupby("semel", as_index=False)
             .agg(grand_total=("grand_total", "sum"),
                  teaching_cost=("teaching_cost", "sum"),
                  students=("students", "sum")))

    valid_students = agg["students"].fillna(0) > 0

    def _per_student(numerator: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = numerator.fillna(0) / agg["students"]
        r = r.replace([np.inf, -np.inf], np.nan)
        r[~valid_students] = np.nan          # empty/zero student counts -> NaN
        return r

    agg["total_budget_per_student"] = _per_student(agg["grand_total"])
    agg["teaching_budget_per_student"] = _per_student(agg["teaching_cost"])

    diag = {
        "n_budget_schools": int(len(agg)),
        "students_zero": int((~valid_students).sum()),
        "total_valid": int(agg["total_budget_per_student"].notna().sum()),
        "total_nonzero": int((agg["total_budget_per_student"].fillna(0) > 0).sum()),
        "total_median": float(agg["total_budget_per_student"].median()),
        "teach_valid": int(agg["teaching_budget_per_student"].notna().sum()),
        "teach_nonzero": int((agg["teaching_budget_per_student"].fillna(0) > 0).sum()),
        "teach_median": float(agg["teaching_budget_per_student"].median()),
    }
    return agg, diag


def merge_budget(step4: pd.DataFrame, budget: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strict left-join on `semel`; return (consolidated matrix, join diagnostics)."""
    step4 = step4.copy()
    step4["semel"] = pd.to_numeric(step4["semel"], errors="coerce").astype("Int64")
    cols = ["semel", "grand_total", "teaching_cost", "students",
            "total_budget_per_student", "teaching_budget_per_student"]
    merged = step4.merge(budget[cols], on="semel", how="left")

    matched_rows = int(merged["total_budget_per_student"].notna().sum())
    matched_schools = int(merged.loc[merged["total_budget_per_student"].notna(), "semel"].nunique())
    total_schools = int(step4["semel"].nunique())
    diag = {
        "rows_total": int(len(merged)),
        "rows_with_budget": matched_rows,
        "rows_with_budget_pct": round(100 * matched_rows / len(merged), 2),
        "schools_total": total_schools,
        "schools_matched": matched_schools,
        "schools_matched_pct": round(100 * matched_schools / total_schools, 2),
    }
    return merged, diag


# --------------------------------------------------------------------------- #
# feature-matrix builders                                                     #
# --------------------------------------------------------------------------- #
def _encode(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    X = df[numeric].copy()
    for c in categorical:
        dummies = pd.get_dummies(df[c].astype("Int64").astype("object"),
                                 prefix=c, dummy_na=False)
        X = pd.concat([X, dummies.astype(int)], axis=1)
    return X


def build_xy(df: pd.DataFrame, target: str, cfg: dict[str, Any], *, with_budget: bool):
    """Build (X, y, groups) for one target on the budget-matched, complete-case rows.

    The SAME row mask is used for the before/after pair so the comparison is
    apples-to-apples: rows must have the target, all base features, AND a valid
    budget_per_student (even for the 'before' model) so both train on identical data.
    """
    feats = cfg["features"]
    numeric = list(feats["base_numeric"])
    if with_budget:
        numeric = numeric + list(feats["budget_numeric"])
    cats = feats["categorical"]

    # Identical row mask for before/after: target + base numerics + budget all present.
    need = [target] + feats["base_numeric"] + feats["budget_numeric"]
    keep = df[need].notna().all(axis=1)

    sub = df[keep].reset_index(drop=True)
    X = _encode(sub, numeric, cats)
    y = sub[target].reset_index(drop=True)
    groups = sub[feats["group_col"]].astype(int).reset_index(drop=True)
    return X, y, groups
