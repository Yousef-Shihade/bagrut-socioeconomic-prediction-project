"""
feature_selection.py — collinearity (VIF) + Boruta selection.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Two rubric items:
  * "features exploration and handling – collinearity": VIF on the numeric
    candidates exposes the cluster <-> index_value redundancy (r = 0.97); we then
    drop index_value and keep `cluster`.
  * "Generate feature selection": Boruta (all-relevant wrapper around a Random
    Forest) isolates the stable municipal predictors per target.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compatibility shim: some Boruta builds reference removed NumPy aliases.
for _a, _t in [("float", float), ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _a):
        setattr(np, _a, _t)
from boruta import BorutaPy  # noqa: E402


def compute_vif(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """VIF for the numeric candidate features (incl. the collinear index_value)."""
    cols = cfg["collinearity"]["vif_features"]
    X = df[cols].dropna().astype(float)
    X = X.assign(_const=1.0)  # intercept so VIF is well-defined
    vif = []
    for i, c in enumerate(X.columns):
        if c == "_const":
            continue
        vif.append({"feature": c, "VIF": variance_inflation_factor(X.values, i)})
    out = pd.DataFrame(vif).sort_values("VIF", ascending=False).reset_index(drop=True)
    out["flag"] = np.where(out["VIF"] >= cfg["collinearity"]["vif_threshold"],
                           "HIGH (collinear)", "ok")
    return out


def run_boruta(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run Boruta on one target; return confirmed / tentative / rejected lists."""
    fs = cfg["feature_selection"]
    seed = cfg["seed"]
    rf = RandomForestRegressor(n_estimators=fs["boruta_estimators"],
                               random_state=seed, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators=fs["boruta_estimators"],
                      max_iter=fs["boruta_max_iter"], perc=fs.get("boruta_perc", 90),
                      random_state=seed, verbose=0)
    boruta.fit(X.values.astype(float), y.values.astype(float))

    cols = np.array(X.columns)
    confirmed = cols[boruta.support_].tolist()
    tentative = cols[boruta.support_weak_].tolist()
    rejected = [c for c in cols if c not in confirmed and c not in tentative]
    ranking = dict(zip(cols.tolist(), boruta.ranking_.tolist()))

    # Features actually used downstream: the top Boruta tier (rank 1 = confirmed).
    # If <2 reach rank 1, keep the 3 best-ranked so the SES variable (cluster) is
    # always retained — never the 12 sparse settlement-type dummies.
    rank1 = [c for c in cols.tolist() if ranking[c] == 1]
    if len(rank1) >= 2:
        selected = rank1
    else:
        selected = [c for c, _ in sorted(ranking.items(), key=lambda kv: kv[1])[:3]]

    return {"confirmed": confirmed, "tentative": tentative, "rejected": rejected,
            "selected": selected, "ranking": ranking}
