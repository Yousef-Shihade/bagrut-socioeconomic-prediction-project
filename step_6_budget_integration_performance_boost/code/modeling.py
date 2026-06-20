"""
modeling.py — Step 6 tournament + tuned champion (mirrors Step 5 exactly).

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Identical model zoo, scoring, GroupKFold(semel) CV, and RandomizedSearchCV tuning
as Step 5 — so any change in the leaderboard is attributable to the new
`budget_per_student` feature, not to a different modeling protocol.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def build_models(seed: int) -> dict[str, Any]:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed)),
        "SGD (linear SVM)": make_pipeline(
            StandardScaler(),
            SGDRegressor(random_state=seed, max_iter=3000, tol=1e-4,
                         early_stopping=True)),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed,
                                              n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=seed),
    }


def cv_metrics(model, X, y, groups, n_splits: int) -> dict[str, float]:
    cv = GroupKFold(n_splits=n_splits)
    res = cross_validate(model, X, y, groups=groups, cv=cv, scoring=_SCORING,
                         n_jobs=-1, return_train_score=False)
    return {
        "RMSE": float(-res["test_rmse"].mean()), "RMSE_std": float(res["test_rmse"].std()),
        "MAE": float(-res["test_mae"].mean()),
        "R2": float(res["test_r2"].mean()), "R2_std": float(res["test_r2"].std()),
    }


def run_tournament(X, y, groups, cfg) -> pd.DataFrame:
    rows = []
    for name, model in build_models(cfg["seed"]).items():
        m = cv_metrics(model, X, y, groups, cfg["modeling"]["cv_splits"])
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def tune_champion(X, y, groups, cfg) -> dict[str, Any]:
    """RandomizedSearchCV on HistGradientBoosting under GroupKFold (Step-5 grid)."""
    seed = cfg["seed"]
    cv = GroupKFold(n_splits=cfg["modeling"]["cv_splits"])
    param_dist = {
        "learning_rate": [0.02, 0.05, 0.1, 0.2],
        "max_iter": [150, 250, 350, 500],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 40, 60],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_depth": [None, 3, 5],
    }
    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=seed),
        param_distributions=param_dist, n_iter=cfg["modeling"]["tuning_iter"],
        scoring="r2", cv=cv, random_state=seed, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)

    best = search.best_estimator_
    tuned = cv_metrics(best, X, y, groups, cfg["modeling"]["cv_splits"])
    return {"best_estimator": best, "best_params": search.best_params_,
            "cv_best_r2": float(search.best_score_), "tuned_metrics": tuned}
