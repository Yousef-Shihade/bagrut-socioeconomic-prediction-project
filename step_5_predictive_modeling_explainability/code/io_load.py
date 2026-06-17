"""
io_load.py — Step 5 input loading & feature-matrix construction.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Loads Step 4's cleaned modeling table and builds, per target, the predictor
matrix X (municipal/socioeconomic features, one-hot encoded settlement type),
the target vector y, and the GroupKFold groups (``semel``). index_value is
deliberately NOT a predictor (collinearity with cluster — handled in
feature_selection.py); it is still loaded so the VIF report can show why.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    return (ROOT / rel_path).resolve()


def load_cleaned(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load Step 4 cleaned table; optionally drop consensus outliers."""
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["cleaned_in"]), encoding=cfg["io"]["encoding"])
    flag = cfg["features"].get("exclude_outlier_flag")
    if flag and flag in df.columns:
        df = df[~df[flag].astype(bool)].reset_index(drop=True)
    return df


def encode_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return the full encoded predictor frame (numeric + one-hot categoricals)."""
    feats = cfg["features"]
    num = feats["numeric"]
    cats = feats["categorical"]
    X = df[num].copy()
    for c in cats:
        dummies = pd.get_dummies(df[c].astype("Int64").astype("object"),
                                 prefix=c, dummy_na=False)
        X = pd.concat([X, dummies.astype(int)], axis=1)
    return X


def build_xy(df: pd.DataFrame, target: str, cfg: dict[str, Any]):
    """Return (X, y, groups) for one target, dropping rows with a missing target."""
    group_col = cfg["features"]["group_col"]
    X_all = encode_features(df, cfg)
    keep = df[target].notna()
    X = X_all[keep].reset_index(drop=True)
    y = df.loc[keep, target].reset_index(drop=True)
    groups = df.loc[keep, group_col].reset_index(drop=True)
    return X, y, groups


if __name__ == "__main__":
    cfg = load_config()
    df = load_cleaned(cfg)
    print(f"[io_load] cleaned (outliers excluded): {df.shape}")
    for t in cfg["targets"]:
        X, y, g = build_xy(df, t, cfg)
        print(f"  {t:30s} X={X.shape}  y={len(y)}  groups={g.nunique()}")
    print("  feature columns:", list(encode_features(df, cfg).columns))
