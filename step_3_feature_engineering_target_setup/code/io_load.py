"""
io_load.py — Step 3 input loading.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Loads the central config and Step 2's merged master table
(``merged_bagrut_ses.csv``), which is at the subject-cell grain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Step root = the directory that contains config.yaml (one level above code/).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(rel_path: str) -> Path:
    """Resolve a config-relative path against this step's root."""
    return (ROOT / rel_path).resolve()


def load_merged(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return the Step 2 merged table (subject-cell grain)."""
    cfg = cfg or load_config()
    df = pd.read_csv(resolve(cfg["paths"]["merged_in"]), encoding=cfg["io"]["encoding"])

    required = ["semel", "year", "subject", "grade", "takers", "studyunits"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"merged_bagrut_ses.csv missing columns {missing}. "
                       "Re-run Step 2 first.")
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = load_merged(cfg)
    print(f"[io_load] merged: {df.shape}")
