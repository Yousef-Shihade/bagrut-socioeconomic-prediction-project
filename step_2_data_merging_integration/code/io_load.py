"""
io_load.py — Step 2 input loading.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Loads the central config and the two cleaned caches produced by Step 1
(``bagrut_clean.csv`` and ``ses_clean.csv``). No re-cleaning happens here — Step 2
trusts Step 1's normalised keys (``city_norm`` / ``locality_norm``).
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


def load_clean_inputs(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(bagrut_clean, ses_clean)`` as DataFrames."""
    cfg = cfg or load_config()
    enc = cfg["io"]["encoding"]
    bag = pd.read_csv(resolve(cfg["paths"]["bagrut_clean"]), encoding=enc)
    ses = pd.read_csv(resolve(cfg["paths"]["ses_clean"]), encoding=enc)

    # Sanity guards: the keys Step 2 relies on must be present.
    for col, df, name in [
        (cfg["matching"]["bagrut_key"], bag, "bagrut_clean"),
        (cfg["matching"]["ses_key"], ses, "ses_clean"),
    ]:
        if col not in df.columns:
            raise KeyError(f"Expected key column '{col}' missing from {name}. "
                           f"Re-run Step 1 first.")
    return bag, ses


if __name__ == "__main__":
    cfg = load_config()
    bag, ses = load_clean_inputs(cfg)
    print(f"[io_load] bagrut_clean: {bag.shape}")
    print(f"[io_load] ses_clean:    {ses.shape}")
