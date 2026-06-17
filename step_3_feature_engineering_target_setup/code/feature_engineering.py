"""
feature_engineering.py — re-grain to school level + engineer targets.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Transforms the Step 2 subject-cell table into a SCHOOL-level (semel x year) table
with four engineered targets and the static CBS predictors:

  Target 1 (takers-weighted average grade, observed cells only):
      math_avg_grade      = Sum(grade*takers) / Sum(takers)   over Math cells
      english_avg_grade   = Sum(grade*takers) / Sum(takers)   over English cells
    Weighting by ``takers`` and using only observed grades stops tiny
    privacy-suppressed cells from distorting a school's baseline (the missing
    grades are small-cohort censored — see Step 1's analysis).

  Target 2 (advanced-track participation rate, takers are always observed):
      math_5unit_participation    = takers(5u) / takers(3u+4u+5u)   for Math
      english_5unit_participation = takers(5u) / takers(3u+4u+5u)   for English

We never impute a target here: a grade target is NaN only when every cell for that
subject/school/year was suppressed; a participation target is NaN only when the
school had no academic-track (3/4/5u) takers in that subject.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _aggregate_subject(
    df: pd.DataFrame,
    subject: str,
    prefix: str,
    grain: list[str],
    part_units: list[int],
    adv_units: int,
) -> pd.DataFrame:
    """Vectorised per-(grain) aggregation for one subject.

    Returns columns: <prefix>_avg_grade, <prefix>_5unit_participation,
    <prefix>_takers_total, indexed by ``grain``.
    """
    s = df[df["subject"] == subject].copy()
    if s.empty:
        return pd.DataFrame(
            columns=[f"{prefix}_avg_grade", f"{prefix}_5unit_participation",
                     f"{prefix}_takers_total"]
        )

    # Numerator/denominator pieces for the takers-weighted grade (observed only).
    s["_grade_x_takers"] = s["grade"] * s["takers"]            # NaN where grade NaN
    s["_takers_observed"] = s["takers"].where(s["grade"].notna(), 0)
    # Numerator/denominator pieces for the participation rate (takers complete).
    s["_takers_adv"] = s["takers"].where(s["studyunits"] == adv_units, 0)
    s["_takers_track"] = s["takers"].where(s["studyunits"].isin(part_units), 0)

    agg = s.groupby(grain).agg(
        _gxt=("_grade_x_takers", "sum"),          # sum() skips NaN -> observed only
        _t_obs=("_takers_observed", "sum"),
        _t_adv=("_takers_adv", "sum"),
        _t_track=("_takers_track", "sum"),
        _t_total=("takers", "sum"),
    )

    # Safe divisions: 0/0 -> NaN (no observed grade / no track takers).
    avg_grade = np.where(agg["_t_obs"] > 0, agg["_gxt"] / agg["_t_obs"].replace(0, np.nan),
                         np.nan)
    participation = np.where(agg["_t_track"] > 0,
                             agg["_t_adv"] / agg["_t_track"].replace(0, np.nan), np.nan)

    out = pd.DataFrame({
        f"{prefix}_avg_grade": avg_grade,
        f"{prefix}_5unit_participation": participation,
        f"{prefix}_takers_total": agg["_t_total"].astype(int),
    }, index=agg.index)
    return out


def build_school_level(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build the school-level (semel x year) feature/target table."""
    grain = cfg["grain"]
    core = cfg["subjects"]["core"]
    part_units = cfg["subjects"]["participation_units"]
    adv_units = cfg["subjects"]["advanced_units"]

    math = _aggregate_subject(df, core["math"], "math", grain, part_units, adv_units)
    eng = _aggregate_subject(df, core["english"], "english", grain, part_units, adv_units)

    # Outer join keeps any school-year that has at least one core subject.
    targets = math.join(eng, how="outer")

    # Static metadata per grain (CBS features are constant within a semel;
    # school/city identifiers taken as the first occurrence).
    meta_cols = cfg["cbs_features"] + cfg["id_columns"]
    meta = df.groupby(grain)[meta_cols].first()

    school_level = meta.join(targets, how="right").reset_index()

    # Tidy column order: keys -> ids -> CBS features -> targets -> support.
    target_cols = ["math_avg_grade", "english_avg_grade",
                   "math_5unit_participation", "english_5unit_participation"]
    support_cols = ["math_takers_total", "english_takers_total"]
    ordered = (grain + cfg["id_columns"] + cfg["cbs_features"]
               + target_cols + support_cols)
    ordered = [c for c in ordered if c in school_level.columns]
    school_level = school_level[ordered]
    return school_level.sort_values(grain).reset_index(drop=True)
