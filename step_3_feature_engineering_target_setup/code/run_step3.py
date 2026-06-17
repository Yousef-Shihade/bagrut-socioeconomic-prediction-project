"""
run_step3.py — Step 3 orchestrator (Feature Engineering & Target Setup).

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Pipeline:
    load Step-2 merged -> aggregate to school level (semel x year)
    -> engineer 4 targets -> write school_level_features_targets.csv
    -> 3 target-exploration plots -> validation summary

Usage:
    python code/run_step3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import feature_engineering as fe  # noqa: E402
import visualize  # noqa: E402
from io_load import load_config, load_merged, resolve  # noqa: E402

TARGET_COLS = ["math_avg_grade", "english_avg_grade",
               "math_5unit_participation", "english_5unit_participation"]


def _hr(c: str = "=") -> str:
    return c * 72


def main() -> None:
    cfg = load_config()

    print(_hr())
    print("STEP 3 — FEATURE ENGINEERING & TARGET SETUP")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    # --- 1. Load + aggregate ---------------------------------------------- #
    merged = load_merged(cfg)
    school = fe.build_school_level(merged, cfg)

    # --- 2. Persist -------------------------------------------------------- #
    resolve(cfg["paths"]["out_data"]).mkdir(parents=True, exist_ok=True)
    out_path = resolve(cfg["paths"]["school_level_out"])
    school.to_csv(out_path, index=False, encoding="utf-8-sig")

    # --- 3. Visuals -------------------------------------------------------- #
    plot_paths = visualize.run(school, cfg)

    # --- 4. Validation summary -------------------------------------------- #
    n = len(school)
    print("\n[1] GRAIN CHANGE")
    print(f"    subject-cell rows (Step 2) : {len(merged):,}")
    print(f"    school-level rows          : {n:,}  (grain = {cfg['grain']})")
    print(f"    distinct schools (semel)   : {school['semel'].nunique():,}")
    print(f"    years                      : "
          f"{sorted(school['year'].unique().tolist())}")
    print(f"    final shape                : {school.shape}")

    print("\n[2] ENGINEERED TARGETS — coverage & range")
    print("    target                          non-null         mean     min     max")
    for c in TARGET_COLS:
        s = school[c]
        print(f"    {c:31s} {s.notna().sum():5d} ({s.notna().mean()*100:4.1f}%)  "
              f"{s.mean():7.3f} {s.min():7.2f} {s.max():7.2f}")

    print("\n[3] CBS PREDICTORS retained (per school, constant within semel)")
    print(f"    columns: {cfg['cbs_features']}")
    print(f"    rows with socioeconomic cluster : {school['cluster'].notna().sum():,} "
          f"({school['cluster'].notna().mean()*100:.1f}%)")
    print("    NOTE: district/מחוז is NOT in the CBS extract -> not included "
          "(documented, not fabricated).")

    print("\n[4] FEASIBILITY — Pearson r(cluster, target)  [higher = more SES-linked]")
    d = school[school["cluster"].notna()]
    for c in TARGET_COLS:
        r = d[["cluster", c]].corr().iloc[0, 1]
        print(f"    cluster vs {c:31s} r = {r:+.3f}")

    print("\n[5] OUTPUTS")
    print(f"    {out_path.name}  ({school.shape[0]:,} x {school.shape[1]})")
    print(f"    columns: {list(school.columns)}")
    print("    graphs:")
    for p in plot_paths:
        print(f"      - {Path(p).name:30s} ({Path(p).stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 3 COMPLETE ✔   (awaiting signal for Step 4)")
    print(_hr())


if __name__ == "__main__":
    main()
