"""
run_step6.py — orchestrator for the budget-integration performance-boost stage.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Flow:
  1. Load + robustly parse the Ministry budget workbook; engineer budget_per_student.
  2. Strict `semel` join onto the finalized Step-4 matrix -> 3-dataset matrix.
  3. Inventory: final row count + full feature column list; save understanding plots.
  4. Re-run the exact Step-5 GroupKFold tournament; tune HGB BEFORE (SES only) and
     AFTER (+budget) on IDENTICAL budget-matched rows -> isolate the budget effect.
  5. Save SHAP / predicted-vs-actual / before-after plots, serialized champions,
     leaderboards, and a console summary report.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget_features as bf
import explain as ex
import io_load as io
import modeling as ml

ROOT = Path(__file__).resolve().parents[1]


def _p(rel: str) -> Path:
    return (ROOT / rel).resolve()


def main() -> None:
    cfg = io.load_config()
    seed = cfg["seed"]
    targets = cfg["targets"]
    g_und = _p(cfg["paths"]["out_graphs_understanding"])
    g_perf = _p(cfg["paths"]["out_graphs_performance"])
    models_dir = _p(cfg["paths"]["out_models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 6 — INSTITUTIONAL BUDGET INTEGRATION & PERFORMANCE BOOST")
    print("=" * 78)

    # --- 1. Parse + engineer ------------------------------------------------
    budget_raw = io.load_budget_raw(cfg)
    print(f"\n[1] Budget workbook parsed (style-error bypassed): {budget_raw.shape[0]} "
          f"institutions x {budget_raw.shape[1]} cols")
    budget, bdiag = bf.engineer_budget(budget_raw, cfg)
    print(f"    DUAL-BUDGET ratios engineered for {bdiag['n_budget_schools']} schools "
          "(empty Gefen column dropped):")
    print(f"      total_budget_per_student    : {bdiag['total_nonzero']} nonzero "
          f"({100*bdiag['total_nonzero']/bdiag['n_budget_schools']:.1f}%), median={bdiag['total_median']:,.0f}")
    print(f"      teaching_budget_per_student : {bdiag['teach_nonzero']} nonzero "
          f"({100*bdiag['teach_nonzero']/bdiag['n_budget_schools']:.1f}%), median={bdiag['teach_median']:,.0f}")

    # --- 2. Strict semel join ----------------------------------------------
    step4 = io.load_cleaned(cfg)
    merged, jdiag = bf.merge_budget(step4, budget)
    print(f"\n[2] Strict semel join onto Step-4 matrix:")
    print(f"    schools matched: {jdiag['schools_matched']}/{jdiag['schools_total']} "
          f"({jdiag['schools_matched_pct']}%)  |  rows with budget: "
          f"{jdiag['rows_with_budget']}/{jdiag['rows_total']} ({jdiag['rows_with_budget_pct']}%)")
    merged.to_csv(_p(cfg["paths"]["consolidated_out"]), index=False, encoding="utf-8-sig")

    # --- 3. Pipeline inventory ---------------------------------------------
    base_feats = cfg["features"]["base_numeric"]
    budget_feats = cfg["features"]["budget_numeric"]
    cat = cfg["features"]["categorical"]
    print("\n[3] CONSOLIDATED 3-DATASET MATRIX INVENTORY")
    print(f"    final shape: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print(f"    school-profile feature space (model inputs):")
    print(f"      numeric (SES)  : {base_feats}")
    print(f"      numeric (BUDGET): {budget_feats}")
    print(f"      categorical   : {cat} (one-hot)")
    print(f"    full column list: {list(merged.columns)}")

    print("\n    saving dataset-understanding plots ...")
    ex.plot_budget_distribution(merged, g_und)
    ex.plot_budget_by_cluster(merged, g_und)
    ex.plot_correlation_heatmap(merged, g_und)
    ex.plot_missingness(merged, g_und)

    # --- 4. Re-run tournament + before/after tuned champions ---------------
    print("\n[4] RE-RUN MODELING TOURNAMENT (GroupKFold by semel)")
    tournaments, comparison_rows, champions = {}, [], {}
    for t in targets:
        Xb, yb, gb = bf.build_xy(merged, t, cfg, with_budget=False)
        Xa, ya, ga = bf.build_xy(merged, t, cfg, with_budget=True)  # same rows

        lb = ml.run_tournament(Xa, ya, ga, cfg)          # full 4-model board (after)
        lb.insert(0, "target", t)
        tournaments[t] = lb

        before = ml.tune_champion(Xb, yb, gb, cfg)["tuned_metrics"]
        after = ml.tune_champion(Xa, ya, ga, cfg)
        champions[t] = after["best_estimator"]
        am = after["tuned_metrics"]

        comparison_rows.append({
            "target": t, "n_rows": len(ya), "n_schools": ga.nunique(),
            "R2_before": before["R2"], "R2_after": am["R2"], "dR2": am["R2"] - before["R2"],
            "RMSE_before": before["RMSE"], "RMSE_after": am["RMSE"],
            "MAE_before": before["MAE"], "MAE_after": am["MAE"],
        })
        print(f"    {t:30s} R2: {before['R2']:+.3f} -> {am['R2']:+.3f}  "
              f"(dR2 {am['R2']-before['R2']:+.3f}) | RMSE {before['RMSE']:.3f} -> {am['RMSE']:.3f}")

        joblib.dump(after["best_estimator"], models_dir / f"{t}_hgb_budget.joblib")

    comparison = pd.DataFrame(comparison_rows)
    full_board = pd.concat(tournaments.values(), ignore_index=True)
    comparison.to_csv(_p(cfg["paths"]["comparison_out"]), index=False, encoding="utf-8-sig")
    full_board.to_csv(_p(cfg["paths"]["leaderboard_out"]), index=False, encoding="utf-8-sig")

    # --- 5. Performance visuals --------------------------------------------
    print("\n[5] saving model-performance plots (before/after, SHAP, pred-vs-actual) ...")
    ex.plot_before_after(comparison, g_perf)
    for t in targets:
        Xa, ya, ga = bf.build_xy(merged, t, cfg, with_budget=True)
        ex.shap_beeswarm(champions[t], Xa, t, cfg, g_perf)
        ex.plot_predicted_vs_actual(champions[t], Xa, ya, ga, t, cfg, g_perf)

    # --- summary report -----------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY REPORT — DID INSTITUTIONAL BUDGET BREAK THE CEILING?")
    print("=" * 78)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(comparison[["target", "n_rows", "n_schools",
                          "R2_before", "R2_after", "dR2"]].to_string(index=False))
    best = comparison.loc[comparison["dR2"].idxmax()]
    print(f"\n  largest gain : {best['target']}  (dR2 {best['dR2']:+.3f})")
    print(f"  mean dR2 across 4 targets: {comparison['dR2'].mean():+.4f}")
    print("\nArtifacts:")
    print(f"  - consolidated matrix : {cfg['paths']['consolidated_out']}")
    print(f"  - before/after table  : {cfg['paths']['comparison_out']}")
    print(f"  - tournament board    : {cfg['paths']['leaderboard_out']}")
    print(f"  - champions           : models/*_hgb_budget.joblib")
    print(f"  - graphs              : graphs/dataset_understanding/ , graphs/model_performance/")
    print("\nStep 6 complete.")


if __name__ == "__main__":
    main()
