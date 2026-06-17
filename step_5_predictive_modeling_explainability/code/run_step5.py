"""
run_step5.py — Step 5 orchestrator (Modeling & Explainability).

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Per the Presentation 3+4 rubric:
    load -> VIF collinearity -> Boruta feature selection
    -> model tournament (GroupKFold CV) -> tune champion (HGB)
    -> serialize models -> SHAP plots -> leaderboard -> report

Usage:
    python code/run_step5.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

import explain  # noqa: E402
import feature_selection as fs  # noqa: E402
import modeling  # noqa: E402
from io_load import build_xy, encode_features, load_cleaned, load_config, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 78


def main() -> None:
    cfg = load_config()
    graphs = resolve(cfg["paths"]["out_graphs"])
    models_dir = resolve(cfg["paths"]["out_models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    targets = cfg["targets"]

    print(_hr())
    print("STEP 5 — PREDICTIVE MODELING & EXPLAINABILITY")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    df = load_cleaned(cfg)
    print(f"\n[DATA] modeling rows {len(df)} (Step-4 consensus outliers excluded) | "
          f"features {list(encode_features(df, cfg).columns)}")

    # ---------------- Collinearity (VIF) --------------------------------- #
    vif = fs.compute_vif(df, cfg)
    print("\n[COLLINEARITY] VIF on numeric candidates:")
    for _, r in vif.iterrows():
        print(f"    {r['feature']:14s} VIF={r['VIF']:8.2f}  {r['flag']}")
    dropped = cfg["collinearity"]["drop_for_collinearity"]
    print(f"    -> dropping {dropped} (collinear with 'cluster', r=0.97); keeping 'cluster'.")

    leaderboards: dict[str, pd.DataFrame] = {}
    tuned_store: dict[str, dict] = {}
    boruta_store: dict[str, dict] = {}
    plots: list[Path] = []

    for target in targets:
        print("\n" + _hr("-"))
        print(f"TARGET: {target}")
        X, y, groups = build_xy(df, target, cfg)
        print(f"    n={len(y)}  schools(groups)={groups.nunique()}  features={X.shape[1]}")

        # ----- Boruta feature selection ----- #
        bor = fs.run_boruta(X, y, cfg)
        boruta_store[target] = bor
        print(f"    Boruta confirmed : {bor['confirmed']}")
        print(f"    Boruta tentative : {bor['tentative']}")
        Xsel = X[bor["selected"]]
        print(f"    -> using {len(bor['selected'])} selected feature(s) for models")

        # ----- Tournament (CV) ----- #
        lb = modeling.run_tournament(Xsel, y, groups, cfg)
        leaderboards[target] = lb
        print("    tournament (GroupKFold CV):")
        for _, r in lb.iterrows():
            print(f"      {r['model']:22s} R2={r['R2']:+.3f}  RMSE={r['RMSE']:.3f}  "
                  f"MAE={r['MAE']:.3f}")

        # ----- Tune champion (HGB) ----- #
        tuned = modeling.tune_champion(Xsel, y, groups, cfg)
        tuned_store[target] = {**tuned, "features": list(Xsel.columns)}
        tm = tuned["tuned_metrics"]
        print(f"    TUNED HistGradientBoosting -> R2={tm['R2']:+.3f} "
              f"RMSE={tm['RMSE']:.3f} MAE={tm['MAE']:.3f}")
        print(f"      best params: {tuned['best_params']}")

        # ----- Serialize final model ----- #
        model_path = models_dir / f"{target}_hgb.joblib"
        joblib.dump({"model": tuned["best_estimator"], "features": list(Xsel.columns),
                     "target": target, "cv_metrics": tm}, model_path)

        # ----- SHAP ----- #
        plots.append(explain.shap_beeswarm(tuned["best_estimator"], Xsel, target, cfg, graphs))

    # ---------------- Leaderboard artefacts ------------------------------ #
    plots.append(explain.plot_leaderboard(leaderboards, graphs))
    rows = []
    for t, lb in leaderboards.items():
        for _, r in lb.iterrows():
            rows.append({"target": t, **r.to_dict()})
    lb_all = pd.DataFrame(rows)
    lb_all.to_csv(resolve(cfg["paths"]["leaderboard_out"]), index=False, encoding="utf-8-sig")

    # ---------------- Final report -------------------------------------- #
    print("\n" + _hr())
    print("FINAL CROSS-VALIDATED LEADERBOARD (champion = tuned HistGradientBoosting)")
    print(_hr())
    print(f"{'target':32s}{'best model':24s}{'R2':>8}{'RMSE':>9}{'MAE':>9}")
    for t in targets:
        tm = tuned_store[t]["tuned_metrics"]
        print(f"{t:32s}{'HGB (tuned)':24s}{tm['R2']:+8.3f}{tm['RMSE']:9.3f}{tm['MAE']:9.3f}")

    print("\n[ARTEFACTS]")
    print(f"    serialized models : {len(targets)} -> {models_dir.name}/*.joblib")
    print(f"    leaderboard csv   : {Path(cfg['paths']['leaderboard_out']).name}")
    print("    graphs:")
    for p in plots:
        print(f"      - {Path(p).name:38s} ({Path(p).stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 5 COMPLETE ✔")
    print(_hr())


if __name__ == "__main__":
    main()
