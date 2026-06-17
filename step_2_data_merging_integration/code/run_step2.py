"""
run_step2.py — Step 2 orchestrator (Data Merging & Integration).

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Pipeline:
    load Step-1 caches -> dedup CBS -> 4-pass alignment -> merge
    -> write merged_bagrut_ses.csv + city_mapping_log.csv -> 3 plots -> report

Usage:
    python code/run_step2.py        (from this step folder, or via full path)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import matching  # noqa: E402
import visualize  # noqa: E402
from io_load import load_clean_inputs, load_config, resolve  # noqa: E402


def _hr(c: str = "=") -> str:
    return c * 72


def main() -> None:
    cfg = load_config()
    m = cfg["matching"]

    print(_hr())
    print("STEP 2 — DATA MERGING & INTEGRATION")
    print(cfg["project"]["title"])
    print("Authors: " + " & ".join(cfg["project"]["authors"]))
    print(_hr())

    # --- 1. Load Step-1 caches -------------------------------------------- #
    bag, ses = load_clean_inputs(cfg)

    # --- 2. Dedup CBS so the join cannot explode rows --------------------- #
    ses_dedup = matching.dedup_ses(ses, m["ses_key"], m["ses_dedup_by"])

    # --- 3. Four-pass alignment ------------------------------------------- #
    mapping = matching.build_city_mapping(bag, ses_dedup, ses, cfg)

    # --- 4. Merge (retain all Bagrut rows) -------------------------------- #
    merged = matching.merge(bag, ses, mapping, cfg)

    # --- 5. Persist outputs ----------------------------------------------- #
    out_data = resolve(cfg["paths"]["out_data"])
    out_data.mkdir(parents=True, exist_ok=True)
    merged_path = resolve(cfg["paths"]["merged_out"])
    mapping_path = resolve(cfg["paths"]["mapping_out"])
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    mapping.sort_values(["stage", "n_records"], ascending=[True, False]) \
           .to_csv(mapping_path, index=False, encoding="utf-8-sig")

    # --- 6. Visuals ------------------------------------------------------- #
    plot_paths = visualize.run(merged, cfg)

    # --- 7. Verification + bias logging ----------------------------------- #
    total = len(merged)
    matched = int(merged["matched_code"].notna().sum())
    with_cluster = int(merged["cluster"].notna().sum())
    rec_by_stage = merged["match_stage"].value_counts()
    key_by_stage = mapping["stage"].value_counts()

    print("\n[1] CBS DEDUPLICATION (prevents row explosion)")
    print(f"    CBS rows {len(ses)} -> {len(ses_dedup)} unique localities "
          f"(kept max {m['ses_dedup_by']} per name)")

    print("\n[2] STAGE-BY-STAGE YIELD            cities      records     cum.match%")
    cum = 0
    for stage in ["exact", "structural", "crosswalk", "fuzzy", "unmatched"]:
        nrec = int(rec_by_stage.get(stage, 0))
        nkey = int(key_by_stage.get(stage, 0))
        if stage != "unmatched":
            cum += nrec
        cum_pct = cum / total * 100
        tag = "" if stage != "unmatched" else "   (residual)"
        print(f"    {stage:11s}              {nkey:4d}      {nrec:8,d}     "
              f"{cum_pct:6.2f}%{tag}")

    print("\n[3] TOTAL MATCH RATE")
    print(f"    records matched to a CBS locality : {matched:,}/{total:,} "
          f"= {matched/total*100:.2f}%")
    print(f"    records with a usable cluster 1-10: {with_cluster:,}/{total:,} "
          f"= {with_cluster/total*100:.2f}%  "
          f"(gap = matched-but-unranked '..' localities)")

    print("\n[4] FUZZY MATCHES ACCEPTED (token_sort_ratio >= "
          f"{m['fuzzy_threshold']}, conservative)")
    fz = mapping[mapping["stage"] == "fuzzy"].sort_values("n_records", ascending=False)
    for _, r in fz.iterrows():
        print(f"    {r['city_norm']!r:24s} -> {r['matched_locality_norm']!r:18s} "
              f"score={r['score']}  ({int(r['n_records'])} recs)")

    print("\n[5] UNMATCHED RESIDUALS (kept in output with NaN SES — Possible Data Bias)")
    um = mapping[mapping["stage"] == "unmatched"].sort_values("n_records", ascending=False)
    print(f"    {len(um)} city keys, {int(um['n_records'].sum()):,} records "
          f"({um['n_records'].sum()/total*100:.2f}%). These are youth villages / "
          "regional-council\n    schools / localities absent from the CBS index "
          "(non-municipal by nature):")
    for _, r in um.iterrows():
        print(f"      - {r['city_norm']!r:22s} {int(r['n_records']):4d} records")

    # Bias direction: compare cohort sizes.
    med_m = merged.loc[merged["cluster"].notna(), "takers"].median()
    med_u = merged.loc[merged["cluster"].isna(), "takers"].median()
    print(f"\n    bias check — median takers: matched={med_m:.0f} vs "
          f"unmatched={med_u:.0f}  "
          f"({'no strong size bias' if abs(med_m-med_u) < med_m*0.5 else 'size-skewed'})")

    print("\n[6] FINAL MERGED FILE")
    print(f"    {merged_path.name}: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    print(f"    columns: {list(merged.columns)}")
    print(f"    mapping audit log: {mapping_path.name} ({len(mapping)} cities)")
    print(f"    row-explosion check: input {len(bag):,} -> output {len(merged):,}  "
          f"({'OK, 1:many preserved' if len(merged) == len(bag) else 'EXPLOSION!'})")

    print("\n[7] GRAPHS")
    for p in plot_paths:
        print(f"    - {Path(p).name:34s} ({Path(p).stat().st_size/1024:6.1f} KB)")

    print("\n" + _hr())
    print("STEP 2 COMPLETE ✔   (awaiting signal for Step 3: Feature Engineering)")
    print(_hr())


if __name__ == "__main__":
    main()
