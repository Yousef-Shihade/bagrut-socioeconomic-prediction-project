# Step 3 — Feature Engineering & Target Setup (Grain Aggregation)

**Project:** Predicting Israeli High School Bagrut Success Using Socioeconomic Data
**Authors:** Yousef Shehade & Shada Esawi

> Self-contained milestone folder. Step 3 re-grains Step 2's subject-cell master
> table to the **school level (`semel` × `year`)** and engineers the four targets
> the project is built around, keeping the static CBS socioeconomic predictors.

---

## 1. What Step 3 accomplishes

- Re-grains `merged_bagrut_ses.csv` (69,638 subject-cell rows) → **3,731
  school-year rows** (1,022 distinct schools × 2013–2016).
- Engineers **4 targets**: takers-weighted average grades (Math, English) and
  advanced-track (5-unit) participation rates (Math, English).
- Retains the static CBS predictors (cluster, index value, population,
  settlement form) mapped onto each school row.
- Produces 3 exploratory plots of the targets vs the socioeconomic cluster.

**Output:** `data/school_level_features_targets.csv` (**3,731 × 17**).

---

## 2. Directory structure

```
step_3_feature_engineering_target_setup/
├── README.md
├── config.yaml                       # paths, grain, subject defs, feature lists
├── code/
│   ├── io_load.py                    # load config + Step-2 merged table
│   ├── feature_engineering.py        # build_school_level: aggregation + 4 targets
│   ├── visualize.py                  # the 3 target-exploration plots
│   └── run_step3.py                  # orchestrator + validation summary
├── data/
│   └── school_level_features_targets.csv
└── graphs/
    ├── cluster_vs_participation.png
    ├── cluster_vs_avg_grade.png
    └── target_distributions.png
```

Run: `python code/run_step3.py` (paths anchored to the step folder; CWD-independent).

---

## 3. Modeling grain decision — `semel` × `year`

We aggregate to **one row per school per year** rather than one row per school:

- **Volume:** 3,731 rows vs ~1,022 — roughly 4× the training data for the
  baseline ML models.
- **Structure:** preserves year-over-year variation and keeps the door open for
  the optional **Trajectory Analysis** method (a 2013→2016 path per school).
- **Leakage guard:** because each school recurs across years with identical CBS
  features, downstream modeling must use **`GroupKFold(semel)`** so the same
  school never spans train/test. (Flagged here for Steps 4–5.)

CBS features were verified **constant within each `semel`** (0 schools with
conflicting clusters), so taking the first value per school is exact.

---

## 4. Engineered features — data dictionary

| Column | Type | Description |
|---|---|---|
| `semel` | id | School institution code (grain key). |
| `year` | id | Exam year 2013–2016 (grain key). |
| `school`, `city`, `city_norm` | id | School name / raw city / normalised city key. |
| `match_stage` | id | How the school's city matched CBS in Step 2 (exact/structural/crosswalk/fuzzy/unmatched) — for quality filtering. |
| **`cluster`** | feature | **CBS socioeconomic cluster 1–10** (main predictor). NaN for unmatched localities. |
| `index_value` | feature | CBS continuous socioeconomic index value. |
| `population` | feature | Locality population. |
| `locality_form` | feature | CBS settlement-type code (צורת יישוב). |
| `ses_locality_name` | feature | Matched CBS locality name. |
| **`math_avg_grade`** | target | **Takers-weighted** mean Math grade, observed cells only: `Σ(grade·takers)/Σ(takers)`. |
| **`english_avg_grade`** | target | Same, for English. |
| **`math_5unit_participation`** | target | `takers(5u) / takers(3u+4u+5u)` for Math ∈ [0,1]. |
| **`english_5unit_participation`** | target | Same, for English ∈ [0,1]. |
| `math_takers_total`, `english_takers_total` | support | Total test-takers per subject (volume / denominator transparency). |

### Why takers-weighted, observed-only grades?
The raw `grade` is ~21 % missing because the source **suppresses small-cohort
averages** (privacy; missingness depends on `takers` — see Step 1). Weighting by
`takers` and averaging only observed cells means a school's baseline reflects its
real students and is **not** distorted by tiny suppressed cells. We therefore do
**not** impute these targets: `*_avg_grade` is NaN only if every cell for that
subject/school/year was suppressed.

---

## 5. Target coverage & ranges (from the run)

| Target | Non-null | Mean | Min | Max |
|---|--:|--:|--:|--:|
| `math_avg_grade` | 3,292 (88.2 %) | 78.68 | 50.09 | 98.18 |
| `english_avg_grade` | 3,280 (87.9 %) | 80.81 | 40.81 | 97.46 |
| `math_5unit_participation` | 3,668 (98.3 %) | 0.087 | 0.00 | 0.87 |
| `english_5unit_participation` | 3,688 (98.8 %) | 0.325 | 0.00 | 1.00 |

Participation targets are near-complete (`takers` is never suppressed); grade
targets are ~88 % complete (small-cohort suppression). 99.1 % of rows carry a
socioeconomic cluster.

**Distribution note (Plot 3):** grades are ~normal; **`math_5unit_participation`
is strongly zero-inflated** (many schools offer no 5-unit Math) and both rates are
[0,1]-bounded — relevant for model/transform choices in Step 4+.

---

## 6. Feasibility — socioeconomic signal

Pearson r between `cluster` and each target (higher ⇒ more SES-linked):

| Target | r(cluster) |
|---|--:|
| `english_5unit_participation` | **+0.346** |
| `english_avg_grade` | **+0.306** |
| `math_5unit_participation` | +0.219 |
| `math_avg_grade` | +0.176 |

All four rise with socioeconomic status, confirming analytical feasibility.
**English is markedly more SES-sensitive than Math** on both grade and
participation — i.e. **Math looks more "resilient" to socioeconomic disparity**,
directly informing the secondary research question.

---

## 7. Important data limitation — district / מחוז

The downloaded CBS extract (`downloadFile.xlsx`) contains **no district/מחוז
column** (its columns are locality code, name, settlement-form, population, index
value, cluster). District is therefore **not** included in the feature set — we
document the absence rather than fabricate it. `locality_form` is retained as the
available structural geographic descriptor. If a district feature is required for
Step 4+, it would need an external locality→district crosswalk.

---

## 8. Step 3 verification checklist

- [x] Re-grained 69,638 subject-cells → 3,731 school-year rows (1,022 schools).
- [x] Math & English isolated (exact subject strings; both use units 3/4/5).
- [x] `math_avg_grade` / `english_avg_grade` = takers-weighted, observed-only.
- [x] `math_5unit_participation` / `english_5unit_participation` = 5u / (3u+4u+5u), in [0,1].
- [x] CBS predictors retained (constant per semel); district documented as absent.
- [x] Targets not imputed (NaN where fully suppressed) — methodology preserved.
- [x] 3/3 target-exploration plots saved; positive SES gradient confirmed.
- [x] `school_level_features_targets.csv` (3,731 × 17) written.

**Status: Step 3 complete ✔ — awaiting signal to begin Step 4.**
