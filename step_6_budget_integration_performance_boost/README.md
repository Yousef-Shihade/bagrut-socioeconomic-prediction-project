# Step 6 — Institutional Budget Integration & Performance Boost

**Project:** Predicting Israeli High School Bagrut Success Using Socioeconomic Data
**Authors:** Yousef Shehade & Shada Esawi

> Step 5 proved that **municipal socioeconomic status alone is a bounded
> predictor** of school outcomes (R² ≈ 0.06–0.19). Step 6 tests the natural next
> hypothesis: *does school-level **institutional resourcing** break that ceiling?*
> We bolt a **third dataset** (Ministry of Education school-budget report) onto the
> finalized Step-4 matrix, joining strictly on the school code `semel`, engineer
> **two complementary budget-per-student ratios**, and re-run the **exact Step-5
> GroupKFold tournament**. The answer is **yes — decisively.** A dual-budget
> feature pair **more than doubles** the mean explained variance
> (**mean ΔR² = +0.132**), and on every target the institutional spend ranks at or
> above the municipal cluster.

---

## 1. Directory structure

```
step_6_budget_integration_performance_boost/
├── README.md
├── config.yaml
├── code/
│   ├── io_load.py          # robust budget-xlsx parser (openpyxl colour-patch) + loaders
│   ├── budget_features.py  # DUAL budget-per-student engineering + strict semel join
│   ├── modeling.py         # Step-5 tournament + tuned HGB (identical protocol)
│   ├── explain.py          # understanding + performance visuals (SHAP, pred-vs-actual…)
│   └── run_step6.py        # orchestrator + console summary report
├── data/
│   └── consolidated_3dataset_matrix.csv     # Bagrut × CBS × Budget (3,662 rows)
├── models/                 # 4 budget-augmented tuned HGB champions + leaderboards
└── graphs/
    ├── dataset_understanding/   # dual-budget distributions, budget×cluster, corr, missingness
    └── model_performance/       # before/after R², SHAP beeswarms, predicted-vs-actual
```

Run: `python code/run_step6.py` (paths anchored to the step folder; CWD-independent).
**Steps 1–5 are untouched** — this stage only *reads* Step-4's output.

---

## 2. Robust parsing of the budget workbook

The Ministry file (`datasets/school_budget.xlsx`, sheet `MyWorkSheet-1`, 4,718
institutions × 54 columns) ships a **malformed `styles.xml`** (non-aRGB theme
colours). A vanilla `openpyxl.load_workbook` aborts with
`ValueError: Colors must be aRGB hex values`. We **monkeypatch openpyxl's `RGB`
descriptor with a lenient validator before opening the file**
([io_load.py](code/io_load.py)), so style parsing can never crash the load, then
read in `read_only` + `data_only` mode.

Parsing notes handled in code:
- **Header is row 1; row 2 is a grand-totals row** (`סה"כ`) — dropped.
- Hebrew headers carry stray double-spaces (e.g. `'עלות  שעות הוראה'`) →
  whitespace-normalised so the config column map matches.

---

## 3. Dual-budget micro-feature engineering

The first iteration used a single ratio built on `עלות שעות הוראה + תקציב גפ"ן`.
Diagnostics exposed two flaws: **Gefen is entirely zero** in this extract, and the
**teaching-hours column is zero for ~half the schools** (instructional costs of
many recognised/independent institutions sit elsewhere). We therefore **drop Gefen
completely** and engineer **two independent, un-diluted ratios** from the columns
that actually carry signal:

```
total_budget_per_student    =  'סה"כ תקציב שכר ותשלומים' (col 51)  /  students
teaching_budget_per_student =  'עלות שעות הוראה'        (col 20)  /  students
```

| Feature | Source | Nonzero coverage | Median ₪/student | Captures |
|---|---|--:|--:|---|
| `total_budget_per_student` | grand-total budget (col 51) | **91.1 %** | 15,960 | overall institutional resourcing |
| `teaching_budget_per_student` | teaching-hours cost (col 20) | 51.8 % | 9,491 | **specific instructional staffing** |

**Why both, not one?** They are complementary, not redundant. The grand total is
comprehensive (it covers the schools the teaching column misses), while the
teaching-hours ratio isolates *instructional* spend — the lever that most directly
governs whether a school can staff a **5-unit Math** track. Empirically, keeping
both beats either alone on **all four targets**.

**Robust guards:** division-by-zero / empty student counts (`students ≤ 0`) →
`±inf` is replaced with `NaN` and excluded, never propagated into the models. The
budget is a single fiscal-year snapshot, joined **statically per school** and
broadcast across that school's 2013–2016 rows.

---

## 4. Strict `semel` join & consolidated 3-dataset matrix

Schools and budgets share the **same institutional code**, so this is a clean key
join (unlike the noisy Hebrew-name join of Step 2):

| Join metric | Value |
|---|---|
| Step-4 schools matched to a budget | **979 / 1,011 (96.8 %)** |
| Step-4 rows receiving a budget | **3,604 / 3,662 (98.4 %)** |
| Consolidated matrix | **3,662 rows × 32 columns** |

**Final school-profile feature space (model inputs):**
- **SES (Step 5, "before"):** `cluster`, `log_population`, `year`, one-hot
  `locality_form` (12 settlement-type dummies)
- **NEW institutional (Step 6, "after"):** **`total_budget_per_student`**,
  **`teaching_budget_per_student`**

A key property (see `graphs/dataset_understanding/budget_per_student_by_cluster.png`):
**budget per student is near-orthogonal to the socioeconomic cluster (r ≈ 0.03)** —
the opposite of `index_value` (r = 0.97, dropped in Step 5). That independence is
exactly *why* it adds predictive power: it encodes information the cluster does not.

---

## 5. The re-run tournament — Before vs After (the headline result)

**Methodology — apples-to-apples.** For every target we tune the champion
HistGradientBoosting **twice on the *identical* budget-matched rows**: once on the
SES-only feature set (**Before**) and once with the two budget ratios added
(**After**). Same rows, same GroupKFold(`semel`) folds, same RandomizedSearchCV
grid — so the **ΔR² is attributable to the budget features alone**.

### Tuned HistGradientBoosting champion — R² (CV, GroupKFold by school)

| 🎯 Target | Rows | Before (SES only) | **After (+ dual budget)** | **ΔR²** | RMSE before → after |
|---|--:|--:|--:|--:|--:|
| `english_5unit_participation` | 3,565 | 0.209 | **0.391** | **+0.182** 🔥 | 0.243 → 0.213 |
| `math_5unit_participation` | 3,546 | 0.050 | **0.211** | **+0.161** 🔥 | 0.101 → 0.092 |
| `math_avg_grade` | 3,210 | 0.084 | **0.208** | **+0.123** | 7.228 → 6.716 |
| `english_avg_grade` | 3,188 | 0.160 | **0.220** | **+0.060** | 5.967 → 5.746 |

> **Mean ΔR² across the four targets: +0.132** — more than double the single-budget
> prototype (+0.054). Every target improves materially; **Math 5-unit participation
> quadruples** (0.05 → 0.21) and **English 5-unit selection nearly doubles**.

### Side-by-side: Step 5 baseline vs Step 6 dual-budget

| 🎯 Target | Step 5 — SES only (2-dataset) | **Step 6 — SES + dual budget (3-dataset)** | Gain |
|---|--:|--:|--:|
| `english_5unit_participation` | 0.192 | **0.391** | **+0.199** |
| `math_5unit_participation` | 0.056 | **0.211** | **+0.155** |
| `math_avg_grade` | 0.094 | **0.208** | **+0.114** |
| `english_avg_grade` | 0.150 | **0.220** | **+0.070** |

*(Step-6 "before" numbers differ slightly from Step 5's because they are recomputed
on the budget-matched subset — that is exactly why the same-rows comparison above
is the rigorous one.)*

See [before_after_comparison.csv](models/before_after_comparison.csv) and the full
4-model board [leaderboard_step6_cv.csv](models/leaderboard_step6_cv.csv).

---

## 6. SHAP — overall spend, instructional spend & the municipal cluster

**The institutional budget displaces the municipal cluster exactly where it
matters most.** In the budget-augmented champion for `math_5unit_participation` —
the *single least SES-predictable target in Step 5* — the SHAP ranking is:

1. **`total_budget_per_student`** (#1)
2. **`teaching_budget_per_student`** (#2)
3. `cluster` (#3)

Both engineered budget features outrank the socioeconomic cluster, with high spend
pushing advanced-Math participation up
(`graphs/model_performance/shap_beeswarm_math_5unit_participation.png`).

This supplies the **mechanism** behind the Step-4/Step-5 *"Math is resilient to
socioeconomics"* finding: Math advanced-track selection is governed not by *town
wealth* (`cluster`) but by **school-level resourcing** — overall budget for capacity,
plus targeted instructional spend for staffing the 5-unit track. For English,
budget contributes strongly while `cluster` retains its pull, consistent with
English being the more SES-linked subject throughout the project.

---

## 7. Interpretation — the ceiling, and what broke it

- **Step 5 thesis upheld and sharpened:** municipal SES is a *bounded* predictor.
  Much of the variance it cannot explain is **institutional capacity**, which the
  dual-budget ratios recover.
- **The boost is broad but largest for *selection*:** budget predicts **who is
  offered / enters the advanced track** (a capacity/policy outcome) most strongly,
  but with the better numerators it now lifts **grades** substantially too
  (Math grade R² more than doubled).
- **"Policy overrides postcode" now has direct, quantified evidence:** the
  resources a school commands per pupil overtake its town's socioeconomic rank in
  explaining the most resilient outcomes.

---

## 8. Step 6 verification checklist

- [x] Budget workbook parsed despite the styles.xml colour error (openpyxl patch).
- [x] Grand-totals row dropped; Hebrew headers whitespace-normalised.
- [x] **Two** budget-per-student ratios engineered (grand-total + teaching); empty Gefen dropped.
- [x] inf / div-by-zero sanitised to NaN with robust guards.
- [x] Strict `semel` join (96.8 % schools / 98.4 % rows) → 3,662×32 consolidated matrix saved.
- [x] Final row count + full feature column list logged to console.
- [x] Dataset-understanding plots saved (dual distributions, budget×cluster, correlation, missingness).
- [x] Exact Step-5 tournament re-run; tuned HGB **before vs after on identical rows** with RandomizedSearchCV.
- [x] Comparative before/after + Step-5-baseline leaderboards saved (ΔR² isolated).
- [x] SHAP beeswarms + predicted-vs-actual saved; both budget features outrank `cluster` for Math 5-unit.
- [x] 4 budget-augmented champions serialized to `models/*_hgb_budget.joblib`.

**Status: Step 6 complete ✔ — the dual-budget integration lifts mean R² by +0.132
(more than 2× the single-ratio prototype), quadruples Math advanced-track
predictability, and makes institutional spend the top driver above municipal SES.**
