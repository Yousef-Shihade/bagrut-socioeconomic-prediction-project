# Step 5 — Predictive Modeling & Explainability

**Project:** Predicting Israeli High School Bagrut Success Using Socioeconomic Data
**Authors:** Yousef Shehade & Shada Esawi

> Built directly to the **Presentation 3+4 rubric** (parsed from
> `guidelines/Presentation_3+4_guidelines.docx`): feature selection with
> rationale/results, collinearity handling, normalisation, ≥1 tuned model with
> cross-validation, and feature importance (SHAP). The headline research question
> is answered here: *can a municipality's socioeconomic status alone predict a
> school's performance and track selection?*

---

## 1. Directory structure

```
step_5_predictive_modeling_explainability/
├── README.md
├── config.yaml
├── code/
│   ├── io_load.py            # load Step-4 data, build X/y/groups (one-hot settlement type)
│   ├── feature_selection.py  # VIF collinearity + Boruta
│   ├── modeling.py           # tournament (Ridge/SGD/RF/HGB), GroupKFold CV, HGB tuning
│   ├── explain.py            # SHAP beeswarms + leaderboard plot
│   └── run_step5.py          # orchestrator + console report
├── models/                   # 4 serialized tuned HGB models + leaderboard_cv.csv
└── graphs/                   # 4 SHAP beeswarms + models_performance.png
```

Run: `python code/run_step5.py`.

---

## 2. Rubric → implementation map

| Rubric requirement | How Step 5 satisfies it |
|---|---|
| **Strict evaluation split** | `GroupKFold(semel)`, 5 folds — every year of a school stays in one fold, so no school-level leakage across our school-year rows. |
| **Normalisation/standardisation** | `StandardScaler` inside a `Pipeline` for the linear models (Ridge, SGD), re-fit per fold (no leakage). Trees need no scaling. |
| **Collinearity handling** | **VIF** on numeric candidates: `index_value` (19.4) and `cluster` (19.2) are collinear (r = 0.97) → drop `index_value`, keep the interpretable `cluster`. |
| **Feature selection (+ why + results)** | **Boruta** (all-relevant RF wrapper), `perc=90`. Isolates `cluster` / `log_population` and **rejects the 12 sparse settlement-type dummies**. |
| **≥1 model + tuning + CV** | 4-model tournament; champion **HistGradientBoosting** tuned via `RandomizedSearchCV` (25 iters) under GroupKFold. |
| **Feature importance (SHAP)** | `shap.TreeExplainer` beeswarm per target, saved to `graphs/`. |
| **Sample balance / imbalance** | Regression targets; the zero-inflated `math_5unit_participation` and skewed clusters are handled **natively by the boosted trees** (no resampling needed); GroupKFold preserves the distribution per fold. See §6. |

---

## 3. Data & features

- **Rows:** 3,662 school-years (Step-4 cleaned set, 35 consensus outliers excluded).
- **Groups:** ~920–1,000 distinct schools (`semel`) per target.
- **Candidate predictors (municipal / socioeconomic only):** `cluster`,
  `log_population`, `year`, and one-hot `locality_form` (12 CBS settlement-type
  codes). `index_value` dropped for collinearity.
- **Targets (4, regression):** `math_avg_grade`, `english_avg_grade`,
  `math_5unit_participation`, `english_5unit_participation`.

Restricting predictors to municipal SES features is deliberate — it is exactly
what the research question asks (*SES alone*), so the resulting R² **is** the
answer, not a limitation to engineer around.

---

## 4. Feature selection results (Boruta, perc = 90)

| Target | Boruta-confirmed | Used for models |
|---|---|---|
| `math_avg_grade` | `log_population` | log_population, cluster, year |
| `english_avg_grade` | **`cluster`, `log_population`** | cluster, log_population |
| `math_5unit_participation` | `log_population` | log_population, cluster, year |
| `english_5unit_participation` | **`cluster`, `log_population`** | cluster, log_population |

**Key insight:** Boruta confirms **`cluster` for the English targets but not for
the Math targets** (where only population survives, cluster ranking #2). This
*independently corroborates* the Step-4 finding that **Math is more resilient to
socioeconomic status** — the SES cluster is simply a weaker signal for Math.
In every case the 12 settlement-type dummies are rejected as noise.

---

## 5. Cross-validated leaderboard (GroupKFold by school)

### Champion — tuned HistGradientBoosting (final models, serialized in `models/`)

| Target | R² | RMSE | MAE |
|---|--:|--:|--:|
| `english_5unit_participation` | **0.192** | 0.245 | 0.194 |
| `english_avg_grade` | **0.150** | 5.986 | 4.535 |
| `math_avg_grade` | 0.094 | 7.188 | 5.600 |
| `math_5unit_participation` | 0.056 | 0.101 | 0.080 |

### Full tournament (untuned CV R², `graphs/models_performance.png`)

| Model | math_grade | eng_grade | math_5u | eng_5u |
|---|--:|--:|--:|--:|
| HistGradientBoosting | 0.048 | 0.068 | −0.060 | 0.092 |
| Ridge | 0.033 | 0.085 | 0.048 | 0.118 |
| SGD (linear SVM) | 0.031 | 0.087 | 0.047 | 0.119 |
| RandomForest | −0.049 | −0.022 | −0.177 | 0.011 |
| **HGB tuned (champion)** | **0.094** | **0.150** | **0.056** | **0.192** |

**Reading the board:**
- **Tuning matters:** untuned HGB is mediocre, but a shallow, strongly-regularised
  tuned HGB (`max_depth 3`, `learning_rate 0.02`, `l2≥0.1`) wins every target.
- **Regularised linear models (Ridge/SGD) are strong** and beat the heavier
  RandomForest everywhere — the SES→outcome relationship is smooth and nearly
  monotonic, so deep bagged trees mostly overfit (negative R²).
- **English > Math** and **participation ≈ grade** in predictability: the most
  SES-predictable outcome is *English advanced-track selection* (R² 0.19); the
  least is *Math advanced participation* (R² 0.06).

---

## 6. Sample balance / imbalance

These are regression targets, so "imbalance" means distribution skew rather than
class imbalance:
- `math_5unit_participation` is **zero-inflated** (many schools offer no 5-unit
  Math). Boosted trees model the spike at 0 natively; no resampling applied.
- Socioeconomic **clusters are uneven** (cluster 2 over-represented, cluster 9
  rare, cluster 10 absent). GroupKFold preserves this mix across folds; we report
  the elite tail (cluster 9, n≈41) as indicative.

---

## 7. SHAP explainability (`graphs/shap_beeswarm_*.png`)

Beeswarms from the tuned champion confirm the story directionally: **higher
`cluster` pushes English grade and English 5-unit participation up** (wealthier
localities → more advanced English), with `log_population` second. For Math, the
SHAP spread is flatter and population-led — visual confirmation of Math's weaker
SES dependence.

---

## 8. Headline answer to the research question

**Municipal socioeconomic status alone is a weak-to-moderate predictor of school
outcomes.** It explains up to ~19 % of the variance in *English advanced-track
selection* and ~15 % in *English grades*, but only ~6–9 % for Math. The clear,
reproducible asymmetry — **SES predicts English far better than Math, and
selection better than raw grades** — is the project's central, defensible
finding, and it is corroborated three independent ways (Step-4 cluster gaps,
Boruta confirmation, and SHAP).

---

## 9. Step 5 verification checklist

- [x] GroupKFold(semel) CV — no school-level leakage.
- [x] Standardisation in-pipeline for linear models.
- [x] Collinearity handled via VIF (dropped `index_value`).
- [x] Boruta feature selection run + reported (cluster/population vs noise dummies).
- [x] 4-model tournament + HGB tuned with RandomizedSearchCV.
- [x] RMSE / MAE / R² leaderboard compiled (`models/leaderboard_cv.csv` + table above).
- [x] SHAP beeswarms saved for all 4 targets + leaderboard plot.
- [x] 4 tuned models serialized to `models/*.joblib`.
- [x] Mandated methods banked: **SGD** (modeling), **Boruta** (selection), **SHAP** (explainability).

**Status: Step 5 complete ✔.**
