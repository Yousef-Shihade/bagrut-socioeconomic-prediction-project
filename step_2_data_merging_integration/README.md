# Step 2 — Data Merging & Integration

**Project:** Predicting Israeli High School Bagrut Success Using Socioeconomic Data
**Authors:** Yousef Shehade & Shada Esawi

> Self-contained milestone folder. Step 2 joins the Step-1 cleaned caches into a
> single school-records table enriched with CBS socioeconomic data, using a
> deterministic **multi-stage alignment** that maximises data retention and logs
> every decision for the "Possible Data Biases" slide.

**External data sources**

The raw source files are git-ignored (not redistributed). To reproduce, download
them into the project-root `datasets/` folder and re-run Steps 1–2:

- **Dataset 1 — Bagrut Grades 2013–2016** (Israeli Freedom of Information Law),
  hosted on Kaggle: <https://www.kaggle.com/datasets/emachlev/bagrut-israel/data>
- **Dataset 2 — CBS Socioeconomic Index** (the socioeconomic cluster joined in
  this step): official publication of the **Israel Central Bureau of Statistics**
  (CBS / הלשכה המרכזית לסטטיסטיקה) — <https://www.cbs.gov.il/>.

---

## 1. What Step 2 accomplishes

1. Loads the Step-1 caches (`bagrut_clean.csv`, `ses_clean.csv`).
2. **Deduplicates the CBS table** so the join cannot explode rows.
3. Runs a **4-pass alignment** (Exact → Structural → Crosswalk → Fuzzy) matching
   `city_norm` ↔ `locality_norm`, resolving each city to one CBS `locality_code`.
4. Merges CBS features (cluster, index value, population, …) onto **every** Bagrut
   record (unmatched rows kept with NaN — nothing is silently dropped).
5. Writes `merged_bagrut_ses.csv` + a full `city_mapping_log.csv` audit trail.
6. Produces 3 merge-health / bias plots and a printed verification summary.

**Headline result: 69,246 / 69,638 records matched = 99.44 %** (98.90 % carry a
usable cluster 1–10; the gap is matched-but-CBS-unranked `..` localities).

---

## 2. Directory structure

```
step_2_data_merging_integration/
├── README.md                     # this file
├── config.yaml                   # paths to Step-1 caches + Step-2 outputs, matching params
├── code/
│   ├── io_load.py                # load config + Step-1 caches (key sanity checks)
│   ├── crosswalk.py              # structural_key() + hardcoded CROSSWALK_NAME / CROSSWALK_CODE
│   ├── matching.py               # dedup_ses, build_city_mapping (4 passes), merge
│   ├── visualize.py              # the 3 merge-health plots
│   └── run_step2.py              # orchestrator + verification & bias log
├── data/
│   ├── merged_bagrut_ses.csv     # FINAL merged dataset (69,638 × 18)
│   └── city_mapping_log.csv      # per-city audit: stage, score, matched code/name
└── graphs/
    ├── match_yield_waterfall.png
    ├── missingness_bias_by_size.png
    └── socioeconomic_representation.png
```

Run from this folder: `python code/run_step2.py` (also works via full path from the
project root; paths are anchored to the step folder, not the CWD).

---

## 3. Why the join key is the city name (not `semel`)

The Bagrut `semel` is a **school** code; the CBS `סמל היישוב` is a **locality**
code — different numbering systems with **zero** overlap. The only viable join is
on the normalised locality name produced in Step 1.

### 3a. Mandatory CBS de-duplication (prevents row explosion)
Nine CBS localities collapse to a shared normalised name after Step-1 cleaning —
typically a large city plus a tiny same-named moshav/kibbutz, e.g.:

| normalised name | variant A (kept) | variant B (dropped) |
|---|---|---|
| `טייבה` | city, pop 40,842, cluster 3 | `טייבה (בעמק)`, pop 1,751 |
| `טמרה` | city, pop 32,048, cluster 2 | `טמרה (יזרעאל)`, pop 1,513 |
| `כנרת`, `גבעת חיים`, `עין חרוד`, `בן שמן`, `מרחביה`, `אשדות יעקב`, `קריית יערים` | … | … |

We keep **the most populous** variant per name. Left unhandled, these duplicates
would turn the merge many-to-many and **inflate** the high-volume city records
(טייבה/טמרה are large). Verified: input 69,638 → output 69,638 rows (no explosion).

---

## 4. The multi-stage alignment strategy

Each unique Bagrut city key (315 of them) is resolved in order:

| Stage | Method | Cities | Records | Cum. match % |
|------:|--------|------:|--------:|-------------:|
| 1 — Exact | `city_norm == locality_norm` | 291 | 64,605 | 92.77 % |
| 2a — Structural | equal after `structural_key()` (strip `*`, normalise hyphen spacing) | 4 | 3,425 | 97.69 % |
| 2b — Crosswalk | hardcoded `CROSSWALK_NAME` / `CROSSWALK_CODE` | 12 | 1,035 | 99.18 % |
| 3 — Fuzzy | `token_sort_ratio ≥ 90` + length-ratio guard | 3 | 181 | 99.44 % |
| — Unmatched | kept with NaN CBS fields | 5 | 392 | (0.56 % residual) |

### Stage 2a — structural transform (`crosswalk.structural_key`)
Strips the CBS footnote `*` and collapses whitespace around hyphens, applied to
both sides. Auto-resolves: `מודיעין-מכבים-רעות*`, `בנימינה-גבעת עדה*`, `מולדה*`,
and `תל אביב - יפו` ↔ `תל אביב -יפו`.

### Stage 2b — hardcoded crosswalk (`crosswalk.py`)
- `CROSSWALK_NAME` — genuine spelling / name-pair fixes, e.g. `עופרה→עפרה`,
  `נוה→נווה`, `יהוד-מונוסון→יהוד מונוסון`, `אבו קרינאת יישוב→אבו קורינאת`,
  `ג ש גוש חלב→ג ש`, `פקיעין בוקייעה→פקיעין`, `תראבין א-צאנע ישוב→תרבין א-צאנע *`.
- `CROSSWALK_CODE` — pins the **correct** CBS locality when one name maps to
  several. Critically these target low-population variants that de-dup drops, so
  they are resolved against the **full** CBS table:
  `בן שמן כפר נוער→1084` (youth village, **cluster 1**, not the moshav cluster 9),
  `כנרת קבוצה→57`, `עין חרוד מאוחד→82`, `גבעת חיים איחוד→2018`,
  `קריית יערים מוסד→2039`.

### Stage 3 — conservative fuzzy (`rapidfuzz`)
`token_sort_ratio ≥ 90` plus a length-ratio guard (≥ 0.6). `token_sort_ratio`
was chosen over `WRatio` because `WRatio` inflates substring matches and would
falsely link e.g. `מרכז אזורי שוהם → שוהם` (90) or `כפר זוהרים → זוהר`; the guard
+ scorer reject all of these. **All 3 accepted matches are pure yod-doubling
variants**, each manually confirmed correct:

| Bagrut | → CBS | score |
|---|---|---|
| `צפריה` | `צפרייה` | 90.9 |
| `טובא-זנגריה` | `טובא-זנגרייה` | 95.7 |
| `שומריה` | `שומרייה` | 92.3 |

---

## 5. Possible data biases (the unmatched 0.56 %)

5 city keys / **392 records** remain unmatched. They are **non-municipal by
nature** (so the absence is structural, not a quality defect):

| City key | Records | Why unmatched |
|---|--:|---|
| `מקווה ישראל` | 194 | agricultural youth village (institution, no locality cluster) |
| `בתי ספר של מרחבים` | 91 | "schools of Merhavim" — regional-council schools, not a town |
| `מרכז אזורי שוהם` | 43 | regional center serving many localities |
| `כפר זוהרים` | 35 | absent from the CBS index |
| `קדמה` | 29 | educational community, absent from the CBS index |

**Bias direction check:** median `takers` is **21 (matched) vs 23 (unmatched)** —
essentially identical, so the data loss is **not biased by school size** (it is
not systematically dropping small or large schools). The loss is concentrated in
boarding/regional institutions, which we document rather than force-match.

> **Representation note (Plot 3):** at the record level, clusters **9–10 are
> heavily under-represented** (1.5 % / 0.0 %) while clusters 2, 7, 8 dominate.
> High-SES localities are few and rarely host large public high schools. This is
> a real-world representation limitation to flag for modelling, not a merge bug.

---

## 6. Output schema — `merged_bagrut_ses.csv` (69,638 × 18)

Bagrut columns (`grade`, `takers`, `studyunits`, `year`, `subject`, `city`,
`school`, `semel`, `city_norm`) **+** merge metadata (`match_stage`,
`fuzzy_score`, `matched_code`) **+** CBS features (`ses_locality_code`,
`ses_locality_name`, `locality_form`, `population`, `index_value`, `cluster`).

`city_mapping_log.csv` records, for all 315 cities: `stage`, `score`,
`matched_locality_norm`, `matched_code`, `n_records` — the full audit trail.

---

## 7. Dependencies

`pandas`, `numpy`, `pyyaml`, `matplotlib`, `seaborn`, and **`rapidfuzz` 3.14.5**
(installed this step). Anaconda Python 3.11.

---

## 8. Step 2 verification checklist

- [x] Step-1 caches loaded; required keys present.
- [x] CBS de-duplicated 1,208 → 1,199; most-populous variant kept (טייבה→city).
- [x] 4-pass alignment executed; per-stage yield logged.
- [x] **Total match rate 99.44 %** (target was 95–98 %); cluster coverage 98.90 %.
- [x] Code-pinned variants verified (בן שמן כפר נוער → cluster 1, not 9).
- [x] Fuzzy stage conservative — 3 matches, all correct, 0 false positives.
- [x] **No row explosion**: 69,638 → 69,638.
- [x] `merged_bagrut_ses.csv` (69,638 × 18) + `city_mapping_log.csv` written.
- [x] 3/3 merge-health plots saved to `graphs/`.
- [x] Unmatched residuals logged with bias direction (no size bias).

**Status: Step 2 complete ✔ — awaiting signal to begin Step 3 (Feature Engineering & Target Setup).**
