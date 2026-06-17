"""
matching.py — Multi-stage Bagrut <-> CBS alignment & merge.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Deterministic four-pass alignment (per UNIQUE Bagrut city key), each city is
resolved to exactly one CBS ``locality_code`` so the final join is strictly
one-CBS-locality-to-many-Bagrut-records (no row explosion):

    Stage 1  exact       — city_norm == locality_norm (1:1)
    Stage 2a structural  — equal after structural_key() (strip '*', hyphen spacing)
    Stage 2b crosswalk   — hardcoded CROSSWALK_NAME / CROSSWALK_CODE
    Stage 3  fuzzy        — token_sort_ratio >= threshold, with a length guard

Anything still unresolved is labelled ``unmatched`` and kept in the output with
NaN socioeconomic fields (we never silently drop rows — Step 3 decides filtering).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

import crosswalk as cw

_SCORERS = {
    "token_sort_ratio": fuzz.token_sort_ratio,
    "WRatio": fuzz.WRatio,
    "ratio": fuzz.ratio,
}


def dedup_ses(ses: pd.DataFrame, key: str, by: str) -> pd.DataFrame:
    """Collapse CBS rows to one per normalised name, keeping the largest ``by``.

    Several CBS localities can share a normalised name (e.g. the city טייבה,
    pop 40,842, vs the moshav טייבה (בעמק), pop 1,751). Keeping the most populous
    row picks the locality a high-school's ``city`` almost certainly refers to and
    prevents a many-to-many join from duplicating Bagrut records.
    """
    ordered = ses.sort_values(by, ascending=False, na_position="last")
    return ordered.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def build_city_mapping(
    bagrut: pd.DataFrame,
    ses_dedup: pd.DataFrame,
    ses_full: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Resolve every unique Bagrut city key to a CBS locality_code + stage label.

    ``ses_dedup`` (one row per normalised name) drives the name-based stages;
    ``ses_full`` (all CBS rows) drives the CROSSWALK_CODE stage, because a pinned
    code may belong to a low-population variant that dedup intentionally dropped
    (e.g. a youth village out-weighed by a same-named moshav).

    Returns a per-city audit log with columns:
        city_norm, n_records, stage, score, matched_locality_norm, matched_code
    """
    m = cfg["matching"]
    bkey, skey, scode = m["bagrut_key"], m["ses_key"], m["ses_code"]
    threshold = m["fuzzy_threshold"]
    min_len_ratio = m["fuzzy_min_len_ratio"]
    scorer = _SCORERS[m["fuzzy_scorer"]]

    name2code = dict(zip(ses_dedup[skey], ses_dedup[scode]))
    exact_names = list(name2code.keys())
    struct_index = {cw.structural_key(n): n for n in exact_names}
    # Full code -> name lookup so pinned crosswalk codes always resolve.
    code2name_full = dict(zip(ses_full[scode], ses_full[skey]))

    # Record counts per city (used for yield logging and bias plots).
    counts = bagrut[bkey].value_counts(dropna=True)

    rows: list[dict[str, Any]] = []
    for city in counts.index:
        rec = {"city_norm": city, "n_records": int(counts[city]),
               "stage": "unmatched", "score": pd.NA,
               "matched_locality_norm": pd.NA, "matched_code": pd.NA}

        # Stage 1 — exact.
        if city in name2code:
            rec.update(stage="exact", matched_locality_norm=city,
                       matched_code=name2code[city])
            rows.append(rec); continue

        # Stage 2a — structural transform on both sides.
        skey_struct = cw.structural_key(city)
        if skey_struct in struct_index:
            target = struct_index[skey_struct]
            rec.update(stage="structural", matched_locality_norm=target,
                       matched_code=name2code[target])
            rows.append(rec); continue

        # Stage 2b — hardcoded crosswalk (name first, then code).
        if city in cw.CROSSWALK_NAME:
            target = cw.CROSSWALK_NAME[city]
            if target in name2code:
                rec.update(stage="crosswalk", matched_locality_norm=target,
                           matched_code=name2code[target])
                rows.append(rec); continue
        if city in cw.CROSSWALK_CODE:
            code = cw.CROSSWALK_CODE[city]
            if code in code2name_full:
                rec.update(stage="crosswalk", matched_code=code,
                           matched_locality_norm=code2name_full[code])
                rows.append(rec); continue

        # Stage 3 — conservative fuzzy.
        cand = process.extractOne(city, exact_names, scorer=scorer)
        if cand is not None and cand[1] >= threshold:
            cand_name = cand[0]
            len_ratio = min(len(city), len(cand_name)) / max(len(city), len(cand_name))
            if len_ratio >= min_len_ratio:
                rec.update(stage="fuzzy", score=round(float(cand[1]), 1),
                           matched_locality_norm=cand_name,
                           matched_code=name2code[cand_name])
                rows.append(rec); continue

        rows.append(rec)  # unmatched

    return pd.DataFrame(rows)


def merge(
    bagrut: pd.DataFrame,
    ses: pd.DataFrame,
    mapping: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Attach stage labels and CBS features to every Bagrut record.

    All Bagrut rows are retained; unmatched rows carry NaN socioeconomic fields.
    """
    m = cfg["matching"]
    bkey, scode = m["bagrut_key"], m["ses_code"]
    feat_cols = cfg["ses_feature_columns"]

    # city -> (stage, code, score) from the audit log.
    city_info = mapping.set_index("city_norm")
    out = bagrut.copy()
    out["match_stage"] = out[bkey].map(city_info["stage"]).fillna("unmatched")
    out["fuzzy_score"] = out[bkey].map(city_info["score"])
    out["matched_code"] = out[bkey].map(city_info["matched_code"])

    # Bring CBS features in on the locality code (one CBS row per code).
    ses_feats = ses.drop_duplicates(subset=scode)[feat_cols].rename(
        columns={"locality_name": "ses_locality_name",
                 "locality_code": "ses_locality_code"})
    out = out.merge(ses_feats, left_on="matched_code", right_on="ses_locality_code",
                    how="left")
    return out
