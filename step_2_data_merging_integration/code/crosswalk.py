"""
crosswalk.py — Stage 2 structural crosswalk for the Bagrut <-> CBS merge.

Project: Predicting Israeli High School Bagrut Success Using Socioeconomic Data
Authors: Yousef Shehade & Shada Esawi

Stage 2 of the multi-stage alignment handles *systematic* differences between the
two sources that an exact key match cannot. It has two parts:

1. ``structural_key()`` — a rule-based transform applied to BOTH sides that
   removes the CBS footnote asterisk ("*") and normalises whitespace around
   hyphens. This alone resolves administrative spellings such as:
       מודיעין-מכבים-רעות*  ==  מודיעין-מכבים-רעות
       בנימינה-גבעת עדה*    ==  בנימינה-גבעת עדה
       מולדה*               ==  מולדה
       תל אביב - יפו        ==  תל אביב -יפו   (hyphen spacing)

2. Two hardcoded lookup dictionaries for genuine name/qualifier differences that
   no generic rule can infer. Every entry below was verified by hand against the
   CBS ``ses_clean`` table (see the Step 2 README for the mapping log):

   * ``CROSSWALK_NAME`` — Bagrut ``city_norm`` -> CBS ``locality_norm`` (by name).
     Used for spelling variants and Arabic/Hebrew name pairs.

   * ``CROSSWALK_CODE`` — Bagrut ``city_norm`` -> CBS ``locality_code`` (by id).
     Used where a single normalised name maps to SEVERAL distinct CBS localities
     and we must pin the *correct* one. Example: "בן שמן כפר נוער" is the youth
     village (code 1084, cluster 1), NOT the same-named moshav (code 2013,
     cluster 9). Mapping by code guarantees the right socioeconomic value.
"""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")
_HYPHEN_SPACES_RE = re.compile(r"\s*-\s*")


def structural_key(value: str) -> str:
    """Rule-based Stage-2 transform applied to both key sets.

    Removes the CBS footnote asterisk and collapses whitespace around hyphens so
    administratively-equal names align.
    """
    text = str(value).replace("*", "")
    text = _HYPHEN_SPACES_RE.sub("-", text)
    return _WS_RE.sub(" ", text).strip()


# Bagrut city_norm  ->  CBS locality_norm  (genuine spelling / name-pair fixes)
CROSSWALK_NAME: dict[str, str] = {
    "יהוד-מונוסון":        "יהוד מונוסון",     # hyphen vs space
    "עופרה":               "עפרה",             # vav spelling variant (Ofra)
    "נוה":                 "נווה",             # vav-doubling
    "אבו קרינאת יישוב":    "אבו קורינאת",      # vav spelling + dropped (יישוב) qualifier
    "ג ש גוש חלב":         "ג ש",              # Jish: CBS keeps the short form ג'ש
    "פקיעין בוקייעה":      "פקיעין",           # Peki'in / Buqei'a -> CBS "פקיעין"
    "תראבין א-צאנע ישוב":  "תרבין א-צאנע *",   # Bedouin locality, alef/spelling variant
}

# Bagrut city_norm  ->  CBS locality_code  (pin the correct same-named locality)
CROSSWALK_CODE: dict[str, int] = {
    "כנרת קבוצה":       57,    # Kinneret (Kvutza), cluster 5  — not the Moshava
    "עין חרוד מאוחד":   82,    # Ein Harod (Me'uhad), cluster 6
    "גבעת חיים איחוד":  2018,  # Givat Haim (Ihud), cluster 7
    "בן שמן כפר נוער":  1084,  # Ben Shemen youth village, cluster 1 — not the Moshav
    "קריית יערים מוסד": 2039,  # Kiryat Ye'arim (institution), cluster 2
}
