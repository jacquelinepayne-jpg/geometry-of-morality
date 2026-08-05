"""Generate animal-vs-human concept datasets from animal_human_templates.py.

Writes one CSV per (category pair, axis) to datasets/multi-template/, e.g.
human_companion_harm.csv, human_companion_neutral.csv, ... The first category
in the filename gets label 1. Columns: statement,label,subject,
subject_category,template_id,axis — extra columns are ignored by DataManager.

These pool all templates on an axis into one file, so dataset names carry the
multi-template/ prefix (e.g. 'multi-template/human_farmed_harm'). For the
one-template-per-file variant see make_per_template.py.

Run from the repo root or from datasets/:

    python datasets/make_animal_human.py
"""

import os
from itertools import combinations

import pandas as pd

from animal_human_templates import CATEGORIES, SUBJECTS, TEMPLATES_BY_AXIS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi-template")


def build_rows(axis):
    rows = []
    for template in TEMPLATES_BY_AXIS[axis]:
        for subject, info in SUBJECTS.items():
            rows.append({
                "statement": template["text"].format(subject=info["phrase"]),
                "subject": subject,
                "subject_category": info["category"],
                "template_id": template["template_id"],
                "axis": axis,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for axis in TEMPLATES_BY_AXIS:
        df = build_rows(axis)
        assert df.statement.is_unique, "duplicate sentences found"
        for pos_cat, neg_cat in combinations(CATEGORIES, 2):
            pair = df[df.subject_category.isin([pos_cat, neg_cat])].copy()
            pair["label"] = (pair.subject_category == pos_cat).astype(int)
            pair = pair[["statement", "label", "subject", "subject_category", "template_id", "axis"]]
            path = os.path.join(OUT_DIR, f"{pos_cat}_{neg_cat}_{axis}.csv")
            pair.to_csv(path, index=False)
            print(f"{os.path.basename(path)}: {len(pair)} rows ({pair.label.sum()} pos / {(1 - pair.label).sum()} neg)")
