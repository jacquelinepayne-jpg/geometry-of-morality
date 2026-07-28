"""Generate animal-vs-human concept datasets from animal_human_templates.py.

The unit is **one template sentence per CSV**, mirroring upstream `cities`
("The city of [city] is in [country]." — a single template, one varying slot).
Here the single varying slot is the subject noun. CSVs are organised into one
folder per category pair, so a file is `datasets/{catA}_{catB}/{template_id}.csv`
and its dataset name — what you pass to --datasets and DataManager — is
`{catA}_{catB}/{template_id}`, e.g. `human_farmed/fire_01`. Each holds one row
per subject in the two categories.

    statement                                                        label
    When the barn caught fire, the man was trapped inside...          1
    When the barn caught fire, the pig was trapped inside...          0

Row count per file = 2 x SUBJECTS_PER_CATEGORY (80). Note this is much smaller
than `cities` (1496): cities varies over hundreds of city/country pairs,
whereas here the varying slot is drawn from a fixed 40-word category list. A
single template file is therefore a good PCA / evaluation unit but a thin
training set — train on several templates concatenated (DataManager takes a
list of datasets) and keep held-out templates for testing.

Pooled CSVs per (category pair, axis) — `human_farmed/harm` and friends — are
also written when generating the full set, since generalization.ipynb and
helping_preference.py refer to those. A pooled file is exactly the
concatenation of its templates: do not generate activations for both, that is
the same forward passes twice.

The first category in the filename gets label 1. Columns: statement, label,
subject, subject_category, template_id, family, axis — extra columns are
ignored by DataManager.

Run from the repo root or from datasets/:

    python datasets/make_animal_human.py --pilot     # small test set, 15 CSVs
    python datasets/make_animal_human.py             # everything, 624 + 12 CSVs
    python datasets/make_animal_human.py --templates fire_01 fire_02 \\
                                         --pairs human_farmed

Then, e.g.:

    python generate_acts.py --datasets human_farmed/fire_01 human_farmed/fire_02 \\
                            --model llama-2-13b --layers 12 15 --device cuda:0
"""

import argparse
import os
from itertools import combinations

import pandas as pd

from animal_human_templates import (
    CATEGORIES,
    SUBJECTS,
    TEMPLATES_BY_AXIS,
    family_of,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLUMNS = ["statement", "label", "subject", "subject_category", "template_id", "family", "axis"]

ALL_PAIRS = [f"{a}_{b}" for a, b in combinations(CATEGORIES, 2)]

# Small test set: two templates from one harm family (within-family transfer),
# one from a different harm family (across-family transfer), and two neutral
# (context transfer) — enough to check that a single-template file is a usable
# unit before committing GPU time to all 624.
PILOT_TEMPLATES = ["fire_01", "fire_02", "neglect_01", "morning_01", "rest_01"]
PILOT_PAIRS = ["human_farmed", "human_wild", "companion_farmed"]


def build_rows(axis):
    rows = []
    for template in TEMPLATES_BY_AXIS[axis]:
        for subject, info in SUBJECTS.items():
            rows.append({
                "statement": template["text"].format(subject=info["phrase"]),
                "subject": subject,
                "subject_category": info["category"],
                "template_id": template["template_id"],
                "family": family_of(template["template_id"]),
                "axis": axis,
            })
    return pd.DataFrame(rows)


def write(frame, pair, name):
    """Write datasets/{pair}/{name}.csv; dataset name is f'{pair}/{name}'."""
    os.makedirs(os.path.join(OUT_DIR, pair), exist_ok=True)
    frame[COLUMNS].to_csv(os.path.join(OUT_DIR, pair, f"{name}.csv"), index=False)


def main(templates, pairs, pooled):
    known = {t["template_id"] for ts in TEMPLATES_BY_AXIS.values() for t in ts}
    if templates is not None:
        unknown = sorted(set(templates) - known)
        assert not unknown, f"unknown template_id(s): {unknown}"
    unknown_pairs = sorted(set(pairs) - set(ALL_PAIRS))
    assert not unknown_pairs, f"unknown pair(s): {unknown_pairs}; choose from {ALL_PAIRS}"

    n_written = 0
    for axis in TEMPLATES_BY_AXIS:
        df = build_rows(axis)
        assert df.statement.is_unique, "duplicate sentences found"

        for pos_cat, neg_cat in combinations(CATEGORIES, 2):
            pair_name = f"{pos_cat}_{neg_cat}"
            if pair_name not in pairs:
                continue
            pair = df[df.subject_category.isin([pos_cat, neg_cat])].copy()
            pair["label"] = (pair.subject_category == pos_cat).astype(int)

            selected = pair if templates is None else pair[pair.template_id.isin(templates)]
            for template_id, group in selected.groupby("template_id", sort=False):
                write(group, pair_name, template_id)
                n_written += 1
                print(f"{pair_name}/{template_id}: {len(group)} rows "
                      f"({group.label.sum()} {pos_cat} / {(1 - group.label).sum()} {neg_cat})")

            if pooled:
                write(pair, pair_name, axis)
                n_written += 1
                print(f"{pair_name}/{axis}: {len(pair)} rows (pooled)")

    print(f"\nwrote {n_written} CSVs to {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pilot", action="store_true",
                   help=f"only {PILOT_TEMPLATES} x {PILOT_PAIRS}")
    p.add_argument("--templates", nargs="+", default=None,
                   help="template_ids to emit (default: all)")
    p.add_argument("--pairs", nargs="+", default=None,
                   help=f"category pairs to emit (default: all of {ALL_PAIRS})")
    p.add_argument("--pooled", action="store_true", default=None,
                   help="also write pooled {pair}_{axis}.csv (default: on unless --pilot)")
    args = p.parse_args()

    templates = PILOT_TEMPLATES if args.pilot else args.templates
    pairs = args.pairs or (PILOT_PAIRS if args.pilot else ALL_PAIRS)
    pooled = (not args.pilot) if args.pooled is None else args.pooled

    main(templates, pairs, pooled)
