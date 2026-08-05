"""Generate single-template concept datasets from animal_human_templates.py.

Every CSV here uses exactly ONE scenario template, so the only thing that
varies within a file is the subject word. That removes the between-template
variance that otherwise dominates PCA of the pooled datasets.

Writes one CSV per (category pair, template) to
datasets/single-template/<pair>/<template>.csv:

    datasets/single-template/human_animal/flood_01.csv
    datasets/single-template/human_farmed/flood_01.csv
    datasets/single-template/farmed_wild/morning_01.csv
    ...

Dataset names therefore carry the single-template/ prefix, e.g.
'single-template/human_farmed/flood_01'. For the all-templates-pooled variant
see make_animal_human.py, which writes to datasets/multi-template/.

The first category in the directory name gets label 1. `human_animal` contrasts
humans against a balanced sample drawn evenly from companion/farmed/wild, so
the two sides stay 40 v 40 and no animal subtype dominates the contrast.

Run from the repo root or from datasets/:

    python datasets/make_per_template.py
    python datasets/make_per_template.py --templates fire_01 flood_01 morning_01 rest_01
    python datasets/make_per_template.py --templates all
"""

import argparse
import os
import random
from itertools import combinations

import pandas as pd

from animal_human_templates import CATEGORIES, SUBJECTS, TEMPLATES_BY_AXIS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single-template")

# two harm templates and two neutral templates by default; override with --templates
DEFAULT_TEMPLATES = ["fire_01", "flood_01", "morning_01", "rest_01"]

ANIMAL_CATEGORIES = [c for c in CATEGORIES if c != "human"]
BALANCE_SEED = 0


def templates_by_id():
    return {t["template_id"]: (t, axis) for axis, ts in TEMPLATES_BY_AXIS.items() for t in ts}


def subjects_in(category):
    return [s for s, info in SUBJECTS.items() if info["category"] == category]


def balanced_animal_sample(n):
    """Sample n animal subjects, spread as evenly as possible over the animal
    categories. Seeded so the datasets are reproducible."""
    rng = random.Random(BALANCE_SEED)
    quotas = [n // len(ANIMAL_CATEGORIES)] * len(ANIMAL_CATEGORIES)
    for i in range(n - sum(quotas)):
        quotas[i] += 1
    sampled = []
    for category, quota in zip(ANIMAL_CATEGORIES, quotas):
        sampled += rng.sample(sorted(subjects_in(category)), quota)
    return sampled


def build(template, axis, pos_subjects, neg_subjects):
    rows = []
    for label, subjects in ((1, pos_subjects), (0, neg_subjects)):
        for subject in subjects:
            info = SUBJECTS[subject]
            rows.append({
                "statement": template["text"].format(subject=info["phrase"]),
                "label": label,
                "subject": subject,
                "subject_category": info["category"],
                "template_id": template["template_id"],
                "axis": axis,
            })
    return pd.DataFrame(rows)


def pairings():
    """(directory name, positive subjects, negative subjects) for every contrast."""
    for pos_cat, neg_cat in combinations(CATEGORIES, 2):
        yield f"{pos_cat}_{neg_cat}", subjects_in(pos_cat), subjects_in(neg_cat)
    humans = subjects_in("human")
    yield "human_animal", humans, balanced_animal_sample(len(humans))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES,
                        help="template_ids to build, or 'all' for every template")
    args = parser.parse_args()

    lookup = templates_by_id()
    template_ids = sorted(lookup) if args.templates == ["all"] else args.templates
    unknown = [t for t in template_ids if t not in lookup]
    if unknown:
        raise SystemExit(f"unknown template_id(s): {unknown}\navailable: {sorted(lookup)}")

    for pair_name, pos_subjects, neg_subjects in pairings():
        pair_dir = os.path.join(OUT_DIR, pair_name)
        os.makedirs(pair_dir, exist_ok=True)
        for template_id in template_ids:
            template, axis = lookup[template_id]
            df = build(template, axis, pos_subjects, neg_subjects)
            assert df.statement.is_unique, f"duplicate sentences in {pair_name}/{template_id}"
            assert df.template_id.nunique() == 1, "a per-template dataset must use one template"
            df.to_csv(os.path.join(pair_dir, f"{template_id}.csv"), index=False)
        print(f"{pair_name}/: {len(template_ids)} files, "
              f"{len(pos_subjects)} pos / {len(neg_subjects)} neg each")
