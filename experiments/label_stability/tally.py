"""
Tally rater agreement for the label-stability test on one dataset.

Reads every completed rating sheet in experiments/label_stability/<dataset>/ and
<dataset>/rated/ matching rating_sheet_*.csv (one file per rater pass, e.g. a
hand-filled rating_sheet_me.csv alongside rated/rating_sheet_opus_5-2.csv written by
rate.py; the blank template rating_sheet.csv is ignored). Sheets use
the format the rating prompt requests: statement_id,statement,rating,reason with
rating good/bad/depends (1/0/-1 also accepted).

Joins ratings on statement_id against that directory's current rating_sheet.csv, pulls
the intended label from datasets/morality/<dataset>.csv, and reports:
- per-rater rating counts, depends rate, and agreement with the intended label
- per-statement unanimity and the vote breakdown for every non-unanimous statement
- the pre-registered metrics: % unanimous and depends rate

Writes per-statement results to <dataset>/tally.csv. That file, not a flag in
data_gen.py, is where a frame's wording is recorded as holding up: data_gen.py rates
every frame on every round, so a revised frame is always re-rated rather than coasting
on the verdict its old wording earned.

Sheets are matched to statements by statement_id, which is a position in the current
sheet. Regenerating a dataset reshuffles those positions, so a completed sheet is only
valid against the rating_sheet.csv it was generated from — the statement-text check
below catches the mismatch when it happens.

Usage: python experiments/label_stability/tally.py --dataset care_harm
"""

import argparse
import os
from glob import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

RATING_MAP = {
    'good': 'good', '1': 'good',
    'bad': 'bad', '0': 'bad',
    'depends': 'depends', '-1': 'depends',
}


def normalize_rating(r):
    r = str(r).strip().lower()
    if r in ('', 'nan', 'none'):
        return pd.NA
    try:
        r = str(int(float(r)))  # 1.0 -> "1", 0 -> "0", -1 -> "-1"
    except ValueError:
        pass
    return RATING_MAP.get(r, pd.NA)


def load_rater_sheet(path, current):
    rater = os.path.basename(path)[len('rating_sheet_'):-len('.csv')]
    df = pd.read_csv(path)
    df['rating'] = df['rating'].map(normalize_rating)
    if df['rating'].isna().any():
        bad_ids = df.loc[df['rating'].isna(), 'statement_id'].tolist()
        print(f"WARNING [{rater}]: missing or unrecognized rating for ids {bad_ids}")
    merged = df.merge(current[['statement_id', 'statement']], on='statement_id',
                      suffixes=('', '_current'), how='right', indicator=True)
    missing = merged[merged['_merge'] == 'right_only']['statement_id'].tolist()
    if missing:
        print(f"WARNING [{rater}]: no rating for ids {missing}")
    stale = merged[merged['statement'].notna()
                   & (merged['statement'] != merged['statement_current'])]
    if len(stale) > 0:
        print(f"WARNING [{rater}]: statement text differs from current sheet for ids "
              f"{stale['statement_id'].tolist()} (rated an older revision; "
              f"re-rate those statements)")
    return rater, merged.set_index('statement_id')['rating']


def tally_dataset(dataset_name):
    sheet_dir = os.path.join(HERE, dataset_name)
    dataset_csv = os.path.join(REPO_ROOT, 'datasets', 'morality', f'{dataset_name}.csv')
    sheet_csv = os.path.join(sheet_dir, 'rating_sheet.csv')
    out_csv = os.path.join(sheet_dir, 'tally.csv')
    for path in (dataset_csv, sheet_csv):
        if not os.path.exists(path):
            raise SystemExit(f"no such file: {path}")

    dataset = pd.read_csv(dataset_csv)
    current = pd.read_csv(sheet_csv)
    current = current[['statement_id', 'statement']].merge(
        dataset[['statement', 'label']], on='statement', how='left')
    assert current['label'].notna().all(), \
        f"rating_sheet.csv and {dataset_name}.csv disagree; rerun data_gen.py"
    current['intended'] = current['label'].map({1: 'good', 0: 'bad'})

    # hand-filled sheets sit in <dataset>/, model passes from rate.py in <dataset>/rated/
    paths = sorted(glob(os.path.join(sheet_dir, 'rating_sheet_*.csv'))
                   + glob(os.path.join(sheet_dir, 'rated', 'rating_sheet_*.csv')))
    if not paths:
        raise SystemExit(f"no completed sheets in {sheet_dir} or {sheet_dir}/rated "
                         f"(expected rating_sheet_<rater>.csv)")

    tally = current[['statement_id', 'statement', 'intended']].set_index('statement_id')
    raters = []
    for path in paths:
        rater, ratings = load_rater_sheet(path, current)
        raters.append(rater)
        tally[rater] = ratings

    votes = tally[raters]
    tally['n_depends'] = (votes == 'depends').sum(axis=1)
    tally['unanimous'] = votes.apply(lambda row: row.dropna().nunique() == 1
                                     and 'depends' not in row.values, axis=1)
    tally['matches_intended'] = tally['unanimous'] & votes.apply(
        lambda row: row.dropna().iloc[0] if row.dropna().nunique() == 1 else None,
        axis=1).eq(tally['intended'])

    print(f"\n=== {dataset_name} ===")
    print(f"raters ({len(raters)}): {', '.join(raters)}")

    print("\nper rater:")
    for rater in raters:
        col = votes[rater].dropna()
        agree = (votes[rater] == tally['intended']).sum()
        print(f"  {rater}: good {(col == 'good').sum()}, bad {(col == 'bad').sum()}, "
              f"depends {(col == 'depends').sum()} "
              f"({(col == 'depends').mean():.1%}), "
              f"matches intended label {agree}/{len(tally)}")

    n = len(tally)
    unanimous = tally['unanimous'].sum()
    print("\nlabel-stability metrics (pre-registered thresholds in first_test_submission.md):")
    print(f"  unanimous statements: {unanimous}/{n} = {unanimous / n:.1%} "
          f"(predicted >=90%, falsified <80% after one revision round)")
    print(f"  unanimous and matching intended label: {tally['matches_intended'].sum()}/{n}")
    print(f"  statements with any depends vote: {(tally['n_depends'] > 0).sum()}/{n} "
          f"(tripwire ~30%)")

    disagreements = tally[~tally['unanimous']]
    if len(disagreements) > 0:
        print("\nnon-unanimous statements:")
        for sid, row in disagreements.iterrows():
            votes_str = ', '.join(f"{r}={row[r]}" for r in raters)
            print(f"  [{sid}] (intended {row['intended']}) {row['statement']}")
            print(f"        {votes_str}")
    else:
        print("\nall statements unanimous")

    tally.to_csv(out_csv)
    print(f"\nwrote {out_csv}")
    return tally


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', required=True,
                        help='dataset name(s), e.g. care_harm neg_care_harm')
    args = parser.parse_args()
    for name in args.dataset:
        tally_dataset(name)
