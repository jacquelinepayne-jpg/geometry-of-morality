"""
Sentiment pass over a moral-valence dataset (preview of assumption A5, run as the
free add-on to the label-stability test).

Question: how strongly does surface sentiment correlate with the moral label? High
correlation is expected on a naive batch and motivates the planned
sentiment-anticorrelated control set. See the project log entry for A1.

Classifiers:
- VADER (lexicon-based, always runs): compound score in [-1, 1].
- twroberta (runs only if transformers + torch are installed): roberta fine-tuned
  on tweet sentiment, signed score P(positive) - P(negative). 3-class, so the
  signed score discards the neutral mass, which is kept in its own column.
  Chosen over a movie-review checkpoint (distilbert SST-2) because review
  polarity is the wrong construct for A5: SST-2 scored these statements on
  whether the sentence reads like a good review, not on affect.

Reported per classifier at both the item level and the within-pair level. The
within-pair number is the one to quote: the dataset is matched pairs, and a
linear probe has a bias term, so a constant offset in the score is free. Item
level sign accuracy understates a classifier whose threshold is merely shifted.

Outputs experiments/label_stability/<dataset>/sentiment_scores.csv and prints a
summary.

Usage: python experiments/label_stability/sentiment_check.py --dataset care_harm
"""

import argparse
import os

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

TRANSFORMER_MODELS = {
    'twroberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
}
NEUTRAL_BAND = 0.05  # |score| below this counts as neutral for sign accuracy


def summarize(df, score_col):
    scores, labels = df[score_col].values, df['label'].values
    r = np.corrcoef(labels, scores)[0, 1]
    neutral = np.abs(scores) < NEUTRAL_BAND
    sign_correct = (scores > 0) == (labels == 1)
    print(f"\n=== {score_col} ===")
    print(f"correlation with moral label (point-biserial r): {r:.3f}")
    print(f"mean score: good {scores[labels == 1].mean():+.3f}, "
          f"bad {scores[labels == 0].mean():+.3f}")
    print(f"sign predicts label: {sign_correct.mean():.1%} "
          f"({neutral.sum()} statements in neutral band |score| < {NEUTRAL_BAND})")

    # within-pair: does the good member outscore its bad twin, same sentence frame?
    piv = df.pivot(index='pair_id', columns='label', values=score_col)
    gap = piv[1] - piv[0]
    # rank separation, threshold-free: P(random good scores above random bad)
    auc = np.mean([[g > b for b in scores[labels == 0]] for g in scores[labels == 1]])
    print(f"within-pair: good outscores bad in {(gap > 0).sum()}/{len(gap)} pairs, "
          f"mean gap {gap.mean():+.3f}")
    print(f"AUC (threshold-free, what a probe with a bias term can exploit): {auc:.3f}")

    inverted = gap[gap < 0].sort_values()
    if len(inverted) > 0:
        print("pairs where the bad member scores higher (sentiment fights the label):")
        for pair_id in inverted.index:
            good = df[(df.pair_id == pair_id) & (df.label == 1)].statement.iloc[0]
            bad = df[(df.pair_id == pair_id) & (df.label == 0)].statement.iloc[0]
            print(f"  pair {pair_id} (gap {inverted[pair_id]:+.3f})")
            print(f"    good {piv[1][pair_id]:+.3f}  {good}")
            print(f"    bad  {piv[0][pair_id]:+.3f}  {bad}")


def score_dataset(dataset_name):
    dataset_csv = os.path.join(REPO_ROOT, 'datasets', f'{dataset_name}.csv')
    if not os.path.exists(dataset_csv):
        raise SystemExit(f"no such file: {dataset_csv}")
    out_csv = os.path.join(HERE, dataset_name, 'sentiment_scores.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    print(f"\n######## {dataset_name} ########")
    df = pd.read_csv(dataset_csv)

    vader = SentimentIntensityAnalyzer()
    df['vader_compound'] = [vader.polarity_scores(s)['compound'] for s in df['statement']]
    summarize(df, 'vader_compound')

    try:
        from transformers import pipeline
    except ImportError:
        print("\ntransformers/torch not installed, skipping the transformer passes "
              "(see the venv recipe in the project log)")
    else:
        for key, model_name in TRANSFORMER_MODELS.items():
            clf = pipeline('sentiment-analysis', model=model_name, top_k=None)
            results = clf(df['statement'].tolist())
            signed, neutral = [], []
            for res in results:
                # label case and class count vary by checkpoint: sst2 is 2-class
                # POSITIVE/NEGATIVE, twitter-roberta is 3-class lowercase with neutral
                probs = {d['label'].lower(): d['score'] for d in res}
                signed.append(probs.get('positive', 0.0) - probs.get('negative', 0.0))
                neutral.append(probs.get('neutral', float('nan')))
            df[f'{key}_signed'] = signed
            if not np.isnan(neutral).all():
                df[f'{key}_neutral'] = neutral
            summarize(df, f'{key}_signed')

    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', required=True,
                        help='dataset name(s), e.g. care_harm neg_care_harm')
    args = parser.parse_args()
    for name in args.dataset:
        score_dataset(name)
