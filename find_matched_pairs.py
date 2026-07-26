"""
Find the good/bad pairs in a moral dataset whose two members tokenize to the same number of
tokens — the pairs `patching.py` can actually use, since activation patching is by position
index and a length mismatch makes the two runs' positions incomparable.

Loads only the tokenizer, so this runs in seconds on any machine with HuggingFace access —
no GPU, no model weights. Run it before picking the pair to hardcode into `patching.py`.

Pairs are printed shortest differing span first. The span is the number of trailing token
positions from the first difference to the end of the prompt, which is exactly what
`patching.py` sweeps: a shorter span means fewer forward passes and a tighter localization.

Usage:
    python find_matched_pairs.py --model llama-2-13b
    python find_matched_pairs.py --model llama-2-13b --dataset neg_care_harm --all
    python find_matched_pairs.py --model llama-2-13b --csv matched_pairs.csv
"""

import argparse
import configparser

import pandas as pd
from transformers import AutoTokenizer

from moral_prompts import READOUT, load_dataset, token_matched_pairs


def matched_pairs_table(model_name, dataset='care_harm'):
    """DataFrame of the token-length-matched pairs, one row per pair, shortest differing span
    first. Columns: pair_id, frame_id, n_tokens, n_differ, good, bad."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    if model_name not in config:
        raise ValueError(f'{model_name} is not a section in config.ini; '
                         f'have {[s for s in config.sections()]}')
    tokenizer = AutoTokenizer.from_pretrained(config[model_name]['weights_directory'])

    df = load_dataset(dataset)
    rows = []
    for pair_id, n_tokens, n_differ in token_matched_pairs(tokenizer, df):
        group = df[df['pair_id'] == pair_id]
        rows.append({
            'pair_id': pair_id,
            'frame_id': int(group['frame_id'].iloc[0]),
            'n_tokens': n_tokens,
            'n_differ': n_differ,
            'good': group[group['label'] == 1]['statement'].iloc[0],
            'bad': group[group['label'] == 0]['statement'].iloc[0],
        })
    return pd.DataFrame(rows, columns=['pair_id', 'frame_id', 'n_tokens', 'n_differ',
                                       'good', 'bad'])


def main(model_name, dataset='care_harm', limit=20, show_all=False, csv=None):
    table = matched_pairs_table(model_name, dataset)
    total_pairs = load_dataset(dataset)['pair_id'].nunique()

    print(f'readout: {READOUT!r} (appended to each statement before counting tokens)')
    print(f'{len(table)}/{total_pairs} pairs in {dataset} are token-length matched under '
          f'{model_name}\n')

    if table.empty:
        print('None. Every pair changes the token count, so no pair can be patched by '
              'position. Widen the dataset or hand-write a length-matched pair.')
        return table

    # one frame can contribute several name-variants of the same wording; the frame count is
    # the number of genuinely distinct scenarios available
    print(f'covering {table["frame_id"].nunique()} distinct frames\n')

    shown = table if show_all else table.head(limit)
    print(f'{"pair_id":>8} {"frame":>6} {"n_tokens":>9} {"n_differ":>9}  statements')
    for row in shown.itertuples():
        print(f'{row.pair_id:>8} {row.frame_id:>6} {row.n_tokens:>9} {row.n_differ:>9}  '
              f'good: {row.good}')
        print(f'{"":>8} {"":>6} {"":>9} {"":>9}  bad:  {row.bad}')
    if not show_all and len(table) > limit:
        print(f'\n... {len(table) - limit} more; pass --all to see them')

    if csv is not None:
        table.to_csv(csv, index=False)
        print(f'\nwrote {csv}')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=str, default='llama-2-13b',
                        help='config.ini section whose tokenizer decides the matching')
    parser.add_argument('--dataset', type=str, default='care_harm')
    parser.add_argument('--limit', type=int, default=20, help='how many pairs to print')
    parser.add_argument('--all', action='store_true', dest='show_all',
                        help='print every matched pair, ignoring --limit')
    parser.add_argument('--csv', type=str, default=None,
                        help='also write the full table to this path')
    args = parser.parse_args()

    main(args.model, args.dataset, args.limit, args.show_all, args.csv)
