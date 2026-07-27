"""Probe-accuracy-per-layer sweep on cached activations.

Runs entirely on activations cached by generate_acts.py (use --layers -1 there),
so it is CPU-cheap and needs no GPU or model download. For every cached layer it
trains a probe and reports accuracy (fraction of statements whose predicted
label matches the csv label; datasets are balanced, so chance = 0.5):

  * iid      : accuracy on a random held-out split of the train datasets —
               "is the category linearly decodable at this layer at all?"
  * subj     : accuracy on rows whose subject words were held out of training
               (needs `subject`/`subject_category` columns) — rules out the
               probe just memorizing token identities.
  * transfer : accuracy on each --val_datasets, probe trained on all of
               --train_datasets — e.g. train neutral, test harm.

Use the curves to pick config.ini values: probe_layer where accuracy saturates,
intervene_layer where it starts to rise.

Results append to experimental_outputs/layer_sweep_results.json; plot them with
layer_sweep.ipynb.

Example:
    python layer_sweep.py --model llama-2-13b \
        --train_datasets human_farmed_neutral \
        --val_datasets human_farmed_harm human_wild_harm human_wild_neutral
"""

import argparse
import json
import os
import random
import re
from glob import glob

import pandas as pd
import torch as t

from probes import LRProbe, MMProbe
from utils import collect_acts

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(ROOT, 'experimental_outputs', 'layer_sweep_results.json')


def cached_layers(datasets, model, noperiod=False):
    """Layers with activations cached for every dataset involved."""
    layers = None
    for dataset in datasets:
        directory = os.path.join(ROOT, 'acts', model)
        if noperiod:
            directory = os.path.join(directory, 'noperiod')
        directory = os.path.join(directory, dataset)
        found = {int(m.group(1)) for f in glob(os.path.join(directory, 'layer_*_0.pt'))
                 if (m := re.search(r'layer_(\d+)_0\.pt$', f))}
        if not found:
            raise ValueError(f"No cached activations for {dataset} ({model}); run generate_acts.py first.")
        layers = found if layers is None else layers & found
    return sorted(layers)


def load_layer(datasets, model, layer, noperiod, device):
    """Concatenated acts and labels for a list of datasets at one layer."""
    acts, labels = [], []
    for dataset in datasets:
        acts.append(collect_acts(dataset, model, layer, noperiod=noperiod, center=True, device=device))
        df = pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset}.csv'))
        labels.append(t.Tensor(df['label'].values).to(device))
    return t.cat(acts), t.cat(labels)


def load_csvs(datasets):
    return pd.concat([pd.read_csv(os.path.join(ROOT, 'datasets', f'{d}.csv')) for d in datasets],
                     ignore_index=True)


def accuracy(probe, acts, labels, iid=False):
    return (probe.pred(acts, iid=iid) == labels).float().mean().item()


def make_iid_split(n_rows, split):
    """Boolean train mask for a random split, fixed once so all layers share it."""
    return t.randperm(n_rows) < int(split * n_rows)


def pick_heldout_subjects(train_df, heldout_frac):
    """Sample subject words to hold out, per category. Returns (subjects, test row mask),
    or ([], None) if the csvs have no subject columns or holding out is disabled."""
    if heldout_frac <= 0 or not {'subject', 'subject_category'}.issubset(train_df.columns):
        return [], None
    heldout = []
    for _, subjects in train_df.groupby('subject_category')['subject'].unique().items():
        k = max(1, round(heldout_frac * len(subjects)))
        heldout += random.sample(sorted(subjects), k)
    return heldout, t.tensor(train_df['subject'].isin(heldout).values)


def sweep_layer(ProbeClass, args, layer, iid_train_mask, subj_test_mask):
    """Train and evaluate probes at one layer. Returns {'iid': acc,
    'heldout_subjects': acc, 'transfer': {dataset: acc}}."""
    acts, labels = load_layer(args.train_datasets, args.model, layer, args.noperiod, args.device)
    res = {}

    # iid: random split of the train datasets
    probe = ProbeClass.from_data(acts[iid_train_mask], labels[iid_train_mask], device=args.device)
    res['iid'] = accuracy(probe, acts[~iid_train_mask], labels[~iid_train_mask], iid=True)

    # held-out subject words
    if subj_test_mask is not None:
        probe = ProbeClass.from_data(acts[~subj_test_mask], labels[~subj_test_mask], device=args.device)
        res['heldout_subjects'] = accuracy(probe, acts[subj_test_mask], labels[subj_test_mask])

    # transfer to other datasets, trained on everything
    if args.val_datasets:
        probe = ProbeClass.from_data(acts, labels, device=args.device)
        res['transfer'] = {}
        for val_dataset in args.val_datasets:
            val_acts, val_labels = load_layer([val_dataset], args.model, layer, args.noperiod, args.device)
            res['transfer'][val_dataset] = accuracy(probe, val_acts, val_labels)

    return res


def format_row(res, args, subj):
    row = [f"{res['iid']:.3f}"]
    if subj:
        row.append(f"{res['heldout_subjects']:.3f}")
    row += [f"{res['transfer'][d]:.3f}" for d in args.val_datasets]
    return row


def save_results(out):
    data = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
    data.append(out)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def main(args):
    ProbeClass = {'MMProbe': MMProbe, 'LRProbe': LRProbe}[args.probe]
    random.seed(args.seed)
    t.manual_seed(args.seed)

    all_datasets = args.train_datasets + [d for d in args.val_datasets if d not in args.train_datasets]
    layers = args.layers or cached_layers(all_datasets, args.model, args.noperiod)

    # fixed splits, shared across layers so the curves are comparable
    train_df = load_csvs(args.train_datasets)
    iid_train_mask = make_iid_split(len(train_df), args.split)
    heldout_subjects, subj_test_mask = pick_heldout_subjects(train_df, args.heldout_frac)

    print(f'layers: {layers}')
    if heldout_subjects:
        print(f'held-out subjects: {sorted(heldout_subjects)}')
    header = ['layer', 'iid'] + (['subj'] if heldout_subjects else []) + args.val_datasets
    print('\t'.join(header))

    results = {}
    for layer in layers:
        results[layer] = sweep_layer(ProbeClass, args, layer, iid_train_mask, subj_test_mask)
        print('\t'.join([str(layer)] + format_row(results[layer], args, bool(heldout_subjects))))

    save_results({
        'model': args.model,
        'probe': args.probe,
        'train_datasets': args.train_datasets,
        'val_datasets': args.val_datasets,
        'noperiod': args.noperiod,
        'split': args.split,
        'seed': args.seed,
        'heldout_subjects': sorted(heldout_subjects),
        'results': results,
    })
    print(f'appended results to {RESULTS_FILE}; plot them with layer_sweep.ipynb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama-2-13b')
    parser.add_argument('--probe', default='MMProbe', choices=['MMProbe', 'LRProbe'])
    parser.add_argument('--train_datasets', nargs='+', default=['human_farmed_neutral'])
    parser.add_argument('--val_datasets', nargs='+', default=[],
                        help='datasets to evaluate transfer on (trained on --train_datasets)')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='layers to sweep; defaults to every layer cached for all datasets')
    parser.add_argument('--split', type=float, default=0.8, help='train fraction for the iid split')
    parser.add_argument('--heldout_frac', type=float, default=0.25,
                        help='fraction of subject words per category to hold out (0 to skip)')
    parser.add_argument('--noperiod', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    main(parser.parse_args())
