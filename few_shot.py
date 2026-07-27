import torch as t
import pandas as pd
import os
from generate_acts import load_model
from tqdm import tqdm
import argparse
import json

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_few_shot_accuracy(datasets, model, n_shots=5, batch_size=32, calibrated=True, remote=True,
                          pos_token='TRUE', neg_token='FALSE', suffix='', save_diffs=False):
    """Compute the few-shot accuracy of the model on the given datasets.
    Returns a list of dictionaries with experimental results, namely:
    * The dataset used.
    * The number of shots in the few shot prompt.
    * The few shot prompt used.
    * The accuracy of the model.
    * The calibration constant, if calibrated=True.
    pos_token/neg_token : the completion tokens scored (label 1 -> pos_token).
    suffix : appended to each statement before the completion (e.g. a question).
    save_diffs : also write per-query P(pos) - P(neg) to
        experimental_outputs/few_shot_diffs/<dataset>.csv, keeping all csv columns.
    Use n_shots=0 for a zero-shot readout (no examples, no label leakage).
    """

    # change padding sight to right
    model.tokenizer.padding_side = 'right'

    outs = []
    for dataset in datasets:
        out = {
            'dataset' : dataset,
            'n_shots' : n_shots
            }

        # prepare data and prompt
        data_directory = os.path.join(ROOT, 'datasets', f"{dataset}.csv")
        df = pd.read_csv(data_directory)
        shots = df.sample(n_shots)
        queries = df.drop(shots.index)

        prompt = ''
        for _, shot in tqdm(shots.iterrows(), desc=f'Processing {dataset}'):
            prompt += f'{shot["statement"]}{suffix} '
            if bool(shot['label']):
                prompt += f'{pos_token}\n'
            else:
                prompt += f'{neg_token}\n'

        out['shots'] = shots['statement'].tolist()
        out['prompt'] = prompt
        out['pos_token'], out['neg_token'], out['suffix'] = pos_token, neg_token, suffix

        # get completions and evaluate accuracy
        true_idx, false_idx = model.tokenizer.encode(f' {pos_token}')[-1], model.tokenizer.encode(f' {neg_token}')[-1]
        diffs = []
        for batch_idx in range(0, len(queries), batch_size):
            batch = (prompt + queries.iloc[batch_idx:batch_idx+batch_size]['statement'] + suffix).tolist()

            batch_lens = [len(model.tokenizer.encode(query)) for query in batch]
            with model.trace(batch, remote=remote):
                logits = model.lm_head.output
                logits = logits[t.arange(len(batch)), t.tensor(batch_lens) - 1, :]
                probs = logits.softmax(-1)
                diffs.append((probs[:, true_idx] - probs[:, false_idx]).save())
        diffs = t.cat(diffs)


        # if calibrated, compute calibration constant
        if calibrated:
            gamma = t.sort(diffs).values[len(diffs) // 2]
            out['gamma'] = gamma.item()
        else:
            gamma = 0
        
        # get predicted labels
        predicted_labels = diffs > gamma
        ground_truth = t.tensor(queries['label'].values, device=predicted_labels.device).bool()
        
        acc = (predicted_labels == ground_truth).float().mean().item()
        out['acc'] = acc

        if save_diffs:
            diffs_df = queries.copy()
            diffs_df['p_diff'] = diffs.float().cpu().numpy()
            diffs_dir = os.path.join(ROOT, 'experimental_outputs', 'few_shot_diffs')
            os.makedirs(diffs_dir, exist_ok=True)
            diffs_df.to_csv(os.path.join(diffs_dir, f'{dataset}.csv'), index=False)

        outs.append(out)

    return outs

if __name__ == '__main__':
    """
    Compute the few-shot accuracy of the model on the given datasets and save results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to evaluate on')
    parser.add_argument('--model', type=str, default='llama-2-70b', help='model size to evaluate')
    parser.add_argument('--n_shots', type=int, default=5, help='number of shots to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to use')
    parser.add_argument('--uncalibrated', action='store_true', default=False, help='set flag if using uncalibrated few shot')
    parser.add_argument('--device', default='remote', help='device to use')
    parser.add_argument('--pos_token', default='TRUE', help='completion token for label 1')
    parser.add_argument('--neg_token', default='FALSE', help='completion token for label 0')
    parser.add_argument('--suffix', default='', help='appended to each statement before the completion, e.g. a question')
    parser.add_argument('--save_diffs', action='store_true', default=False,
                        help='write per-query P(pos)-P(neg) to experimental_outputs/few_shot_diffs/<dataset>.csv')

    args = parser.parse_args()

    model = load_model(args.model, device=args.device)

    outs = get_few_shot_accuracy(args.datasets, model, args.n_shots, args.batch_size, not args.uncalibrated, args.device == 'remote',
                                 pos_token=args.pos_token, neg_token=args.neg_token, suffix=args.suffix, save_diffs=args.save_diffs)
    for out in outs:
        out['model'] = args.model

    # save results
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'r') as f:
        data = json.load(f)
    data.extend(outs)
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'w') as f:
        json.dump(data, f, indent=4)