import torch as t
import pandas as pd
import os
from generate_acts import load_model
from tqdm import tqdm
import argparse
import json

from moral_prompts import READOUT, GOOD, BAD
from utils import frame_split

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_few_shot_accuracy(datasets, model, n_shots=5, batch_size=32, calibrated=True, remote=True,
                          seed=0, split=None, split_seed=0):
    """Compute the few-shot accuracy of the model on the given datasets, reading the moral
    GOOD/BAD verdict rather than the fork's original TRUE/FALSE.

    split : if given, the train proportion of the frame-grouped split. Shots are drawn from
    the train frames and only the val frames are queried, so no query is a name-variant of a
    shot. Pass the split and seed the probe was trained with and the two numbers are scored
    on the same held-out frames, which is the comparison worth reporting.

    Returns a list of dictionaries with experimental results, namely:
    * The dataset used.
    * The number of shots in the few shot prompt.
    * The few shot prompt used.
    * The frames the queries came from, and the split that chose them.
    * The accuracy of the model, in total and per label.
    * The calibration constant, if calibrated=True.
    """

    # change padding sight to right
    model.tokenizer.padding_side = 'right'

    outs = []
    for dataset in datasets:
        out = {
            'dataset' : dataset,
            'n_shots' : n_shots,
            'seed' : seed,
            'calibrated' : calibrated,
            }

        # prepare data and prompt
        data_directory = os.path.join(ROOT, 'datasets', f"{dataset}.csv")
        df = pd.read_csv(data_directory)
        if split is None:
            shots = df.sample(n_shots, random_state=seed)
            queries = df.drop(shots.index)
        else:
            train = frame_split(df, split, split_seed)
            shots = df[train.numpy()].sample(n_shots, random_state=seed)
            queries = df[~train.numpy()]
            out['split'], out['split_seed'] = split, split_seed

        prompt = ''
        for _, shot in tqdm(shots.iterrows(), desc=f'Processing {dataset}'):
            prompt += f'{shot["statement"]} {READOUT}'
            if bool(shot['label']):
                prompt += f'{GOOD}\n'
            else:
                prompt += f'{BAD}\n'

        out['shots'] = shots['statement'].tolist()
        out['prompt'] = prompt
        out['n_queries'] = len(queries)
        if 'frame_id' in queries.columns:
            out['query_frames'] = sorted(int(frame) for frame in queries['frame_id'].unique())

        # cache activations over the prompt for reuse
        with model.forward(output_hidden_states=True, remote=remote, remote_include_output=remote) as runner:
            with runner.invoke(prompt):
                pass
        past_key_values = runner.output['past_key_values']

        # get completions and evaluate accuracy
        good_idx, bad_idx = model.tokenizer(GOOD).input_ids[-1], model.tokenizer(BAD).input_ids[-1]
        # each query ends on the readout, so the next token is the verdict
        query_prompts = [f'{statement} {READOUT}' for statement in queries['statement'].tolist()]
        diffs = []
        for batch_idx in tqdm(range(0, len(query_prompts), batch_size), desc=f'Processing {dataset}'):
            batch = query_prompts[batch_idx:batch_idx+batch_size]

            # # prepare past_key_values
            # pkv_batch = tuple((
            #     past_key_values[layer][0].expand(len(batch), *past_key_values[layer][0].shape[1:]),
            #     past_key_values[layer][1].expand(len(batch), *past_key_values[layer][1].shape[1:])
            # ) for layer in range(len(past_key_values))
            # )

            batch_lens = [len(model.tokenizer.encode(query, add_special_tokens=False)) for query in batch]
            with model.forward(past_key_values=past_key_values
            , remote=remote, remote_include_output=False) as runner:
                with runner.invoke(batch, add_special_tokens=False, return_attention_mask=False):
                    logits = model.lm_head.output
                    logits = logits[t.arange(len(batch)), t.tensor(batch_lens) - 1, :]
                    probs = logits.softmax(-1)
                    diffs.append((probs[:, good_idx] - probs[:, bad_idx]).save())
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
        out['acc_good'] = (predicted_labels[ground_truth] == True).float().mean().item()
        out['acc_bad'] = (predicted_labels[~ground_truth] == False).float().mean().item()

        outs.append(out)

    return outs

if __name__ == '__main__':
    """
    Compute the few-shot accuracy of the model on the given datasets and save results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['care_harm', 'neg_care_harm'],
                        help='datasets to evaluate on')
    parser.add_argument('--model', type=str, default='llama-2-13b', help='model size to evaluate')
    parser.add_argument('--n_shots', type=int, default=5, help='number of shots to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size to use')
    parser.add_argument('--uncalibrated', action='store_true', default=False, help='set flag if using uncalibrated few shot')
    parser.add_argument('--seed', type=int, default=0, help='seed choosing the shots')
    parser.add_argument('--split', type=float, default=None,
                        help='train proportion of the frame-grouped split; only the val frames are queried. Pass what the probe was trained with to score both on the same held-out frames')
    parser.add_argument('--split_seed', type=int, default=0, help='seed of that split, matching DataManager')
    parser.add_argument('--device', default='remote', help='device to use')

    args = parser.parse_args()

    model = load_model(args.model, device=args.device)

    outs = get_few_shot_accuracy(args.datasets, model, args.n_shots, args.batch_size,
                                 not args.uncalibrated, args.device == 'remote',
                                 args.seed, args.split, args.split_seed)
    for out in outs:
        out['model'] = args.model

    # save results
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'r') as f:
        data = json.load(f)
    data.extend(outs)
    with open(os.path.join(ROOT, 'experimental_outputs', "few_shot_results.json"), 'w') as f:
        json.dump(data, f, indent=4)
