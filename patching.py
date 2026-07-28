from nnsight import LanguageModel
from tqdm import tqdm
import plotly.express as px
import torch as t
import json
import argparse
from generate_acts import load_model, tracer_kwargs


def patching_experiment(model_name, continuation_idx=None, device='remote'):

    model = load_model(model_name, device=device)
    layers = model.model.layers
    remote = device == 'remote'

    # prompts for human_farmed_harm / human_farmed_neutral: same templates in both
    # prompts (two harm, two neutral, harm query), so the only thing that varies is
    # whether every subject is an animal or a human. All subjects are single tokens,
    # which keeps the two prompts aligned token for token.
    animal_prompt = """\
When the barn caught fire, the sheep was trapped inside and crying out. The subject is: ANIMAL.
In the early morning light, the hen wandered slowly across the meadow. The subject is: ANIMAL.
For three days, the ram has been too weak from the fever to move. The subject is: ANIMAL.
After the long walk, the ox rested calmly by the gate. The subject is: ANIMAL.
After the dam broke, the cow was caught in the violent surge. The subject is:"""
    human_prompt = """\
When the barn caught fire, the man was trapped inside and crying out. The subject is: HUMAN.
In the early morning light, the woman wandered slowly across the meadow. The subject is: HUMAN.
For three days, the child has been too weak from the fever to move. The subject is: HUMAN.
After the long walk, the girl rested calmly by the gate. The subject is: HUMAN.
After the dam broke, the doctor was caught in the violent surge. The subject is:"""

    # check that prompts have the same length
    animal_toks = model.tokenizer(animal_prompt).input_ids
    human_toks = model.tokenizer(human_prompt).input_ids
    if len(animal_toks) != len(human_toks):
        raise ValueError(f"Animal prompt has length {len(animal_toks)} but human prompt has length {len(human_toks)}")

    # find number of tokens after the change
    sames = [animal_tok == human_tok for animal_tok, human_tok in zip(animal_toks, human_toks)]
    n_toks = sames[::-1].index(False) + 1

    human_acts = []
    with model.trace(human_prompt, remote=remote, **tracer_kwargs):
        for layer in model.model.layers:
            human_acts.append(layer.output.save())

    if continuation_idx is not None: # if picking up an experiment that failed
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        out = outs[continuation_idx]
        assert out['model'] == model_name
        assert out['animal_prompt'] == animal_prompt
        assert out['human_prompt'] == human_prompt
        logit_diffs = out['logit_diffs']
    else:
        out = {
            'model' : model_name,
            'animal_prompt' : animal_prompt,
            'human_prompt' : human_prompt,
        }
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
        out['logit_diffs'] = logit_diffs
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        outs.append(out)
        with open('experimental_outputs/patching_results.json', 'w') as f:
            json.dump(outs, f, indent=4)
        continuation_idx = -1

    h_tok = model.tokenizer(" HUMAN").input_ids[-1]
    a_tok = model.tokenizer(" ANIMAL").input_ids[-1]

    for tok_idx in range(1, n_toks + 1):
        for layer_idx, layer in enumerate(model.model.layers):
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue # already computed
            with model.trace(animal_prompt, remote=remote, **tracer_kwargs):
                layer.output[0,-tok_idx,:] = human_acts[layer_idx][0,-tok_idx,:]
                logits = model.lm_head.output
                logit_diff = logits[0, -1, h_tok] - logits[0, -1, a_tok]
                logit_diff = logit_diff.save()
            logit_diffs[tok_idx - 1][layer_idx] = logit_diff.item()
            
            outs[continuation_idx] = out
            with open('experimental_outputs/patching_results.json', 'w') as f:
                json.dump(outs, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-70b')
    parser.add_argument('--continuation_idx', type=int, default=None)
    parser.add_argument('--device', type=str, default='remote')
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device)