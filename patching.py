from nnsight import LanguageModel
from tqdm import tqdm
import plotly.express as px
import torch as t
import json
import argparse
from generate_acts import load_model


def patching_experiment(model_name, continuation_idx=None, device='remote',
                        true_prompt=None, false_prompt=None, pos_token='YES', neg_token='NO'):
    """
    true_prompt/false_prompt : a minimal pair of prompts differing only in a few tokens
        (must tokenize to the same length). Defaults to the hardcoded animal/human pair.
    pos_token/neg_token : completion tokens whose logit difference is tracked
        (each must be a single token with a leading space).
    """

    model = load_model(model_name, device=device)
    layers = model.model.layers
    remote = device == 'remote'

    # prompt for animal/human data (fire_01 template; "man" and "pig" are both single
    # tokens, so the two prompts tokenize to equal length). Zero-shot, YES/NO readout.
    default_false_prompt = "When the barn caught fire, the pig was trapped inside and crying out. Should the rescuers make saving this individual their top priority? Answer:"
    default_true_prompt = "When the barn caught fire, the man was trapped inside and crying out. Should the rescuers make saving this individual their top priority? Answer:"

    if false_prompt is None:
        false_prompt = default_false_prompt
    if true_prompt is None:
        true_prompt = default_true_prompt

    # check that prompts have the same length
    false_toks = model.tokenizer(false_prompt).input_ids
    true_toks = model.tokenizer(true_prompt).input_ids
    if len(false_toks) != len(true_toks):
        raise ValueError(f"False prompt has length {len(false_toks)} but true prompt has length {len(true_toks)}")

    # find number of tokens after the change
    sames = [false_tok == true_tok for false_tok, true_tok in zip(false_toks, true_toks)]
    n_toks = sames[::-1].index(False) + 1

    true_acts = []
    with model.trace(true_prompt, remote=remote):
        for layer in model.model.layers:
            true_acts.append(layer.output.save())

    if continuation_idx is not None: # if picking up an experiment that failed
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        out = outs[continuation_idx]
        assert out['model'] == model_name
        assert out['false_prompt'] == false_prompt
        assert out['true_prompt'] == true_prompt
        logit_diffs = out['logit_diffs']
    else:
        out = {
            'model' : model_name,
            'false_prompt' : false_prompt,
            'true_prompt' : true_prompt,
            'pos_token' : pos_token,
            'neg_token' : neg_token,
        }
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
        out['logit_diffs'] = logit_diffs
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        outs.append(out)
        with open('experimental_outputs/patching_results.json', 'w') as f:
            json.dump(outs, f, indent=4)
        continuation_idx = -1

    t_tok = model.tokenizer(f" {pos_token}").input_ids[-1]
    f_tok = model.tokenizer(f" {neg_token}").input_ids[-1]

    for tok_idx in range(1, n_toks + 1):
        for layer_idx, layer in enumerate(model.model.layers):
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue # already computed
            with model.trace(false_prompt, remote=remote, scan=True):
                layer.output[0,-tok_idx,:] = true_acts[layer_idx][0,-tok_idx,:]
                logits = model.lm_head.output
                logit_diff = logits[0, -1, t_tok] - logits[0, -1, f_tok]
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
    parser.add_argument('--true_prompt', type=str, default=None,
                        help='prompt whose activations are patched in (e.g. the human version of a scenario)')
    parser.add_argument('--false_prompt', type=str, default=None,
                        help='prompt run with patched activations (e.g. the animal version); must tokenize to the same length as --true_prompt')
    parser.add_argument('--pos_token', type=str, default='YES', help='completion token for the positive class (e.g. YES or TRUE)')
    parser.add_argument('--neg_token', type=str, default='NO', help='completion token for the negative class (e.g. NO or FALSE)')
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device,
                        true_prompt=args.true_prompt, false_prompt=args.false_prompt,
                        pos_token=args.pos_token, neg_token=args.neg_token)