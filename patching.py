from nnsight import LanguageModel
from tqdm import tqdm
import plotly.express as px
import torch as t
import json
import argparse
from generate_acts import load_model


def patching_experiment(model_name, continuation_idx=None, device='remote'):

    model = load_model(model_name, device=device)
    layers = model.model.layers
    remote = device == 'remote'

    # # prompt for cities
    # false_prompt = "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Chicago is in Canada. This statement is:"
    # true_prompt = "The city of Tokyo is in Japan. This statement is: TRUE\nThe city of Hanoi is in Poland. This statement is: FALSE\nThe city of Toronto is in Canada. This statement is:"

    # # prompt for larger_than
    # false_prompt = "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-one is larger than eighty-four. This statement is:"
    # true_prompt = "Fifty-one is larger than seventy-five. This statement is: FALSE\nEighty-two is larger than sixty-four. This statement is: TRUE\nEighty-six is larger than eighty-four. This statement is:"

    # prompt for care_harm (pair_id 0, frame_id 0); the shots are one instance each from
    # five other frames, so the prefix never shows a minimal pair
    good_prompt = """\
On the crowded bus, Felix offered his seat to the injured passenger by the door. This action is: GOOD
When the toddler tripped on the sidewalk, Nina pointed and laughed at his scraped knee. This action is: BAD
After his little sister failed her exam, Arjun stayed up late helping her prepare for the retake. This action is: GOOD
On the first day at the new school, Leo shared his lunch with the girl sitting alone. This action is: GOOD
Before leaving for the weekend, Tom deliberately poured bleach into the water in his goldfish's tank. This action is: BAD
After the storm, Carmen carried her elderly neighbor's groceries up the apartment stairs. This action is:"""
    bad_prompt = """\
On the crowded bus, Felix offered his seat to the injured passenger by the door. This action is: GOOD
When the toddler tripped on the sidewalk, Nina pointed and laughed at his scraped knee. This action is: BAD
After his little sister failed her exam, Arjun stayed up late helping her prepare for the retake. This action is: GOOD
On the first day at the new school, Leo shared his lunch with the girl sitting alone. This action is: GOOD
Before leaving for the weekend, Tom deliberately poured bleach into the water in his goldfish's tank. This action is: BAD
After the storm, Carmen hurled her elderly neighbor's groceries down the apartment stairs. This action is:"""


    # check that prompts have the same length
    good_toks = model.tokenizer(good_prompt).input_ids
    bad_toks = model.tokenizer(bad_prompt).input_ids
    if len(good_toks) != len(bad_toks):
        raise ValueError(f"Good prompt has length {len(good_toks)} but bad prompt has length {len(bad_toks)}")

    # find number of tokens from the first change to the end. The pair differs in more than
    # one place (the verb and the direction), so sweeping back to the *last* difference would
    # skip the verb, which is the token carrying the valence.
    sames = [good_tok == bad_tok for good_tok, bad_tok in zip(good_toks, bad_toks)]
    n_toks = len(sames) - sames.index(False)

    true_acts = []
    with model.forward(remote=remote, remote_include_output=False) as runner:
        with runner.invoke(good_prompt):
            for layer in model.model.layers:
                true_acts.append(layer.output.save())

    if continuation_idx is not None: # if picking up an experiment that failed
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        out = outs[continuation_idx]
        assert out['model'] == model_name
        assert out['good_prompt'] == good_prompt
        assert out['bad_prompt'] == bad_prompt
        logit_diffs = out['logit_diffs']
    else:
        out = {
            'model' : model_name,
            'good_prompt' : good_prompt,
            'bad_prompt' : bad_prompt,
        }
        logit_diffs = [[None for _ in range(len(layers))] for _ in range(n_toks)]
        out['logit_diffs'] = logit_diffs
        with open('experimental_outputs/patching_results.json', 'r') as f:
            outs = json.load(f)
        outs.append(out)
        with open('experimental_outputs/patching_results.json', 'w') as f:
            json.dump(outs, f, indent=4)
        continuation_idx = -1

    g_tok = model.tokenizer(" GOOD").input_ids[-1]
    b_tok = model.tokenizer(" BAD").input_ids[-1]

    for tok_idx in range(1, n_toks + 1):
        for layer_idx, layer in enumerate(model.model.layers):
            if logit_diffs[tok_idx - 1][layer_idx] is not None:
                continue # already computed
            with model.forward(remote=remote, remote_include_output=False) as runner:
                with runner.invoke(bad_prompt, scan=True) as invoker:
                    layer.output[0,-tok_idx,:] = true_acts[layer_idx][0,-tok_idx,:]
                    logits = model.lm_head.output
                    logit_diff = logits[0, -1, g_tok] - logits[0, -1, b_tok]
                    logit_diff = logit_diff.save()
            logit_diffs[tok_idx - 1][layer_idx] = logit_diff.item()
            
            outs[continuation_idx] = out
            with open('experimental_outputs/patching_results.json', 'w') as f:
                json.dump(outs, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-2-13b')
    parser.add_argument('--continuation_idx', type=int, default=None)
    parser.add_argument('--device', type=str, default='remote')
    args = parser.parse_args()

    patching_experiment(args.model, args.continuation_idx, args.device)