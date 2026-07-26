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

    # prompt for care_harm (pair_id 115, frame_id 17); the shots are one instance each from
    # five other frames (1, 2, 4, 6, 3), so the prefix never shows a minimal pair
    #
    # The pair was chosen from find_matched_pairs.py, which is what the tokenizer actually
    # admits. The previous target (pair_id 0, "Carmen carried/hurled her elderly neighbor's
    # groceries up/down the apartment stairs") is NOT token-length matched under the Llama-2
    # tokenizer and would have tripped the length check below on the first run.
    #
    # Of the eight frames that do match, this one is the closest analog to the original's
    # Chicago/Toronto pair: the two statements differ in a single contiguous span ("gently
    # iced" / "deliberately kicked") and are token-identical everywhere after it, so the
    # sweep watches valence propagate forward through a shared tail. Most of the other
    # candidates differ in two separated places (frame 12's "invited...to join" vs
    # "blocked...from joining"), which leaves the heatmap unable to say which site the
    # effect came from.
    #
    # The frame-13 bleach shot that used to sit in the prefix was swapped for a frame-3 shot:
    # "deliberately" occurs in exactly two frames of care_harm, 13 and 17, so keeping it would
    # have put the target's own adverb in the context labelled BAD.
    good_prompt = """\
On the crowded bus, Felix offered his seat to the injured passenger by the door. This action is: GOOD
When the toddler tripped on the sidewalk, Nina pointed and laughed at his scraped knee. This action is: BAD
After his little sister failed her exam, Arjun stayed up late helping her prepare for the retake. This action is: GOOD
On the first day at the new school, Leo shared his lunch with the girl sitting alone. This action is: GOOD
During the night shift, the nurse hid the call buttons from the patients on the recovery ward. This action is: BAD
After practice, Coach Ethan gently iced the sprained ankle of the youngest player on the team. This action is:"""
    bad_prompt = """\
On the crowded bus, Felix offered his seat to the injured passenger by the door. This action is: GOOD
When the toddler tripped on the sidewalk, Nina pointed and laughed at his scraped knee. This action is: BAD
After his little sister failed her exam, Arjun stayed up late helping her prepare for the retake. This action is: GOOD
On the first day at the new school, Leo shared his lunch with the girl sitting alone. This action is: GOOD
During the night shift, the nurse hid the call buttons from the patients on the recovery ward. This action is: BAD
After practice, Coach Ethan deliberately kicked the sprained ankle of the youngest player on the team. This action is:"""


    # check that prompts have the same length
    good_toks = model.tokenizer(good_prompt).input_ids
    bad_toks = model.tokenizer(bad_prompt).input_ids
    if len(good_toks) != len(bad_toks):
        raise ValueError(f"Good prompt has length {len(good_toks)} but bad prompt has length {len(bad_toks)}")

    # find number of tokens from the first change to the end. Sweeping back only as far as the
    # *last* difference would skip the valence-carrying verb whenever a pair differs in more
    # than one place, so the sweep always starts at the first difference. Note this is a
    # different quantity from the n_differ column find_matched_pairs.py sorts on, which
    # measures from the last difference — that column ranks candidates, it does not predict
    # how many forward passes the sweep costs.
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