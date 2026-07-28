"""Run patching.py over many pos/neg prompt pairs.

Calls `patching.patching_experiment` unchanged — same grid, same
experimental_outputs/patching_results.json format, same per-cell incremental saving
and resume. This just builds the prompt pairs, loads the model once instead of once
per pair, and keeps going if one pair fails.

A pair is one scenario template with only the subject noun swapped, plus a readout
question:

    true : "When the barn caught fire, the man was trapped inside and crying out.
            Should the rescuers make saving this individual their top priority? Answer:"
    false: "When the barn caught fire, the cow was trapped inside and crying out. ..."

Run from the repo root (patching.py uses paths relative to it):

    python experiments/patching_sweep.py --model llama-2-13b --device cuda:0
    python experiments/patching_sweep.py --model llama-2-13b --device cuda:0 \
        --templates fire_01 --subject-pairs man:cow man:dog --readouts rescue suffering
    python experiments/patching_sweep.py --summary
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "datasets"))

from animal_human_templates import SUBJECTS, TEMPLATES_BY_AXIS  # noqa: E402
from patching import patching_experiment  # noqa: E402

RESULTS_PATH = os.path.join(REPO_ROOT, "experimental_outputs", "patching_results.json")

# Readout question appended after the scenario, plus the completion tokens whose logit
# difference is tracked. 'rescue' is patching.py's hardcoded default question, so
# fire_01 + man:cow + rescue reproduces that default exactly.
#
# Varying the readout is the point of sweeping: if the same (position, layer) cells carry
# the effect for 'rescue' and for 'suffering', that is one species representation feeding
# several downstream judgments rather than an artifact of one question's phrasing.
READOUTS = {
    "rescue": "Should the rescuers make saving this individual their top priority? Answer:",
    "urgent": "Is this an emergency that demands immediate action? Answer:",
    "suffering": "Is the individual described here suffering? Answer:",
    "cost": "Would it be worth spending a large sum of money to help? Answer:",
}

DEFAULT_TEMPLATES = ["fire_01", "neglect_01", "trap_01", "morning_01"]

# Patching needs both prompts to tokenize to equal length, so the two subjects must have
# the same token count. These are short common nouns that should each be one token; any
# pair that turns out not to match is reported and skipped, not fatal.
DEFAULT_SUBJECT_PAIRS = ["man:cow", "man:dog", "man:rat", "boy:dog", "girl:hen"]


def template_text(template_id):
    for templates in TEMPLATES_BY_AXIS.values():
        for template in templates:
            if template["template_id"] == template_id:
                return template["text"]
    raise KeyError(f"unknown template_id: {template_id}")


def build_specs(templates, subject_pairs, readouts):
    specs = []
    for template_id in templates:
        text = template_text(template_id)
        for pair in subject_pairs:
            pos_key, neg_key = pair.split(":")
            for key in (pos_key, neg_key):
                if key not in SUBJECTS:
                    raise KeyError(f"unknown subject: {key}")
            for readout in readouts:
                question = READOUTS[readout]
                specs.append({
                    "name": f"{template_id}/{pos_key}-vs-{neg_key}/{readout}",
                    "true_prompt": f"{text.format(subject=SUBJECTS[pos_key]['phrase'])} {question}",
                    "false_prompt": f"{text.format(subject=SUBJECTS[neg_key]['phrase'])} {question}",
                })
    return specs


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return []
    with open(RESULTS_PATH) as f:
        return json.load(f)


def find_existing(results, model_name, spec):
    """Index of a prior run of this exact pair, or None."""
    for i, out in enumerate(results):
        if (out.get("model") == model_name
                and out.get("true_prompt") == spec["true_prompt"]
                and out.get("false_prompt") == spec["false_prompt"]):
            return i
    return None


def is_complete(out):
    grid = out.get("logit_diffs") or []
    return bool(grid) and all(cell is not None for row in grid for cell in row)


def peak(out):
    """(value, position_from_end, layer) of the largest-magnitude logit difference."""
    best = (0.0, None, None)
    for tok_idx, row in enumerate(out.get("logit_diffs") or []):
        for layer_idx, value in enumerate(row):
            if value is not None and abs(value) > abs(best[0]):
                best = (value, -(tok_idx + 1), layer_idx)
    return best


def print_summary(model_name=None):
    results = load_results()
    if not results:
        print(f"no results in {RESULTS_PATH}")
        return
    print(f"{'#':>3}  {'model':<14} {'done':>7}  {'peak':>8} {'pos':>4} {'layer':>5}  prompt")
    for i, out in enumerate(results):
        if model_name and out.get("model") != model_name:
            continue
        grid = out.get("logit_diffs") or []
        n_cells = sum(len(row) for row in grid)
        n_done = sum(1 for row in grid for cell in row if cell is not None)
        value, position, layer = peak(out)
        done = "yes" if is_complete(out) else f"{n_done}/{n_cells}"
        print(f"{i:>3}  {out.get('model', '?'):<14} {done:>7}  {value:>8.3f} "
              f"{position if position is not None else '-':>4} "
              f"{layer if layer is not None else '-':>5}  {out.get('false_prompt', '')[:56]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="llama-2-13b")
    parser.add_argument("--device", default="remote")
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    parser.add_argument("--subject-pairs", nargs="+", default=DEFAULT_SUBJECT_PAIRS,
                        help="pos:neg subject keys, e.g. man:cow. Both subjects must "
                             "tokenize to the same length or the pair is skipped.")
    parser.add_argument("--readouts", nargs="+", default=["rescue"], choices=sorted(READOUTS))
    parser.add_argument("--limit", type=int, default=None, help="cap the number of runs")
    parser.add_argument("--redo", action="store_true", help="recompute pairs already finished")
    parser.add_argument("--summary", action="store_true", help="print results so far and exit")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)  # patching.py writes to a relative path

    if args.summary:
        print_summary(args.model)
        return

    specs = build_specs(args.templates, args.subject_pairs, args.readouts)
    if args.limit:
        specs = specs[:args.limit]

    from generate_acts import load_model
    model = load_model(args.model, device=args.device)

    failed = []
    for n, spec in enumerate(specs, 1):
        results = load_results()
        idx = find_existing(results, args.model, spec)
        if idx is not None and is_complete(results[idx]) and not args.redo:
            print(f"[{n}/{len(specs)}] {spec['name']} — already complete, skipping")
            continue
        resume = None if (idx is None or args.redo) else idx
        print(f"[{n}/{len(specs)}] {spec['name']} — {'resuming' if resume is not None else 'running'}")
        try:
            patching_experiment(
                args.model,
                continuation_idx=resume,
                device=args.device,
                true_prompt=spec["true_prompt"],
                false_prompt=spec["false_prompt"],
                model=model,
            )
        except Exception as e:
            # most often unequal token length between the two prompts; keep going so one
            # bad subject pair does not waste the rest of the sweep
            print(f"    ! failed: {e}")
            failed.append((spec["name"], str(e)))

    if failed:
        print(f"\n{len(failed)} pair(s) failed:")
        for name, err in failed:
            print(f"  {name}: {err}")

    print()
    print_summary(args.model)


if __name__ == "__main__":
    main()
