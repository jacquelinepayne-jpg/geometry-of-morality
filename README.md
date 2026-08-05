# The Geometry of Morality

This repository is forked from the repository https://github.com/saprmarks/geometry-of-truth associated to the paper [*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*](https://arxiv.org/abs/2310.06824) by Samuel Marks and Max Tegmark.

This fork extends their methodology to test whether moral valence (good/bad) is linearly represented in LLM activations the same way true/false is, and whether that representation holds up for more complex moral concepts and across different moral subjects (human vs. animal). The goal is to establish whether morality is a stable, controllable direction in activation space.

The current experiments focus on the **moral subject** question: given the same scenario, does the model represent *who* it is happening to (human, companion animal, farmed animal, wild animal) as a linear direction, and does that direction change under harm scenarios versus neutral ones?

## Set-up

Navigate to the location that you want to clone this repo to, clone and enter the repo, and install requirements.

```
git clone git@github.com:jacquelinepayne-jpg/geometry-of-morality.git
cd geometry-of-morality
pip install -r requirements.txt
```

Before doing anything, you'll need to generate activations for the datasets. You should have your own LLaMA weights stored on the machine where you cloned this repo. Put the absolute path for the directory containing your LLaMA weights in the file `config.ini`; Huggingface repos are also supported.

Once that's done, you can generate the LLaMA activations for the datasets you'd like to work with with a command like

```
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets single-template/human_farmed/flood_01 single-template/farmed_wild/flood_01 --device cuda:0
```

These activations will be stored in the `acts` directory. If you want to save activations for all layers, simply use `--layers -1`. Omit `--device` to run remotely on NDIF via `nnsight`.

`setup_host.sh` provisions a fresh GPU box (e.g. a Vast.ai instance); `documentation/compute_estimate.md` has the GPU/storage/cost plan.

## Datasets

A dataset name is a path relative to `datasets/`, without the `.csv`, so it may contain a subdirectory (e.g. `single-template/human_farmed/flood_01`, `truth/cities`). The same name is used for the activation cache directory. Every CSV has at minimum `statement` and `label` (1/0) columns; the animal/human CSVs carry extra `subject`, `subject_category`, `template_id`, and `axis` columns, which are used for grouped train/val splits and for coloring plots.

The animal/human data comes in two groupings, one folder each:

* **`datasets/single-template/<pair>/<template>.csv`** — one scenario template per file, so the only thing varying within a file is the subject word. This removes the between-template variance that otherwise dominates PCA of the pooled sets. `human_animal` contrasts humans against a balanced 40-subject sample drawn evenly from the three animal categories. Written by `datasets/make_per_template.py`.
* **`datasets/multi-template/<pair>_<axis>.csv`** — all templates on an axis pooled into one file (`human_farmed_harm.csv`, `companion_wild_neutral.csv`, …). Written by `datasets/make_animal_human.py`.

In both, the first category in the pair name gets label 1 — in `{catA}_{catB}`, label 1 = catA.

Supporting files:

* `datasets/animal_human_templates.py` — source of truth for both groupings: 160 subjects (40 each for human / companion / farmed / wild) crossed with 100 scenario templates (60 harm, 40 neutral). The docstring lists the authoring constraints that keep the contrast clean (one `{subject}` slot, no pronouns, no subject word appearing in a template, scenarios plausible for every subject).
* `datasets/truth/` — the original *Geometry of Truth* datasets (`truth/cities`, `truth/neg_cities`, `truth/larger_than`, …), plus `make_conj_disj.py`.
* `datasets/morality/` — the moral-valence datasets: matched good/bad minimal pairs across four moral foundations (`morality/care_harm`, `morality/fairness_cheating`, `morality/honesty_deception`, `morality/loyalty_betrayal`), plus `morality/neg_care_harm`, the lexical-shortcut control. label 1 = morally good. `datasets/morality/moral_common.py` holds the shared machinery — frame validation, name substitution, negation — and each foundation's `<name>/data_gen.py` supplies its frames. Regenerate with `python datasets/morality/<name>/data_gen.py`.

Regenerate with `python datasets/make_animal_human.py` and `python datasets/make_per_template.py`.

## Label stability

The moral labels are validated rather than assumed: `experiments/label_stability/` holds a blind rating round per dataset, where every frame is re-rated on every round.

* `rate.py` — sends a dataset's `rating_prompt.txt` to each model rater via OpenRouter (needs `OPENROUTER_API_KEY`), three passes each, writing to `<dataset>/rated/`.
* `tally.py` — reads every completed sheet, joins on `statement_id`, and reports per-rater agreement, per-statement unanimity, and the pre-registered metrics. Writes `<dataset>/tally.csv`.
* `sentiment_check.py` — VADER (and a roberta pass if torch is installed) over a dataset, to see how far surface sentiment tracks the moral label. Writes `<dataset>/sentiment_scores.csv`.

```
python experiments/label_stability/tally.py --dataset care_harm
python experiments/label_stability/sentiment_check.py --dataset care_harm
```

`documentation/first_test_submission.md` states the pre-registered thresholds; `documentation/project_log.md` records each round's result.

## Files

* `dataexplorer.ipynb`: PCA visualizations of the datasets — single-template panels for every contrast pair, cross-basis projections (does a direction found in one dataset separate another?), harm-vs-neutral orientation checks, and a sweep of where the human/animal split emerges across layers. Figures are saved under `dataexplorer/plots/`.
* `generalization.ipynb`: trains probes on one contrast and evaluates on the others, producing the generalization matrix. Splits are grouped by `subject` so no subject word appears in both train and val.
* `patching_single_template.py` / `patching_single_template.ipynb`: activation patching between a matched animal prompt and human prompt (identical templates, single-token subjects), measuring the shift in the HUMAN − ANIMAL logit difference per (token, layer); the notebook plots the result.
* `patching_multi_template.py` / `patching_multi_template.ipynb`: the multi-template counterpart — same patching mechanic, but a zero-shot YES/NO moral readout ("should the rescuers make saving this individual their top priority?") rather than a HUMAN/ANIMAL category readout. Results go to `experimental_outputs/multi-template/patching_results.json`, figures to `dataexplorer/plots/multi-template/`.
* `interventions.py`: causal intervention experiments — adds or subtracts a probe direction across a layer range during live forward passes.
* `few_shot.py`: calibrated 5-shot baseline.
* `logprobs.py`: log-probabilities of statements under the model.
* `probes.py`: probe classes (`LRProbe`, `MMProbe`, `CCSProbe`).
* `utils.py` and `visualization_utils.py`: utilities for managing datasets and producing visualizations.

Results are appended to JSON files in `experimental_outputs/`, under a subfolder per dataset grouping (`single-template/`, `multi-template/`) matching the one in `datasets/`. A script will fail if its output file doesn't exist yet, so create it containing `[]` before the first run.

Figures follow the same split: `dataexplorer/plots/single-template/` and `dataexplorer/plots/multi-template/`. The multi-template results and figures were ported from the `animal-vs-human-results` branch and were computed on the earlier 20-subject-per-category generation, so they don't correspond to the current 40-subject CSVs; re-run before comparing them against anything current.
