# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth) (Marks & Tegmark, *The Geometry of Truth*, arXiv:2310.06824). The fork's goal is to test whether **moral valence** (good/bad) is linearly represented in LLM activations the way true/false is. Planning docs live in `documentation/` (see `compute_estimate.md` for the model/GPU/cost plan: LLaMA-2 7B/13B/70B on Vast.ai, starting with 13B).

## Setup and Commands

```bash
pip install -r requirements.txt
```

Everything downstream depends on cached activations, which must be generated first:

```bash
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets care_harm neg_care_harm --device cuda:0
# --layers -1 saves all layers; omit --device to run remotely via NDIF (device defaults to "remote")
```

Experiment scripts (all use argparse; see each `__main__` block for options):

```bash
python few_shot.py --datasets care_harm neg_care_harm --model llama-2-13b --device cuda:0   # calibrated 5-shot baseline
python interventions.py --model llama-2-13b --intervention add --device cuda:0        # causal interventions
python patching.py --model llama-2-13b --device cuda:0                                # activation patching
python logprobs.py --model llama-2-13b --dataset care_harm --device cuda:0             # statement log-probs
```

Probe training/generalization experiments are in `generalization.ipynb`; dataset visualizations in `dataexplorer.ipynb`.

There are no tests or linters.

## Key Conventions

- **`config.ini` defines models.** Each section (e.g. `[llama-2-13b]`) maps a model name (as passed to `--model`) to a HuggingFace repo or local weights path, plus per-model `probe_layer`, `intervene_layer`, and `noperiod` settings used by `interventions.py`.
- **`--device` defaults to `"remote"`** in every script, meaning execution on the NDIF server via `nnsight`. Pass `cuda:0` (or similar) for local runs, which load weights in bf16 with `device_map="auto"`.
- **Activations are cached on disk** under `acts/<model>/<dataset>/layer_<L>_<batch_idx>.pt` in batches of 25 statements (`ACTS_BATCH_SIZE` in `utils.py`), saving the residual-stream output at the last token position. `noperiod` variants go in an extra `noperiod/` subdirectory. `utils.collect_acts` reads them back; it raises if activations for a dataset haven't been generated yet.
- **Datasets** are CSVs at the top level of `datasets/` with at minimum `statement` and `label` (1 = morally good, 0 = morally bad) columns. **`care_harm` is the working dataset** — pass it bare, as `--datasets care_harm`. The other moral foundations (`fairness_cheating`, `loyalty_betrayal`, `honesty_deception`) follow the same format once written. Each has a `datasets/<name>/data_gen.py` generator built on the shared frame machinery in `datasets/moral_common.py` (matched good/bad minimal-pair templates instantiated with several agent names; extra columns `pair_id`, `frame_id`). `DataManager` groups train/val splits by `frame_id` when that column exists, so name-variants of one frame never straddle the split — the within-dataset effective *n* is the frame count, not the row count. Negated variants are prefixed `neg_`.
- **The moral `neg_` sets are the lexical-shortcut control**, and each `data_gen.py` writes its `neg_<name>.csv` alongside `<name>.csv` — there is no separate generator to run. They negate by agentive refusal ("refused to hurl her elderly neighbor's groceries down the apartment stairs", label 1) rather than plain "did not", because plain negation of a harm is morally neutral and would make half the flipped labels indefensible. Train on `<name>` and test on `neg_<name>`, the way the original work paired a dataset with its negation; a probe that has only learned the valence of the action verb scores at or below chance there. The negated set is rated on its own wording rather than inheriting its source's verdict, so it gets its own sheet under `experiments/label_stability/neg_<name>/`.
- **Negation is per-frame opt-in, so the two files are not row-aligned.** A frame supplies `neg_good`/`neg_bad` only if refusing its good act is blameworthy on its own — a role duty, or serious harm at trivial cost. Refusing a discretionary kindness — company, comfort, an invitation, courtesy (a bus seat, a suitcase), charity (your own lunch, your own soup), or care for an animal you do not own — is merely unkind, not wrong, so those frames keep their pair in `<name>` and omit their negations; `care_harm` runs 38 frames against `neg_care_harm`'s 22. A frame's bad act must also use a **positive-form harm verb** to opt in, since negating an omission gives a double negative ("refused to ignore the cries of…") that entails only that the omission stopped; `validate()` enforces this against the `OMISSION_VERBS` blocklist. `frame_id` and `pair_id` carry over from the source frame, so `frame_id` has gaps in the negated file. Unlike truth, moral valence has no truth-functional negation — refusal pulls both a good and a bad act toward the neutral middle — so label 1 in a `neg_` file means "not blameworthy" at least as much as "praiseworthy". Report generalization to it with the signed projection onto the probe direction, not accuracy alone: compressed-but-positive means the labels went neutral, confidently negative means the probe was riding the action verb.
- **Label stability is validated per dataset, per round, and recorded outside the generator.** Every `data_gen.py` run writes `experiments/label_stability/<name>/{rating_sheet.csv,rating_prompt.txt}` covering *every* frame — one name-instantiation each, shuffled so a rater never sees a minimal pair side by side. `rate.py --dataset <name>` sends that prompt verbatim through OpenRouter to each model rater in its `RATERS` dict (Claude Opus 5, Gemini 3.6 Flash, GPT-5.6 Luna), three passes each at temperature 1.0, and writes `<name>/rated/rating_sheet_<rater>[-<pass>].csv`; hand-filled sheets go in `<name>/` itself as `rating_sheet_<rater>.csv`. Then `python experiments/label_stability/tally.py --dataset <name>` reads both locations and writes `<name>/tally.csv` with per-statement unanimity. That file is the record of which wording holds up; there is deliberately no `validated` flag in the frame list, so a frame whose wording is edited is always re-rated instead of coasting on a verdict its old wording earned. `sentiment_check.py --dataset <name>` is the A5 preview add-on. `experiments/label_stability/pilot/` is the frozen first round (see `documentation/first_test_submission.md`); its `tally.py` hardcodes pilot paths and is part of that record, not the pipeline.
- **Experiment results append to JSON files** in `experimental_outputs/`; scripts read the existing file and append, so the file must exist (e.g. containing `[]`) before a first run.

## Architecture

The pipeline is two-phase: (1) run forward passes once to cache activations, (2) do all analysis on the cached tensors (cheap, CPU-friendly).

- `generate_acts.py` — extraction: loads a model via nnsight, traces each statement, saves last-token residual-stream activations per layer. Also provides `load_model`, reused by the experiment scripts.
- `utils.py` — `DataManager` is the central data abstraction: loads cached activations + CSV labels per dataset, handles train/val splits, centering/scaling, concatenation across datasets, and PCA projection. Notebooks and scripts build on it.
- `probes.py` — three probe classes with a shared interface (`from_data`, `pred`, `.direction`): `LRProbe` (logistic regression), `MMProbe` (mass-mean), `CCSProbe` (contrast-consistent search, needs paired pos/neg datasets). `.direction` is what the intervention experiments add/subtract in the residual stream.
- `interventions.py` — trains a probe on cached activations, then adds/subtracts the (norm-calibrated) probe direction across a layer range (`intervene_layer`..`probe_layer` from `config.ini`) during live forward passes. Probe class is selected by name via `eval(args.probe)`. **Still truth-shaped and not yet migrated:** it scores the shift in P(` TRUE`) − P(` FALSE`) on a hardcoded few-shot prompt about Spanish vocabulary, and its `--val_dataset` still defaults to `geometry_of_truth/sp_en_trans`. Both need replacing with a moral readout (e.g. P(` GOOD`) − P(` BAD`) over a care/harm few-shot prompt) before the causal step means anything for moral valence. `--train_datasets` already defaults to `care_harm neg_care_harm`.
- `patching.py` and `few_shot.py` also carry true/false wording inherited from the fork; check their prompts before quoting results on moral data.
- `visualization_utils.py` — plotly helpers for the notebooks.
