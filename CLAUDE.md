# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth) (Marks & Tegmark, *The Geometry of Truth*, arXiv:2310.06824). The fork tests whether **moral valence** (good/bad) is linearly represented in LLM activations the way true/false is.

The active line of work is the **moral subject** question: holding the scenario fixed, is *who it happens to* (human / companion animal / farmed animal / wild animal) a linear direction, and does it look different under harm scenarios than neutral ones? Planning docs live in `documentation/` (`compute_estimate.md` has the model/GPU/cost plan: LLaMA-2 7B/13B/70B on Vast.ai, starting with 13B). `setup_host.sh` provisions a fresh GPU instance.

## Setup and Commands

```bash
pip install -r requirements.txt
```

Everything downstream depends on cached activations, which must be generated first:

```bash
python generate_acts.py --model llama-2-13b --layers 8 10 12 --datasets single-template/human_farmed/flood_01 truth/cities --device cuda:0
# --layers -1 saves all layers; omit --device to run remotely via NDIF (device defaults to "remote")
```

Regenerate datasets from the templates (rarely needed; the CSVs are checked in):

```bash
python datasets/make_animal_human.py       # datasets/multi-template/{pair}_{axis}.csv
python datasets/make_per_template.py       # datasets/single-template/{pair}/{template}.csv
```

Experiment scripts (all use argparse; see each `__main__` block for options):

```bash
python few_shot.py --datasets truth/cities --model llama-2-13b --device cuda:0   # calibrated 5-shot baseline
python interventions.py --model llama-2-13b --intervention add --device cuda:0   # causal interventions
python patching_single_template.py --model llama-2-13b --device cuda:0                           # animal-vs-human activation patching
python logprobs.py --model llama-2-13b --dataset truth/cities --device cuda:0     # statement log-probs
```

Probe training/generalization is in `generalization.ipynb`, dataset PCA visualizations in `dataexplorer.ipynb`, and patching plots in `patching_single_template.ipynb`.

There are no tests or linters.

## Key Conventions

- **A dataset name is a path relative to `datasets/`, without `.csv`** — so it can include a subdirectory: `single-template/human_farmed/flood_01`, `multi-template/human_farmed_harm`, `truth/cities`. The same string is the activation cache subdirectory. Nothing special-cases the slash; `load_statements`, `DataManager.add_dataset`, and `collect_acts` all just join it onto a root.
- **`config.ini` defines models.** Each section (e.g. `[llama-2-13b]`) maps a model name (as passed to `--model`) to a HuggingFace repo or local weights path, plus per-model `probe_layer`, `intervene_layer`, and `noperiod` settings used by `interventions.py` and read by the notebooks.
- **`--device` defaults to `"remote"`** in every script, meaning execution on the NDIF server via `nnsight`. Pass `cuda:0` (or similar) for local runs, which load weights in bf16 with `device_map="auto"`.
- **Activations are cached on disk** under `acts/<model>/<dataset>/layer_<L>_<batch_idx>.pt` in batches of 25 statements (`ACTS_BATCH_SIZE` in `utils.py`), saving the residual-stream output at the last token position. `noperiod` variants go in an extra `noperiod/` subdirectory. `utils.collect_acts` reads them back; it raises if activations for a dataset haven't been generated yet.
- **Datasets** are CSVs with at minimum `statement` and `label` (1/0) columns; extra columns are carried through and ignored by `DataManager` unless named. The animal/human CSVs add `subject`, `subject_category`, `template_id`, `axis`.
- **Grouped splits matter here.** Pass `group='subject'` to `add_dataset` for the animal/human data so no subject word lands in both train and val — a row-wise split lets the probe memorize the subject token and inflates val accuracy. Grouped splits are stratified by label to keep both sides balanced.
- **Label polarity is positional.** In `{catA}_{catB}`, label 1 = catA. A probe trained on `human_*` and evaluated on `farmed_wild` is being asked a different question, so below-chance accuracy there is real signal, not a bug.
- **Ported results live in their own folder.** `experimental_outputs/multi-template/` and `dataexplorer/plots/multi-template/` hold the patching results and PCA figures carried over from the `animal-vs-human-results` branch; they were computed on the earlier 20-subject-per-category generation, not the current 40, so they do not correspond to the CSVs now in `datasets/multi-template/` and are not comparable to a fresh run. Re-run `patching_multi_template.py` before comparing them against anything current.
- **Experiment results append to JSON files** in `experimental_outputs/<grouping>/`, mirroring the `datasets/` grouping folders — `single-template/patching_results.json` for the 5-shot category readout, `multi-template/patching_results.json` for the zero-shot moral readout. Scripts read the existing file and append, so the file must exist (containing `[]`) before a first run; each script holds its path in a `RESULTS_PATH` constant.

## Architecture

The pipeline is two-phase: (1) run forward passes once to cache activations, (2) do all analysis on the cached tensors (cheap, CPU-friendly).

- `generate_acts.py` — extraction: loads a model via nnsight, traces each statement, saves last-token residual-stream activations per layer. Also provides `load_model`, reused by the experiment scripts.
- `datasets/animal_human_templates.py` — source of truth for the animal/human data: 160 subjects (40 per category: human, companion, farmed, wild) × 100 scenario templates (60 harm, 40 neutral). The module docstring lists the authoring constraints (one `{subject}` slot, never sentence-initial; no pronouns; no template contains a subject word; scenarios plausible for every subject; no category-ambiguous subject words). Honor these when adding templates or subjects.
- **Animal/human data is split into two groupings, one folder each.** `datasets/single-template/<pair>/<template>.csv` isolates one scenario template per file; `datasets/multi-template/<pair>_<axis>.csv` pools every template on an axis. Dataset names carry the folder as a prefix, and so do the `acts/` cache paths. Keep new animal/human data in whichever folder matches its template structure rather than at the `datasets/` root.
- `datasets/make_animal_human.py` — writes the pooled grouping to `datasets/multi-template/`, one CSV per (category pair, axis), e.g. `human_farmed_harm.csv`.
- `datasets/make_per_template.py` — writes the single-template grouping to `datasets/single-template/<pair>/<template>.csv`, where the only thing varying within a file is the subject word. Defaults to `fire_01 flood_01 morning_01 rest_01` (2 harm, 2 neutral). `human_animal` balances the animal side by sampling evenly across the three animal categories.
- `datasets/truth/` — the original Geometry-of-Truth datasets and `make_conj_disj.py`.
- `utils.py` — `DataManager` is the central data abstraction: loads cached activations + CSV labels per dataset, handles train/val splits (row-wise or grouped), centering/scaling, concatenation across datasets, and PCA projection.
- `probes.py` — three probe classes with a shared interface (`from_data`, `pred`, `.direction`): `LRProbe` (logistic regression), `MMProbe` (mass-mean), `CCSProbe` (contrast-consistent search, needs paired pos/neg datasets). `.direction` is what the intervention experiments add/subtract in the residual stream.
- `interventions.py` — trains a probe on cached activations, then adds/subtracts the (norm-calibrated) probe direction across a layer range (`intervene_layer`..`probe_layer` from `config.ini`) during live forward passes, measuring the shift in P(TRUE) − P(FALSE). Probe class is selected by name via `eval(args.probe)`.
- `patching_multi_template.py` — the multi-template counterpart to `patching_single_template.py`. Same patching mechanic, but a zero-shot YES/NO moral readout instead of the 5-shot HUMAN/ANIMAL category readout, writing to `experimental_outputs/multi-template/patching_results.json` (`patching_single_template.py` writes to `single-template/`). The two ask different questions (does the model represent the category vs. does that representation drive its stated helping preference), so neither replaces the other and their results stay in separate files. Its patch loop passes `scan=True` explicitly rather than using the module-level `tracer_kwargs`; that is what produced the checked-in results.
- `patching_single_template.py` — patches residual-stream activations from a human prompt into a token-aligned animal prompt (same four few-shot templates, single-token subjects, harm query) and records the HUMAN − ANIMAL logit difference for every (token, layer). Writes incrementally and supports `--continuation_idx` to resume a failed run.
- `visualization_utils.py` — plotly helpers (`TruthData.from_datasets(...).plot(...)`) for the notebooks; figures land in `dataexplorer/plots/`.
