# The Geometry of Morality

This repository is forked from the repository https://github.com/saprmarks/geometry-of-truth associated to the paper [*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*](https://arxiv.org/abs/2310.06824) by Samuel Marks and Max Tegmark.

This fork extends their methodology to test whether moral valence (good/bad) is linearly represented in LLM activations the same way true/false is, and whether that representation holds up for more complex moral concepts and across different moral subjects (human vs. animal). The goal is to establish whether morality is a stable, controllable direction in activation space.

Two lines of experiment run in parallel:

* **Moral valence** — is good/bad itself a direction? Matched good/bad minimal pairs over four moral foundations (`datasets/morality/`), with the labels validated by a blind rating round rather than assumed.
* **Moral subject** — holding the scenario fixed, does the model represent *who* it is happening to (human, companion animal, farmed animal, wild animal) as a linear direction, and does that direction change under harm scenarios versus neutral ones? (`datasets/single-template/`, `datasets/multi-template/`)

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

* Activations are written to `acts/`.
* `--layers -1` saves every layer.
* Omit `--device` to run remotely on NDIF via `nnsight`.

On a rented GPU instance (e.g. Vast.ai), `setup_host.sh` handles clone, install, and weight download once you're on the machine — see its usage header. `documentation/compute_estimate.md` has the GPU/storage/cost plan.

## Datasets

A dataset name is its path relative to `datasets/` without the `.csv`, so it can include a subdirectory (`single-template/human_farmed/flood_01`, `truth/cities`). That same name is the activation cache directory. Every CSV has `statement` and `label` (1/0).

| Folder | Contents | Extra columns |
| --- | --- | --- |
| `morality/` | Good/bad statement pairs over four moral foundations: `care_harm`, `fairness_cheating`, `honesty_deception`, `loyalty_betrayal`. Label 1 = morally good. | `pair_id`, `frame_id` |
| `morality/neg_care_harm` | Refusal-form versions of the `care_harm` frames, flipping the label while keeping the original verb. | `pair_id`, `frame_id` |
| `single-template/<pair>/<template>.csv` | Animal/human statements, one scenario template per file, so only the subject word varies. Seven category pairs × four templates (`fire_01`, `flood_01`, `morning_01`, `rest_01` — two harm, two neutral). | `subject`, `subject_category`, `template_id`, `axis` |
| `multi-template/<pair>_<axis>.csv` | The same statements pooled, all templates on one axis per file (`human_farmed_harm.csv`, …). | same |
| `truth/` | The original *Geometry of Truth* datasets (`cities`, `neg_cities`, `larger_than`, …), plus `make_conj_disj.py`. | varies |

Details:

* **Morality pairs.** The two statements share a sentence frame and end on the same words, differing only in the middle. Each frame is instantiated with several agent names, all sharing a `frame_id`. `moral_common.py` builds them; each foundation's `<name>/data_gen.py` supplies the frames.
* **Animal/human labels are positional.** In `{catA}_{catB}`, label 1 = catA.
* **Source of both animal/human groupings:** `datasets/animal_human_templates.py` — 160 subjects (40 each: human, companion, farmed, wild) × 100 templates (60 harm, 40 neutral). Its docstring lists the constraints they were written to.

Regenerate with `datasets/make_animal_human.py`, `datasets/make_per_template.py`, or `datasets/morality/<name>/data_gen.py`.

## Label stability

`experiments/label_stability/` holds a rating round per morality dataset: models label the statements blind, and the ratings are compared to the intended labels. Each dataset folder holds `rating_prompt.txt`, the blank `rating_sheet.csv`, completed sheets in `rated/`, and the outputs below.

| Script | Does | Writes |
| --- | --- | --- |
| `rate.py` | Sends `rating_prompt.txt` to each model rater via OpenRouter, three passes each. Needs `OPENROUTER_API_KEY`. | `<dataset>/rated/` |
| `tally.py` | Joins ratings on `statement_id`; reports per-rater agreement, per-statement unanimity, and the "depends" rate. | `<dataset>/tally.csv` |
| `sentiment_check.py` | Scores statements with VADER, plus a RoBERTa tweet-sentiment model if torch is installed, per item and within pairs. | `<dataset>/sentiment_scores.csv` |

```
python experiments/label_stability/tally.py --dataset care_harm
python experiments/label_stability/sentiment_check.py --dataset care_harm
```

## Files

| File | Does |
| --- | --- |
| `generate_acts.py` | Runs the model over a dataset, saves the last-token residual stream per layer to `acts/`. |
| `dataexplorer.ipynb` | PCA plots: per-pair single-template panels, one dataset projected onto another's direction, harm vs. neutral, layer sweep. → `dataexplorer/plots/` |
| `generalization.ipynb` | Trains probes on one contrast, evaluates on the others → generalization matrix. Splits grouped by `subject`. |
| `patching_single_template.py` / `.ipynb` | Patches activations from a human prompt into a matched animal prompt; records HUMAN − ANIMAL logit difference per (token, layer). |
| `patching_multi_template.py` / `.ipynb` | Same patching, zero-shot YES/NO readout ("should the rescuers make saving this individual their top priority?") instead of the 5-shot HUMAN/ANIMAL one. |
| `interventions.py` | Trains a probe, adds/subtracts its direction across a layer range during live forward passes, measures the change in P(TRUE) − P(FALSE). |
| `few_shot.py` | Calibrated 5-shot baseline. |
| `logprobs.py` | Log-probabilities of statements under the model. |
| `probes.py` | Probe classes: `LRProbe`, `MMProbe`, `CCSProbe`. |
| `utils.py`, `visualization_utils.py` | Dataset loading/splitting and plotting helpers. |

Outputs:

* Results append to JSON in `experimental_outputs/<grouping>/`, mirroring the `datasets/` folders. A script fails if its output file doesn't exist, so create it containing `[]` before the first run.
* Figures split the same way: `dataexplorer/plots/single-template/` and `.../multi-template/`.
* The multi-template results and figures were ported from the `animal-vs-human-results` branch and computed on the earlier 20-subject-per-category generation, so they don't match the current 40-subject CSVs. Re-run before comparing them to anything current.
