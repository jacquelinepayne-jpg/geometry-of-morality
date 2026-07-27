# Animal-vs-Human Concept Experiments — Design Notes

*Written 2026-07-26, on branch `animal-vs-human`.*

## The pivot

The original plan (test whether **moral valence** is linearly represented, like
truth in Marks & Tegmark) stalled on data generation: producing good/bad
statement pairs that slot into the Geometry-of-Truth (GoT) pipeline cleanly was
hard. The project pivoted to a sharper, more tractable question:

> **Is the human/animal category linearly represented in LLM activations, and
> is that representation causally upstream of the model's preference for
> helping humans over animals?**

This keeps the moral-cognition angle (moral status / speciesism) while using a
category contrast that is easy to generate cleanly with templates.

## Dataset design

Source of truth: `datasets/animal_human_templates.py` (adapted from an earlier
Gemma/TransformerLens prototype on `~/Desktop/data/`; the tokenizer span
machinery was dropped because GoT reads **last-token** activations, so no
subject-token indices — and no HF token — are needed).

- **80 subjects, 20 per category**: `human`, `companion`, `farmed`, `wild`.
  All are single-noun phrases of the form "the X".
- **100 templates**: 60 *harm/suffering* scenarios (fire, road accident,
  illness, abandonment, entrapment, injury, flood, weather, neglect, attack)
  + 40 *neutral* scenarios (resting, eating, walking, being observed, ...).
- Authoring constraints (enforced by asserts in the templates file):
  - exactly one `{subject}` slot, never sentence-initial;
  - no pronouns referring back to the subject (pronoun confound);
  - every scenario plausible for every subject, so scenario content cancels
    in cross-category contrasts;
  - templates end with a period (GoT `--noperiod` strips the last char);
  - no subject word may appear in any template text (two originals were
    edited for this: "stray dogs" → "strays", "aggressive bull" →
    "aggressive animal").

**Why 20 subjects per category instead of 3:** with 3 subject words a probe
could just memorize token identities. 20 per category enables the key control:
evaluate probes on *held-out subject words*. Subject diversity, not raw row
count, is the scientifically load-bearing quantity (cf. `cities` with ~1,500
distinct cities).

### Generated CSVs (all in `datasets/`, GoT format: `statement,label` + extra
columns `subject,subject_category,template_id,axis` that `DataManager` ignores)

- `datasets/make_animal_human.py` → 12 pairwise files
  `{catA}_{catB}_{harm|neutral}.csv` (first category in the name = label 1).
  Harm files: 2,400 rows; neutral: 1,600. All balanced 50/50.
- `datasets/make_help_continuations.py` → `help_continuations.csv`
  (19,200 rows) and `help_continuations_pilot.csv` (2,880 rows; the original
  12 subjects, chosen because GPU budget is limited).

### The harm/neutral split matters

The neutral templates are not just extra data — they are the analog of GoT's
`neg_cities` generalization test *and* a causal-hygiene device: the
intervention direction should be trained on **neutral** contexts (no
harm/helping content) and tested on **harm** decisions, so any behavioral
shift can't be attributed to a "harm salience" component in the probe
direction.

## Experiment plan (in cost order)

### 1. Helping-preference baseline (`logprobs.py`, zero code changes)

`help_continuations*.csv` statements are *scenario + subject-free
continuation* ("Everyone nearby dropped what they were doing and rushed to
help." / "Nobody thought the situation was worth interrupting the day for.").
Two design tricks make whole-statement logprob sums usable:

- continuations never mention the subject → identical string for all subjects;
- each scenario is scored with both **help** and **dismiss** continuations,
  and the analysis uses the difference, which cancels the scenario's own
  logprob: `diff = log P(help|scen) − log P(dismiss|scen)`.

```bash
python logprobs.py --model llama-2-13b --dataset help_continuations_pilot --device cuda:0
```

```python
df = pd.read_csv('experimental_outputs/logprobs/help_continuations_pilot.csv')
p = df.pivot_table(index=['template_id','subject','subject_category'],
                   columns='continuation_type', values='logprob')
p['help_pref'] = p['help'] - p['dismiss']
p.groupby('subject_category').help_pref.agg(['mean','sem'])
```

Note `logprobs.py` runs one unbatched forward pass per statement (pilot ≈ 2,880
passes; full set ≈ 19,200 — batch it before running the full set).
`experimental_outputs/logprobs/` had to be created (script crashes on save
otherwise; the other results JSONs must likewise pre-exist, containing `[]`).

### 2. Zero-shot decision curve (`few_shot.py`)

`few_shot.py` was parameterized: `--pos_token`, `--neg_token`, `--suffix`,
`--save_diffs`, and `--n_shots 0` support. Defaults reproduce the original
TRUE/FALSE behavior exactly.

```bash
python few_shot.py --datasets human_farmed_harm --model llama-2-13b --n_shots 0 --save_diffs \
  --suffix " Should the rescuers make saving this individual their top priority? Answer:" \
  --pos_token YES --neg_token NO --device cuda:0
```

- **Zero-shot is mandatory here**: sampled shots would pair human→YES /
  animal→NO and teach the model the bias being measured.
- `--save_diffs` writes per-query P(YES)−P(NO) with all CSV columns to
  `experimental_outputs/few_shot_diffs/<dataset>.csv` → groupby
  `subject_category` gives the helping-preference curve
  (expected ordering to test: human > companion > farmed > wild).
- Absolute P(YES) calibration doesn't matter; only within-template,
  cross-category differences are interpreted.

### 3. Cache activations + probe layer sweep

```bash
python generate_acts.py --model llama-2-13b --layers -1 \
  --datasets human_farmed_neutral human_farmed_harm human_wild_harm --device cuda:0
```

Then (CPU, free) sweep probe accuracy across layers with `layer_sweep.py`:

```bash
python layer_sweep.py --model llama-2-13b \
  --train_datasets human_farmed_neutral \
  --val_datasets human_farmed_harm human_wild_harm human_wild_neutral
```

It auto-detects which layers are cached, trains a probe per layer (MMProbe by
default; `--probe LRProbe` also supported), prints a table, and appends to
`experimental_outputs/layer_sweep_results.json`. `layer_sweep.ipynb` renders
the saved runs as a color-graded table and accuracy-vs-layer plots. Three
numbers per layer:

- **In-distribution accuracy** = is the category linearly decodable at layer L
  → sets `probe_layer` (where the curve saturates) and `intervene_layer`
  (where it rises) in `config.ini`. The existing llama-2-13b values were tuned
  for *truth* and may not fit; a basic category may emerge earlier.
- **Held-out subject accuracy** (`subj` column; train on 75% of subject words,
  test on the held-out 25%) = rules out token-identity memorization.
- **Transfer accuracy** (train neutral → test harm; train human_farmed → test
  human_wild) = the actual "one shared direction" claim, analog of GoT's
  cities → sp_en_trans transfer.

### 4. Causal intervention (`interventions.py`) — the headline experiment

`interventions.py` was parameterized the same way (`--pos_token`,
`--neg_token`, `--suffix`, `--prompt`; pass `--prompt ""` for zero-shot,
omit for the original hardcoded sp_en_trans prompts).

```bash
python interventions.py --model llama-2-13b --probe MMProbe \
  --train_datasets human_farmed_neutral --val_dataset human_farmed_harm \
  --subset false --intervention add --prompt "" \
  --suffix "Should the rescuers make saving this individual their top priority? Answer:" \
  --pos_token YES --neg_token NO --device cuda:0
```

Reads: train the human-direction probe on neutral-context activations, add it
to the residual stream on **animal** harm scenarios (`--subset false`), watch
P(YES)−P(NO).

**Required baselines and controls (run before/alongside):**

- `--intervention none --subset false`: animal baseline — the starting point.
- `--intervention none --subset true`: human baseline — the ceiling; effect
  size is "moved X% of the way from animal baseline to human baseline".
  Without it you can't tell a targeted effect from "perturbation raises
  P(YES) for everything".
- The `none` runs also sanity-check the readout: `tot` = P(YES)+P(NO) must be
  reasonably high, else the model isn't answering the question format at all
  (fix the suffix before renting more GPU time). And if the two baselines are
  equal, there is no preference gap to explain.
- Symmetric run: `--intervention subtract --subset true` (humans should move
  toward the animal baseline).
- `--probe random`: random direction, same norm calibration — if it shifts
  P(YES) as much as the trained probe, the effect is generic perturbation,
  not the human/animal representation.

### 5. Activation patching (`patching.py` / `patching.ipynb`) — stretch

`patching.py` now takes `--true_prompt`, `--false_prompt`, `--pos_token`,
`--neg_token`; the hardcoded default is an animal/human minimal pair
(fire_01, "the man" vs "the pig", YES/NO readout), with the old sp_en_trans
pair kept in a comment; `patching.ipynb` loads
the tokenizer for whatever model is recorded in the results. Purpose here is **not** layer selection (the probe sweep does that for free) but
mechanistic localization: patch a minimal pair (same template, "the man" vs
"the pig", rescue question, YES/NO readout) layer×token and see where the
category information flows and when the readout stops depending on it. Output
is a *logit difference* (no accuracy anywhere in this script). Constraints:
both subjects must tokenize to equal length under the LLaMA-2 tokenizer (the
script raises otherwise); it patches from the first differing token to the
end, which is fine for mid-sentence subjects. Cost ≈ n_layers × n_toks single
forward passes *per prompt pair*.

## Code changes made (all defaults preserve original truth-experiment behavior)

| File | Change |
|---|---|
| `few_shot.py` | `--pos_token/--neg_token/--suffix/--save_diffs`; `--n_shots 0` (skips KV-cache prompt pass, adds BOS to queries); records tokens/suffix in results JSON |
| `interventions.py` | same token/suffix args; `--prompt` override; `prepare_data` takes suffix; suffix length recomputed so the intervention still targets end-of-statement tokens; tokens/suffix recorded in results JSON |
| `datasets/animal_human_templates.py` | new — subjects, harm+neutral templates, help/dismiss continuations, asserts |
| `datasets/make_animal_human.py` | new — writes the 12 pairwise CSVs |
| `datasets/make_help_continuations.py` | new — writes full + pilot logprobs datasets |

| `patching.py` | `--true_prompt/--false_prompt/--pos_token/--neg_token`; default is now the hardcoded animal/human fire_01 pair with YES/NO (sp_en_trans pair kept in a comment) |
| `patching.ipynb` | tokenizer loaded from the model recorded in the results (via config.ini) instead of hardcoded 70b |
| `dataexplorer.ipynb` | model → llama-2-13b, all cells point at the 12 animal/human CSVs; fixed stale `model_size=` calls; layer-sweep cell uses layers 4/8/12/16/24 |
| `generalization.ipynb` | train medleys = neutral→harm human/farmed and human/wild; val = all 12 CSVs; CCS cell marked skip (needs negation pairs); plotting cells work with a single model in `models` |

| `layer_sweep.py` | new — per-layer probe accuracy (iid / held-out subjects / transfer) on cached acts; used to pick `probe_layer` and `intervene_layer` |
| `layer_sweep.ipynb` | new — displays saved sweep runs: per-layer accuracy table and accuracy-vs-layer plots |

`generate_acts.py`, `utils.py`, `probes.py`: unchanged. `matplotlib` added to
`requirements.txt` (the notebooks already needed it).

## Gotchas / reminders

- YES and NO must each be a **single token** with a leading space under the
  LLaMA-2 tokenizer (they are; re-check via `tokenizer.encode` if the readout
  tokens or suffix are ever reworded).
- Everything is uncommitted on `animal-vs-human` as of writing.
- GPU strategy: one Vast.ai session runs steps 1–4 back-to-back (scripts were
  prepared in advance precisely so a single rental suffices). Start with
  llama-2-7b or the pilot CSV if budget-squeezed; `--device remote` (NDIF) is
  a zero-rental fallback if access is available.
- Probing/analysis after `generate_acts.py` is CPU-cheap and local — download
  the `acts/` directory before killing the GPU box.
