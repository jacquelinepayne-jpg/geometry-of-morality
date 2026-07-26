# Project Log

## 2026-07-25 — Assumptions

Assumptions underlying the project, grouped by category. Revisit as experiments land;
mark each ✓ (validated), ✗ (falsified), or ~ (revised) as evidence comes in.

### A. Dataset Construction & Labeling

1. ✓ Moral minimal pairs can have a stable, largely-agreed-upon valence label, in the way
   "Paris is in France" has a stable truth value. (Validated 2026-07-25, see the label
   stability test entry below.)
2. I can separate statements whose moral valence is obvious and uncontroversial from
   statements whose valence depends on cultural, political, or philosophical position.
3. Labels derived from or checked against the Moral Foundations Dictionary reflect genuine
   moral judgment, not just lexical sentiment already baked into the dictionary's word list.
4. I can build several such datasets, one per moral topic/foundation (e.g. honesty,
   care/harm, fairness/cheating, loyalty/betrayal), each with its own fixed template and
   non-overlapping vocabulary. Sentences will be matched in structure and length within
   each pair, and roughly matched in dataset size across topics, so no single topic
   dominates when results are averaged.
5. ~ LLM-assisted generation plus manual review can produce ~1,000+ statements per topic
   (comparable to `cities` at ~1,500) without label noise or systematic lexical shortcuts —
   i.e., the label is never readable off a single valence-laden word or the final tokens
   of the statement. (Revised 2026-07-25: the single-word version holds, VADER orders only
   13 of 20 pairs correctly, but a sentence-level sentiment classifier orders 20 of 20 at
   AUC 0.887. "Lexical shortcut" is too narrow a framing; the shortcut is sentence-level
   harm language. See the transformer-half entry below.)
6. Negation pairs work for moral claims: "X is not morally wrong" is a valid label-flipped
   counterpart to "X is morally wrong," even though moral negation is semantically messier
   than factual negation ("not wrong" ≠ "good" — it may only mean "permissible"). This is
   required for the CCS probe and negation-consistency analyses to carry over.
7. ~ Enough natural-sounding pleasant-but-immoral and unpleasant-but-moral statements exist
   to build a held-out sentiment-anticorrelated evaluation set, and off-the-shelf sentiment
   classifiers (e.g. VADER, RoBERTa-sentiment) are reliable enough to certify the
   decorrelation quantitatively rather than by my own (biased) intuition. (Revised
   2026-07-25: VADER is not adequate as a certifier on this material — 11 of 40 statements
   match no lexicon entry at all and score exactly 0.000. Use twitter-roberta. The
   "pleasant-but-immoral / unpleasant-but-moral" framing also assumes a two-sided axis;
   measured sentiment here is one-sided, so the set needs harmful acts in flat language,
   not just harmful acts in pleasant language.)

### B. Representational (does a "morality direction" exist at all?)

1. A linear probe trained on these pairs will find a direction that separates good/bad
   statements within each moral topic.
2. Each direction reflects the moral content itself, not just the shared template
   structure the pairs are built from.
3. Each direction is specific to the moral concept, not just general sentiment/valence.
   Two independent checks are assumed jointly sufficient to rule out the sentiment
   confound: (a) generalization accuracy on the held-out anticorrelated set, and
   (b) cosine similarity between the moral direction and a sentiment direction trained
   on morally neutral pleasant/unpleasant statements.
4. A probe trained on one moral topic will generalize to some other moral topics at
   above-chance accuracy, but not necessarily all of them.
5. The transfer pattern is interpretable as the finding itself: uniformly low cross-topic
   transfer → topic-specific features with no shared moral direction; uniformly high →
   a single unified moral direction; high within clusters but low between → multiple
   distinct moral directions.
6. The last-token residual stream is the right readout point for moral valence. Unlike
   the original paper, I will validate this with a small token-position pilot (~200
   statements, probes at each position-from-end across a few layers) rather than inherit
   it. A mid-sentence accuracy peak would itself be evidence of a lexical shortcut, so
   the pilot doubles as a confound diagnostic.
7. A layer exists where moral valence is linearly decodable. The `probe_layer` values in
   `config.ini` were tuned for *truth* and are assumed not to transfer automatically —
   I will run my own layer sweep (`--layers -1`) for moral datasets.
8. A probe can't distinguish "the model represents X as wrong" from "the model knows that
   people describe X as wrong" (descriptive norm knowledge). I assume the representational
   claim is worth making either way, but the write-up must not conflate the two.

### C. Causal Validation

1. Assuming A and B hold, the direction(s) found are causally implicated, not just
   correlational: intervening on them should shift the model's outputs from treating a
   vice statement as virtuous and vice versa, tested on out-of-distribution data. The
   primary method is directional intervention (adding/subtracting the norm-calibrated
   probe direction in the residual stream, per `interventions.py`), which matches the
   steering-vector motivation; activation patching (`patching.py`) is a supplementary
   localization tool, not the main causal test.
2. If I find multiple distinct moral directions rather than one, causal intervention must
   be tested per-cluster rather than assuming a single direction works universally —
   steering one direction might flip behavior on topics that share it while doing nothing
   on topics that don't.
3. The original intervention protocol transfers with one adaptation: the P(TRUE) − P(FALSE)
   metric becomes a moral analog (e.g. P(good) − P(bad)), which assumes a
   judgment-eliciting prompt suffix and target-token pair exist that elicit clean one-token
   moral judgments from the model. This is a design decision I own, not inherited.
4. The `intervene_layer`..`probe_layer` range tuned for truth is a reasonable starting
   point for moral interventions, but may need its own tuning.

### D. Reporting & Scientific Integrity

1. If the causal intervention piece doesn't produce a strong result, a rigorous
   representational-only finding is still a complete and worthwhile release, as long as
   it is explicitly framed as a claim about representation, not controllability, and the
   two aren't conflated in the write-up.
2. Whether I find no shared direction, one unified direction, or multiple clustered
   directions, I'll report that pattern as the finding — rather than treating only some
   outcomes as "real" results and others as failures to be minimized or re-run until they
   look cleaner.

### E. Practical / Resourcing

1. I will have enough compute through Vast.ai, which is fast and cheap enough to run these
   experiments in reasonable time without being too expensive — including storage for
   activations across multiple datasets and layers (see `compute_estimate.md`), not just
   GPU time.
2. I have (or will be granted) access to the gated `meta-llama/Llama-2-*-hf` weights on
   HuggingFace, and the pinned dependencies (`nnsight`, `transformers`, etc.) still
   install and run correctly on current infrastructure.

### F. Existing Work & Positioning

1. The original Geometry of Truth finding is robust and will reproduce on my setup
   (llama-2-13b, this fork, current library versions).
2. Known critiques of truth-probing (probes latching onto salient non-truth features;
   CCS finding arbitrary contrastive features rather than truth) apply to the inherited
   methodology. I assume the matched-syntax controls, sentiment checks, and causal step
   address the same class of objection for morality.
3. Related work (moral-foundations probing of LLMs, contrastive activation addition /
   steering-vector papers that include behavioral traits) doesn't already answer this
   specific question. The per-foundation clustering analysis with matched sentiment
   controls and causal validation is assumed to be a novel contribution — a literature
   positioning pass is needed to confirm before release.

## 2026-07-25 — Label stability test (A1): result

Setup. 40 care/harm statements in 20 matched good/bad pairs, generated by
`datasets/care_harm/data_gen.py`. Blind ratings from me plus GPT-5.5, Claude Opus 5,
and Gemini 3.1 Pro, three runs each. Materials and raw ratings live in
`experiments/label_stability/`.

Result. 37 of 40 statements got unanimous labels across all ten passes (92.5%).
The prediction was 90% or more, so A1 survives. My "depends" rate was 5% and the
models' was under 1%, far below the 30% tripwire. The LLM majority matched my
intended label on all 40 statements, so no labels in `care_harm.csv` changed.

The three failures matched the predicted failure mode of unstated context.
GPT-5.5 read "locked the students inside the classroom" after a fire alarm as a
possible lockdown and said depends in all three runs. I flagged both halves of the
wheelchair pair because inviting someone to a kickball game they may not be able to
play is not clearly caring, and blocking them is not clearly cruel.

Revision round (the one allowed by the pre-registration). The fire alarm became
visible smoke so the danger is real and not possibly a drill. Kickball became a card
game the new kid can actually play. Statement ids 5, 6, 9, and 23 changed. Note that
I raised the wheelchair objection myself, so my own re-rating of that pair is not
independent. The models' re-ratings are the real check there.

Post-revision result (same day). Seven fresh passes on the full revised sheet, from
me plus two runs each of GPT-5.5 mini, Claude Opus 5, and Gemini 3.1 Pro, tallied by
`experiments/label_stability/tally.py`. 39 of 40 statements unanimous (97.5%), all
matching the intended labels. The wheelchair pair is fully resolved. The one holdout
is the revised fire statement. One Opus 5 run said depends because locking students
away from a smoke-filled hallway could be sheltering them rather than trapping them,
which is a fair reading since shelter-in-place is real fire protocol. The revision
budget is spent, so this stands as the final A1 number: 97.5% unanimous after one
revision round. A1 is validated. The locked-in pair should be replaced or dropped
when the full dataset is generated, since harm through confinement invites a
protective reading unless the harmful intent is unmistakable.

Lessons for the dataset template. A good care act has to meet the recipient's
actual need, so each statement must make the need and the act's fit to it explicit.
Also, each model gave identical answers across its three runs, so repeat runs add
nothing. Future rounds should use one run per model and more model families instead.

## 2026-07-25 — Sentiment pass on care_harm (A5 preview): VADER half

I predicted sentiment would correlate strongly with the moral label in this naive
batch. At the lexicon level that prediction was wrong. VADER's correlation with the
label is r = 0.325. Good statements average a neutral 0.00 compound score, bad ones
only -0.20, and sentiment sign predicts the label at 62.5%, barely above chance.

The cause is the matched-pair design itself. Good care acts respond to a need, so
the good statements are full of negative words like injured, failed, smoke, tears,
and freezing. A lexicon scorer reads those words, not the deed. "Leo gave water to
the injured bird" scores -0.73 while "Priya pointed and laughed at his scraped knee"
scores +0.46.

Interpretation stays narrow. The lexical-shortcut version of the sentiment confound
is already weak in this batch, which is good news for B3. The sentence-level version
is untested, and the model-internal sentiment a probe could latch onto is closer to
a transformer's judgment than to a word list. The transformer half of the check
(`experiments/label_stability/sentiment_check.py`) runs automatically where
transformers and torch are installed, e.g. the Vast.ai host. The anticorrelated
control set (A5, B3) is still needed either way. Scores are saved in
`experiments/label_stability/sentiment_scores.csv`.

## 2026-07-25 — Sentiment pass on care_harm (A5 preview): transformer half

It ran on the laptop, not on Vast.ai. The classifiers here are 66M-125M parameter
models scoring 40 short sentences, which is seconds of CPU work; nothing about this
check needed the GPU host. Environment notes at the end.

The transformer half reverses the reading I took from the VADER half. I wrote that
the lexical version of the confound is weak in this batch and called that good news
for B3. The first clause holds, the conclusion was premature.
`cardiffnlp/twitter-roberta-base-sentiment-latest` correlates with the moral label at
r = 0.655, orders 20 of 20 pairs correctly (paired t = 6.91, p < 0.00001), and
separates good from bad at AUC 0.887, against VADER's 13 of 20. The confound is weak
at the word level and strong at the sentence level.

AUC is the number to watch rather than sign accuracy. A linear probe carries a bias
term, so a constant offset in the sentiment score costs it nothing and only the
ranking matters. Sign accuracy is 77.5% here and understates the problem, because the
whole score distribution sits below zero.

The confound is one-sided, which I did not anticipate. Good statements average -0.025
and bad ones -0.364, so good acts are not scored positive, they are scored null. The
highest P(positive) anywhere in the 40 is 0.20 and mean P(neutral) is 0.695. What the
classifier detects is harm language, present on the bad side and absent on the good
side, not valence running in both directions. That is the same cause I named in the
VADER entry, since both members of a pair carry the need's negative wording, but the
sentence-level model sees past the need to the deed on the bad side only.

Consequences for A7 and B3. "Sentiment-anticorrelated" is underspecified for a
one-sided axis: it cannot be anticorrelated symmetrically against a direction that is
only active one way. Two control types are needed instead. Kind acts in harm-loaded
language, which the narrowest pairs already approximate (Priya and the scraped knee,
Leo and the injured bird, Omar and the sleeping bag, all separated by under 0.09),
and harmful acts in flat or bureaucratic language, of which this batch has none. The
second type is what would actually break a harm detector, so it is the priority. A
cheaper test comes first though: if the probe direction is harm detection rather than
moral valence, it should fail to separate good statements from morally neutral ones,
since both sit at zero on this axis. That needs no new labels.

SST-2 was tried and dropped. `distilbert-base-uncased-finetuned-sst-2-english` hit the
same AUC 0.887 but inverted the Priya pair by 1.9, scoring "pointed and laughed at his
scraped knee" at +0.97. It is fine-tuned on movie reviews, so its axis is review
polarity rather than affect: the right number from the wrong instrument. It is also
badly calibrated for this use, saturating 26 of 40 statements above |0.9| where
twitter-roberta saturates none. Only twitter-roberta is kept in the script.

`sentiment_check.py` now reports the within-pair count, mean gap, AUC and any inverted
pairs alongside the item-level numbers, since the dataset is matched pairs and the
within-pair figure is the one to quote. `sentiment_scores.csv` gains
`twroberta_signed` and `twroberta_neutral`; the neutral mass is kept because the
signed score discards it and a zero would otherwise be ambiguous between confidently
neutral and evenly torn.

Environment, and a version split to be aware of. The Cardiff repo ships only
`pytorch_model.bin`, and transformers 4.57.6 refuses to `torch.load` a `.bin` without
torch >= 2.6 (CVE-2025-32434). This Intel Mac caps at torch 2.2.2, because PyTorch
stopped building x86-64 macOS wheels after that, so the local run uses transformers
4.44.2 and numpy < 2 inside a separate venv (`.venv-sentiment`, kept separate because
torch 2.2 and numpy 2 cannot coexist and the repo environment needs numpy 2 for pandas
3). On the Vast.ai host, where torch is current, `requirements.txt` as pinned runs the
script unchanged. Converting the checkpoint to safetensors once would remove the split
if the local and remote versions ever need to match. `.gitignore` covers `.venv` but
not `.venv-sentiment`, which is about 1 GB and should be added before the next commit.
(Done: added in the dataset-scaling pass below.)

## 2026-07-25 — Full-size datasets for the first experiment (A4, A5)

Built the four per-foundation datasets from A4, all generated by
`datasets/<name>/data_gen.py` on shared machinery in `datasets/moral_common.py`:

- `care_harm` — 614 statements (41 frames; the 19 surviving pilot frames plus a
  replacement for the locked-classroom pair and 21 new frames)
- `fairness_cheating` — 640 statements (40 frames) - to be done
- `loyalty_betrayal` — 640 statements (40 frames) - to be done
- `honesty_deception` — 640 statements (40 frames) - to be done

Design. Each frame is a hand-written matched good/bad minimal pair with a {name}
slot for the agent, instantiated with 8 different names — the moral analog of the
entity substitution that gives `cities` its size (there, one template reused 1,500
times; here, 40 templates reused 8 times each). All pilot constraints are enforced
programmatically per frame: shared final words >= 2 (median 3-4), word-count gap
<= 3, needs/rules/facts stated explicitly per the A1 lessons, no
harm-through-confinement. Sizes sit between `sp_en_trans` (354) and `companies`
(1,200); reaching the ~1,000+ of A5 means writing more frames (the honest path —
name variants add rows but not scenario diversity) or raising `names_per_frame`.

Foundation vocabularies are kept apart by design rule (care: needs and injuries;
fairness: shares, turns, scores, rules; loyalty: friends, teams, secrets,
promises; honesty: claims, records, disclosure), which matters for the cross-topic
transfer analysis in B4/B5. No quantitative overlap check yet.

Leakage guard. Name variants of one frame are near-duplicates, so a row-level
train/val split would leak. The CSVs carry a `frame_id` column and
`utils.DataManager.add_dataset` now groups its split by `frame_id` when the column
exists (row-level behavior unchanged for the geometry_of_truth datasets). The
within-dataset "effective n" is still the frame count, not the row count — quote
frame counts when reporting.

Label check before any activations are generated. Every frame whose exact wording
was not A1-validated goes on a fresh blind rating sheet:
`experiments/label_stability/<dataset>/rating_sheet.csv` plus a paste-ready
`rating_prompt.txt` (one instantiation per frame; 22+40+40+40 = 142 frames, 284
statements). Per the A1 lesson, run one pass per model across more model families
rather than repeat runs. These sheets live in per-dataset subdirectories, and the completed pilot round
(blank sheet, seven scored rater sheets, prompt, tally.py, tally.csv, and the
pilot-batch sentiment_scores.csv) moved intact into
`experiments/label_stability/pilot/` as a frozen record. Rerunning the pilot
tally.py now trips its consistency assert against the current care_harm.csv —
correctly, since the smoke pair was replaced; tally.csv preserves the result.
tally.py still hardcodes the pilot paths and needs generalizing before tallying
round 2 (`sentiment_check.py` stays at the top level as the reusable tool). The
transformer sentiment check (`sentiment_check.py`) is also still care_harm-only
and should be rerun per dataset — the harm-language confound found on care_harm
plausibly reappears as cheat/betrayal/deception language on the new sets. The
anticorrelated control sets from the revised A5/A7 (harmful acts in flat
bureaucratic language, kind acts in harm-loaded language) remain unbuilt.

## 2026-07-26 — Round 2 label stability on the full care_harm frames, and a rebuild of the negated set

Setup. First round on the full frame list rather than the pilot's 20 pairs: 41 frames,
so 82 statements per sheet for `care_harm` and `neg_care_harm`. Ten rater passes each —
me plus three passes apiece of Claude Opus 5, Gemini 3.6 Flash, and GPT-5.6 Luna,
collected through OpenRouter by the new `experiments/label_stability/rate.py`. It sends
each dataset's `rating_prompt.txt` verbatim at temperature 1.0 and writes
`<dataset>/rated/rating_sheet_<rater>[-<pass>].csv`; `tally.py` was generalized off the
pilot's hardcoded paths and now reads both `<dataset>/` and `<dataset>/rated/`.

Result. `care_harm` 81 of 82 unanimous (98.8%), the single dissent a "depends" of mine.
`neg_care_harm` 71 of 82 (86.6%) counting my passes, 76 of 82 (92.7%) across the nine
model passes alone. Every unanimous statement matched its intended label, and — the
number that matters most — **not one statement in either set drew both a "good" and a
"bad" vote.** All 15 dissenting votes were "depends", spread over 11 statements. So no
label in either CSV was wrong; the negated set is merely softer than its source.

The dissent was mine. 9 of the 15 depends votes are mine, 4 are one Opus 5 pass, 2 are
one GPT-5.6 Luna pass, and the remaining seven passes had none. The 86.6% is therefore
measuring my threshold against the models' more than it is measuring rater instability,
and both numbers should be quoted with that split disclosed. This also refutes the
pilot's lesson that repeat runs of one model add nothing: pass 3 of Opus 5 produced four
depends where passes 1 and 2 produced none, on identical input. Repeat passes stay.

Where the dissent concentrated. Four of the ten statements containing a temporal tail
("...until the paramedics arrived", "...before the patients arrived") were
non-unanimous, against seven of the other 72. `refused to X until Y` admits the reading
"declined, then did it once Y happened", which is not the intended meaning at all. All
five such frames were reworded to tails that cannot take scope under a refusal (`while`,
`as`, `through`).

Negation has no truth-functional form for morality. This is the structural finding of
the round, and it explains the softness rather than excusing it. `neg_cities` works
because ¬P is true exactly when P is false — same axis, exact flip. Moral valence has
three regions, and refusal maps into the middle one from both sides: refusing a harm
lands near neutral, and refusing a kindness lands near neutral unless a duty makes it
blameworthy. So **label 1 in a `neg_` file means "not blameworthy" at least as much as
"praiseworthy"**, which is a different axis at the positive end than `care_harm`'s. The
consequence for B-series reporting: generalization to `neg_care_harm` cannot be read off
accuracy alone, because a mid-range number is ambiguous between "the direction does not
transfer" and "the direction transfers fine but neutral items sit near the boundary".
Report the signed projection onto the probe direction alongside it. The two readings make
opposite predictions — compressed-but-positive projections mean the labels went neutral,
confidently negative ones mean the probe was riding the action verb. Worth pre-registering
before 13B.

Rebuild of the negated set. Reading the frames against that finding turned up four defect
classes beyond the temporal tails: negations that strand a verb in front of the refusal
("pulled over and refused to share a jug of water"); refusals whose blameworthiness rests
on unstated capacity (refusing to pay a stranger's lunch balance presupposes the means);
bad acts phrased as omissions, whose negation is a double negative that entails only that
the omission stopped ("refused to ignore the cries of the elderly man"); and frames whose
setup states no need at all, so refusing the good act has nothing to be blameworthy about.
The last of these violates the stated-need rule the pilot already established.

Fix. Negation is now **per-frame opt-in** rather than all-or-nothing: a frame supplies
`neg_good`/`neg_bad` only if refusing its good act is blameworthy in its own right, which
takes a role duty or serious harm at trivial cost. Refusing a discretionary kindness —
company, comfort, an invitation, courtesy, charity, care for an animal one does not own —
is unkind but not wrong, so those frames keep their pair in `care_harm` and drop out of
the negated set. This preserves training data (the source statements are fine; `care_harm`
rated 98.8%) while making every negated label defensible. Two frames were dropped outright
for stating no need (the stray dog on the porch, reading aloud at the nursing home), one
was removed after rewriting (the wheelchair ramp, where a pre-existing blockage meant
refusing the bad act was mere non-aggravation), three had their bad act rewritten into a
positive-form harm verb, and two pronoun bugs were fixed (a cataphoric "her first day"
clashing with a male agent; an unanchored "she" that could attach to the agent and invert
the sentence). The wheelchair was also cut from the recess frame, where it did no work for
the label and so sat in the input as a spurious feature.

`validate()` now enforces the omission rule against an `OMISSION_VERBS` blocklist, so the
double-negative class fails at generation rather than at rating time. The stated-need and
blameworthy-refusal rules remain judgment calls the rating sheet adjudicates, and are
written into its docstring as design rules.

Sizes after regeneration. `care_harm` 552 statements, 276 pairs, 38 frames, sheet of 76.
`neg_care_harm` 296 statements, 148 pairs, 22 frames, sheet of 44 — 16 of the 38 frames
declined negation. Two limitations to carry forward: 296 rows is thin for a held-out
control, and if it proves too thin the fix is more role-duty frames rather than readmitting
courtesy ones; and the 22 survivors skew hard toward role duty and emergency rescue, so
label 0 in the negated set leans toward *neglect of duty* rather than *failure of care*,
and the set concentrates on severe physical harm. That last point is where A5's sentiment
confound should be strongest, and the A5 pass should be read with it in mind.

Record. The round-2 sheets were not archived before regeneration, so the numbers above are
the only surviving record of that round — quote them from here. Round 3 has not been run
yet; the datasets and blank sheets are regenerated and awaiting it.

## 2026-07-26 — Sentiment pass on the full sets (A5): the negated set is already a control

Re-ran `sentiment_check.py` on the regenerated `care_harm` (552 statements, 276 pairs, 38
frames) and on `neg_care_harm` (296, 148, 22) — the 25 July A5 entries above were on the
40-statement pilot and their numbers should be quoted as pilot numbers. Scores now live per
dataset at `experiments/label_stability/<dataset>/sentiment_scores.csv`.

| dataset | classifier | r | mean good | mean bad | AUC | frames good>bad |
|---|---|---|---|---|---|---|
| care_harm | VADER | +0.151 | −0.089 | −0.209 | 0.564 | 20/38 (15 tied) |
| care_harm | twroberta | +0.561 | −0.061 | −0.311 | **0.855** | 36/38 |
| neg_care_harm | VADER | −0.088 | −0.288 | −0.224 | **0.370** | 3/22 (9 tied) |
| neg_care_harm | twroberta | +0.300 | −0.408 | −0.517 | 0.667 | 19/22 |

`care_harm` at scale confirms the pilot: AUC 0.855 against the pilot's 0.887, so the
sentence-level confound survived quadrupling the frame count and is not an artifact of the
19 pilot frames. It stays one-sided in exactly the way the pilot entry described — good
statements average −0.061 rather than positive, mean neutral mass is 0.707, so the
classifier is detecting harm language on the bad side rather than valence in both
directions. The operational consequence is unchanged and now firmer: a moral probe scoring
around 0.85 on held-out `care_harm` frames is statistically indistinguishable from a
sentiment probe, so that result alone cannot be reported as evidence for a moral direction.

VADER on `care_harm` should be dropped from the reading rather than interpreted. Its AUC of
0.564 is not evidence the lexical confound is weak; 138 of 552 statements score exactly
0.0 and 106 of 276 pairs are exact ties, because the lexicon has no entry for *hurled*,
*cooked*, *pocketed*, or *refilled*. It cannot confound what it cannot see. This corrects
the framing of the 25 July VADER entry, which read a low lexical correlation as good news
before the transformer half arrived; the more accurate statement is that the instrument was
blind, not that the confound was absent.

The finding I did not expect is on `neg_care_harm`, and it reverses the prediction at the
end of the entry above. I wrote that the negated set's concentration on severe physical
harm should make the sentiment confound *strongest* there. It is weakest there: twroberta
drops from 0.855 to 0.667, and VADER goes actively inverted at AUC 0.370, below chance.
The one-sided harm-detector hypothesis explains this cleanly, which is a point in its
favour. Label 1 in the negated set is a refusal of harm ("refused to laugh at the moaning
or drain the bedside glass"), so harm language is present on *both* sides of every pair —
a harm detector has nothing left to separate, and a lexicon that simply counts harm words
now points at the wrong label.

So `care_harm` → `neg_care_harm` transfer is a stronger result than the robustness check it
was scoped as, and it comes free. A probe that holds its direction there cannot be riding a
lexical sentiment cue, because that cue is anticorrelated with the labels; and it cannot be
a pure harm detector, because harm language is symmetric across the pair. Report it as a
headline with the signed projection, per the reporting rule for the negated sets. This is a
second instance of the cheap test proposed in the pilot entry (separating good from morally
neutral), obtainable with no new labels, so both should run before any new control set is
written. The two anticorrelated control types from A7/B3 are still needed — neither of
these tests supplies harmful acts in flat language — but they are no longer the only way to
attack the confound, which lowers the priority of writing them.

The highest-value items in the sets are the frames where sentiment ties or inverts, and
they invert for a reason worth preserving: shared vocabulary across the pair. Frame 36 is
good "stopped to bandage her scrapes and call for help" (−0.147) against bad "pocketed the
fallen phone she needed to call for help" (−0.121) — the same helping phrase on both sides,
only the beneficiary differs. Same for the broth pair, where VADER's only scored word in
the good member is *weak* from the shared setup clause while the bad member's *laughed*
(+2.0, the lexicon not knowing it is laughing *at* someone) nearly cancels *moaning* and
*drained*. These are not wording defects to fix. They are the items where a correct
classification is informative, and the anticorrelated control set should be written to look
like them.

Also removed `truncation=True` from the pipeline call, which never fired — the statements
are one sentence and the Cardiff config declares no `model_max_length`, so it only emitted
a warning.

## 2026-07-26 — Negation deferred for the other three foundations, and what unanimity cannot see

Cut `neg_fairness_cheating`, `neg_honesty_deception`, and `neg_loyalty_betrayal` to get to a
first probe result sooner. The three CSVs and their blank rating sheets are deleted, the
frames have had their `neg_good`/`neg_bad` stripped, and `build()` now takes
`write_negations=False` so re-running a generator does not quietly bring them back. Nothing
was lost: the three original generators are copied verbatim to
`documentation/deferred_negations/`, and the shipped positive CSVs are byte-identical after
the strip (md5 unchanged), so this moved no data that any experiment reads. `care_harm`
keeps its negation untouched.

The reasoning is cost, not doubt. A negated set is rated on its own wording rather than
inheriting its source's verdict, so each one costs a full rating round, and three rounds
buys nothing until the `care_harm` result says whether negated sets earn their keep. The
consequence to carry: `neg_care_harm` is now the *only* anticorrelated-sentiment control in
the project. Given `care_harm` sits at twroberta AUC 0.855, a probe scoring near that on
held-out `care_harm` frames is still indistinguishable from a sentiment probe, and the
cross-topic transfer to `fairness_cheating` has to do more work than planned — it is a
weaker control for this purpose, since harm and blame language is shared across foundations
in a way it is not shared between `care_harm` and its own negation.

**Correction to the round-2 entry above.** That entry closes with "Round 3 has not been run
yet". It has: `experiments/label_stability/neg_care_harm/rated/` is timestamped 26 July
11:58–12:00, after the regeneration, and all 44 tally statements match the current CSV. The
result is 44/44 unanimous, 100% matching intended labels, zero `depends`, across 9 passes
(3 models × 3). `care_harm` is 76/76 on the same terms. Quote these rather than the round-2
numbers.

**The more useful finding is that this clean sheet does not mean what it looks like.**
Unanimity measures whether raters agree on the *sign* of a label. It cannot see the thing
that is actually odd about the negated set, which is that its two poles are not the same
construct. Label 0 is a genuine moral verdict — "refused to guide every last student out of
the classroom" as smoke fills it. Label 1 is mostly *refraining from an atrocity*: refused
to pour bleach in the fish tank, refused to slap the resident, refused to snatch the only
sleeping bag from the shivering hiker. Every rater will unanimously call those good, and
every rater will be reporting "not blameworthy" rather than "praiseworthy". So the 100% is
real and also blind, and a fourth binary round would buy nothing.

This is the same compression the 25 July entry predicted from the three-region argument,
but arriving from the labels rather than from the theory, which is worth separating: the
earlier prediction was about where the *probe* would land, and this is a property of the
*label set* that holds regardless of what the probe does. It also revises the note that
`neg_` accuracy can be read as accuracy — it can be read as accuracy against the labels, but
the positive label there is a weaker claim than `care_harm`'s, so a drop in accuracy is not
by itself evidence the direction failed to transfer. The signed-projection reporting rule
already covers this and should be treated as required rather than supplementary.

If the negated set gets more verification later, the question to change is the scale, not
the number of rounds: re-rate on **praiseworthy / permissible / blameworthy** and count how
many label-1 items land in the middle bucket. That measures the defect directly, and it is a
cheap test on 44 statements.

Two smaller corrections. The `loyalty_betrayal` docstring claimed "27 of 40 frames opt in";
the true count is 26, confirmed by an AST count against the preserved generator. The 27 came
from dividing the old negated CSV's row count by 8 names per frame, which is not valid —
frames with no `{name}` slot expand to a single row. And one wording defect survives in
`neg_care_harm` frame 16: "refused to pour out the water of the elderly man living upstairs"
should be "the elderly man's water". It is not worth a re-rate on its own, but fix it
whenever that set is next regenerated.

## 2026-07-26 — Label stability across all four foundations (A1), and one frame that will not converge

Rated and tallied the three remaining topics, so every shipping dataset now has a full round:
three model raters × three passes at temperature 1.0.

| dataset | n | unanimous | matches intended | "depends" |
|---|---|---|---|---|
| care_harm | 76 | 100% | 100% | 0 |
| neg_care_harm | 44 | 100% | 100% | 0 |
| fairness_cheating | 80 | 100% | 100% | 0 |
| honesty_deception | 80 | 100% | 100% | 0 |
| loyalty_betrayal | 80 | 98.8% | 98.8% | 0 |

Zero "depends" in 360 statement-ratings, which is the headline: A1 holds at scale and across
domains, not just on `care_harm`. The frame machinery transfers — allocation, assertion, and
trust vocabulary all produce labels raters agree on as readily as injury vocabulary does.
This clears the way for the cross-topic transfer test, which now has three rated topics
rather than the two that were the stated minimum.

**The single exception is worth more than the 79 that passed.** `loyalty_betrayal` frame 24:

> good: "…{name} kept ordering weekly from the friend's hardware store."
> bad:  "…{name} steered every neighbor away from the friend's hardware store."

Gemini and Opus called the bad member *bad* on all three passes. GPT-5.6 Luna called it
*good* on all three. That is not sampling noise — it is a perfectly reproducible split at the
level of the rater, 3/3 against 3/3, and it is the first of its kind in this project. Every
prior disagreement has been a single stray pass.

**First diagnosis, and it was wrong.** I read it as structural ambiguity: the sentence holds
two stores (the megastore and the friend's), and nothing forces which one "away from"
attaches to, so it is readable as diverting neighbours *from the megastore* — loyalty rather
than betrayal. The fix named the destination explicitly:

> bad: "…{name} sent every neighbor there instead of the friend's hardware store."

Regeneration was surgical, which is worth recording as a property of the generator: because
`frame_names` seeds its RNG on `sha256(frame.good)` and only `frame.bad` changed, the eight
agent names were preserved, and because the sheet shuffle is `random_state=seed` over a fixed
80-row frame, every `statement_id` stayed put. Net change: 8 rows in the CSV, 1 row in the
sheet, nothing else. Editing one frame's bad half costs exactly one re-rate and leaves the
rest of the round comparable.

**The reword changed nothing.** After re-rating, GPT-5.6 Luna still reads it *good* on all
three passes, Gemini and Opus still *bad*. Removing the attachment ambiguity left the split
bit-identical, which falsifies the wording hypothesis cleanly.

**Revised reading: this is a real moral disagreement, not a defect.** The frame's own inline
comment already says it — "nobody owes a friend their custom." Directing neighbours to a
cheaper store trades a small loss to one friend against a real gain to many neighbours, so
the verdict depends on how much weight partiality carries against impartial benefit. That is
a live ethical question, and it is arguably *the* contested question inside the loyalty
foundation rather than an incidental one. The raters are not making an error; they are
holding different positions.

This is exactly the class assumption A2 says to exclude — statements whose valence depends on
philosophical position rather than being obvious and uncontroversial — so the frame should be
dropped rather than reworded a second time. 39 frames is no loss. Rewording again would only
be searching for phrasing that hides a disagreement the sentence genuinely contains.

**The methodological point, which is the durable one.** Three passes of a *single* rater would
have returned 3/3 unanimity here and shipped a contested frame as validated, whichever model
had been chosen — Opus and Gemini would have said "clearly bad", GPT would have said "clearly
good", and each would have looked perfectly stable. It took *disagreeing raters* to surface
it. The multi-model design was adopted for robustness against one model's quirks; its real
value turns out to be detecting moral contestedness, which no amount of repeated sampling
from one rater can see. Worth stating in the write-up: unanimity across three passes of one
model measures a model's *consistency*, not a statement's *stability*, and the two come apart
precisely on the items that matter most.

Sharpens the caveat from the negation entry above, which noted unanimity cannot see when a
set's two poles are different constructs. Both are the same lesson: unanimity is a floor, not
a proof, and it goes blind in different ways depending on what is being asked of it.
