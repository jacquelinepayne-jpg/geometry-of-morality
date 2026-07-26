# Assumptions

## A. Dataset Construction & Labeling

1. Moral minimal pairs can carry a stable, widely agreed-upon good/bad label, the way
   "Paris is in France" has a stable truth value, and I can separate uncontroversial
   statements from culturally or politically contested ones.
2. I can build several small datasets, one per moral topic (e.g. honesty, care/harm,
   fairness, loyalty), each with its own template, matched in structure and length
   within pairs and roughly matched in size across topics. I will build one topic first
   and expand only if it shows a probe-able direction.
3. LLM-assisted generation plus manual review can produce a few hundred clean statements
   per topic whose label is not readable off a single loaded word or the sentence's
   final words.
4. Negation works for moral claims. "X is not morally wrong" is a valid label-flipped
   counterpart to "X is morally wrong," even though "not wrong" may only mean
   "permissible."
5. Enough natural-sounding pleasant-but-immoral and unpleasant-but-moral statements
   exist to build a sentiment-anticorrelated evaluation set, and off-the-shelf sentiment
   classifiers can verify the decorrelation.

## B. Representational (does a "morality direction" exist at all?)

1. A linear probe will find a direction separating good from bad statements within each
   moral topic.
2. Each direction reflects the moral content, not the shared template structure the
   pairs are built from.
3. Each direction is specific to morality, not generic sentiment. Two checks are assumed
   jointly sufficient to rule this out: accuracy on the sentiment-anticorrelated set,
   and low similarity to an independently trained sentiment direction.
4. Probes will transfer across some moral topics but not necessarily all, and the
   transfer pattern (no shared direction, one unified direction, or clusters) is itself
   the finding.
5. The last token is the right place to read out moral valence. I will validate this
   with a small token-position pilot rather than inherit it.
6. Some layer linearly encodes moral valence. The best layer for truth need not be the
   best layer for morality, so I will run my own layer sweep.
7. A probe cannot distinguish "the model judges X wrong" from "the model knows people
   call X wrong." The finding is worth reporting either way, but the write-up must not
   conflate the two.

## C. Causal Validation

1. The directions I find are causally implicated, not just correlational. Adding or
   subtracting them during a forward pass should shift the model's judgment of vice and
   virtue statements on out-of-distribution data. If multiple directions emerge, each is
   tested separately.
2. The original intervention metric transfers with one adaptation, replacing
   P(TRUE) − P(FALSE) with a moral analog such as P(good) − P(bad). This assumes a
   prompt exists that elicits clean one-token moral judgments from the model.
3. The layer range used for truth interventions is a reasonable starting point but may
   need its own tuning.

## D. Reporting & Scientific Integrity

1. If the causal step is weak, a rigorous representational-only finding is still worth
   releasing, framed as a claim about representation rather than controllability.
2. Whatever transfer pattern appears, I will report it as the finding rather than
   treating some outcomes as real results and others as failures to re-run until they
   look cleaner.

## E. Practical / Resourcing

1. Rented cloud GPUs are fast and affordable enough for these experiments, including
   storage for cached activations, not just GPU time.
2. I have or can get access to the LLaMA-2 weights, and the required libraries install
   and run on current infrastructure.

## F. Existing Work & Positioning

1. The original Geometry of Truth result is robust and will reproduce on my setup,
   verified before any moral experiments run so pipeline bugs can't masquerade as
   findings.
2. Known critiques of truth-probing apply to the methodology I inherit. I assume my
   matched controls, sentiment checks, and causal step address the same objections for
   morality.
3. Existing moral-probing and steering-vector work does not already answer this specific
   question. A literature pass will confirm novelty before release.
