# First Cheap Test

## What I will test first and why

I will test assumption A1, that moral minimal pairs can carry a stable, widely
agreed-upon good/bad label. This is the crux by both criteria. It is high-stakes
because every later stage assumes stable labels exist. The probing, transfer, and
intervention experiments all train on those labels, so if they are not stable the
analogy to truth breaks at the root. It is also genuinely uncertain, since "widely
agreed upon" is doing a lot of work in a domain famous for disagreement. Other
assumptions are either high-stakes but expensive to test (a probe finding a direction
requires the full pipeline) or cheap to test but ones I am already confident in
(compute and model access). A1 is the only assumption that is both cheap to test now
and fatal if wrong.

## What will I be testing?

Whether independent raters agree on the moral label of my statements. I will draft 40
care/harm statements (20 matched good/bad pairs) using the planned template, then
collect blind labels from three or four LLMs from different model families plus myself.
Each rater marks a statement good, bad, or depends-on-context. I will measure the rate
of unanimous agreement and the rate of "depends" flags. As a free add-on, I will run a
sentiment classifier over the same statements to preview how strongly sentiment
correlates with the moral label. Estimated cost is one to two hours and no GPU.

## What do I expect the results to be, and why?

I expect 90% or more of the statements to receive unanimous labels, with failures
concentrated in statements that smuggle in unstated context, such as punishment, white
lies, or competing obligations. I expect this because the statements are deliberately
chosen to be uncontroversial, and existing moral-judgment datasets report high
annotator agreement on clear-cut cases. I also expect sentiment to correlate strongly
with the moral label in this first batch. That result is expected and motivates the
planned sentiment-anticorrelated control set rather than arguing against the project.

## What result would make me change course?

Below roughly 80% unanimous agreement, after one round of revising the worst
statements, the assumption is falsified at the scale I need. I would then either narrow
to a single ultra-clear subdomain, such as gratuitous harm versus costly helping, or
reframe the project around the contested/uncontested boundary itself. Agreement between
80% and 95% means the assumption survives with adjustment, and I would tighten the
template and add a screening pass for context-dependence to the dataset pipeline. A
"depends" rate above roughly 30%, even with high agreement elsewhere, would also force
a template redesign, since it would mean my statements lean on unstated context.

## What the results were

A1 is validated. On the revised sheet, 39 of 40 statements drew unanimous labels
(97.5%), and every unanimous label matched the one I intended. Across seven rater
passes, myself plus two runs each of GPT-5.5 mini, Claude Opus 5, and Gemini 3.1 Pro,
mean pairwise agreement was 99.3% and the lowest pairwise agreement was 97.5%. Only
one of 280 individual judgments was "depends", a rate of 0.36% against the 30% level
that would have forced a template redesign. That clears the 90% I predicted and sits
above the 95% band where the assumption would have survived only with adjustment.

This is the post-revision number and should be read as such. The first round used
"kickball game" for the wheelchair pair and "when the fire alarm sounded" for the
classroom pair, and I flagged both as depends myself. Rewording them to "card game"
and "as smoke filled the hallway" removed the confounds: the wheelchair pair is now
unanimous. My pre-registration allows one revision round, and the budget is now spent,
so 97.5% stands as the final figure. The single holdout is the revised fire statement,
where one Opus 5 run read locking students away from smoke as shelter-in-place rather
than entrapment. That is a fair reading, and the item should be replaced when the full
dataset is generated, since harm through confinement invites a protective reading
unless the intent is unmistakable.

The sentiment add-on confirmed the correlation I predicted but not the mechanism.
VADER, the lexicon scorer, was the weaker instrument and ordered only 13 of 20 pairs
correctly, indistinguishable from chance. A sentence-level classifier
(cardiffnlp/twitter-roberta-base-sentiment-latest) correlated with the label at
r = 0.655, ordered all 20 of 20 pairs correctly, and separated good from bad at
AUC 0.887. Since a linear probe carries a bias term, AUC rather than raw sign accuracy
is the quantity that describes what a probe could exploit, and 0.887 is high enough
that the confound has to be treated as real.

The shape of that correlation matters more than its size. The classifier assigns good
statements a mean signed score of -0.025 and bad statements -0.364. Good acts are not
scored positive; they are scored null. The highest P(positive) anywhere in the dataset
is 0.20, and mean P(neutral) is 0.695. What the classifier detects is therefore not
moral valence but harm language, present on the bad side and absent on the good side.

## What this changes

The planned control set needs a sharper specification than "sentiment-anticorrelated".
An axis with only one active direction cannot be anticorrelated against symmetrically.
Two distinct control types are needed instead: kind acts described in harm-loaded
language, which the current pairs already approximate at their narrowest sentiment
gaps (Priya and the scraped knee, Leo and the injured bird, Omar and the sleeping bag,
all separated by less than 0.09), and harmful acts described in flat or bureaucratic
language, of which this batch contains none. The second type is the one that would
actually break a harm detector, so it is the priority for B3.

The one-sided confound also suggests a cheaper falsification test than a full control
set. If the probe direction is harm detection rather than moral valence, it should fail
to separate good statements from morally neutral ones, since both sit at zero on the
sentiment axis. That test needs no new labels and is more diagnostic than a generic
sentiment control, so it should run before the anticorrelated set is built.

Finally, A1 and A5 are not independent, and my pre-registration treated them as if they
were. The same choice that produced 97.5% agreement, writing deliberately
uncontroversial statements, is what loads them with affect: unambiguous harm is
described with words like screams, shredded, stomped, and hurled. Flattening the
language to weaken the sentiment confound should be expected to push agreement down
toward the 80-95% band. If that happens in a later round it will not mean the template
failed, and I should not treat it as evidence against A1 without checking whether the
confound fix caused it.
