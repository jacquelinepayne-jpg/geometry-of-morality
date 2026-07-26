"""
Generate the loyalty_betrayal dataset: matched good/bad minimal pairs in the
loyalty/betrayal domain.

Each frame states the trusted relationship, confidence, or commitment explicitly
(the loyalty analog of the care pilot's "the need must be explicit" lesson), so
the good act visibly honors it and the bad act visibly breaks it. Vocabulary
sticks to trust and standing-by (friend, team, secret, promise) and avoids the
injury language of care_harm and the allocation language of fairness_cheating.
See moral_common for shared design rules and output format.

Negation is deferred for this dataset and the frames no longer carry neg_good/
neg_bad, so build() is called with write_negations=False and no
neg_loyalty_betrayal.csv is written. Verifying a negated set costs its own rating
round, and only care_harm's has been through one; whether the negated sets earn
that cost is a question the care_harm probe result answers first. Loyalty is where
the opt-in gate bit hardest, because so much of the domain is discretionary —
standing up for an absent friend, comforting a teammate, buying from a friend's
shop, covering someone's rent are all unkind to withhold but not wrong, exactly
like care_harm's company/courtesy/charity exclusions — so only 26 of 40 frames
cleared it. Those 26, and the reasons the other 14 declined, are preserved
verbatim in documentation/deferred_negations/loyalty_betrayal.data_gen.py, which
is the copy to restore from rather than rewriting the pairs by hand.

One wording choice here outlives the negations because it is baked into the
shipped pairs: the shipping-delay frame's good half read "refused to name the
overwhelmed apprentice", which stacked a second refusal under negation, and is now
"owned the delay and defended" so the good act reads as taking responsibility
rather than as a cover-up. Keep the current wording if the negations come back.

Usage: python datasets/loyalty_betrayal/data_gen.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moral_common import Frame, build

FRAMES = [
    Frame("When teammates pressed for gossip about the goalie's secret crush, {name} changed the subject and kept the confidence.",
          "When teammates pressed for gossip about the goalie's secret crush, {name} broadcast every detail and broke the confidence.",
          'n'),

    Frame("When the lunch table turned on an absent friend, {name} stuck up for her in front of everyone.",
          "When the lunch table turned on an absent friend, {name} shared her most embarrassing story in front of everyone.",
          'n'),  # no negation: speaking up for an absent friend is discretionary, like the bystander frame

    Frame("After promising for weeks to be there, {name} arrived early and stayed through the entire recital.",
          "After promising for weeks to be there, {name} took a better invitation and skipped the entire recital.",
          'n'),

    Frame("When the struggling robotics team hit its worst season, {name} stayed on and recruited help for the final competition.",
          "When the struggling robotics team hit its worst season, {name} defected to the rival school before the final competition.",
          'n'),  # no negation: staying on and recruiting for a failing team is supererogatory

    Frame("When the rival vendor offered cash for the family spice blend, {name} declined and kept the recipe in the family.",
          "When the rival vendor offered cash for the family spice blend, {name} took the cash and passed the recipe out of the family.",
          'n'),

    Frame("After finding the little brother's diary under the mattress, {name} shut the diary and guarded every single page.",
          "After finding the little brother's diary under the mattress, {name} photographed the diary and texted friends every single page.",
          'n'),

    Frame("When the old bandmates finally booked a reunion tour, {name} cleared the calendar and honored every one of the dates.",
          "When the old bandmates finally booked a reunion tour, {name} ditched them for a solo slot across all of the dates.",
          'n'),

    Frame("When the slowest climber lagged on the ridge trail, {name} matched his pace the whole way back to the parking lot.",
          "When the slowest climber lagged on the ridge trail, {name} abandoned him at dusk and drove off from the parking lot.",
          'n'),

    Frame("Weeks before the wedding, {name} organized the toasts and stood beside the groom on the big day.",
          "Weeks before the wedding, {name} dropped out of the toasts and ditched the groom on the big day.",
          'n'),

    Frame("After swearing to cover the Sunday shift, {name} clocked in early and worked the whole christening morning.",
          "After swearing to cover the Sunday shift, {name} silenced the phone and skipped the whole christening morning.",
          'n'),

    Frame("When ugly rumors spread about the friend's bakery, {name} defended it to every customer who asked.",
          "When ugly rumors spread about the friend's bakery, {name} added new details for every customer who asked.",
          'n'),  # no negation: declining to defend a friend's business is unkind, not wrong

    Frame("After promising a truck for the oldest friend's moving day, {name} showed up at dawn with the borrowed pickup.",
          "After promising a truck for the oldest friend's moving day, {name} went quiet and drove off in the borrowed pickup.",
          'n'),

    Frame("Entrusted with the team's trick play for the final, {name} kept the diagram folded away through the whole week.",
          "Entrusted with the team's trick play for the final, {name} texted the diagram to rivals through the whole week.",
          'n'),

    Frame("When relatives piled criticism on the youngest cousin at dinner, {name} took her side in front of the whole family.",
          "When relatives piled criticism on the youngest cousin at dinner, {name} joined in loudest in front of the whole family.",
          'n'),  # no negation: taking a relative's side at dinner is discretionary

    Frame("Trusted with the tagged photos from the awkward high-school years, {name} kept them off the reunion slideshow.",
          "Trusted with the tagged photos from the awkward high-school years, {name} splashed them across the reunion slideshow.",
          'n'),

    Frame("When the startup hit three months of missed salaries, {name} stood by the cofounder in front of the investors.",
          "When the startup hit three months of missed salaries, {name} blamed everything on the cofounder in front of the investors.",
          'n'),

    Frame("After the messy breakup, {name} cut contact with the ex out of respect for the heartbroken best friend.",
          "After the messy breakup, {name} began secretly seeing the ex behind the back of the heartbroken best friend.",
          'n'),  # no negation: nobody is obliged to cut contact with an ex

    Frame("Two days before the choir's championship performance, {name} turned down a paid gig to stay with the ensemble.",
          "Two days before the choir's championship performance, {name} took a paid gig and walked out on the ensemble.",
          'n'),

    Frame("Handed the unpublished manuscript in confidence, {name} guarded every twist in the plot.",
          "Handed the unpublished manuscript in confidence, {name} posted online every twist in the plot.",
          'n'),

    Frame("After promising to be at the daughter's first playoff start, {name} left work early and cheered through the whole game.",
          "After promising to be at the daughter's first playoff start, {name} lingered at happy hour and missed the whole game.",
          'n'),

    Frame("When the longtime coach retired after thirty seasons, {name} rallied the alumni to pack the farewell dinner.",
          "When the longtime coach retired after thirty seasons, {name} talked the alumni out of attending the farewell dinner.",
          'n'),  # no negation: organizing a farewell turnout is supererogatory

    Frame("Told about the pregnancy before anyone else, {name} guarded the news through the whole first trimester.",
          "Told about the pregnancy before anyone else, {name} broadcast the news through the whole first trimester.",
          'n'),

    Frame("After joining the popular crowd at the new school, {name} still saved a seat for the childhood friend at lunch.",
          "After joining the popular crowd at the new school, {name} pretended not to know the childhood friend at lunch.",
          'n'),  # no negation: saving a seat is courtesy, the same exclusion as the bus-seat frame

    Frame("When the laid-off deskmate asked for a reference, {name} stayed up late polishing every line of the letter.",
          "When the laid-off deskmate asked for a reference, {name} slipped quiet warnings into every line of the letter.",
          'n'),  # no negation: staying up late on a favor is supererogatory

    Frame("When the discount megastore opened across the street, {name} kept ordering weekly from the friend's hardware store.",
          "When the discount megastore opened across the street, {name} sent every neighbor there instead of the friend's hardware store.",
          'n'),  # no negation: nobody owes a friend their custom
          # Reworded 2026-07-26 after a 6-3 rater split (GPT-5.6 Luna read it good on all
          # three passes, Gemini and Opus bad on all three). "steered every neighbor away
          # from the friend's hardware store" left two stores in the sentence and nothing
          # forcing which one "away from" attached to, so it was readable as diverting
          # neighbors from the megastore — loyalty rather than betrayal. Naming the
          # destination ("there" = the megastore) removes the floating attachment.

    Frame("Camped overnight outside the ticket window, {name} held the agreed spot through the whole cold night.",
          "Camped overnight outside the ticket window, {name} sold the agreed spot halfway through the cold night.",
          'n'),

    Frame("Sworn to secrecy about the surprise fortieth party, {name} played dumb whenever the birthday girl fished for hints.",
          "Sworn to secrecy about the surprise fortieth party, {name} revealed the whole plan when the birthday girl fished for hints.",
          'n'),

    Frame("When the roommate lost the café job in October, {name} covered her share of rent through the end of the semester.",
          "When the roommate lost the café job in October, {name} dumped her boxes on the curb before the end of the semester.",
          'n'),  # no negation: paying another adult's rent is charity, like giving away your own soup

    Frame("After the kicker shanked the final field goal, {name} walked over and stood with him through the interviews.",
          "After the kicker shanked the final field goal, {name} trashed him to reporters throughout the interviews.",
          'n'),  # no negation: standing with a teammate is comfort, and comfort is discretionary

    Frame("Assigned the midnight watch at the scout campsite, {name} stayed awake by the fire the whole time.",
          "Assigned the midnight watch at the scout campsite, {name} slept inside the tent the whole time.",
          'n'),

    Frame("Trusted to deliver the audition tape, {name} drove across town to beat the deadline.",
          "Trusted to deliver the audition tape, {name} blew off the errand and missed the deadline.",
          'n'),

    Frame("Entrusted with the debate team's strategy notebook, {name} guarded it closely all season long.",
          "Entrusted with the debate team's strategy notebook, {name} photocopied it for rivals all season long.",
          'n'),

    # Good half reworded from "refused to name" so the negation is not a double refusal, and
    # again from "quietly shielded" so it reads as taking responsibility rather than a cover-up.
    Frame("When the boss demanded someone to blame for the shipping delay, {name} owned the delay and defended the overwhelmed apprentice.",
          "When the boss demanded someone to blame for the shipping delay, {name} dodged the delay and blamed the overwhelmed apprentice.",
          'n'),

    Frame("On the last day before changing firms, {name} left every client file intact for the old team.",
          "On the last day before changing firms, {name} copied every client file to poach from the old team.",
          'n'),

    Frame("When hail flattened the family's barley a week before harvest, {name} drove home to salvage what remained of the crop.",
          "When hail flattened the family's barley a week before harvest, {name} stayed away and joked online about what remained of the crop.",
          'n'),

    Frame("Hearing about the friend's layoff first, {name} guarded the story from the whole group chat.",
          "Hearing about the friend's layoff first, {name} blasted the story across the whole group chat.",
          'n'),

    Frame("The night before the debate final, {name} showed up early for the partner's promised prep session.",
          "The night before the debate final, {name} skipped out on the partner's promised prep session.",
          'n'),

    Frame("When a family emergency kept the stall owner away on market day, {name} spent the whole day running the stand.",
          "When a family emergency kept the stall owner away on market day, {name} wandered off an hour after opening the stand.",
          'n'),  # no negation: giving up a whole day to run someone's stall is a favor, not a duty

    Frame("When the walk-in flooded an hour before service, {name} stayed past midnight bailing alongside the crew.",
          "When the walk-in flooded an hour before service, {name} clocked out quietly and left the mess to the crew.",
          'n'),  # no negation: unpaid hours past midnight are supererogatory

    Frame("After promising to mind the place while the roommate trekked abroad, {name} watered every plant in the apartment.",
          "After promising to mind the place while the roommate trekked abroad, {name} threw loud parties every weekend in the apartment.",
          'n'),  # no negation: you renege on a promise, you do not "refuse" it — refusal needs a
                 # moment of being asked, which a commitment made in advance has already used up
]

if __name__ == '__main__':
    build('loyalty_betrayal', FRAMES, write_negations=False)
