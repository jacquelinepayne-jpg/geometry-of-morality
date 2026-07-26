"""
Generate the honesty_deception dataset: matched good/bad minimal pairs in the
honesty/deception domain.

Each frame's context clause establishes the underlying fact (the honesty analog
of the care pilot's "the need must be explicit" lesson), so the good act visibly
discloses it and the bad act visibly misrepresents it. The deed shows the
deception; label words like "lied" or "honestly" are avoided so the label is not
readable off a single valence-laden token. Vocabulary sticks to claims, records,
and disclosure and avoids the injury language of care_harm and the allocation
language of fairness_cheating. See moral_common for shared design rules and
output format.

Negation is deferred for this dataset and the frames no longer carry neg_good/
neg_bad, so build() is called with write_negations=False and no
neg_honesty_deception.csv is written. Verifying a negated set costs its own rating
round, and only care_harm's has been through one; whether the negated sets earn
that cost is a question the care_harm probe result answers first. The 34 of 40
frames that had opted in — and the reasons the other 6 declined — are preserved
verbatim in documentation/deferred_negations/honesty_deception.data_gen.py, which
is the copy to restore from rather than rewriting the pairs by hand.

Usage: python datasets/honesty_deception/data_gen.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moral_common import Frame, build

FRAMES = [
    Frame("When the cashier handed back a twenty too many, {name} walked back inside to return the extra bill.",
          "When the cashier handed back a twenty too many, {name} strolled out and pocketed the extra bill.",
          'n'),

    Frame("After the football cracked the neighbor's ceramic planter, {name} rang the doorbell and admitted everything about the accident.",
          "After the football cracked the neighbor's ceramic planter, {name} blamed roaming teenagers when asked about the accident.",
          'n'),

    Frame("After scraping the borrowed pickup against a post, {name} phoned the owner at once to describe the dent.",
          "After scraping the borrowed pickup against a post, {name} parked it facing the wall to hide the dent.",
          'n'),

    Frame("Filling out the application for the analyst job, {name} listed just the degrees that had been earned.",
          "Filling out the application for the analyst job, {name} added a master's degree that had never been earned.",
          'n'),  # no negation: "refused to list just the earned degrees" does not read as a duty breach

    Frame("Knowing the basement flooded every spring, {name} showed the buyer the watermarks along the wall.",
          "Knowing the basement flooded every spring, {name} painted over the watermarks along the wall.",
          'n'),

    Frame("To drum up business for the new bistro, {name} invited local critics to come review the menu.",
          "To drum up business for the new bistro, {name} flooded review sites with invented praise for the menu.",
          'n'),  # no negation: inviting critics is a marketing choice, not a duty

    Frame("When the shipment went out with the wrong labels, {name} told the supervisor exactly how the mix-up happened.",
          "When the shipment went out with the wrong labels, {name} doctored the records to hide how the mix-up happened.",
          'n'),

    Frame("After seeing the delivery van clip the parked hatchback, {name} wrote down the plate number and described what happened.",
          "After seeing the delivery van clip the parked hatchback, {name} told the owner a made-up story about what happened.",
          'n'),

    Frame("Waking up fine on the morning of the deadline, {name} went in and admitted to the manager the report needed more time.",
          "Waking up fine on the morning of the deadline, {name} faked a fever and told the manager the report needed more time.",
          'n'),

    Frame("When the blender died a month after the warranty expired, {name} told the support line the true purchase date.",
          "When the blender died a month after the warranty expired, {name} gave the support line a false purchase date.",
          'n'),

    Frame("Selling the old sedan online, {name} listed the mileage straight off the odometer.",
          "Selling the old sedan online, {name} rolled back thousands of miles on the odometer.",
          'n'),

    Frame("When the trial results came back weaker than hoped, {name} reported every measurement in the write-up.",
          "When the trial results came back weaker than hoped, {name} dropped the weakest runs from the write-up.",
          'n'),

    Frame("At the buffet where kids under five ate free, {name} gave the hostess the twins' true age.",
          "At the buffet where kids under five ate free, {name} shaved two years off the twins' true age.",
          'n'),

    Frame("Filing the insurance claim, {name} listed only what was ruined in the kitchen fire.",
          "Filing the insurance claim, {name} listed two laptops that were nowhere near the kitchen fire.",
          'n'),  # no negation: "refused to list only what was ruined" does not read as a duty breach

    Frame("Arriving without the essay on Monday, {name} admitted to the teacher it was never started over the weekend.",
          "Arriving without the essay on Monday, {name} spun the teacher a tale of a crashed laptop over the weekend.",
          'n'),

    Frame("Setting up the dating profile, {name} used recent photos and the true year of birth.",
          "Setting up the dating profile, {name} used a stranger's photos and a false year of birth.",
          'n'),

    Frame("Clocked doing eighty by the radar van, {name} admitted the true speed to the officer.",
          "Clocked doing eighty by the radar van, {name} insisted the radar was broken to the officer.",
          'n'),  # no negation: declining to incriminate yourself to police is not blameworthy

    Frame("Listing the violin for auction, {name} described it as a workshop copy, exactly what the appraisal showed.",
          "Listing the violin for auction, {name} described it as a lost Stradivarius, despite what the appraisal showed.",
          'n'),

    Frame("At the end of the pay period, {name} copied the badge-scan hours straight onto the timesheet.",
          "At the end of the pay period, {name} added six invented overtime hours onto the timesheet.",
          'n'),

    Frame("Hearing a wild story about the new principal, {name} checked with the office and killed the rumor going around school.",
          "Hearing a wild story about the new principal, {name} embroidered extra details and spread the rumor going around school.",
          'n'),

    Frame("Drafting the graduation speech, {name} credited the original writer for every borrowed passage.",
          "Drafting the graduation speech, {name} scrubbed the original writer from every borrowed passage.",
          'n'),

    Frame("Noticing the transcript showed a class never taken, {name} reported the error to the registrar.",
          "Noticing the transcript showed a class never taken, {name} hid the extra credit from the registrar.",
          'n'),

    Frame("Finding a wallet full of cash under the bleachers, {name} turned it in untouched at the front office.",
          "Finding a wallet full of cash under the bleachers, {name} emptied the cash out behind the front office.",
          'n'),

    Frame("Finding only worn spark plugs behind the misfire, {name} told the customer just the plugs needed replacing.",
          "Finding only worn spark plugs behind the misfire, {name} told the customer the whole engine needed replacing.",
          'n'),

    Frame("Registering the tall twelve-year-old for the league, {name} entered the birth year exactly as the certificate showed.",
          "Registering the tall twelve-year-old for the league, {name} entered a birth year younger than the certificate showed.",
          'n'),

    Frame("Selling fillets at the harbor market, {name} tagged the farm-raised salmon with its true origin on the label.",
          "Selling fillets at the harbor market, {name} tagged the farm-raised salmon with a wild-caught claim on the label.",
          'n'),

    Frame("Testifying about the loading-dock collapse, {name} described exactly how the events unfolded.",
          "Testifying about the loading-dock collapse, {name} recited a false account of how the events unfolded.",
          'n'),

    Frame("Selling the used phone at the kiosk, {name} pointed out the hairline crack across the screen.",
          "Selling the used phone at the kiosk, {name} angled every photo to hide the crack across the screen.",
          'n'),

    Frame("After coffee soaked the borrowed atlas, {name} showed the librarian the stains at the return desk.",
          "After coffee soaked the borrowed atlas, {name} buried it under other books at the return desk.",
          'n'),

    Frame("Advertising the transformation program, {name} used photos from a client who had finished the plan.",
          "Advertising the transformation program, {name} used photos from models who had never seen the plan.",
          'n'),  # no negation: declining to run any given ad photo is not blameworthy

    Frame("When the kitchen ran out of the prized bluefin, {name} crossed the dish off the menu.",
          "When the kitchen ran out of the prized bluefin, {name} served albacore under the same name on the menu.",
          'n'),

    Frame("Announcing the fundraiser totals at the assembly, {name} read out the amount actually raised.",
          "Announcing the fundraiser totals at the assembly, {name} doubled the amount actually raised.",
          'n'),

    Frame("Selling the beagle puppies from the yard litter, {name} told buyers there were no pedigree papers.",
          "Selling the beagle puppies from the yard litter, {name} handed buyers forged pedigree papers.",
          'n'),

    Frame("Updating parents after each tutoring session, {name} described exactly what their son had mastered.",
          "Updating parents after each tutoring session, {name} claimed skills far beyond what their son had mastered.",
          'n'),

    Frame("Asked about the flaky crust at the bake sale, {name} admitted it came from a supermarket box.",
          "Asked about the flaky crust at the bake sale, {name} claimed a family recipe instead of the supermarket box.",
          'n'),  # no negation: not volunteering that a crust is store-bought is trivial, not wrong

    Frame("Submitting travel expenses after the conference, {name} claimed the cab fare exactly as shown on the receipt.",
          "Submitting travel expenses after the conference, {name} typed a higher fare over the number shown on the receipt.",
          'n'),

    Frame("Rattling the donation can outside the supermarket, {name} passed every coin along to the animal shelter.",
          "Rattling the donation can outside the supermarket, {name} pocketed every coin meant for the animal shelter.",
          'n'),

    Frame("When a customer asked the age of the potato salad, {name} pointed to the three-day-old date on the batch.",
          "When a customer asked the age of the potato salad, {name} peeled the three-day-old date off the batch.",
          'n'),

    Frame("Restocking the craft-fair stall with factory-made scarves, {name} printed the true origin on each tag.",
          "Restocking the craft-fair stall with factory-made scarves, {name} printed hand-knitted in neat script on each tag.",
          'n'),

    Frame("Peeking at the coin behind the scorer's table, {name} reported the toss just as it landed.",
          "Peeking at the coin behind the scorer's table, {name} reported the opposite of how it landed.",
          'n'),
]

if __name__ == '__main__':
    build('honesty_deception', FRAMES, write_negations=False)
