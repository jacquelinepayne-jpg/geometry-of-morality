"""
Generate the fairness_cheating dataset: matched good/bad minimal pairs in the
fairness/cheating domain.

Each frame states the governing rule, agreement, or entitlement explicitly (the
fairness analog of the care pilot's "the need must be explicit" lesson), so the
good act visibly follows it and the bad act visibly subverts it for the agent's
own or a favorite's benefit. Vocabulary sticks to allocation and rules (shares,
turns, scores, queues, prizes) and avoids the injury language of care_harm and
the assertion language of honesty_deception. See moral_common for shared design
rules and output format.

Negation is deferred for this dataset and the frames no longer carry neg_good/
neg_bad, so build() is called with write_negations=False and no
neg_fairness_cheating.csv is written. Verifying a negated set costs its own rating
round, and only care_harm's has been through one; whether the negated sets earn
that cost is a question the care_harm probe result answers first. The 34 of 40
frames that had opted in — and the reasons the other 6 declined — are preserved
verbatim in documentation/deferred_negations/fairness_cheating.data_gen.py, which
is the copy to restore from rather than rewriting the pairs by hand.

Usage: python datasets/morality/fairness_cheating/data_gen.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moral_common import Frame, build

FRAMES = [
    Frame("When the Saturday tips were counted at closing, {name} split the jar evenly among the whole waitstaff.",
          "When the Saturday tips were counted at closing, {name} pocketed half the jar while splitting it among the whole waitstaff.",
          'n'),

    Frame("After the group project earned the top grade, {name} listed every member's name on the final report.",
          "After the group project earned the top grade, {name} erased the other members' names from the final report.",
          'n'),

    Frame("Whenever the family board game got close, {name} counted out every move exactly as the dice were rolled.",
          "Whenever the family board game got close, {name} slid the pieces two extra spaces after the dice were rolled.",
          'n'),

    Frame("When a leaked answer key circulated before the chemistry final, {name} deleted the file and studied the textbook instead.",
          "When a leaked answer key circulated before the chemistry final, {name} memorized the file and skipped the textbook instead.",
          'n'),  # no negation: declining to delete a file someone else sent is not blameworthy

    Frame("Halfway through the charity 10k, {name} followed every marked turn all the way around the official course.",
          "Halfway through the charity 10k, {name} cut through the park to skip a mile of the official course.",
          'n'),

    Frame("At the crowded pharmacy counter, {name} took a ticket and waited like everyone else in the line.",
          "At the crowded pharmacy counter, {name} shoved forward and cut ahead of everyone else in the line.",
          'n'),

    Frame("When the director praised the ad campaign, {name} credited the concept sketches to the junior designer.",
          "When the director praised the ad campaign, {name} claimed the concept sketches and never mentioned the junior designer.",
          'n'),

    Frame("While drawing the raffle winner at the school fundraiser, {name} shook the box and drew blind from the folded names.",
          "While drawing the raffle winner at the school fundraiser, {name} fished out a cousin's ticket from the folded names.",
          'n'),

    Frame("When the apartment chore chart came around again, {name} scrubbed the bathroom on the assigned week.",
          "When the apartment chore chart came around again, {name} swapped the names to dodge the assigned week.",
          'n'),

    Frame("At the end of the carpool month, {name} paid an equal share of the fuel receipts.",
          "At the end of the carpool month, {name} dodged paying any share of the fuel receipts.",
          'n'),

    Frame("Keeping score at the bowling league, {name} entered all frames exactly as they were bowled.",
          "Keeping score at the bowling league, {name} shaved pins off rivals' frames as they were bowled.",
          'n'),

    Frame("When the pizzas arrived for the study group, {name} took two slices and left an even share for the others.",
          "When the pizzas arrived for the study group, {name} grabbed six slices and left only crusts for the others.",
          'n'),  # no negation: refusing your own fair share of a shared pizza wrongs nobody

    Frame("After losing the coin toss for the last parking spot, {name} backed away and left the space for the winning driver.",
          "After losing the coin toss for the last parking spot, {name} swerved in and took the space from the winning driver.",
          'n'),

    Frame("When the reserved study rooms opened for finals week, {name} signed up for one slot on the waiting list.",
          "When the reserved study rooms opened for finals week, {name} scribbled fake names to hog slots on the waiting list.",
          'n'),  # no negation: declining to book a study room is not blameworthy

    Frame("In the friendly poker night, {name} played every hand straight as the cards were dealt.",
          "In the friendly poker night, {name} slid an ace into a sleeve as the cards were dealt.",
          'n'),

    Frame("On the club scorecard after the round, {name} wrote down every stroke taken on the back nine.",
          "On the club scorecard after the round, {name} scratched three strokes off the total for the back nine.",
          'n'),

    Frame("Sharing the console with the younger cousins, {name} handed over the controller when each timer ran out.",
          "Sharing the console with the younger cousins, {name} clung to the controller long after each timer ran out.",
          'n'),

    Frame("When the community garden assigned the sunny plots by lottery, {name} planted in the plot the drawing allowed.",
          "When the community garden assigned the sunny plots by lottery, {name} moved the stakes to widen the plot the drawing allowed.",
          'n'),  # no negation: leaving your own lottery plot unplanted harms nobody

    Frame("When the dinner bill came for the table of six, {name} chipped in a full share of the total.",
          "When the dinner bill came for the table of six, {name} dodged the check and skipped a share of the total.",
          'n'),

    Frame("When pairing the chess tournament brackets, {name} matched the players strictly by their posted ratings.",
          "When pairing the chess tournament brackets, {name} handed friends easy draws regardless of the posted ratings.",
          'n'),

    Frame("Playing banker in the marathon Monopoly game, {name} paid every salary straight from the bank.",
          "Playing banker in the marathon Monopoly game, {name} snuck extra hundreds out of the bank.",
          'n'),

    Frame("Opening the shared office fridge at noon, {name} took only the lunch bag labeled with the matching initials.",
          "Opening the shared office fridge at noon, {name} took a coworker's lunch bag ignoring the matching initials.",
          'n'),  # no negation: refusing to take your own labeled lunch is not blameworthy

    Frame("Working the till at the charity bake sale, {name} counted back exact change for every purchase.",
          "Working the till at the charity bake sale, {name} shortchanged the busiest customers on every purchase.",
          'n'),

    Frame("Counting the ballots for the club presidency, {name} recorded each vote as marked in the final tally.",
          "Counting the ballots for the club presidency, {name} slipped a rival's ballots out of the final tally.",
          'n'),

    Frame("Dividing the arcade tokens between the twins, {name} counted out matching stacks for each child.",
          "Dividing the arcade tokens between the twins, {name} palmed a fistful while counting out stacks for each child.",
          'n'),

    Frame("Selling tomatoes at the farmers market stall, {name} weighed every bag on the certified scale.",
          "Selling tomatoes at the farmers market stall, {name} propped a hidden weight on the certified scale.",
          'n'),

    Frame("When the holiday schedule went up in the stockroom, {name} volunteered for an even rotation of the overnight shifts.",
          "When the holiday schedule went up in the stockroom, {name} shuffled a new hire into all of the overnight shifts.",
          'n'),  # no negation: the good act is volunteering, supererogatory by construction

    Frame("Collecting money for the retirement gift, {name} spent every dollar on the engraved watch.",
          "Collecting money for the retirement gift, {name} skimmed twenty dollars while buying the engraved watch.",
          'n'),

    Frame("Grading the midterm essays as a teaching assistant, {name} marked every essay against the posted rubric.",
          "Grading the midterm essays as a teaching assistant, {name} marked friends a grade above the posted rubric.",
          'n'),

    Frame("Out on the lake before dawn, {name} stopped casting after reaching the posted catch limit.",
          "Out on the lake before dawn, {name} stuffed a cooler far beyond the posted catch limit.",
          'n'),

    Frame("Driving tourists in from the airport, {name} took the direct highway and charged the metered fare.",
          "Driving tourists in from the airport, {name} circled extra miles to inflate the metered fare.",
          'n'),

    Frame("When the lemonade stand closed for the day, {name} counted the coins into even piles for the partner.",
          "When the lemonade stand closed for the day, {name} pocketed the coins and counted out pennies for the partner.",
          'n'),

    Frame("With no umpire at the club tennis final, {name} called the close serves exactly where they landed.",
          "With no umpire at the club tennis final, {name} called winning serves out no matter where they landed.",
          'n'),

    Frame("When overtime hours were handed out on the dock, {name} rotated the extra shifts straight down the crew list.",
          "When overtime hours were handed out on the dock, {name} handed the extra shifts to favorites ignoring the crew list.",
          'n'),

    Frame("Running the ring toss at the summer carnival, {name} paid out a prize for each of the winning throws.",
          "Running the ring toss at the summer carnival, {name} pocketed the tickets and denied all of the winning throws.",
          'n'),

    Frame("When the tenants moved out of the spotless apartment, {name} returned every cent of the deposit.",
          "When the tenants moved out of the spotless apartment, {name} invented scratches to withhold the deposit.",
          'n'),

    Frame("After the group agreed to divide the firewood they had hauled, {name} stacked equal bundles beside each cabin door.",
          "After the group agreed to divide the firewood they had hauled, {name} dragged the biggest bundles behind one cabin door.",
          'n'),

    Frame("Updating the classroom sticker chart on Friday, {name} gave each pupil the stars their reading had earned.",
          "Updating the classroom sticker chart on Friday, {name} gave a favorite pupil stars no reading had earned.",
          'n'),

    Frame("Splitting the quarterly bonus pool among the sales team, {name} divided it by the deals each member had closed.",
          "Splitting the quarterly bonus pool among the sales team, {name} handed a golf buddy shares that other members had closed.",
          'n'),

    Frame("Hosting the trivia night championship, {name} read each team the questions from the same sealed pack.",
          "Hosting the trivia night championship, {name} slipped one team the questions from the same sealed pack.",
          'n'),
]

if __name__ == '__main__':
    build('fairness_cheating', FRAMES, write_negations=False)
