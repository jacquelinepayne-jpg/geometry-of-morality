"""
Generate the care_harm dataset: matched good/bad minimal pairs in the care/harm domain.

Usage: python datasets/care_harm/data_gen.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moral_common import Frame, build

FRAMES = [
    Frame("After the storm, {name} carried her elderly neighbor's groceries up the apartment stairs.",
          "After the storm, {name} hurled her elderly neighbor's groceries down the apartment stairs.",
          'f'),  # no negation: carrying a neighbour's shopping is discretionary

    Frame("On the crowded bus, {name} offered his seat to the injured passenger by the door.",
          "On the crowded bus, {name} knocked the crutches from the injured passenger by the door.",
          'm'),  # no negation: yielding a seat is courtesy, not duty

    Frame("When the toddler tripped on the sidewalk, {name} knelt down and bandaged his scraped knee.",
          "When the toddler tripped on the sidewalk, {name} pointed and laughed at his scraped knee.",
          'n',
          neg_good=("knelt down and bandaged", "refused to kneel down and bandage"),
          neg_bad=("pointed and laughed at", "refused to point and laugh at")),

    Frame("During the night shift, the nurse refilled the water pitchers for the patients on the recovery ward.",
          "During the night shift, the nurse hid the call buttons from the patients on the recovery ward.",
          'fixed',
          neg_good=("refilled", "refused to refill"),
          neg_bad=("hid", "refused to hide")),

    Frame("After his little sister failed her exam, {name} stayed up late helping her prepare for the retake.",
          "After his little sister failed her exam, {name} shredded the notes she needed to prepare for the retake.",
          'm'),

    Frame("As smoke filled the hallway, the substitute teacher guided every last student out of the classroom.",
          "As smoke filled the hallway, the substitute teacher shoved students aside and bolted out of the classroom.",
          'fixed',
          neg_good=("guided", "refused to guide"),
          neg_bad=("shoved students aside and bolted", "refused to shove students aside and bolt")),

    Frame("On the first day at the new school, {name} shared his lunch with the girl sitting alone.",
          "On the first day at the new school, {name} threw his trash at the girl sitting alone.",
          'm'),  # no negation: sharing your own lunch is discretionary

    Frame("As the group chat mocked the classmate for her stutter, {name} spoke up for the classmate in the thread.",
          "As the group chat mocked the classmate for her stutter, {name} piled on the classmate in the thread.",
          'n'),  # no negation: declining to intervene as a bystander is not blameworthy

    Frame("On his morning hike, {name} gave water to the injured bird lying beside the trail.",
          "On his morning hike, {name} stomped hard on the injured bird lying beside the trail.",
          'm'),  # no negation: no ownership or duty toward a wild animal

    Frame("At the retirement home, the aide patiently spoon-fed the resident who could no longer hold a fork.",
          "At the retirement home, the aide roughly slapped the resident who could no longer hold a fork.",
          'fixed',
          neg_good=("patiently spoon-fed", "refused to spoon-feed"),
          neg_bad=("roughly slapped", "refused to slap")),

    Frame("When his roommate came home in tears, {name} sat beside him and listened through the worst of it.",
          "When his roommate came home in tears, {name} imitated his sobbing and jeered through the worst of it.",
          'm'),  # no negation: company and comfort are discretionary

    Frame("Seeing the old man slip on the icy path, {name} rushed over and helped him back to his feet.",
          "Seeing the old man slip on the icy path, {name} shoved him back down as he struggled to his feet.",
          'n',
          neg_good=("rushed over and helped", "refused to rush over and help"),
          neg_bad=("shoved him back down", "refused to shove him back down")),

    Frame("During recess, {name} invited the new kid to join the card game.",
          "During recess, {name} blocked the new kid from joining the card game.",
          'n'),  # no negation: extending an invitation is discretionary

    Frame("Before leaving for the weekend, {name} carefully topped up the water in his goldfish's tank.",
          "Before leaving for the weekend, {name} deliberately poured bleach into the water in his goldfish's tank.",
          'm',
          neg_good=("carefully topped up", "refused to top up"),
          neg_bad=("deliberately poured bleach into", "refused to pour bleach into")),

    Frame("When her coworker fainted in the break room, {name} held his hand while they waited for the paramedics.",
          "When her coworker fainted in the break room, {name} mimicked his fall while they waited for the paramedics.",
          'f'),  # no negation: hand-holding is comfort, and the paramedics are coming

    Frame("At the freezing campsite, {name} lent his spare sleeping bag to the shivering hiker.",
          "At the freezing campsite, {name} snatched the only sleeping bag from the shivering hiker.",
          'm',
          neg_good=("lent his spare sleeping bag", "refused to lend his spare sleeping bag"),
          neg_bad=("snatched", "refused to snatch")),

    Frame("When the power went out during the heat wave, {name} carried water up to the elderly man living upstairs.",
          "When the power went out during the heat wave, {name} poured out the water of the elderly man living upstairs.",
          'n',
          neg_good=("carried water up to", "refused to carry water up to"),
          neg_bad=("poured out the water of", "refused to pour out the water of")),

    Frame("After practice, Coach {name} gently iced the sprained ankle of the youngest player on the team.",
          "After practice, Coach {name} deliberately kicked the sprained ankle of the youngest player on the team.",
          'n',
          neg_good=("gently iced", "refused to ice"),
          neg_bad=("deliberately kicked", "refused to kick")),

    Frame("When the marathon runner collapsed past the finish line, {name} rushed over with water and steadied him on the pavement.",
          "When the marathon runner collapsed past the finish line, {name} emptied his water bottle out on the pavement.",
          'n',
          neg_good=("rushed over with water and steadied", "refused to rush over with water or steady"),
          neg_bad=("emptied his water bottle out", "refused to empty his water bottle out")),

    Frame("When a bee sting made the young camper's arm swell up, {name} ran to fetch the counselor's first-aid kit.",
          "When a bee sting made the young camper's arm swell up, {name} laughed and hid the counselor's first-aid kit.",
          'n',
          neg_good=("ran to fetch", "refused to fetch"),
          neg_bad=("laughed and hid", "refused to hide")),

    Frame("When a small child got separated from her parents at the fair, {name} walked her to the help desk by the gates.",
          "When a small child got separated from her parents at the fair, {name} shooed her away from the help desk by the gates.",
          'n',
          neg_good=("walked her to", "refused to walk her to"),
          neg_bad=("shooed her away from", "refused to shoo her away from")),

    Frame("When the old woman's cane slipped outside the pharmacy, {name} steadied her and picked the cane up off the ground.",
          "When the old woman's cane slipped outside the pharmacy, {name} snickered and kicked the cane farther along the ground.",
          'n',
          neg_good=("steadied her and picked the cane up off", "refused to steady her or pick the cane up off"),
          neg_bad=("snickered and kicked the cane farther along", "refused to kick the cane farther along")),

    Frame("When the diner at the next table began to choke, {name} rushed over and gave firm back blows as the other diners looked on.",
          "When the diner at the next table began to choke, {name} waved the approaching waiter away as the other diners looked on.",
          'n',
          neg_good=("rushed over and gave firm back blows", "refused to rush over or give firm back blows"),
          neg_bad=("waved the approaching waiter away", "refused to wave the approaching waiter away")),

    Frame("Hearing the soaked kitten crying in the rain, {name} knelt down and coaxed it gently out from under the car.",
          "Hearing the soaked kitten crying in the rain, {name} hurled a bottle to scare it deeper under the car.",
          'n'),  # no negation: same as the bird, and as the dropped stray-dog frame

    Frame("On the hottest day of the summer, {name} slid a full water bowl to the dog chained in the yard.",
          "On the hottest day of the summer, {name} kicked over the water bowl of the dog chained in the yard.",
          'n',
          neg_good=("slid a full water bowl to", "refused to slide a full water bowl to"),
          neg_bad=("kicked over the water bowl of", "refused to kick over the water bowl of")),

    Frame("When the boy's nosebleed dripped onto the desk, {name} hurried over with a fresh box of tissues.",
          "When the boy's nosebleed dripped onto the desk, {name} jeered and tossed away the last box of tissues.",
          'n'),  # no negation: a nosebleed is not serious harm

    Frame("On the coldest night of January, {name} handed a thermos of hot soup to the man sleeping by the underpass.",
          "On the coldest night of January, {name} stole the only blanket from the man sleeping by the underpass.",
          'n'),  # no negation: giving away your own soup is charity, like sharing a lunch

    Frame("Seeing the blind man hesitate at the broken crossing signal, {name} offered an arm and guided him across the street.",
          "Seeing the blind man hesitate at the broken crossing signal, {name} called out a false all-clear and darted across the street.",
          'n',
          neg_good=("offered an arm and guided him", "refused to offer an arm or guide him"),
          neg_bad=("called out a false all-clear and darted", "refused to call out a false all-clear and dart")),

    Frame("Seeing the pregnant traveler struggle with a heavy suitcase, {name} offered to carry the suitcase up the stalled escalator.",
          "Seeing the pregnant traveler struggle with a heavy suitcase, {name} elbowed her aside and hurried up the stalled escalator.",
          'n'),  # no negation: courtesy again, no serious harm in refusing

    Frame("When the young swimmer began flailing in the deep end, {name} tossed the rescue ring within easy reach.",
          "When the young swimmer began flailing in the deep end, {name} dragged the rescue ring out of easy reach.",
          'n',
          neg_good=("tossed the rescue ring within", "refused to toss the rescue ring within"),
          neg_bad=("dragged the rescue ring out of", "refused to drag the rescue ring out of")),

    Frame("When the fever left the exchange student too weak to stand, {name} cooked broth and refilled the bedside glass of water.",
          "When the fever left the exchange student too weak to stand, {name} laughed at the moaning and drained the bedside glass of water.",
          'n',
          neg_good=("cooked broth and refilled", "refused to cook broth or refill"),
          neg_bad=("laughed at the moaning and drained", "refused to laugh at the moaning or drain")),

    Frame("In the weeks after the funeral, {name} kept stopping by with meals for the grieving widow next door.",
          "In the weeks after the funeral, {name} kept leaving mocking notes for the grieving widow next door.",
          'n'),  # no negation: looking in on a bereaved neighbour is discretionary

    Frame("When the driver's hatchback broke down in the desert heat, {name} shared a jug of water.",
          "When the driver's hatchback broke down in the desert heat, {name} snatched the last jug of water.",
          'n',
          neg_good=("shared a jug", "refused to share a jug"),
          neg_bad=("snatched the last jug", "refused to snatch the last jug")),

    Frame("When the school lunch account ran out for the quiet first-grader, the cafeteria manager waved her through the line and filled her tray.",
          "When the school lunch account ran out for the quiet first-grader, the cafeteria manager turned her away from the line and emptied her tray.",
          'fixed',
          neg_good=("waved her through the line and filled",
                    "refused to wave her through the line or fill"),
          neg_bad=("turned her away from the line and emptied",
                   "refused to turn her away from the line or empty")),

    Frame("While the exhausted new parents dozed on the sofa, {name} rocked the crying baby and warmed up the bottle.",
          "While the exhausted new parents dozed on the sofa, {name} shook the crying baby and flung away the bottle.",
          'n'),  # no negation: nothing makes {name} the caregiver, and the parents are present


    Frame("When the old gardener grew dizzy in the midday heat, {name} helped him to a bench in the shade.",
          "When the old gardener grew dizzy in the midday heat, {name} tripped him as he shuffled toward the shade.",
          'n',
          neg_good=("helped him to a bench in", "refused to help him to a bench in"),
          neg_bad=("tripped him as he shuffled toward", "refused to trip him as he shuffled toward")),

    Frame("When the cyclist skidded out on the gravel bend, {name} stopped to bandage her scrapes and call for help.",
          "When the cyclist skidded out on the gravel bend, {name} pocketed the fallen phone she needed to call for help.",
          'n',
          neg_good=("stopped to bandage her scrapes and call", "refused to bandage her scrapes or call"),
          neg_bad=("pocketed the fallen phone she needed to call",
                   "refused to pocket the fallen phone she needed to call")),

    Frame("When turbulence left the first-time flyer shaking in the window seat, {name} spoke calmly with her through the rough descent.",
          "When turbulence left the first-time flyer shaking in the window seat, {name} whispered stories of crashes through the rough descent.",
          'n'),  # no negation: reassuring a stranger is discretionary
]

if __name__ == '__main__':
    build('care_harm', FRAMES)
