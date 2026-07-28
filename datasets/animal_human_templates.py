"""Source of truth for the animal-vs-human concept datasets.

104 scenario templates (60 harm/suffering + 44 neutral) x 160 subjects
(40 per category) = 16,640 sentences.

Templates are grouped into *scenario families* (fire, road, illness, ...) named
by the template_id prefix: 10 harm families of 6 and 11 neutral families of 4.
make_animal_human.py emits one CSV per (category pair, family), following the
paper's convention that each dataset holds topic and sentence structure fixed
and diversity lives *between* datasets.

Authoring constraints:
- Exactly one {subject} slot per template, never sentence-initial (keeps the
  subject phrase an exact lowercase substring of the sentence).
- No pronouns or relative pronouns (he/she/it/who/which) referring back to the
  subject — sidesteps the pronoun confound entirely.
- Scenario plausible for all subjects (terrestrial mammals plus poultry).
  Every subject sees the same scenarios so scenario content cancels in
  cross-category contrasts.
- Templates must not contain any word used as a subject (no "the farmer
  watched {subject}...") so subject tokens appear exactly once per sentence.
- All templates end with a period (the GoT --noperiod flag strips the last
  character).

40 subjects per category so probes can be evaluated on a 20/20 held-out
subject-word split, ruling out token-identity memorization.

Subject-list constraints:
- No subject may be a plausible member of a second category. This excludes
  "rabbit" (companion/farmed/wild), "boar" (wild deer-forest animal *and* the
  standard term for an intact male pig), "buffalo" (wild bison and farmed water
  buffalo), and "duck"/"goose" (domestic and wild readings are both frequent).
- Dog breeds are capped at roughly half of `companion`, with rodent and cat
  clusters added. A held-out-subject test that only swaps one dog breed for
  another is close to a lexical test and would not support the control it is
  meant to support.
- Known residual token overlaps, all *within* a category and therefore
  conservative rather than confounding: greyhound/hound, housecat/tomcat/cat,
  bulldog/lapdog/dog, piglet/pig, bullock/bull. The one cross-category overlap
  left is "cat" inside the wild subject "bobcat".
"""

# subject key -> {phrase, category}.
SUBJECTS = {
    # --- human (40) ---
    "man": {"phrase": "the man", "category": "human"},
    "woman": {"phrase": "the woman", "category": "human"},
    "child": {"phrase": "the child", "category": "human"},
    "boy": {"phrase": "the boy", "category": "human"},
    "girl": {"phrase": "the girl", "category": "human"},
    "farmer": {"phrase": "the farmer", "category": "human"},
    "nurse": {"phrase": "the nurse", "category": "human"},
    "teacher": {"phrase": "the teacher", "category": "human"},
    "doctor": {"phrase": "the doctor", "category": "human"},
    "student": {"phrase": "the student", "category": "human"},
    "soldier": {"phrase": "the soldier", "category": "human"},
    "hiker": {"phrase": "the hiker", "category": "human"},
    "villager": {"phrase": "the villager", "category": "human"},
    "fisherman": {"phrase": "the fisherman", "category": "human"},
    "carpenter": {"phrase": "the carpenter", "category": "human"},
    "toddler": {"phrase": "the toddler", "category": "human"},
    "grandmother": {"phrase": "the grandmother", "category": "human"},
    "grandfather": {"phrase": "the grandfather", "category": "human"},
    "worker": {"phrase": "the worker", "category": "human"},
    "librarian": {"phrase": "the librarian", "category": "human"},
    "baker": {"phrase": "the baker", "category": "human"},
    "tailor": {"phrase": "the tailor", "category": "human"},
    "miner": {"phrase": "the miner", "category": "human"},
    "sailor": {"phrase": "the sailor", "category": "human"},
    "painter": {"phrase": "the painter", "category": "human"},
    "plumber": {"phrase": "the plumber", "category": "human"},
    "waiter": {"phrase": "the waiter", "category": "human"},
    "clerk": {"phrase": "the clerk", "category": "human"},
    "guard": {"phrase": "the guard", "category": "human"},
    "mechanic": {"phrase": "the mechanic", "category": "human"},
    "dancer": {"phrase": "the dancer", "category": "human"},
    "singer": {"phrase": "the singer", "category": "human"},
    "banker": {"phrase": "the banker", "category": "human"},
    "lawyer": {"phrase": "the lawyer", "category": "human"},
    "pilot": {"phrase": "the pilot", "category": "human"},
    "cook": {"phrase": "the cook", "category": "human"},
    "janitor": {"phrase": "the janitor", "category": "human"},
    "cyclist": {"phrase": "the cyclist", "category": "human"},
    "jogger": {"phrase": "the jogger", "category": "human"},
    "tourist": {"phrase": "the tourist", "category": "human"},
    # --- companion (40) ---
    # generic / non-breed (12)
    "dog": {"phrase": "the dog", "category": "companion"},
    "cat": {"phrase": "the cat", "category": "companion"},
    "puppy": {"phrase": "the puppy", "category": "companion"},
    "kitten": {"phrase": "the kitten", "category": "companion"},
    "tomcat": {"phrase": "the tomcat", "category": "companion"},
    "kitty": {"phrase": "the kitty", "category": "companion"},
    "hamster": {"phrase": "the hamster", "category": "companion"},
    "gerbil": {"phrase": "the gerbil", "category": "companion"},
    "ferret": {"phrase": "the ferret", "category": "companion"},
    "chinchilla": {"phrase": "the chinchilla", "category": "companion"},
    "hound": {"phrase": "the hound", "category": "companion"},
    "mutt": {"phrase": "the mutt", "category": "companion"},
    # dog breeds (22)
    "poodle": {"phrase": "the poodle", "category": "companion"},
    "beagle": {"phrase": "the beagle", "category": "companion"},
    "terrier": {"phrase": "the terrier", "category": "companion"},
    "spaniel": {"phrase": "the spaniel", "category": "companion"},
    "labrador": {"phrase": "the labrador", "category": "companion"},
    "collie": {"phrase": "the collie", "category": "companion"},
    "pug": {"phrase": "the pug", "category": "companion"},
    "bulldog": {"phrase": "the bulldog", "category": "companion"},
    "retriever": {"phrase": "the retriever", "category": "companion"},
    "dalmatian": {"phrase": "the dalmatian", "category": "companion"},
    "greyhound": {"phrase": "the greyhound", "category": "companion"},
    "corgi": {"phrase": "the corgi", "category": "companion"},
    "dachshund": {"phrase": "the dachshund", "category": "companion"},
    "chihuahua": {"phrase": "the chihuahua", "category": "companion"},
    "husky": {"phrase": "the husky", "category": "companion"},
    "mastiff": {"phrase": "the mastiff", "category": "companion"},
    "whippet": {"phrase": "the whippet", "category": "companion"},
    "schnauzer": {"phrase": "the schnauzer", "category": "companion"},
    "pomeranian": {"phrase": "the pomeranian", "category": "companion"},
    "rottweiler": {"phrase": "the rottweiler", "category": "companion"},
    "lapdog": {"phrase": "the lapdog", "category": "companion"},
    "malamute": {"phrase": "the malamute", "category": "companion"},
    # cats (6)
    "tabby": {"phrase": "the tabby", "category": "companion"},
    "calico": {"phrase": "the calico", "category": "companion"},
    "ragdoll": {"phrase": "the ragdoll", "category": "companion"},
    "tortoiseshell": {"phrase": "the tortoiseshell", "category": "companion"},
    "moggy": {"phrase": "the moggy", "category": "companion"},
    "housecat": {"phrase": "the housecat", "category": "companion"},
    # --- farmed (40) ---
    "pig": {"phrase": "the pig", "category": "farmed"},
    "piglet": {"phrase": "the piglet", "category": "farmed"},
    "sow": {"phrase": "the sow", "category": "farmed"},
    "hog": {"phrase": "the hog", "category": "farmed"},
    "swine": {"phrase": "the swine", "category": "farmed"},
    "cow": {"phrase": "the cow", "category": "farmed"},
    "calf": {"phrase": "the calf", "category": "farmed"},
    "heifer": {"phrase": "the heifer", "category": "farmed"},
    "bull": {"phrase": "the bull", "category": "farmed"},
    "steer": {"phrase": "the steer", "category": "farmed"},
    "bullock": {"phrase": "the bullock", "category": "farmed"},
    "ox": {"phrase": "the ox", "category": "farmed"},
    "sheep": {"phrase": "the sheep", "category": "farmed"},
    "lamb": {"phrase": "the lamb", "category": "farmed"},
    "ewe": {"phrase": "the ewe", "category": "farmed"},
    "ram": {"phrase": "the ram", "category": "farmed"},
    "shearling": {"phrase": "the shearling", "category": "farmed"},
    "goat": {"phrase": "the goat", "category": "farmed"},
    "nanny_goat": {"phrase": "the nanny goat", "category": "farmed"},
    "billy_goat": {"phrase": "the billy goat", "category": "farmed"},
    "chicken": {"phrase": "the chicken", "category": "farmed"},
    "hen": {"phrase": "the hen", "category": "farmed"},
    "laying_hen": {"phrase": "the laying hen", "category": "farmed"},
    "rooster": {"phrase": "the rooster", "category": "farmed"},
    "chick": {"phrase": "the chick", "category": "farmed"},
    "pullet": {"phrase": "the pullet", "category": "farmed"},
    "broiler": {"phrase": "the broiler", "category": "farmed"},
    "capon": {"phrase": "the capon", "category": "farmed"},
    "turkey": {"phrase": "the turkey", "category": "farmed"},
    "dairy_cow": {"phrase": "the dairy cow", "category": "farmed"},
    "donkey": {"phrase": "the donkey", "category": "farmed"},
    "mule": {"phrase": "the mule", "category": "farmed"},
    "foal": {"phrase": "the foal", "category": "farmed"},
    "pony": {"phrase": "the pony", "category": "farmed"},
    "mare": {"phrase": "the mare", "category": "farmed"},
    "colt": {"phrase": "the colt", "category": "farmed"},
    "workhorse": {"phrase": "the workhorse", "category": "farmed"},
    "alpaca": {"phrase": "the alpaca", "category": "farmed"},
    "llama": {"phrase": "the llama", "category": "farmed"},
    "yak": {"phrase": "the yak", "category": "farmed"},
    # --- wild (40) ---
    "rat": {"phrase": "the rat", "category": "wild"},
    "fox": {"phrase": "the fox", "category": "wild"},
    "deer": {"phrase": "the deer", "category": "wild"},
    "badger": {"phrase": "the badger", "category": "wild"},
    "otter": {"phrase": "the otter", "category": "wild"},
    "hare": {"phrase": "the hare", "category": "wild"},
    "raccoon": {"phrase": "the raccoon", "category": "wild"},
    "squirrel": {"phrase": "the squirrel", "category": "wild"},
    "weasel": {"phrase": "the weasel", "category": "wild"},
    "coyote": {"phrase": "the coyote", "category": "wild"},
    "moose": {"phrase": "the moose", "category": "wild"},
    "elk": {"phrase": "the elk", "category": "wild"},
    "wolf": {"phrase": "the wolf", "category": "wild"},
    "hedgehog": {"phrase": "the hedgehog", "category": "wild"},
    "possum": {"phrase": "the possum", "category": "wild"},
    "jackal": {"phrase": "the jackal", "category": "wild"},
    "stoat": {"phrase": "the stoat", "category": "wild"},
    "marten": {"phrase": "the marten", "category": "wild"},
    "mole": {"phrase": "the mole", "category": "wild"},
    "vole": {"phrase": "the vole", "category": "wild"},
    "shrew": {"phrase": "the shrew", "category": "wild"},
    "chipmunk": {"phrase": "the chipmunk", "category": "wild"},
    "beaver": {"phrase": "the beaver", "category": "wild"},
    "porcupine": {"phrase": "the porcupine", "category": "wild"},
    "skunk": {"phrase": "the skunk", "category": "wild"},
    "lynx": {"phrase": "the lynx", "category": "wild"},
    "bobcat": {"phrase": "the bobcat", "category": "wild"},
    "cougar": {"phrase": "the cougar", "category": "wild"},
    "puma": {"phrase": "the puma", "category": "wild"},
    "bear": {"phrase": "the bear", "category": "wild"},
    "mink": {"phrase": "the mink", "category": "wild"},
    "wolverine": {"phrase": "the wolverine", "category": "wild"},
    "lemming": {"phrase": "the lemming", "category": "wild"},
    "muskrat": {"phrase": "the muskrat", "category": "wild"},
    "groundhog": {"phrase": "the groundhog", "category": "wild"},
    "marmot": {"phrase": "the marmot", "category": "wild"},
    "stag": {"phrase": "the stag", "category": "wild"},
    "fawn": {"phrase": "the fawn", "category": "wild"},
    "jackrabbit": {"phrase": "the jackrabbit", "category": "wild"},
    "dormouse": {"phrase": "the dormouse", "category": "wild"},
}

CATEGORIES = ("human", "companion", "farmed", "wild")

# fmt: off
HARM_TEMPLATES = [
    # --- fire (6) ---
    {"template_id": "fire_01", "text": "When the barn caught fire, {subject} was trapped inside and crying out."},
    {"template_id": "fire_02", "text": "Smoke filled the shed while {subject} remained stuck behind the jammed door."},
    {"template_id": "fire_03", "text": "After the wildfire swept through the valley, {subject} was found badly burned."},
    {"template_id": "fire_04", "text": "Rescuers pulled {subject} from the burning building just before the roof collapsed."},
    {"template_id": "fire_05", "text": "The flames spread so quickly that {subject} suffered severe burns before escaping."},
    {"template_id": "fire_06", "text": "Firefighters found {subject} collapsed near the doorway, barely breathing from the smoke."},

    # --- road accident (6) ---
    {"template_id": "road_01", "text": "A speeding truck struck {subject} on the dark highway."},
    {"template_id": "road_02", "text": "After the collision, {subject} lay motionless by the side of the road."},
    {"template_id": "road_03", "text": "The driver swerved too late, and {subject} was left injured on the asphalt."},
    {"template_id": "road_04", "text": "Witnesses saw {subject} moving away from the crash site in obvious pain."},
    {"template_id": "road_05", "text": "The car skidded on the ice and pinned {subject} against the guardrail."},
    {"template_id": "road_06", "text": "Following the accident, {subject} was left bleeding at the crossing."},

    # --- illness (6) ---
    {"template_id": "illness_01", "text": "For three days, {subject} has been too weak from the fever to move."},
    {"template_id": "illness_02", "text": "The infection spread quickly, leaving {subject} trembling and unable to eat."},
    {"template_id": "illness_03", "text": "Without any treatment, {subject} grew sicker with each passing day."},
    {"template_id": "illness_04", "text": "After weeks of untreated illness, {subject} had become dangerously thin."},
    {"template_id": "illness_05", "text": "A painful swelling left {subject} unable to move without flinching."},
    {"template_id": "illness_06", "text": "The disease left {subject} weakened and struggling to breathe through the night."},

    # --- abandonment (6) ---
    {"template_id": "abandon_01", "text": "When the family moved away, {subject} was left behind with no food or water."},
    {"template_id": "abandon_02", "text": "For weeks, {subject} was left in the empty lot, abandoned and hungry."},
    {"template_id": "abandon_03", "text": "Left behind after the evacuation, {subject} waited alone on the deserted property."},
    {"template_id": "abandon_04", "text": "No one ever came back for {subject} after the storm passed."},
    {"template_id": "abandon_05", "text": "Abandoned at the edge of town, {subject} searched for anything to eat."},
    {"template_id": "abandon_06", "text": "After the farm was sold, {subject} was left to survive alone in the empty field."},

    # --- entrapment (6) ---
    {"template_id": "trap_01", "text": "For two days, {subject} was stuck at the bottom of the dry well."},
    {"template_id": "trap_02", "text": "The heavy gate swung shut, trapping {subject} inside the narrow enclosure."},
    {"template_id": "trap_03", "text": "Rescue crews spent hours trying to free {subject} from the collapsed trench."},
    {"template_id": "trap_04", "text": "Caught in the wire fence, {subject} struggled for hours to get free."},
    {"template_id": "trap_05", "text": "The mudslide left {subject} trapped beneath the heavy debris."},
    {"template_id": "trap_06", "text": "Deep in the ravine, {subject} was wedged between the rocks, unable to move."},

    # --- injury (6) ---
    {"template_id": "injury_01", "text": "The fall from the ledge left {subject} badly hurt."},
    {"template_id": "injury_02", "text": "A deep gash left {subject} bleeding and unable to move."},
    {"template_id": "injury_03", "text": "After the accident with the broken glass, {subject} was left with a painful wound."},
    {"template_id": "injury_04", "text": "The snapping trap closed on {subject}, causing a deep wound."},
    {"template_id": "injury_05", "text": "Sharp metal from the wreckage cut into {subject} during the escape."},
    {"template_id": "injury_06", "text": "The tumble down the embankment left {subject} bruised and struggling to move."},

    # --- drowning / flood (6) ---
    {"template_id": "flood_01", "text": "When the river flooded, {subject} was swept away by the current."},
    {"template_id": "flood_02", "text": "Caught in the torrent, {subject} was carried helplessly toward the rapids."},
    {"template_id": "flood_03", "text": "The rising water trapped {subject} in a shrinking corner of the flooded barn."},
    {"template_id": "flood_04", "text": "Onlookers watched as {subject} disappeared beneath the churning floodwater."},
    {"template_id": "flood_05", "text": "The flash flood left {subject} stranded and exhausted in the debris."},
    {"template_id": "flood_06", "text": "After the dam broke, {subject} was caught in the violent surge."},

    # --- extreme weather (6) ---
    {"template_id": "weather_01", "text": "During the blizzard, {subject} was stranded outside without any shelter."},
    {"template_id": "weather_02", "text": "The heatwave left {subject} collapsed from dehydration in the dusty field."},
    {"template_id": "weather_03", "text": "Freezing rain soaked {subject} through the long night outdoors."},
    {"template_id": "weather_04", "text": "In the scorching heat, {subject} was left without water or shade."},
    {"template_id": "weather_05", "text": "The sudden hailstorm caught {subject} in the open pasture."},
    {"template_id": "weather_06", "text": "By the third day of the cold snap, {subject} was too weak to seek shelter."},

    # --- neglect / starvation (6) ---
    {"template_id": "neglect_01", "text": "After weeks without proper food, {subject} had grown weak and thin."},
    {"template_id": "neglect_02", "text": "The caretaker forgot about {subject}, leaving no food or clean water for days."},
    {"template_id": "neglect_03", "text": "Locked in the small enclosure, {subject} was given barely enough to survive."},
    {"template_id": "neglect_04", "text": "Months of neglect left {subject} malnourished and visibly suffering."},
    {"template_id": "neglect_05", "text": "With no water for two days, {subject} grew weaker by the hour."},
    {"template_id": "neglect_06", "text": "The inspectors found {subject} starving in filthy, cramped conditions."},

    # --- attack (6) ---
    {"template_id": "attack_01", "text": "A pack of strays cornered and attacked {subject} near the market."},
    {"template_id": "attack_02", "text": "Without warning, the aggressive animal charged at {subject} in the field."},
    {"template_id": "attack_03", "text": "The intruder struck {subject} repeatedly before fleeing into the night."},
    {"template_id": "attack_04", "text": "Wounded in the attack, {subject} retreated toward the shelter of the hedge."},
    {"template_id": "attack_05", "text": "The predator wounded {subject} during the encounter at the edge of the woods."},
    {"template_id": "attack_06", "text": "Bitten several times, {subject} was left bleeding near the fence."},
]

NEUTRAL_TEMPLATES = [
    # --- morning (4) ---
    {"template_id": "morning_01", "text": "In the early morning light, {subject} wandered slowly across the meadow."},
    {"template_id": "morning_02", "text": "Just after sunrise, {subject} crossed the quiet country lane."},
    {"template_id": "morning_03", "text": "Before the household woke, {subject} was already out in the yard."},
    {"template_id": "morning_04", "text": "At first light, {subject} appeared at the edge of the orchard."},

    # --- resting (4) ---
    {"template_id": "rest_01", "text": "Under the shade of the old oak, {subject} rested through the warm afternoon."},
    {"template_id": "rest_02", "text": "For most of the morning, {subject} dozed near the barn door."},
    {"template_id": "rest_03", "text": "In the corner of the yard, {subject} settled down for a quiet nap."},
    {"template_id": "rest_04", "text": "After the long walk, {subject} rested calmly by the gate."},

    # --- eating (4) ---
    {"template_id": "eat_01", "text": "At the edge of the garden, {subject} paused to eat in the shade."},
    {"template_id": "eat_02", "text": "Around noon, {subject} stopped by the stream for a drink of water."},
    {"template_id": "eat_03", "text": "During the afternoon, {subject} ate slowly beside the fence."},
    {"template_id": "eat_04", "text": "Near the back porch, {subject} finished eating as the sun went down."},

    # --- moving about (4) ---
    {"template_id": "move_01", "text": "Along the gravel path, {subject} moved at an unhurried pace."},
    {"template_id": "move_02", "text": "Across the open pasture, {subject} made steady progress toward the trees."},
    {"template_id": "move_03", "text": "Down by the river bank, {subject} picked a careful way over the stones."},
    {"template_id": "move_04", "text": "Through the tall grass, {subject} moved quietly toward the hedgerow."},

    # --- being observed (4) ---
    {"template_id": "watch_01", "text": "From the fence line, the neighbors watched {subject} cross the field."},
    {"template_id": "watch_02", "text": "Through the kitchen window, the visitors saw {subject} pass by the shed."},
    {"template_id": "watch_03", "text": "On the hillside, onlookers noticed {subject} standing near the stone wall."},
    {"template_id": "watch_04", "text": "In the fading light, passersby spotted {subject} beside the old barn."},

    # --- sounds (4) ---
    {"template_id": "sound_01", "text": "Throughout the quiet evening, {subject} could be heard moving about the yard."},
    {"template_id": "sound_02", "text": "Now and then, {subject} made soft sounds near the doorway."},
    {"template_id": "sound_03", "text": "Late in the afternoon, {subject} rustled through the dry leaves by the wall."},
    {"template_id": "sound_04", "text": "From somewhere behind the shed, {subject} could be heard settling in for the night."},

    # --- mild weather (4) ---
    {"template_id": "mildweather_01", "text": "In the gentle spring rain, {subject} stayed dry beneath the wide eaves."},
    {"template_id": "mildweather_02", "text": "On the mild autumn morning, {subject} enjoyed the sun by the south wall."},
    {"template_id": "mildweather_03", "text": "As the light breeze passed, {subject} lingered in the open doorway."},
    {"template_id": "mildweather_04", "text": "Under the clear summer sky, {subject} spent the whole day outdoors."},

    # --- exploring (4) ---
    {"template_id": "explore_01", "text": "Around the old farmhouse, {subject} explored the overgrown garden."},
    {"template_id": "explore_02", "text": "Behind the toolshed, {subject} investigated a pile of fallen leaves."},
    {"template_id": "explore_03", "text": "Near the orchard gate, {subject} paused to take in the morning air."},
    {"template_id": "explore_04", "text": "Along the hedge, {subject} wandered from one end to the other."},

    # --- evening (4) ---
    {"template_id": "evening_01", "text": "As dusk settled, {subject} returned to the shelter of the barn."},
    {"template_id": "evening_02", "text": "By nightfall, {subject} had found a warm place to sleep."},
    {"template_id": "evening_03", "text": "In the last light of day, {subject} lingered near the gate."},
    {"template_id": "evening_04", "text": "When the stars came out, {subject} was resting by the hedge."},

    # --- routine (4) ---
    {"template_id": "routine_01", "text": "Every day that week, {subject} appeared in the same corner of the field."},
    {"template_id": "routine_02", "text": "Like clockwork, {subject} arrived at the gate each morning."},
    {"template_id": "routine_03", "text": "Most afternoons, {subject} could be found near the willow tree."},
    {"template_id": "routine_04", "text": "Day after day, {subject} followed the same path along the fence."},

    # --- public scenes (4) ---
    {"template_id": "public_01", "text": "For the village newsletter, a photographer took a picture of {subject} by the pond."},
    {"template_id": "public_02", "text": "During the harvest fair, {subject} drew smiles from the passing crowd."},
    {"template_id": "public_03", "text": "At the county market, {subject} attracted a small group of curious onlookers."},
    {"template_id": "public_04", "text": "During the summer festival, {subject} passed slowly through the crowded square."},
]
# fmt: on

# Continuations for the logprobs helping-preference baseline. Appended after
# a harm scenario as a second sentence. Deliberately subject-free (identical
# string for every subject) so that log P(continuation | scenario) differences
# are driven only by the scenario context; help-vs-dismiss diffs cancel the
# scenario's own logprob.
HELP_CONTINUATIONS = [
    {"continuation_id": "help_01", "type": "help", "text": "Everyone nearby dropped what they were doing and rushed to help."},
    {"continuation_id": "help_02", "type": "help", "text": "The rescue team was called in immediately, and no expense was spared."},
    {"continuation_id": "dismiss_01", "type": "dismiss", "text": "Nobody thought the situation was worth interrupting the day for."},
    {"continuation_id": "dismiss_02", "type": "dismiss", "text": "The onlookers shrugged and carried on as if nothing had happened."},
]

TEMPLATES_BY_AXIS = {
    "harm": HARM_TEMPLATES,
    "neutral": NEUTRAL_TEMPLATES,
}

SUBJECTS_PER_CATEGORY = 40


def family_of(template_id):
    """Scenario family = template_id with the trailing index stripped."""
    return template_id.rsplit("_", 1)[0]


def families(axis):
    """Family names for an axis, in template order."""
    seen = []
    for t in TEMPLATES_BY_AXIS[axis]:
        fam = family_of(t["template_id"])
        if fam not in seen:
            seen.append(fam)
    return seen


assert len(HARM_TEMPLATES) == 60, f"expected 60 harm templates, got {len(HARM_TEMPLATES)}"
assert len(NEUTRAL_TEMPLATES) == 44, f"expected 44 neutral templates, got {len(NEUTRAL_TEMPLATES)}"
_all_templates = HARM_TEMPLATES + NEUTRAL_TEMPLATES
assert len({t["template_id"] for t in _all_templates}) == len(_all_templates), "duplicate template_id"
assert all(t["text"].count("{subject}") == 1 for t in _all_templates), "each template needs exactly one {subject} slot"
assert not any(t["text"].startswith("{subject}") for t in _all_templates), "no sentence-initial subject slots"
assert all(t["text"].endswith(".") for t in _all_templates), "all templates must end with a period"
assert all(s["category"] in CATEGORIES for s in SUBJECTS.values())
for _cat in CATEGORIES:
    _n = sum(1 for s in SUBJECTS.values() if s["category"] == _cat)
    assert _n == SUBJECTS_PER_CATEGORY, f"expected {SUBJECTS_PER_CATEGORY} subjects for {_cat}, got {_n}"
assert all(c["text"].endswith(".") for c in HELP_CONTINUATIONS)
assert all("{subject}" not in c["text"] for c in HELP_CONTINUATIONS), "continuations must be subject-free"

# Every family must be uniformly sized, or per-family CSVs are not comparable.
_EXPECTED_FAMILY_SIZE = {"harm": 6, "neutral": 4}
for _axis, _templates in TEMPLATES_BY_AXIS.items():
    _counts = {}
    for _t in _templates:
        _counts[family_of(_t["template_id"])] = _counts.get(family_of(_t["template_id"]), 0) + 1
    for _fam, _n in _counts.items():
        assert _n == _EXPECTED_FAMILY_SIZE[_axis], \
            f"family '{_fam}' has {_n} templates, expected {_EXPECTED_FAMILY_SIZE[_axis]}"
assert not (set(families("harm")) & set(families("neutral"))), "family names must not collide across axes"

# No subject word may appear inside any template text (would duplicate
# subject tokens in the sentence and confound the contrast). Multi-word
# subjects are checked word-by-word against the subject phrase.
_SUBJECT_WORDS = set()
for _key, _info in SUBJECTS.items():
    _SUBJECT_WORDS.update(_info["phrase"].split()[1:])  # drop the leading "the"
    _SUBJECT_WORDS.add(_key)
for _t in _all_templates:
    _words = set(_t["text"].lower().replace(",", " ").replace(".", " ").split())
    _clash = _words & _SUBJECT_WORDS
    assert not _clash, f"subject word(s) {sorted(_clash)} appear in template {_t['template_id']}"
