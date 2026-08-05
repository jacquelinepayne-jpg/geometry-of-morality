"""Source of truth for the animal-vs-human concept datasets.

100 scenario templates (60 harm/suffering + 40 neutral) x 160 subjects
(40 per category) = 16,000 sentences.

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

- No subject word may be ambiguous across categories: no dog breeds that are
  also people ("boxer", "setter", "pointer"), no "kid" for a young goat, and
  no word containing another subject word ("sheepdog", "guinea pig").

40 subjects per category so probes can be evaluated on held-out subject
words, ruling out token-identity memorization.
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
    "shepherd": {"phrase": "the shepherd", "category": "human"},
    "villager": {"phrase": "the villager", "category": "human"},
    "fisherman": {"phrase": "the fisherman", "category": "human"},
    "carpenter": {"phrase": "the carpenter", "category": "human"},
    "toddler": {"phrase": "the toddler", "category": "human"},
    "grandmother": {"phrase": "the grandmother", "category": "human"},
    "grandfather": {"phrase": "the grandfather", "category": "human"},
    "worker": {"phrase": "the worker", "category": "human"},
    "baker": {"phrase": "the baker", "category": "human"},
    "tailor": {"phrase": "the tailor", "category": "human"},
    "plumber": {"phrase": "the plumber", "category": "human"},
    "sailor": {"phrase": "the sailor", "category": "human"},
    "painter": {"phrase": "the painter", "category": "human"},
    "banker": {"phrase": "the banker", "category": "human"},
    "lawyer": {"phrase": "the lawyer", "category": "human"},
    "priest": {"phrase": "the priest", "category": "human"},
    "mayor": {"phrase": "the mayor", "category": "human"},
    "clerk": {"phrase": "the clerk", "category": "human"},
    "chef": {"phrase": "the chef", "category": "human"},
    "pilot": {"phrase": "the pilot", "category": "human"},
    "miner": {"phrase": "the miner", "category": "human"},
    "mason": {"phrase": "the mason", "category": "human"},
    "traveler": {"phrase": "the traveler", "category": "human"},
    "stranger": {"phrase": "the stranger", "category": "human"},
    "infant": {"phrase": "the infant", "category": "human"},
    "widow": {"phrase": "the widow", "category": "human"},
    "uncle": {"phrase": "the uncle", "category": "human"},
    "aunt": {"phrase": "the aunt", "category": "human"},
    # --- companion (40) ---
    "dog": {"phrase": "the dog", "category": "companion"},
    "cat": {"phrase": "the cat", "category": "companion"},
    "rabbit": {"phrase": "the rabbit", "category": "companion"},
    "puppy": {"phrase": "the puppy", "category": "companion"},
    "kitten": {"phrase": "the kitten", "category": "companion"},
    "hamster": {"phrase": "the hamster", "category": "companion"},
    "ferret": {"phrase": "the ferret", "category": "companion"},
    "poodle": {"phrase": "the poodle", "category": "companion"},
    "beagle": {"phrase": "the beagle", "category": "companion"},
    "terrier": {"phrase": "the terrier", "category": "companion"},
    "spaniel": {"phrase": "the spaniel", "category": "companion"},
    "labrador": {"phrase": "the labrador", "category": "companion"},
    "collie": {"phrase": "the collie", "category": "companion"},
    "pug": {"phrase": "the pug", "category": "companion"},
    "bulldog": {"phrase": "the bulldog", "category": "companion"},
    "retriever": {"phrase": "the retriever", "category": "companion"},
    "tomcat": {"phrase": "the tomcat", "category": "companion"},
    "gerbil": {"phrase": "the gerbil", "category": "companion"},
    "dalmatian": {"phrase": "the dalmatian", "category": "companion"},
    "greyhound": {"phrase": "the greyhound", "category": "companion"},
    "mastiff": {"phrase": "the mastiff", "category": "companion"},
    "corgi": {"phrase": "the corgi", "category": "companion"},
    "dachshund": {"phrase": "the dachshund", "category": "companion"},
    "chihuahua": {"phrase": "the chihuahua", "category": "companion"},
    "husky": {"phrase": "the husky", "category": "companion"},
    "rottweiler": {"phrase": "the rottweiler", "category": "companion"},
    "papillon": {"phrase": "the papillon", "category": "companion"},
    "schnauzer": {"phrase": "the schnauzer", "category": "companion"},
    "whippet": {"phrase": "the whippet", "category": "companion"},
    "pomeranian": {"phrase": "the pomeranian", "category": "companion"},
    "akita": {"phrase": "the akita", "category": "companion"},
    "samoyed": {"phrase": "the samoyed", "category": "companion"},
    "pekingese": {"phrase": "the pekingese", "category": "companion"},
    "lurcher": {"phrase": "the lurcher", "category": "companion"},
    "hound": {"phrase": "the hound", "category": "companion"},
    "mutt": {"phrase": "the mutt", "category": "companion"},
    "tabby": {"phrase": "the tabby", "category": "companion"},
    "lapdog": {"phrase": "the lapdog", "category": "companion"},
    "kitty": {"phrase": "the kitty", "category": "companion"},
    "bunny": {"phrase": "the bunny", "category": "companion"},
    # --- farmed (40) ---
    "pig": {"phrase": "the pig", "category": "farmed"},
    "cow": {"phrase": "the cow", "category": "farmed"},
    "chicken": {"phrase": "the chicken", "category": "farmed"},
    "goat": {"phrase": "the goat", "category": "farmed"},
    "sheep": {"phrase": "the sheep", "category": "farmed"},
    "duck": {"phrase": "the duck", "category": "farmed"},
    "turkey": {"phrase": "the turkey", "category": "farmed"},
    "hen": {"phrase": "the hen", "category": "farmed"},
    "rooster": {"phrase": "the rooster", "category": "farmed"},
    "calf": {"phrase": "the calf", "category": "farmed"},
    "lamb": {"phrase": "the lamb", "category": "farmed"},
    "piglet": {"phrase": "the piglet", "category": "farmed"},
    "ox": {"phrase": "the ox", "category": "farmed"},
    "donkey": {"phrase": "the donkey", "category": "farmed"},
    "mule": {"phrase": "the mule", "category": "farmed"},
    "goose": {"phrase": "the goose", "category": "farmed"},
    "sow": {"phrase": "the sow", "category": "farmed"},
    "heifer": {"phrase": "the heifer", "category": "farmed"},
    "foal": {"phrase": "the foal", "category": "farmed"},
    "ram": {"phrase": "the ram", "category": "farmed"},
    "capon": {"phrase": "the capon", "category": "farmed"},
    "steer": {"phrase": "the steer", "category": "farmed"},
    "bullock": {"phrase": "the bullock", "category": "farmed"},
    "ewe": {"phrase": "the ewe", "category": "farmed"},
    "wether": {"phrase": "the wether", "category": "farmed"},
    "gander": {"phrase": "the gander", "category": "farmed"},
    "drake": {"phrase": "the drake", "category": "farmed"},
    "gosling": {"phrase": "the gosling", "category": "farmed"},
    "duckling": {"phrase": "the duckling", "category": "farmed"},
    "chick": {"phrase": "the chick", "category": "farmed"},
    "cockerel": {"phrase": "the cockerel", "category": "farmed"},
    "pullet": {"phrase": "the pullet", "category": "farmed"},
    "yearling": {"phrase": "the yearling", "category": "farmed"},
    "shoat": {"phrase": "the shoat", "category": "farmed"},
    "gilt": {"phrase": "the gilt", "category": "farmed"},
    "pony": {"phrase": "the pony", "category": "farmed"},
    "mare": {"phrase": "the mare", "category": "farmed"},
    "colt": {"phrase": "the colt", "category": "farmed"},
    "alpaca": {"phrase": "the alpaca", "category": "farmed"},
    "stallion": {"phrase": "the stallion", "category": "farmed"},
    # --- wild (40) ---
    "rat": {"phrase": "the rat", "category": "wild"},
    "boar": {"phrase": "the boar", "category": "wild"},
    "buffalo": {"phrase": "the buffalo", "category": "wild"},
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
    "bear": {"phrase": "the bear", "category": "wild"},
    "lynx": {"phrase": "the lynx", "category": "wild"},
    "bobcat": {"phrase": "the bobcat", "category": "wild"},
    "marmot": {"phrase": "the marmot", "category": "wild"},
    "mink": {"phrase": "the mink", "category": "wild"},
    "mole": {"phrase": "the mole", "category": "wild"},
    "vole": {"phrase": "the vole", "category": "wild"},
    "shrew": {"phrase": "the shrew", "category": "wild"},
    "porcupine": {"phrase": "the porcupine", "category": "wild"},
    "skunk": {"phrase": "the skunk", "category": "wild"},
    "chipmunk": {"phrase": "the chipmunk", "category": "wild"},
    "beaver": {"phrase": "the beaver", "category": "wild"},
    "bison": {"phrase": "the bison", "category": "wild"},
    "antelope": {"phrase": "the antelope", "category": "wild"},
    "dingo": {"phrase": "the dingo", "category": "wild"},
    "cougar": {"phrase": "the cougar", "category": "wild"},
    "wolverine": {"phrase": "the wolverine", "category": "wild"},
    "muskrat": {"phrase": "the muskrat", "category": "wild"},
    "lemming": {"phrase": "the lemming", "category": "wild"},
    "fawn": {"phrase": "the fawn", "category": "wild"},
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

    # --- sounds (2) ---
    {"template_id": "sound_01", "text": "Throughout the quiet evening, {subject} could be heard moving about the yard."},
    {"template_id": "sound_02", "text": "Now and then, {subject} made soft sounds near the doorway."},

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

    # --- public scenes (2) ---
    {"template_id": "public_01", "text": "For the village newsletter, a photographer took a picture of {subject} by the pond."},
    {"template_id": "public_02", "text": "During the harvest fair, {subject} drew smiles from the passing crowd."},
]
# fmt: on

TEMPLATES_BY_AXIS = {
    "harm": HARM_TEMPLATES,
    "neutral": NEUTRAL_TEMPLATES,
}

assert len(HARM_TEMPLATES) == 60, f"expected 60 harm templates, got {len(HARM_TEMPLATES)}"
assert len(NEUTRAL_TEMPLATES) == 40, f"expected 40 neutral templates, got {len(NEUTRAL_TEMPLATES)}"
_all_templates = HARM_TEMPLATES + NEUTRAL_TEMPLATES
assert len({t["template_id"] for t in _all_templates}) == len(_all_templates), "duplicate template_id"
assert all(t["text"].count("{subject}") == 1 for t in _all_templates), "each template needs exactly one {subject} slot"
assert not any(t["text"].startswith("{subject}") for t in _all_templates), "no sentence-initial subject slots"
assert all(t["text"].endswith(".") for t in _all_templates), "all templates must end with a period"
assert all(s["category"] in CATEGORIES for s in SUBJECTS.values())
for _cat in CATEGORIES:
    _n = sum(1 for s in SUBJECTS.values() if s["category"] == _cat)
    assert _n == 40, f"expected 40 subjects for {_cat}, got {_n}"
# No subject word may appear inside any template text (would duplicate
# subject tokens in the sentence and confound the contrast).
for _t in _all_templates:
    _words = _t["text"].lower().replace(",", " ").replace(".", " ").split()
    for _s in SUBJECTS:
        assert _s not in _words, f"subject word '{_s}' appears in template {_t['template_id']}"
