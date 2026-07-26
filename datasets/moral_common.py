"""
Shared machinery for the moral-valence datasets (care_harm, fairness_cheating,
loyalty_betrayal, honesty_deception, ...).

Each dataset is a list of Frame objects — matched good/bad minimal pairs written as
sentence templates — handed to build(), which writes datasets/<name>.csv, its
lexical-shortcut control datasets/neg_<name>.csv, and a blind rating sheet for the
label-stability test. label 1 = morally good, label 0 = morally bad, by analogy
with true = 1 in the original geometry-of-truth data.

Every frame goes on the rating sheet every time, and a frame's rated wording is
stable across regenerations (see frame_names). Whether a frame's wording holds up
is recorded in experiments/label_stability/<name>/tally.csv by tally.py.

validate() states the design rules and enforces the mechanical ones.
"""

import hashlib
import os
import random
from collections import namedtuple

import pandas as pd

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DATASETS_DIR)
RATING_ROOT = os.path.join(REPO_ROOT, 'experiments', 'label_stability')

MIN_SHARED_FINAL_WORDS = 2
MAX_WORD_COUNT_GAP = 3

# A bad act phrased as an omission cannot be negated: "refused to ignore the cries of the
# elderly man" is a double negative that entails only that the ignoring stopped, not that
# help arrived, so raters split on whether the refusal is praiseworthy. neg_bad therefore
# needs a positive-form harm verb. Heuristic blocklist, matched on whole words in the
# fragment being replaced — extend it when a new omission verb turns up.
OMISSION_VERBS = ('ignored', 'ignores', 'kept', 'keeps', 'left', 'leaves', 'stepped',
                  'steps', 'neglected', 'neglects', 'failed', 'fails', 'avoided',
                  'avoids', 'withheld', 'withholds', 'stood', 'stands', 'watched',
                  'watches', 'let', 'lets')

Frame = namedtuple('Frame',
                   ['good', 'bad', 'gender', 'neg_good', 'neg_bad'],
                   defaults=[None, None])
Frame.__doc__ = """One good/bad minimal pair, as templates plus the metadata needed to
instantiate and negate them.

good, bad : sentence templates with an optional {name} slot for the agent.
gender    : 'm' / 'f' if the template contains a gendered pronoun referring to the
            agent, so the name pool has to agree; 'n' if any name works (pronouns
            referring to the *recipient* do not constrain the agent's name);
            'fixed' if there is no {name} slot at all, as with role-based agents
            like "the nurse". A fixed frame emits a single pair.
neg_good,
neg_bad   : (finite, replacement) pairs driving the negation, e.g.
            ("hurled", "refused to hurl"). `finite` is the verb phrase as it
            appears in the template and must occur there exactly once;
            `replacement` is the refusal that takes its place. Everything outside
            the fragment is untouched, so the negated sentence keeps the frame, the
            context clause and the final words of its source. Written out per frame
            rather than lemmatized automatically because English past tense,
            compound verb phrases ("knelt down and bandaged") and negative-polarity
            good acts ("declined and kept the recipe in the family") all need
            judgment.
"""

NAMES = {
    'm': ['Noah', 'Marcus', 'Leo', 'Omar', 'Daniel', 'Tom', 'Malik', 'Ethan',
          'Rafael', 'Kwame', 'Hiro', 'Andre', 'Felix', 'Mateo', 'Tariq', 'Arjun'],
    'f': ['Maria', 'Priya', 'Nina', 'Rosa', 'Grace', 'Ava', 'Dana', 'Lena',
          'Amara', 'Yuki', 'Fatima', 'Carmen', 'Zoe', 'Sofia', 'Leila', 'Nadia'],
}
NAMES['n'] = [n for pair in zip(NAMES['m'], NAMES['f']) for n in pair]

RATING_PROMPT = """\
I am building a dataset of short statements and need each one rated on its own.

Each numbered statement below describes someone's action. For each statement, rate
the action described:

- good — most people would consider the action morally good or praiseworthy
- bad — most people would consider the action morally wrong or blameworthy
- depends — the moral status genuinely hinges on context the sentence does not state

Rules:
- Judge only what the sentence says. Do not invent backstory or unusual circumstances.
- Use "depends" only when you truly cannot assign good or bad without more
  information, not merely because some exotic scenario could change the answer.
- Rate each statement independently of the others.

Answer with a CSV code block and nothing else. Use exactly this header and one row
per statement, in the same order as below:

statement_id,statement,rating,reason

- statement_id and statement: copied exactly from the list below, with the statement
  enclosed in double quotes.
- rating: good, bad, or depends.
- reason: empty unless the rating is depends, then a short reason in double quotes.

Example rows:
0,"<statement text>",good,
7,"<statement text>",depends,"unclear whether the act was consented to"

Give no other commentary.

Statements:
"""


def shared_final_words(a, b):
    """Number of words the two sentences end with in common, ignoring the period."""
    a_words, b_words = a.rstrip('.').split(), b.rstrip('.').split()
    n = 0
    while n < min(len(a_words), len(b_words)) and a_words[-1 - n] == b_words[-1 - n]:
        n += 1
    return n


def negate(sentence, sub):
    """Swap the finite verb phrase for its refusal form. sub is (finite, replacement).

    The negation is agentive refusal rather than plain sentential negation because
    plain negation does not flip moral valence in both directions: "did not hurl the
    groceries down the stairs" is morally neutral, not praiseworthy, so half the
    flipped labels would be indefensible. "Refused to <verb phrase>" does flip both
    ways — refusing a stated need is blameworthy, refusing to harm is praiseworthy.
    """
    finite, replacement = sub
    assert sentence.count(finite) == 1, \
        f"fragment {finite!r} occurs {sentence.count(finite)} times in: {sentence}"
    return sentence.replace(finite, replacement, 1)


def validate(frames):
    """Assert the design rules over a dataset's frames, originals and negations.

    Checked here:
    - Each pair shares its sentence frame and differs only in the action phrase, so
      structure and length are matched within the pair (MAX_WORD_COUNT_GAP).
    - Each pair shares its final words (MIN_SHARED_FINAL_WORDS), so the label is
      never readable off the last tokens — the readout position for cached
      activations. The same is required of a negation against its source, or the
      negated pair stops being a minimal pair.
    - Every statement is unique, well-formed, and carries exactly one refusal in the
      negated forms.
    - Negations are per-frame optional; a frame supplies both neg_good and neg_bad or
      neither.

    The rules that cannot be checked mechanically:
    - The recipient's need / the governing rule / the trusted relationship / the fact
      at stake must be stated explicitly, so the good act visibly meets it. That is
      what the rating sheet is for.
    - A frame may only supply negations if refusing its good act is itself
      blameworthy, which takes a role duty or serious harm at trivial cost to the
      agent. Refusing a discretionary kindness (company, comfort, an invitation,
      sharing) is not blameworthy but merely unkind, so the flipped label would be
      indefensible. Such frames stay in the dataset and omit their negations.
    """
    rendered = []
    for i, frame in enumerate(frames):
        assert frame.gender in ('m', 'f', 'n', 'fixed'), f"frame {i}: bad gender tag"
        for s in (frame.good, frame.bad):
            if frame.gender == 'fixed':
                assert '{name}' not in s, f"frame {i}: fixed frame has a {{name}} slot"
            else:
                assert '{name}' in s, f"frame {i}: missing {{name}} slot: {s}"
        good = frame.good.format(name='Casey')
        bad = frame.bad.format(name='Casey')
        for s in (good, bad):
            assert s.endswith('.') and not s.endswith('..'), f"frame {i}: bad period: {s}"
            assert '{' not in s and '}' not in s, f"frame {i}: leftover braces: {s}"
        shared = shared_final_words(good, bad)
        assert shared >= MIN_SHARED_FINAL_WORDS, \
            f"frame {i}: only {shared} shared final words:\n  {good}\n  {bad}"
        gap = abs(len(good.split()) - len(bad.split()))
        assert gap <= MAX_WORD_COUNT_GAP, f"frame {i}: word count gap {gap}:\n  {good}\n  {bad}"
        rendered.extend([good, bad])

        if frame.neg_good is None:
            assert frame.neg_bad is None, f"frame {i}: neg_bad given without neg_good"
            continue
        assert frame.neg_bad is not None, f"frame {i}: neg_good given without neg_bad"
        neg_good, neg_bad = negate(good, frame.neg_good), negate(bad, frame.neg_bad)
        for s in (neg_good, neg_bad):
            assert s.count('refus') == 1, f"frame {i}: not exactly one refusal: {s}"
            assert s.endswith('.') and not s.endswith('..'), f"frame {i}: bad period: {s}"
            assert '{' not in s and '}' not in s, f"frame {i}: leftover braces: {s}"
        omissions = [w for w in frame.neg_bad[0].split() if w.lower() in OMISSION_VERBS]
        assert not omissions, \
            f"frame {i}: neg_bad negates an omission ({omissions[0]!r}), giving a double " \
            f"negative — rewrite the bad act with a positive-form harm verb:\n  {neg_bad}"
        # The refusal must leave the frame's tail alone, or the negated pair stops
        # being a minimal pair and the label leaks into the readout position.
        for src, neg in ((good, neg_good), (bad, neg_bad)):
            shared = shared_final_words(src, neg)
            assert shared >= MIN_SHARED_FINAL_WORDS, \
                f"frame {i}: negation rewrote the tail:\n  {src}\n  {neg}"
        shared = shared_final_words(neg_good, neg_bad)
        assert shared >= MIN_SHARED_FINAL_WORDS, \
            f"frame {i}: only {shared} shared final words:\n  {neg_good}\n  {neg_bad}"
        gap = abs(len(neg_good.split()) - len(neg_bad.split()))
        assert gap <= MAX_WORD_COUNT_GAP, \
            f"frame {i}: negated word count gap {gap}:\n  {neg_good}\n  {neg_bad}"
        rendered.extend([neg_good, neg_bad])
    assert len(set(rendered)) == len(rendered), "duplicate statements across frames"


def frame_names(frame, names_per_frame, seed):
    """Pick the agent names one frame is instantiated with; [None] for fixed frames.

    Instantiating a frame with several names is the moral analog of the entity
    substitution in cities: it multiplies dataset size along an axis that provably
    does not carry the label, without touching the action phrase. Names are sampled
    per frame rather than shared across the dataset, so no name is frequent enough to
    become a confound.

    The rng is seeded on a hash of the frame's own text, not on its position in the
    list. Position would mean that inserting or reordering a frame reshuffles the
    names of every frame after it — silently changing which sentence each frame was
    rated on, and making rating sheets from before and after the edit incomparable.
    Keying on the text means a frame's names change only when its wording does, which
    is exactly when it needs re-rating anyway.
    """
    if frame.gender == 'fixed':
        return [None]
    pool = NAMES[frame.gender]
    assert names_per_frame <= len(pool), "names_per_frame exceeds pool size"
    digest = hashlib.sha256(frame.good.encode('utf-8')).hexdigest()
    rng = random.Random(seed * 100003 + int(digest[:16], 16))
    return rng.sample(pool, names_per_frame)


def write_dataset(dataset_name, rows, frames, seed):
    """Write datasets/<name>.csv plus its rating sheet; return the dataframe.

    The csv has columns statement,label,pair_id,frame_id.

    The rating sheet, experiments/label_stability/<name>/{rating_sheet.csv,
    rating_prompt.txt}, holds one instantiation of every frame — the other
    name-variants would cost the rater 8x for the same judgment, since the name does
    not bear the label. Every frame is rated on every round rather than only the ones
    new since the last round: it keeps one dataset's evidence homogeneous (same
    sheet, same neighbours, same raters) and it guarantees that an edited frame is
    re-rated instead of coasting on a verdict its old wording earned.

    The sheet is shuffled so that a frame's good and bad statement are not adjacent
    and the good/bad alternation of the source order is broken: the prompt asks for
    each statement to be judged on its own, and a rater shown the minimal pair side
    by side rates contrastively, which is the confound this test exists to detect.
    """
    df = pd.DataFrame(rows)
    assert df['statement'].is_unique, "duplicate statements after name substitution"
    out_csv = os.path.join(DATASETS_DIR, f'{dataset_name}.csv')
    df.to_csv(out_csv, index=False)

    # Rating materials: the first instantiation of every frame.
    rating = df.groupby('frame_id').head(2)[['statement']]  # first pair per frame
    rating_dir = os.path.join(RATING_ROOT, dataset_name)
    os.makedirs(rating_dir, exist_ok=True)
    rating = rating.sample(frac=1, random_state=seed).reset_index(drop=True)
    rating.insert(0, 'statement_id', rating.index)
    rating['rating'] = ''
    rating['reason'] = ''
    sheet_csv = os.path.join(rating_dir, 'rating_sheet.csv')
    rating.to_csv(sheet_csv, index=False)
    with open(os.path.join(rating_dir, 'rating_prompt.txt'), 'w') as f:
        f.write(RATING_PROMPT)
        for _, row in rating.iterrows():
            f.write(f"{row['statement_id']}: {row['statement']}\n")

    good_lens = df[df.label == 1]['statement'].str.split().str.len()
    bad_lens = df[df.label == 0]['statement'].str.split().str.len()
    print(f"[{dataset_name}] wrote {len(df)} statements "
          f"({len(df) // 2} pairs, {len(frames)} frames) to {out_csv}")
    print(f"[{dataset_name}] rating sheet: {len(rating)} statements "
          f"({len(frames)} frames) in {sheet_csv}")
    print(f"[{dataset_name}] mean words: good {good_lens.mean():.1f}, bad {bad_lens.mean():.1f}")
    return df


def build(dataset_name, frames, names_per_frame=8, seed=0, write_negations=True):
    """Validate frames, write datasets/<name>.csv and neg_<name>.csv, print stats.

    write_negations=False suppresses the negated file and its rating sheet (the name
    avoids shadowing the module-level negate()). The frames keep their
    neg_good/neg_bad and are still validated, so this is a shipping decision, not a
    deletion: flip it back to True and re-run to get the set again. Only care_harm
    currently ships a negation — verifying a negated set costs its own rating round, and
    the other three foundations are deferred until the care_harm result says whether the
    negated sets earn their keep.

    neg_<name> is built only from the frames that supply negations, so it covers a
    subset of the frames and the two files are not row-aligned. Refusal flips moral
    valence only where refusing the good act is blameworthy in its own right — a role
    duty, or serious harm at trivial cost — and a frame whose good act is a
    discretionary kindness keeps its pair here while opting out of the negation (see
    validate). pair_id and frame_id are carried over from the source frame, so
    frame_id has gaps in the negated file; nothing downstream requires them dense.

    neg_<name> is the moral analog of neg_cities and serves as the lexical-shortcut
    control. "Refused to hurl her elderly neighbor's groceries down the apartment
    stairs" carries label 1 while still containing the harm verb, so a probe that has
    merely learned the valence of "hurled" scores at or below chance on it. Train on
    <name> and test on neg_<name>, exactly as the original work trains on cities and
    tests on neg_cities.

    Moral valence has no truth-functional negation the way truth does, but the per-frame
    opt-in above is what handles that: the frames whose refusal would land near neutral
    are exactly the ones filtered out. In the file that ships, both labels are genuine
    verdicts — label 0 is refusing a duty (blameworthy), label 1 is refusing to harm
    (praiseworthy) — so neg_ accuracy can be read as accuracy. Report the signed
    projection onto the probe direction alongside it anyway, since that is what
    separates "the direction transfers" from "the probe was riding the action verb": a
    confidently negative projection means the latter.

    frame_id groups the name-variants of one frame. Train/val splits must group by it
    (utils.DataManager does when the column is present) or accuracy is inflated by
    near-duplicate statements straddling the split.
    """
    validate(frames)
    neg_frames = [f for f in frames if f.neg_good is not None]

    rows, neg_rows = [], []
    pair_id = 0
    for frame_id, frame in enumerate(frames):
        for name in frame_names(frame, names_per_frame, seed):
            good = frame.good if name is None else frame.good.format(name=name)
            bad = frame.bad if name is None else frame.bad.format(name=name)
            rows.append({'statement': good, 'label': 1, 'pair_id': pair_id, 'frame_id': frame_id})
            rows.append({'statement': bad, 'label': 0, 'pair_id': pair_id, 'frame_id': frame_id})
            if frame.neg_good is not None:
                # Labels flip: refusing the good act is blameworthy, refusing the bad
                # act is praiseworthy. Only frames that opt in appear here.
                neg_rows.append({'statement': negate(good, frame.neg_good), 'label': 0,
                                 'pair_id': pair_id, 'frame_id': frame_id})
                neg_rows.append({'statement': negate(bad, frame.neg_bad), 'label': 1,
                                 'pair_id': pair_id, 'frame_id': frame_id})
            pair_id += 1

    df = write_dataset(dataset_name, rows, frames, seed)
    shared = [shared_final_words(f.good, f.bad) for f in frames]
    print(f"[{dataset_name}] shared final words per frame: min {min(shared)}, "
          f"median {sorted(shared)[len(shared) // 2]}")
    if neg_frames and not write_negations:
        print(f"[{dataset_name}] negation suppressed (write_negations=False); {len(neg_frames)} "
              f"frames carry negations but no neg_{dataset_name}.csv is written")
    elif neg_frames:
        print(f"[{dataset_name}] {len(neg_frames)}/{len(frames)} frames opted in to "
              f"negation; {len(frames) - len(neg_frames)} declined (refusing the good "
              f"act is not blameworthy)")
        # The negated set gets its own sheet: the refusal construction is the part of
        # this design most likely to read as neutral to a rater, so it is rated on its
        # own wording rather than inheriting its source's verdict.
        write_dataset(f'neg_{dataset_name}', neg_rows, neg_frames, seed)
    return df
