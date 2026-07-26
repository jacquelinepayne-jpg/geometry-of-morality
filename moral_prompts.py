"""
Shared few-shot prompt machinery for the moral readout.

The fork inherits three scripts whose prompts are true/false shaped — `patching.py`,
`few_shot.py`, `interventions.py`. They need to agree on the readout, or their numbers
are not comparable, so the wording lives here rather than in each script.

Readout. `<statement> This action is: GOOD` / ` BAD`, rather than the original's
`This statement is: TRUE` / ` FALSE`. "Statement" is wrong for moral valence: the
statements in `care_harm` are all *true* descriptions of an act, and the question is
whether the act was good, so asking about the statement invites the model to answer the
truth question instead. GOOD/BAD is also what the causal step in `interventions.py` needs
to score, per the project's migration note.

Token-length matching. The patching experiment requires the two prompts to tokenize to the
same length, since it patches by position index. Which of the 276 `care_harm` pairs satisfy
this depends on the tokenizer, so it is discovered at run time by
`token_matched_pairs()` rather than hardcoded.
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

READOUT = 'This action is:'
GOOD, BAD = ' GOOD', ' BAD'


def load_dataset(dataset_name):
    return pd.read_csv(os.path.join(ROOT, 'datasets', f'{dataset_name}.csv'))


def label_word(label):
    return GOOD.strip() if label == 1 else BAD.strip()


def readout_tokens(tokenizer):
    """The two token ids the metric is read off. Matches the original's `[-1]` convention,
    which takes the final token of the encoding to sidestep the leading-space handling."""
    return tokenizer(GOOD).input_ids[-1], tokenizer(BAD).input_ids[-1]


def few_shot_prefix(df, n_shots=6, seed=0, exclude_frames=()):
    """A balanced GOOD/BAD few-shot prefix, drawn from frames other than the one under test
    so the model is not primed with the target scenario. Deterministic given `seed`.

    Two things the prefix deliberately avoids, both of which would let the model answer
    without representing valence at all:

    * **No minimal pairs.** Each frame contributes at most one shot, so the prefix never
      shows the same scenario labelled both ways. Showing the pair would tell the model in
      context that the task is "spot the changed verb" — the lexical shortcut the whole
      design is trying not to measure.
    * **No positional pattern.** The labels are shuffled rather than alternated or blocked,
      so the next label is not predictable from its slot.

    Returns the prefix string, ending in a newline.
    """
    if n_shots % 2:
        raise ValueError(f'n_shots should be even so the prefix is label-balanced, got {n_shots}')
    pool = df[~df['frame_id'].isin(exclude_frames)]
    # one row per frame per label, so no two shots are name-variants of one another
    pool = pool.groupby(['frame_id', 'label'], as_index=False).first()

    frames = pool['frame_id'].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    if len(frames) < n_shots:
        raise ValueError(f'need {n_shots} distinct frames outside {set(exclude_frames)}, '
                         f'only {len(frames)} available')
    # disjoint frames per label, so no frame appears as both a good and a bad shot
    good_frames, bad_frames = frames[:n_shots // 2], frames[n_shots // 2:n_shots]

    shots = [pool[(pool['frame_id'] == f) & (pool['label'] == label)].iloc[0]
             for label, side in ((1, good_frames), (0, bad_frames)) for f in side]

    # shuffle, rejecting the two degenerate orders a shuffle can land on by chance: fully
    # alternating (the next label is predictable from the slot) and fully blocked (all the
    # good shots then all the bad ones)
    for offset in range(1, 100):
        # seed * 1000 keeps the offset ranges of different seeds from colliding, so a seed
        # sweep actually varies the order rather than converging on one accepted permutation
        order = pd.Series(range(n_shots)).sample(frac=1, random_state=seed * 1000 + offset).tolist()
        labels = [shots[i]['label'] for i in order]
        alternating = all(a != b for a, b in zip(labels, labels[1:]))
        blocked = labels == sorted(labels) or labels == sorted(labels, reverse=True)
        if not alternating and not blocked:
            break
    shots = [shots[i] for i in order]
    return ''.join(f'{s.statement} {READOUT}{GOOD if s.label == 1 else BAD}\n' for s in shots)


def token_matched_pairs(tokenizer, df):
    """Pair ids whose good and bad statements tokenize to the same number of tokens, which is
    what the position-indexed patching sweep requires. Returns a list of
    (pair_id, n_tokens, n_differing_trailing_tokens), shortest differing span first — a
    shorter span means fewer forward passes and a tighter localization."""
    out = []
    for pair_id, group in df.groupby('pair_id'):
        if set(group['label']) != {0, 1}:
            continue
        good = group[group['label'] == 1]['statement'].iloc[0]
        bad = group[group['label'] == 0]['statement'].iloc[0]
        good_toks = tokenizer(f'{good} {READOUT}').input_ids
        bad_toks = tokenizer(f'{bad} {READOUT}').input_ids
        if len(good_toks) != len(bad_toks):
            continue
        sames = [g == b for g, b in zip(good_toks, bad_toks)]
        if all(sames):
            continue  # identical statements, nothing to patch
        n_differing = sames[::-1].index(False) + 1
        out.append((int(pair_id), len(good_toks), n_differing))
    return sorted(out, key=lambda r: r[2])


def patching_prompts(df, pair_id, n_shots=6, seed=0):
    """The matched (good_prompt, bad_prompt) for one pair: identical few-shot prefix, then the
    good or bad member of the pair, ending on the readout so the next token is the verdict."""
    group = df[df['pair_id'] == pair_id]
    if set(group['label']) != {0, 1}:
        raise ValueError(f'pair {pair_id} does not have both a good and a bad member')
    frame_id = group['frame_id'].iloc[0]
    prefix = few_shot_prefix(df, n_shots=n_shots, seed=seed, exclude_frames={frame_id})
    good = group[group['label'] == 1]['statement'].iloc[0]
    bad = group[group['label'] == 0]['statement'].iloc[0]
    return (f'{prefix}{good} {READOUT}', f'{prefix}{bad} {READOUT}')
