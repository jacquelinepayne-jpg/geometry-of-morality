"""
Collect model rater passes for the label-stability test via OpenRouter.

Sends a dataset's rating_prompt.txt verbatim to each model rater, three passes each,
and writes the returned CSV to experiments/label_stability/<dataset>/rated/ under the
filenames tally.py expects: rating_sheet_<rater>.csv for pass 1 and
rating_sheet_<rater>-<n>.csv for later passes (matching the pilot's naming, so each
pass shows up as its own rater column in the tally). Model passes live in rated/ to
keep them separate from hand-filled sheets; tally.py reads both locations.

The prompt is the record of what was asked, so it is sent unmodified and with no
system message. Passes are meant to be independent samples of the same prompt, so
temperature stays at 1.0 by default; dropping it to 0 makes the repeat passes
near-duplicates and the unanimity number meaningless.

Requires OPENROUTER_API_KEY in the environment. Existing sheets are never
overwritten without --overwrite: the completed sheets are the experimental record.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python experiments/label_stability/rate.py --dataset care_harm neg_care_harm
  python experiments/label_stability/rate.py --dataset care_harm --raters opus_5 --passes 1

Then tally as usual:
  python experiments/label_stability/tally.py --dataset care_harm
"""

import argparse
import io
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request

import pandas as pd

from tally import normalize_rating  # same rating vocabulary the tally accepts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# rater key -> OpenRouter model slug. The key becomes the rater name in the sheet
# filename and so in the tally, so keep it short and versioned.
RATERS = {
    'opus_5': 'anthropic/claude-opus-5',
    'gemini_3.6_flash': 'google/gemini-3.6-flash',
    'gpt_5.6_luna': 'openai/gpt-5.6-luna',
}

N_PASSES = 3
TEMPERATURE = 1.0
MAX_TOKENS = 32000  # a full sheet is ~82 rows; reasoning models spend tokens first
N_ATTEMPTS = 4      # per pass, covering rate limits and unparseable replies
FENCE_RE = re.compile(r'```(?:csv)?\s*\n(.*?)```', re.S)


class RatingError(Exception):
    """A pass that came back unusable and is worth retrying."""


def sheet_path(dataset, rater, pass_idx):
    suffix = '' if pass_idx == 1 else f'-{pass_idx}'
    return os.path.join(HERE, dataset, 'rated', f'rating_sheet_{rater}{suffix}.csv')


def canonical(text):
    """Normalize away the differences that are not the rater's fault."""
    text = unicodedata.normalize('NFKC', str(text))
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"')
    return ' '.join(text.split())


def call_openrouter(model, prompt, temperature, api_key):
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens': MAX_TOKENS,
    }).encode()
    request = urllib.request.Request(API_URL, data=body, headers={
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    })
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as e:
        raise RatingError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
    except (urllib.error.URLError, TimeoutError) as e:
        raise RatingError(f"request failed: {e}")

    if 'error' in payload:  # OpenRouter can return 200 with an error body
        raise RatingError(f"api error: {str(payload['error'])[:500]}")
    choice = payload['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise RatingError(f"reply hit the {MAX_TOKENS}-token cap and is truncated")
    usage = payload.get('usage', {})
    print(f"    tokens: prompt {usage.get('prompt_tokens', '?')}, "
          f"completion {usage.get('completion_tokens', '?')}")
    return choice['message']['content'] or ''


def parse_ratings(text, current):
    """Pull the CSV out of a reply and check it against the dataset's blank sheet."""
    match = FENCE_RE.search(text)
    body = match.group(1) if match else text
    try:
        df = pd.read_csv(io.StringIO(body.strip()))
    except Exception as e:
        raise RatingError(f"reply is not parseable CSV ({e}); reply began: "
                          f"{text.strip()[:300]!r}")

    missing_cols = {'statement_id', 'statement', 'rating'} - set(df.columns)
    if missing_cols:
        raise RatingError(f"reply is missing column(s) {sorted(missing_cols)}; "
                          f"got {list(df.columns)}")
    if 'reason' not in df.columns:
        df['reason'] = ''
    df = df[['statement_id', 'statement', 'rating', 'reason']]

    df['statement_id'] = pd.to_numeric(df['statement_id'], errors='coerce')
    if df['statement_id'].isna().any():
        raise RatingError("reply has non-numeric statement_id values")
    df['statement_id'] = df['statement_id'].astype(int)

    expected, got = set(current['statement_id']), set(df['statement_id'])
    if expected != got:
        raise RatingError(f"statement_ids do not match the sheet: missing "
                          f"{sorted(expected - got)}, unexpected {sorted(got - expected)}")
    if df['statement_id'].duplicated().any():
        dupes = df.loc[df['statement_id'].duplicated(), 'statement_id'].tolist()
        raise RatingError(f"duplicate statement_id rows: {sorted(set(dupes))}")

    unrecognized = df.loc[df['rating'].map(normalize_rating).isna(), 'rating']
    if len(unrecognized) > 0:
        raise RatingError(f"unrecognized rating value(s): "
                          f"{sorted(set(unrecognized.astype(str)))}")

    df = df.sort_values('statement_id').reset_index(drop=True)
    df['reason'] = df['reason'].fillna('')

    # The statement text is left exactly as the rater returned it: tally.py compares
    # it against the current sheet to catch a pass that rated an older wording. A
    # mismatch here is a live problem, so surface it now rather than at tally time.
    text_by_id = dict(zip(current['statement_id'], current['statement']))
    drifted = [sid for sid, stmt in zip(df['statement_id'], df['statement'])
               if canonical(stmt) != canonical(text_by_id[sid])]
    if drifted:
        print(f"    WARNING: returned statement text differs from the sheet for ids "
              f"{drifted} (rater altered the statement; check before tallying)")
    return df


def run_pass(dataset, rater, model, pass_idx, prompt, current, args, api_key):
    out_csv = sheet_path(dataset, rater, pass_idx)
    label = f"{rater} pass {pass_idx}/{args.passes}"
    if os.path.exists(out_csv) and not args.overwrite:
        print(f"  {label}: {os.path.basename(out_csv)} exists, skipping "
              f"(--overwrite to replace)")
        return True
    if args.dry_run:
        print(f"  {label}: would call {model} -> {os.path.basename(out_csv)}")
        return True

    print(f"  {label}: calling {model}")
    for attempt in range(1, N_ATTEMPTS + 1):
        try:
            reply = call_openrouter(model, prompt, args.temperature, api_key)
            df = parse_ratings(reply, current)
        except RatingError as e:
            print(f"    attempt {attempt}/{N_ATTEMPTS} failed: {e}")
            if attempt < N_ATTEMPTS:
                time.sleep(2 ** attempt)
            continue
        df.to_csv(out_csv, index=False)
        counts = df['rating'].map(normalize_rating).value_counts()
        print(f"    wrote {out_csv} (good {counts.get('good', 0)}, "
              f"bad {counts.get('bad', 0)}, depends {counts.get('depends', 0)})")
        return True
    print(f"    {label}: giving up after {N_ATTEMPTS} attempts")
    return False


def rate_dataset(dataset, args, api_key):
    sheet_dir = os.path.join(HERE, dataset)
    prompt_txt = os.path.join(sheet_dir, 'rating_prompt.txt')
    sheet_csv = os.path.join(sheet_dir, 'rating_sheet.csv')
    for path in (prompt_txt, sheet_csv):
        if not os.path.exists(path):
            raise SystemExit(f"no such file: {path} (run the dataset's data_gen.py first)")

    os.makedirs(os.path.join(sheet_dir, 'rated'), exist_ok=True)
    with open(prompt_txt) as f:
        prompt = f.read()
    current = pd.read_csv(sheet_csv)

    print(f"\n######## {dataset} ({len(current)} statements) ########")
    failures = []
    for rater in args.raters:
        for pass_idx in range(1, args.passes + 1):
            if not run_pass(dataset, rater, RATERS[rater], pass_idx, prompt,
                            current, args, api_key):
                failures.append(f"{dataset}/{rater} pass {pass_idx}")
    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', nargs='+', required=True,
                        help='dataset name(s), e.g. care_harm neg_care_harm')
    parser.add_argument('--raters', nargs='+', default=list(RATERS),
                        choices=list(RATERS), help='which model raters to run')
    parser.add_argument('--passes', type=int, default=N_PASSES,
                        help=f'passes per rater (default {N_PASSES})')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE,
                        help=f'sampling temperature (default {TEMPERATURE}; keep it '
                             f'nonzero so the repeat passes are independent samples)')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace sheets that already exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the calls that would be made and exit')
    args = parser.parse_args()

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key and not args.dry_run:
        raise SystemExit('OPENROUTER_API_KEY is not set')

    failures = []
    for name in args.dataset:
        failures += rate_dataset(name, args, api_key)
    if failures:
        raise SystemExit('\nfailed passes (rerun to retry):\n  ' + '\n  '.join(failures))
    print('\nall passes done; tally with '
          f"python experiments/label_stability/tally.py --dataset {' '.join(args.dataset)}")
