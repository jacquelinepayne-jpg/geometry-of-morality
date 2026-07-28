"""Generate the helping-preference baseline dataset for logprobs.py.

Each statement is a harm scenario followed by a subject-free continuation
(help or dismiss). Runs through logprobs.py unchanged:

    python logprobs.py --model llama-2-13b --dataset help_continuations --device cuda:0

Analysis: for each (template_id, subject), the helping preference is
logprob(help) - logprob(dismiss), averaged over continuations — the shared
scenario prefix cancels in the difference. Compare across subject_category.

Run from the repo root or from datasets/:

    python datasets/make_help_continuations.py
"""

import os

import pandas as pd

from animal_human_templates import HARM_TEMPLATES, HELP_CONTINUATIONS, SUBJECTS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 3 subjects per category for the cheap pilot run. (rabbit / boar / buffalo
# were dropped from SUBJECTS as cross-category ambiguous.)
PILOT_SUBJECTS = [
    "man", "woman", "child",
    "dog", "cat", "hamster",
    "pig", "cow", "chicken",
    "rat", "fox", "deer",
]

if __name__ == "__main__":
    rows = []
    for template in HARM_TEMPLATES:
        for subject, info in SUBJECTS.items():
            scenario = template["text"].format(subject=info["phrase"])
            for cont in HELP_CONTINUATIONS:
                rows.append({
                    "statement": f"{scenario} {cont['text']}",
                    # label 1 = help, 0 = dismiss, so the CSV stays compatible
                    # with DataManager if probing on these is ever wanted
                    "label": int(cont["type"] == "help"),
                    "subject": subject,
                    "subject_category": info["category"],
                    "template_id": template["template_id"],
                    "continuation_id": cont["continuation_id"],
                    "continuation_type": cont["type"],
                })
    df = pd.DataFrame(rows)
    assert df.statement.is_unique

    pilot = df[df.subject.isin(PILOT_SUBJECTS)]
    for name, frame in [("help_continuations", df), ("help_continuations_pilot", pilot)]:
        frame.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
        n_scen = frame.template_id.nunique() * frame.subject.nunique()
        print(f"{name}.csv: {len(frame)} rows "
              f"({n_scen} scenarios x {len(HELP_CONTINUATIONS)} continuations)")
