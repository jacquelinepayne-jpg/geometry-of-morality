# Multi-template results

Patching results from the pooled (all-templates-in-one-file) animal-vs-human
datasets, ported from the `animal-vs-human-results` branch. The matching
figures are in `dataexplorer/plots/multi-template/`.

`patching_results.json` here is the **zero-shot YES/NO** readout — "Should the
rescuers make saving this individual their top priority?" — patching a human
prompt into an otherwise identical animal prompt. It is a different experiment
from the root `experimental_outputs/patching_results.json`, which is the 5-shot
HUMAN/ANIMAL category readout. Neither supersedes the other.

**Provenance.** These numbers were computed on the earlier data generation:

- **20 subjects per category** (80 total), not the current 40 per category
  (160). `datasets/multi-template/*.csv` on this branch are the regenerated
  40-subject versions, so these results do not correspond to them row for row.
- The **flat dataset layout** that predates the `truth/`, `single-template/`,
  and `multi-template/` groupings, so `dataset` fields carry bare names
  (`human_farmed_harm`, not `multi-template/human_farmed_harm`).

Treat these as a record of the earlier run rather than a baseline to diff
against new results; re-run against the current datasets before comparing.
