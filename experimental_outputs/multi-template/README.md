# Provenance

`patching_results.json` here was ported from the `animal-vs-human-results`
branch and computed on an **earlier data generation**:

- **20 subjects per category** (80 total), not the current 40 per category
  (160). The CSVs in `datasets/multi-template/` are the regenerated 40-subject
  versions, so these results do not correspond to them row for row.
- The **flat dataset layout** that predates the `truth/`, `single-template/`,
  and `multi-template/` groupings.

Re-run `patching_multi_template.py` against the current datasets before
comparing these numbers against anything else.
