# Publication follow-ups

Work that is deferred on purpose. Each item says when it should be done.

## Paper metadata

- [ ] Replace the working title, and confirm the tentative author list and each
  author's affiliations in `CITATION.cff` and `paper/main.tex`.
  *When:* before the first draft goes to the co-authors.
- [ ] Choose the target journal and switch `paper/main.tex` to its document
  class and bibliography style.
  *When:* once the journal is chosen.

## Reproduction

- [ ] **Public reproduction from archived tables (Tier 1).** Archive the
  canonical tables and per-epoch derived files that the part summaries read
  (`paper/reported/part{n}.json`, `inputs`) on Zenodo, with a download script
  that checks each file's SHA-256, so the figures and summaries can be rebuilt
  without the lab's data. Fitted-model pickles are not archival inputs: the
  public path should refit the models from the tables.
  *When:* before submission, or when an outside reader needs to rerun the
  analyses.
- [ ] **Lab pipeline from NWB (Tier 2).** Document the decode, detection and
  table-building steps from Spyglass/NWB as a lab guide, and publish the NWB
  files on DANDI.
  *When:* with the data statement for submission.
- [ ] **Summary provenance.** Make `scripts/reported/summarize_part.py` refuse
  a dirty working tree and record input paths relative to the data root, not
  absolute paths (`part2.json` records `/cumulus/...`; `part3.json` and
  `part4.json` were built at `-dirty` commits). Add a script that compares a
  rebuilt summary with the committed one within stated tolerances.
  *When:* together with Tier 1, or when a number in the paper cannot be traced
  back to a commit.

## Reported values

- [ ] **Held-out test split by SWR (claim 7): confirm which script made it.**
  `part2.json`'s 10 `heldout_*` values are read from
  `Processed-Data/_refresh_20261016/state_evidence/nl_bouts_by_swr_stats.csv`.
  - The Part 2 builder says the tracked
    `scripts/detection/split_state_evidence_by_swr.py` writes that file. The
    script was added in `e78273b5` (2026-10-30).
  - The claims table says an untracked research script made them and that they
    are pending (`paper/claims_table.md:49`, `:112`, `:119–120`).
  - On the cluster, read `nl_bouts_by_swr.provenance.json` beside the CSV: its
    commit and script. Then mark the values pending in
    `reported/part2.py:heldout_by_swr_values`, or correct the claims table.

  *When:* at the next session on the cluster, before the Part 2 rebuild.

## Checks

- [ ] **Type checking.** `mypy src/` reports about 400 errors in 90 files.
  Fix them, then add mypy to CI and pre-commit.
  *When:* as a separate cleanup, before mypy is relied on.
- [ ] **Pin GitHub Actions to commit SHAs.** The workflows use version tags
  (`@v4`), which the action's owner can move.
  *When:* before the repository is made public, or before a workflow gets a
  secret or deploys anything.
