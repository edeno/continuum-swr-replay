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
  (`paper/reported/<summary>.json`, `inputs`) on Zenodo, with a download script
  that checks each file's SHA-256, so the figures and summaries can be rebuilt
  without the lab's data. Fitted-model pickles are not archival inputs: the
  public path should refit the models from the tables.
  *When:* before submission, or when an outside reader needs to rerun the
  analyses.
- [ ] **Lab pipeline from NWB (Tier 2).** Document the decode, detection and
  table-building steps from Spyglass/NWB as a lab guide, and publish the NWB
  files on DANDI.
  *When:* with the data statement for submission.
- [ ] **Summary input paths and rebuild comparison.** Record input paths
  relative to the data root, not absolute paths (a build whose data root is
  outside the checkout records `/cumulus/...`; the current summaries were built in
  a checkout that links `Processed-Data`, so theirs are relative). Add a script that compares a rebuilt summary with the
  committed one within stated tolerances. Record each input file set's glob, not
  a label, so `make release-check` can re-check sets too (today it re-checks only
  single files). (Each build records the code that built it; a summary in
  `paper/reported/` is written only from committed code, and `make release-check`
  flags a stale one.)
  *When:* together with Tier 1, or when a number in the paper cannot be traced
  back to a commit.

## Reported values

- [ ] **Statistics out of the detection and behavior builders.** The events and
  mpfc summaries' statistics are functions in `analysis/`
  (`hippocampus/event_statistics.py`, `inventory_comparison/nonlocal_comparison.py`,
  `pfc/population_statistics.py`, `pfc/timing.py`) and their builders only name
  and describe them. The detection and behavior builders still compute what they
  report. Move each the same way, checking that every value is unchanged by a
  trial rebuild on the cluster.
  *When:* before the Results that cite them are drafted.

- [ ] **Figure 4 and three pfc scripts recompute statistics the mpfc summary
  reports.** `generate_figure04.py` computes the called shares and the
  best-model shares itself (as percents); `scripts/pfc/compare_pfc_cohorts.py`
  (cohort totals, prior-binding share) and
  `scripts/pfc/measure_pfc_null_geometry.py` (the null-geometry numbers) compute
  theirs. Call `analysis/pfc/population_statistics.py` instead.
  *When:* at the next regeneration of Figure 4.

- [ ] **Two faults in the mpfc summary's text, found while moving its
  statistics; the numbers are not affected.** (1) Five timing descriptions read
  "onset is rise_size": in `reported/mpfc.py:timing_values` the variable holding
  the half-maximum rule's wording is reused as a loop variable, so later
  descriptions name the loop's key instead of the rule. (2) The extended tree's
  two input records name `swr_without_replay/swr_without_replay/…`: the label
  passed to `files_record` repeats the folder.
  *When:* at the next rebuild of the mpfc summary.

- [ ] **The events summary reads two inputs from `paper/figures/`.**
  `reported/events.py` reads `paper/figures/reality_all/per_epoch/` (the
  per-epoch spike-evidence metrics) and
  `paper/figures/content_occupancy/content_occupancy_per_session.csv` from the
  checkout. Both folders are ignored by git, so a fresh checkout or worktree does
  not have them and the build fails until they are linked in (the 2026-11-11
  rebuild linked them from the main cluster checkout). Write them under the
  processed-data root and read them from there.
  No script in the repository makes the per-epoch layout the builder reads. The
  current spike-evidence run was launched on the cluster on 2026-10-30, at commit
  `bfb20f35`, by `logs/reality_stored_20261030/run_one.sh` (in the main cluster
  checkout's `logs/`, not in git): for each analysed epoch,
  `scripts/detection/analyze_event_spike_evidence.py --epoch-identifiers <epoch>
  --n-sample 1000000 --likelihood stored --cache-dir
  logs/f1_reality_marks_cache_20261016 --output-dir
  paper/figures/reality_all/per_epoch/<epoch>`, with
  `$CSR_PROCESSED_DATA_DIR` set to the main checkout's `Processed-Data`. Put that
  loop in the repository when the run moves.
  *When:* with the move of the events statistics out of the builder.

- [ ] **Held-out test split by SWR (question 2c.3): confirm which script made it.**
  `events.json`'s 10 `heldout_*` values are read from
  `Processed-Data/_refresh_20261016/state_evidence/nl_bouts_by_swr_stats.csv`.
  - The events builder says the tracked
    `scripts/detection/split_state_evidence_by_swr.py` writes that file. The
    script was added in `e78273b5` (2026-10-30).
  - The claims table says an untracked research script made them and that they
    are pending (`paper/claims_table.md:49`, `:112`, `:119–120` at `c74e56b8`;
    the claims table is retired).
  - On the cluster, read `nl_bouts_by_swr.provenance.json` beside the CSV: its
    commit and script. If the tracked script did not make it, mark the values
    pending in `reported/events.py:heldout_by_swr_values`; if it did, say so in
    question 2c.3's note.

  *When:* at the next session on the cluster, before the events summary is rebuilt.

- [ ] **Open decisions from the retired figure plan** (`paper/figure_claims_plan.md`,
  removed after `cad31719`; it used an older claim numbering, so "D" is its claim
  "our method beats SWR-triggered decoding"). Quoted as it wrote them; confirm
  each is still open, or close it:
  - "Close vs remote: scope of headline claims (default remote-clean)."
  - "Whether D needs a literal traditional-pipeline re-implementation or a
    derived contingency suffices."
  - "Does Fig 3d's 'reward zones' survive the remote-subset + autocorrelation
    controls, or get reframed as 'recent/valued locations'?"

  *When:* when Figures 2 and 3 are designed.

- [ ] **Open analyses from the retired notes.** Quoted from
  `paper/reported/overview.md`, `docs/replay_rate_content_swr_synthesis.md` and
  `docs/per_bin_rate_results.md` (retired; read them at `c74e56b8`). Confirm each,
  or close it:
  - Tests at the recording-day level: a p or q value over sessions treats the
    epochs of one recording day as independent, which they are not; until then,
    the animal counts are the level to read a claim at.
  - Property contrasts (duration, extent, speed, ...) are plain medians unless a
    value says matched duration; non-SWR events are shorter than SWR events.
  - The behavioral-state covariates on rate (value, uncertainty, switch rate,
    block): the `MU_uncertainty` and `is_swr`-contrast rate ladders report fit
    counts and separation shares, not coefficients; any re-derived coefficient
    must state its excluded (separated) share.
  - The `Mvu` conditioning diagnostics were count-response fits; re-derive them on
    the per-event fraction response.
  - The per-arm value-update (realized gain) re-run under the beta covariates has
    not been done.
  - The content arm hierarchy has no held-out cross-validation (model comparison
    rests on ΔAIC on the quasi-likelihood scale).
  - The choice fit (question 3.4) has no provenance sidecar.
  - Not built: the full marked point process (joint intensity, coupling and
    content); a graded ripple-power content rung.
  - Hypotheses that need a test: a continuum rather than two mechanisms (needs
    the peri-onset peak distribution and the joint model); reward shifting SWR
    coupling continuously; value-gating of with-SWR rate.

  *When:* before the Results are drafted.

## Checks

- [ ] **Type checking.** `mypy src/` reports about 400 errors in 90 files.
  Fix them, then add mypy to CI and pre-commit.
  *When:* as a separate cleanup, before mypy is relied on.
- [ ] **Pin GitHub Actions to commit SHAs.** The workflows use version tags
  (`@v4`), which the action's owner can move.
  *When:* before the repository is made public, or before a workflow gets a
  secret or deploys anything.
