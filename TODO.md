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
  The builders read CSV inputs with pandas' default float parser, which is not
  exact and differs by machine: of the 1,492 float values in
  `docs/design/replay_detection_comparison/detection_sweep_summary.csv`, 125 on
  an arm64 laptop differ in the last digit from the exact parse
  (`float_precision="round_trip"`), and the cluster (x86_64) reads one of the
  detection summary's values differently from the laptop. So a summary is
  bit-identical only when rebuilt on the same kind of machine. Reading with the
  round-trip parser would make it exact everywhere and change those values in
  their last digit.
  *When:* together with Tier 1, or when a number in the paper cannot be traced
  back to a commit.

## Reported values

- [ ] **Scripts and figures that compute a reported statistic themselves.** Each
  summary's statistics are functions in `analysis/` (`docs/development.md`,
  "Reported numbers"). Figure 2's panels b and d and Figure 4d call them, and the
  `scripts/pfc/` reports take a statistic whole from them or compute their own.
  These still hold their own copy; make each call the function, comparing its
  output before and after on the cluster:
  - detection: `scripts/figures/generate_memoryless_comparison.py` (the timing by
    duration, the paired part changes, the carving mean, the held-out rate, the
    jump-pass shares and the three refusals), against
    `memoryless_sequence/sweep_statistics.py`.
  - behavior: `scripts/qa/compare_fits.py` (AIC differences, the lag-1 kernel and
    its paired test, the clock shapes, separation shares, arrival-window shares,
    the animal-balanced summaries), `scripts/rate/plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`,
    `plot_model_checking.py` and `fit_per_bin_rate_models.py` (the arrival-window
    shares), `scripts/behavior/run_stay_leave.py` (the pooled GEE's correction),
    `scripts/coupling/fit_swr_power_model.py` and
    `scripts/content/fit_swr_contrast_content.py` (animals positive, signed-rank
    p), against `analysis/replay/coefficient_statistics.py`,
    `rate_fit_statistics.py`, `stop_statistics.py` and `learning.py`.
    `contrast_aggregation.aggregate_contrast` repeats the animal-median and
    sign-count pattern of `session_median_statistics`.

  *When:* at each figure's or report's next regeneration.

- [ ] **Decide the open choices in the reported statistics.**
  `docs/known_issues/reported_statistics_choices.md` lists them: where each is
  made, what the code does, what is open, and what was measured on the current
  data. They were found while the statistics moved into `analysis/` and in the
  review of that code (`.claude/docs/reviews/2026-11-11-reported-numbers-review.md`).
  Write each decision in that file.
  *When:* before the Results that cite the value are drafted.

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
