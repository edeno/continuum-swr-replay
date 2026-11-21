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
  `paper/reported/inputs/detection/detection_sweep_summary.csv`, 125 on
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
  "Reported numbers"). Figure 2's panels b and d and Figure 5d call them, and the
  `scripts/pfc/` reports take the summary's statistics from them (the
  marked-family summary's are `analysis/pfc/marked_family_statistics.py`'s); their
  audit-only tables are still computed in the scripts.
  These still hold their own copy; make each call the function, comparing its
  output before and after on the cluster:
  - events: `scripts/event_evidence/analyze_event_spike_evidence.py` (per-epoch and
    per-animal medians written to CSVs nothing reads).

  *When:* at each figure's or report's next regeneration.

- [x] **Figure 5c: pooled or animal-balanced.** *Decided 2026-11-12: animal-balanced*,
  to match how the paper summarizes its other statistics. Panel c
  (`scripts/figures/generate_figure05.py`, `plot_population_average`) takes each
  unit's curve less its own value at -200 ms and reduces it unit -> session ->
  animal, the mean over animals with the SEM over animals
  (`analysis/pfc/population_statistics.animal_balanced_unit_curve`, the reduction the
  marked-family population curves use). It drew the mean over all units with the SEM
  over units before; over half the units are one animal's (wilbur), and the
  animal-balanced curve peaks elsewhere. The mpfc summary reports no value for this
  curve (its only 5c value is the unit count), so the figure copies no reported
  number. The choice is open to a different method later: the panel's reduction is
  that one call, so changing it means swapping the function.

- [ ] **Refit the content models at the next regeneration.** Two stored content
  outputs predate today's rules. (1) The six `models_validation/` staging bundles
  of the 2026-09-01 regeneration (`replay_content_regen_staging/`,
  `contfrag_replay_content_regen_staging/`) fail `write_regeneration_manifest.py`'s
  convergence gate: their null rung (M0/P0) records no `converged` flag, which the
  gate reads as not converged. The conditional-Poisson null records its flags since
  `745204e8`. (2) The leaf bundle's stored cohort audit
  (`replay_content/models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu`, written 2026-10-16)
  books rung Mu's qvals-gap events under the older audit rule (as own-predictor
  missing); rebuilt with today's code, that row moves to `n_drop_qvals_gap`. No
  reported value reads either. *When:* the next content regeneration, which writes
  fresh staging bundles and fits with the current code.

- [ ] **Refit after the one-owner merge, with the changes that wait for it.** The
  user decided (2026-11-12) to refit once `one-owner` is on master. With the
  content refit above, land the code changes that can only be proven by a refit
  (`.claude/research/one-owner/scope.md`, section D and the d2 line): one seed
  module (a changed key format changes the draws); one update loop in
  `analysis/pfc/batched_glm.py` (`_sweep` and `_sweep_prediction_batch`, and the
  held-out and full-fit loops, are bit-identical on CPU but contract differently,
  so GPU results can move in the last bits); and one ridge-Poisson fitter
  (`ridge_history_cv` is Newton with step halving where `batched_glm` is IRLS; on
  a synthetic design their parameters differ by about 3e-11). Show every reported
  value that moves before merging. *When:* after `one-owner` is merged into master.

- [ ] **Decide the open choices in the reported statistics.**
  `docs/known_issues/reported_statistics_choices.md` lists them: where each is
  made, what the code does, what is open, and what was measured on the current
  data. They were found while the statistics moved into `analysis/` and in the
  review of that code. Write each decision in that file.
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

## Event evidence

The review of the restriction and held-out tetrode runs before the
`swr-restricted-models` merge left these on purpose.

- [ ] **Which trials make the folds.** Both GPU runners make their folds from the
  trials `load_data` returns (the DIO parse); the session cache's trials
  (`replay_content/_session_cache/<epoch>.pkl`) can differ from them and give other
  folds. Each epoch's record now keeps the digest of the trials it used
  (`epoch_inputs["trials"]`), but nothing compares it on a resume.
  *When:* decide which trials are canonical before any rerun of either runner.
- [ ] **A run made from a restriction run of another design is pending, not refused.**
  The window run (`sum_nonlocal_restriction_windows.py`) and the held-out event run
  (`detect_heldout_events.py`) record the restriction run's manifest digest but not its
  design, so the summary can mark their values pending but cannot tell another design
  from a rerun of the same one.
  *When:* when either runner is next changed.
- [ ] **The per-bin facts' spikes outside the canonical windows are checked against no
  other run.** The window run checks the facts' spikes over each canonical window against
  the held-out tetrode run's; the restriction run records no spike count, so the facts'
  spikes elsewhere (the time sums, the held-out events) are not checked.
  *When:* if the restriction run is changed to record its spike counts.
- [ ] **Figure 2 draws panel g when the summary marks its values pending.** The figure
  refuses panel g for another restriction run, per-bin facts run or session cache, but
  not when only the event table changed (the held-out event run's canonical check), nor
  for the restriction run's decodes or SWR source, while the events summary marks the
  values pending in both cases.
  *When:* when Figure 2's checks are next changed.
- [ ] **The supplementary held-out tetrode figure checks no staleness.**
  `scripts/figures/generate_supp_heldout_tetrode.py` refuses a run not on folds of whole
  trials, but draws a run made from another event table or participation run.
  *When:* before the figure is final.
- [ ] **`n_folds` beside `folds`.** `tetrode_holdout.heldout_terms`, `holdout_scores` and
  `nonlocal_restriction.fold_fit_scores` take `n_folds` beside the `folds` array, which
  implies it.
  *When:* at the next change to those functions.
- [ ] **A branch only a test uses.** `tetrode_holdout.capped_groups`' `max_segment_s=None`
  (every segment whole) is used only by a test.
  *When:* at the next change to `capped_groups`.
- [ ] **`--shards` values not below `--n-shards`.** `_per_event_run.select_epochs` selects
  nothing, without a message, for a shard that is not below `--n-shards`.
  *When:* at the next change to the shared runner arguments.
- [ ] **The "block" names.** `tetrode_holdout.fold_blocks`, `FoldTerms.block` and the
  restriction run's per-bin `block` column keep the name of the contiguous design; the
  column is on disk.
  *When:* at the next full rerun of the restriction run.
- [ ] **Two long simulated integration tests.** The two simulated tests in
  `tests/integration/` of the restriction and held-out tetrode runs take about 13 and 23
  minutes.
  *When:* when the suite's time matters (CI).

## Run infrastructure

- [ ] **A resumed run does not compare an epoch's own inputs.** `runs/epoch_records.same_run`
  compares the run-wide header `inputs` only. A runner whose inputs are per epoch (a decode
  frame, a saved detector) reuses an epoch's record after that epoch is re-decoded; only
  `--overwrite` recomputes it. `scripts/event_evidence/score_nonlocal_restriction.py` records
  each epoch's input digests (`epoch_inputs`), which only the events summary reads back (its
  decode digests, `reported.events.restriction_input_staleness`), not the resume;
  `score_tetrode_holdout.py` records only the trials' digest and
  `score_event_likelihood_advantage.py` none. Let a task
  give `epochs_to_run` a per-epoch input function and rerun an epoch whose digests differ.
  A resume does not compare the code either: the commit is recorded (`provenance`) and not
  read back, so an `ok` record made by older scoring code is reused unless the change shows
  in `settings`.
  *When:* before the next re-decode, or when one of these runners is resumed across one or
  across a change to its scoring code.
- [ ] **One GPU-per-worker setup.** `scripts/event_evidence/_per_event_run.use_gpu` repeats the
  environment setup and device check of `runs/device.configure_jax_gpu`, which it cannot call
  because that allows one process per card. Move the shared part into `runs/device` and have
  both use it.
  *When:* when a third runner needs several workers on a card.

## Checks

- [ ] **Type checking.** `mypy src/` reports about 400 errors in 90 files.
  Fix them, then add mypy to CI and pre-commit.
  *When:* as a separate cleanup, before mypy is relied on.
- [ ] **Pin GitHub Actions to commit SHAs.** The workflows use version tags
  (`@v4`), which the action's owner can move.
  *When:* before the repository is made public, or before a workflow gets a
  secret or deploys anything.

## Regression fitter

Follow-ups of the 2026-11-21 cleanup of `analysis/regression/`. The timings are a
reviewer's, on a synthetic session of 40,000 rows and 56 columns, 4 threads.

- [ ] **Prepare each session once per search.** Today every evaluation validates the
  inputs, checks identifiability, rotates the design into the penalties' eigenbasis
  and reduces it to the active columns, and three layers each map back the
  coefficients, `A`, the covariance, the step and the running-away columns. Do the
  preparation once per session and search, with one pair of maps back to the given
  columns. Measured: the rotation takes 28 ms and the two checks 10.5 ms of a 109 ms
  score-only evaluation.
  *When:* before the fitter runs the real models' searches.
- [ ] **The step cap depends on column scaling.** The cap is in coefficient units. A
  prototype (2026-11-21, simulated designs of 600 rows) showed the effect: with one
  N(0,1) column multiplied by 1e-4 or less, the Poisson, Bernoulli and softmax fits end
  at the step limit, and a Poisson fit from a far start ends "no finite maximum" although
  its maximum is finite; multiplied by 1e-2 or by any factor above 1, every fit
  converges to the same optimum. Two replacements were tried and both failed the saved
  failure cases, so the cap stays. A cap of 10 on the change of the linear predictor
  cannot find a separation along a smooth's line and runs out of steps from a saturated
  start. No cap, with halving alone, overshoots until the information underflows: a
  separated session ends "information not positive definite", not "no finite maximum".
  Until a better rule exists, a design must not hold a column whose values are all very
  small: the convention's design check refuses one (planned with the collinearity and
  support checks). A rule not tried: no cap, except where a step would saturate a row.
  *When:* the design check with the convention tools; a new rule only if needed.
- [ ] **Who refits the set-aside sessions.** Refitting a set-aside session with columns
  left out is model policy. Decide whether it belongs to the caller, which owns the
  support rule, with the search returning the set-aside sessions and their directions.
  *When:* when the first model's caller is written.
- [ ] **The Hessian's trace terms by leverages.** For the diagonal families the trace
  terms can use leverages (`sum_i lev_i * (...)`) instead of two Gram products and a
  solve per pair of penalties. Measured: 22.0 ms to 6.6 ms per pair, with a relative
  difference of 8e-14.
  *When:* when a search's cost matters.
- [ ] **Smaller savings in the search.** Seed the grid's first fits from the screen's
  fits; score a halving candidate before computing its derivatives; cache `X'X` and
  `X'y` for the Gaussian.
  *When:* with the trace terms above.
- [ ] **The Gaussian in the Newton loop.** The Gaussian goes through the loop and is
  exempted from its step rules at three places. One exact step does not replace the
  loop: the loop's second step refines the first. A reviewer measured, on a design with
  two nearly equal columns, a largest coefficient error of 3.9e-8 with one step and
  2.9e-11 with the loop, each against a stable least-squares solution; one step also
  returned "converged" for an objective that was not finite. A direct solve needs a
  refinement step and that check. Outside the penalty search, a Gaussian fit stops by an
  absolute tolerance, so an outcome with a very large spread ends at the step limit.
  *When:* with the step cap above.
