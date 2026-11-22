# Publication follow-ups

Work that is deferred on purpose. Each item says when it should be done.

## Paper metadata

- [ ] Replace the working title, and confirm the tentative author list and each
  author's affiliations in `CITATION.cff` and `paper/main.tex`.
  *When:* before the first draft goes to the co-authors.
- [ ] Choose the target journal and switch `paper/main.tex` to its document
  class and bibliography style.
  *When:* once the journal is chosen.

## Figures

- [ ] **Figures that are the supplement of no main figure.** Each supplementary
  figure is the supplement of one main figure (`docs/development.md`, convention
  11). These are not in `paper/main.tex` and are named for no main figure; each needs
  a main figure, or a place as panels in a main figure's supplement, before it
  enters the paper: in `reported/questions.py`, `decoder_validation`
  (`scripts/figures/generate_supp_decoder_validation.py`), `content_estimators`
  (`scripts/replay_models/content/compare_content_estimators.py`),
  `uncertainty_content` (`scripts/figures/generate_uncertainty_content_figure.py`)
  and `swr_contrast` (`scripts/figures/plot_swr_contrast_coefs.py`); and the scripts
  `scripts/figures/generate_supp_pipeline_schematic.py` and
  `generate_supp_pipeline_data_flow.py`.
  *When:* before any of them enters the paper.

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

- [ ] **Panel 2j's caption or text gives what no figure draws.** From
  `paper/reported/events.json`, over the immobile time inside a well visit (from
  the animal's first nose-poke at a reward well to its last poke-out there): the
  share of the full decoder's gain that the SWR-only decoder captures
  (`restriction_immobile_share_captured_by_swr_only`), and the full minus the
  SWR-only decoder per spike inside the events detected from the held-out decode
  (`restriction_heldout_non_swr_events_outside_swr_full_minus_swr_per_spike`,
  `restriction_heldout_swr_events_outside_swr_full_minus_swr_per_spike`), and the
  gain outside SWRs over only the bins whose place has at least 10 and at least
  60 s of training data (`restriction_outside_swr_sampled_ten_s_full_minus_swr_per_spike`,
  `restriction_outside_swr_sampled_minute_full_minus_swr_per_spike`). The gain
  outside SWRs away from the SWRs is in Figure 2's supplement
  (`scripts/figures/generate_supp_figure02.py`, panel j); its shares in and
  outside those events are reported values for the text. The supplement's MUA
  panel (e): most non-local time is below the MUA level
  (`nonlocal_time_below_mua_level_fraction`), and nearly all time above the MUA
  level is decoded non-local (`nonlocal_share_mua_above_level`). The caption
  must not say that this shows non-local detection is independent of
  population bursts: it poses the question the held-out tetrode panels (g, h)
  answer.
  *When:* when the caption of Figure 2 is written.
- [ ] **The library does not repeat itself on one kind of CI machine.** On
  GitHub's runners with an AMD EPYC 7763 (no AVX-512), the fold fit of the
  simulated recording is not the same twice from the same inputs: the
  library's likelihood of fold 0 differed between two calls back to back in
  481,504 of 16,430,000 values (by up to 7.2), and its EM on fold 0 ran 6, 7, 7
  and 12 iterations (transitions up to 4.6e-4 apart; with one XLA thread the
  likelihood repeated and the EM still did not, with a marginal that went
  down twice). Fold 1 repeated. On the runners with an EPYC 9V45 or a Xeon
  8573C, and on the development machine (Xeon Gold 6342, 1, 2 and 96 cores,
  and an A100), every repeat was bit for bit; there a one-step float change of
  the inputs moves the transitions by 4e-10 and a planted leak by 1.4e-2. So
  the cause is below this repository, in JAX's CPU execution or in how the
  library uses it, and it is not found. The two tests that compare two
  computations of the fold fit bit for bit
  (`tests/integration/test_nonlocal_restriction_simulated.py`, marked
  `bit_for_bit`) are not run on GitHub's runners, where a job cannot choose
  its machine, and are exact everywhere else. A repeat made inside the test
  did not tell the machines apart: on such a runner two calls in a row agreed
  after an earlier one had differed (master d6cce684's slow job). Not done: a reproducer that needs only the
  library, and an issue on `non_local_detector` or JAX. The script and the
  runners' reports are in `.claude/research/ci_float_reproducibility/`.
- [ ] **Old session-cache files hold other trials.** Both GPU runners make their
  folds from the trials `load_data` returns (the DIO parse). The current session
  cache (`replay_content/_session_cache/comrie_photogate_trials_v4/`) is a saved
  copy of the same trials: folds made from it equal the stored folds of the
  restriction run in every epoch (checked 2026-11-21). The files of older cache
  contracts beside it (the `.pkl` files directly under `_session_cache/` and
  `comrie_photogate_trials_v3/`) hold trials from before the photogate reparse; a
  review read one of them and took it for the current cache. Each epoch's record now
  keeps the digest of the trials it used (`epoch_inputs["trials"]`), but nothing
  compares it on a resume.
  *When:* delete the old cache files when the lead agrees; compare the digest on a
  resume if the runners are rerun.
- [ ] **The participation run's record names a staging folder.**
  `event_participation/manifest.json` records the paths of its event table and session
  cache under a staging root that held links to the real files (the run of 2026-11-21
  with controls inside well visits). The digests it records are the real files', and no
  code reads those paths. Do not edit the manifest by hand: the held-out tetrode run and
  the events summary record its digest.
  *When:* the paths become correct at the next participation run.
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
- [ ] **Figure 2 draws panel j when only the event table changed.** The figure
  refuses panel j for another restriction run, per-bin facts run or session cache, and
  for a restriction run made from other decode frames or another SWR source
  (`restriction_inputs_note`), but not when only the event table changed (the held-out
  event run's canonical check), while the events summary marks the values pending.
  *When:* when Figure 2's checks are next changed.
- [ ] **Figure 2's supplement, panels g and h, check no staleness.**
  `scripts/figures/generate_supp_figure02.py` refuses a held-out tetrode run not on
  folds of whole trials, but draws a run made from another event table or
  participation run.
  *When:* before the figure is final.
- [ ] **The restriction run's categories still count bins without ripple signal as
  outside an SWR.** `nonlocal_restriction.bin_categories` puts an immobile analyzed bin
  outside every SWR of the source in `immobile_outside_swr` whether or not SWR detection
  was observed there (2,013 of the 64.3 million such bins inside a well visit, in 19
  epochs, in the run of 2026-11). The CPU steps correct it: the per-bin facts record
  `swr_observed`, and every sum outside SWRs uses
  `restriction_windows.outside_swr_bins`. Correcting `bin_categories` itself needs a new
  category (the bins are analyzed and immobile, so neither `excluded` nor
  `immobile_in_swr` is right), which changes the finished run's `category` values and
  its records' `n_bins_by_category` keys, so the run would not gather or resume; it is
  left. The all-immobile values keep those bins (they are immobile time either way).
  *When:* at the next rerun of `score_nonlocal_restriction.py`; give `bin_categories`
  the observed mask then, and keep `outside_swr_bins` as the one rule.
- [ ] **The margin from the SWRs ignores gaps in the ripple band.** The parts of the
  time outside SWRs farther than a margin from them
  (`restriction_totals.swr_margin_bands`) measure the distance to the scoring mask
  only, so a bin beside a stretch where SWR detection was not observed counts as far
  from an SWR.
  *When:* if the margin values are quoted as "far from any SWR".
- [ ] **No reader checks the per-bin facts' SWR source.** `measure_bin_facts.py` records
  the SWR source in its header's settings (`swr_source`), but neither the CPU runners
  nor the events summary compare it with the current source.
  *When:* at the next change to the per-bin facts' readers.
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
- [ ] **Is an event's decoded place where silence alone would put it?** Not done and not planned unless a reviewer asks: compare each event's decoded place with the lowest-rate place bins of the saved encoding model (`decoding/encoding_model.py`), where the ground process alone would put it. The split of the held-out tetrode gain into bins with a spike and silent bins (Figure 2's supplement, panel g) and the events' spike rate against matched Local rest (`non_swr_event_spike_rate_hz`, `non_swr_control_spike_rate_hz`) already speak against a silence-driven estimate; events with a wide estimate (a quarter of no-SWR events, `non_swr_shared_content_hundred_cm_or_more_share_of_events`) are where it could still hold.
  *When:* only if a reviewer asks.

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
- [ ] **The step cap depends on column scaling.** The cap is in coefficient units, so a
  column whose values are all very small ends a fit at the step limit (a prototype,
  2026-11-21: one N(0,1) column multiplied by 1e-4 on simulated designs of 600 rows;
  a Poisson fit from a far start then ends "no finite maximum" although its maximum is
  finite). The design check (`analysis/regression/terms.py:check_column_sizes`) refuses such a
  column before any fit. Two replacement rules failed the saved failure cases: a cap of
  10 on the change of the linear predictor cannot find a separation along a smooth's line
  and runs out of steps from a saturated start; no cap, with halving alone, overshoots
  until the information underflows. A rule not tried: no cap, except where a step would
  saturate a row.
  *When:* only if a design that passes the check still ends at the step limit.
- [ ] **The Hessian's trace terms by leverages.** For the diagonal families the trace
  terms can use leverages (`sum_i lev_i * (...)`) instead of two Gram products and a
  solve per pair of penalties. Measured: 22.0 ms to 6.6 ms per pair, with a relative
  difference of 8e-14.
  *When:* when a search's cost matters.
- [ ] **Smaller savings in the search.** Seed the grid's first fits from the screen's
  fits; score a halving candidate before computing its derivatives; cache `X'X` and
  `X'y` for the Gaussian.
  *When:* with the trace terms above.
- [ ] **Rescale columns inside the fitter.** A second way to remove the step cap's dependence on
  column units, not tried: the fitter scales every column to unit size before the Newton
  steps and maps the coefficients, the penalty and the covariance back. It would remove the
  design's small-column check. It changes the paths of the fits, so it needs the saved
  failure cases and the mgcv comparison again.
  *When:* only if the small-column check gets in the way.
- [ ] **Contrasts built from the design.** `range_mean_contrast` is given its term's range and
  curve rows by the caller, and `SDChangeContrast` is given a pooled mean and SD, although the
  design holds all of them. Note that "per pooled SD" has two meanings in the declared
  list: for a smooth's SD change it is the SD on one role's rows; for a slope it is the SD
  over every level on which a slope of that covariate is fitted.
  *When:* with the first model's design (rate or content).
- [ ] **The across-session band on a curve's coefficients.** Every session shares one design,
  so a curve is `C beta` with one `C`; a band built on the term's coefficients needs no
  grid correlation and has no refusal at a reference point. With five animals it does not
  remove the weak case (more independent directions than animals).
  *When:* with the band's check on the models' own curves.
- [ ] **Collinearity from the Gram matrix.** The measures take one QR of the weighted
  design per session (about 0.18 s at 50,000 rows by 60 columns, a reviewer's timing). A
  factor of the p by p Gram matrix is about ten times cheaper, but it must tolerate a
  column that is zero in a session and it squares the condition number.
  *When:* if the per-session loop's time matters.
- [ ] **The Gaussian in the Newton loop.** The Gaussian goes through the loop and is
  exempted from its step rules at three places. One exact step does not replace the
  loop: the loop's second step refines the first. A reviewer measured, on a design with
  two nearly equal columns, a largest coefficient error of 3.9e-8 with one step and
  2.9e-11 with the loop, each against a stable least-squares solution; one step also
  returned "converged" for an objective that was not finite. A direct solve needs a
  refinement step and that check. Outside the penalty search, a Gaussian fit stops by an
  absolute tolerance, so an outcome with a very large spread ends at the step limit.
  *When:* with the step cap above.
