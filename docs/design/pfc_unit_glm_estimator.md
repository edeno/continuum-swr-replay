# The mPFC unit GLM: a prior on the event kernels, scored out of fold

**What this is.** The decision behind how `scripts/pfc/run_pfc_analysis_bandit.py` fits
and tests each mPFC unit, the checks it passed, and its known limits. The settings and
the sidecar fields are described below ("The bundles"); the Figure 4 population and its
values are question 4.1 in
[../../paper/reported/questions.md](../../paper/reported/questions.md).

## The decision

The unit GLMs were first fit by maximum likelihood, tested by a likelihood ratio against
an asymptotic chi-square, and compared by AIC. Neither part holds for these fits:

- a basis column with no spikes under its support drives its coefficient to minus
  infinity, so for those units the maximum-likelihood estimate does not exist;
- neighbouring 2 ms bins are not independent, so an in-sample deviance drop is worth
  much less than its nominal degrees of freedom.

Since 2026-09-16 the models carry a Gaussian prior on the event-basis coefficients
(σ = <!--v:mpfc.penalty_sigma-->3<!--/v--> on the log-rate scale, loose enough that the
spikes shape any modulation they can support) and are compared by held-out deviance, in
<!--v:mpfc.folds-->5<!--/v--> folds of whole immobility runs, against a null of
<!--v:mpfc.null_draws-->1,000<!--/v--> redraws of the duration-matched non-event control,
with a plus-one p-value. Each class's test uses one fixed subset of its events, those
with a legal control window in their own bout, chosen from timing and duration alone;
the observed statistic and every null draw are read on that subset.

## How it was checked

- **The solver is the maximum-likelihood one at zero penalty** (2026-09-15,
  `scripts/pfc/check_pfc_ridge_lambda0.py`, three sessions). With the penalty set to
  zero, the penalised solver matched the stored statsmodels fits on every cell whose
  maximum-likelihood estimate exists, within 1e-5 in the parameters and 1e-6 in the
  deviance. The cells where it does not exist were reported, not compared; they are
  the ones the prior was introduced for.
- **The prior is loose for this corpus** (2026-09-18, `scripts/pfc/compare_pfc_cohorts.py`
  on the first 71-session batch). The spread of the unpenalised peaks, less their
  estimation error, implied a σ of about 0.7 to 1.3 by curve, against the chosen 3, and
  the penalised peak differed from the unpenalised one by more than its standard error
  for under 3 % of the well-posed units per curve. These are dated numbers of that batch; the
  current share is <!--v:mpfc.prior_binds_percent_min-->2.6<!--/v-->–<!--v:mpfc.prior_binds_percent_max-->4.9<!--/v-->%
  (`paper/reported/mpfc.json`).
- **The old and new inferences are not comparable test by test.** On that batch, the
  likelihood-ratio tests and the held-out tests called comparable fractions of a
  comparable population of units, but they test different things against different
  references. Read them as two accounts of one population, not as a change.

The dated audits with every table are
[2026-09-15-pfc-ridge-lambda0-gate.md](../../.claude/docs/reviews/2026-09-15-pfc-ridge-lambda0-gate.md)
and [2026-09-18-pfc-ridge-heldout.md](../../.claude/docs/reviews/2026-09-18-pfc-ridge-heldout.md).

## Known limits

- **Unit fractions at raw p < 0.05 are descriptive.** Units are not independent
  replicates; population inference goes through session and animal.
- **Sensitivity to σ was measured on one session only** (σ = 2 and 4.5, on
  `j1620210710_02_r1`, in the first batch). Those bundles were not refit on the R11
  re-decode and are no longer on disk.
- **Bundles written before 2026-09-15 have the replay-split curves swapped**:
  `non_local_SWR_modulation` held the non-SWR kernel and the reverse, so a per-class
  modulation magnitude quoted from the split model before that date, the
  maximum-likelihood backup tree included, is mislabelled. Whole-model comparisons never
  read a block and are unaffected.

## What it means today

The current <!--v:mpfc.sessions-->75<!--/v--> at-well bundles are this estimator refit on
the R11 re-decode; the maximum-likelihood tree is kept at
`Processed-Data/_pre_ridge_backup_20260915/`.

## The bundles

`scripts/pfc/run_pfc_analysis_bandit.py` writes, per session,
`Processed-Data/{epoch_identifier}_mPFC_modulation.pkl` and five `*_modulation.npy`
time courses.

- **The statistics.** Per unit: the held-out deviance and the count of separated
  columns per model, each nested pair's held-out difference (total and per fold), an
  empirical p-value per event class and for the engagement comparison, the best model by
  held-out deviance, and per curve the penalised and unpenalised peak with that peak's
  own standard error.
- **The time courses.** `SWR`, `no_SWR`, `non_local_SWR`, `non_local_no_SWR` and
  `matched_control`, the duration-matched non-event baseline. All five are read into the
  Figure 4 population by `analysis/pfc/load_population.py`; the first four are NaN-free
  by its drop rule, while the matched-control curve may carry NaN and is reduced with
  `nanmean`.
- **The cohort.** The bundles are fit on the well-anchored stop cohort, the same
  population as the rate and content ladders, and an unsuffixed bundle is that cohort
  ([cohort_and_timing_rules.md](cohort_and_timing_rules.md)). `_allstops` is the
  unrestricted sensitivity bundle; none has been made for the unit GLM. Produce it with
  `run_pfc_analysis_bandit.py --stop-cohort all_stops` (a full
  <!--v:mpfc.sessions-->75<!--/v-->-session refit), the only way to compare the unit fits
  across populations on artifacts rather than on the unit test that pins the unrestricted
  path. A bundle fit at another prior is written under a `_sigma<value>` name beside the
  canonical one and draws no null; it exists for its modulation curves.

Every bundle carries a `*.provenance.json` sidecar recording `stop_cohort`, the
stop-table SHA-256, and the bins and events before and after the restriction. It also
states:

- **Which events the fit describes:** `is_swr_definition`, the SHA-256 of the canonical
  event table and of the session's cached decode frame (`canonical_event_table_sha256`,
  `data_over_time_sha256`), the cohort-restricted `n_events_total` / `n_events_swr` /
  `n_events_non_swr`, and the matched control's `matched_control_seed` /
  `n_matched_control` / `n_matched_control_unplaced` /
  `matched_control_placement_method`. The runner refuses to skip a bundle that does not
  carry the current values.
- **The estimator and the inference:** `estimator` (`ridge_poisson_glm`), `inference`
  (`blocked_heldout_deviance`), `penalty_sigma_log_rate` with the `implied_prior_sd` it
  means for a curve, `n_basis_per_class`, `n_folds` / `fold_seed` / `fold_sizes` /
  `events_per_fold`, `n_control_draws` / `control_draw_seed` / `null_usable_draws` /
  `min_usable_null_draws` / `null_windows` (per class: the events total, the fixed
  matchable subset actually tested, the excluded remainder, the tested source indices
  and how many windows each draw placed), `empirical_pvalue_method` (`plus_one`),
  `control_null_event_policy` (`fixed_jointly_matchable_subset`) and
  `engagement_null_overlap` (how much of the engagement null's draws land on the shipped
  control).
- **What the fits did:** `n_model_fits` (neuron × model cells that produced a fit at
  all), `n_nonconverged_model_fits` and `n_neurons_with_nonconverged_model` (fits that
  ran to `maxiter` without settling, and how many neurons carry at least one). A fit that
  never settled is non-estimable for the unit that owns it (every statistic reading it is
  NaN), so those counts are the only record of how much was dropped.

A bundle that does not state all of these is refit rather than skipped, and the
population loader also refuses it before figure generation: the canonical bundle name is
unsuffixed, so nothing else tells a penalised bundle from the maximum-likelihood one that
held the same name.

**The canonical settings**, read from the sidecars into `paper/reported/mpfc.json`:
σ = <!--v:mpfc.penalty_sigma-->3<!--/v--> (implied per-lag sd
<!--v:mpfc.prior_sd_per_lag-->2.07<!--/v-->, a rate multiplier of
×<!--v:mpfc.prior_multiplier_per_lag_low-->0.017<!--/v-->–×<!--v:mpfc.prior_multiplier_per_lag_high-->57<!--/v-->
at 95 %; sustained sd <!--v:mpfc.prior_sd_sustained-->0.90<!--/v-->,
×<!--v:mpfc.prior_multiplier_sustained_low-->0.17<!--/v-->–×<!--v:mpfc.prior_multiplier_sustained_high-->5.9<!--/v-->),
K = <!--v:mpfc.folds-->5<!--/v--> folds and R = <!--v:mpfc.null_draws-->1,000<!--/v--> null
draws. The finite-sample Monte Carlo p-value is `(r + 1)/(R + 1)`, so with
<!--v:mpfc.null_draws-->1,000<!--/v--> usable draws its minimum is
1/<!--v:mpfc.min_pvalue_denominator-->1,001<!--/v--> ≈ <!--v:mpfc.min_pvalue-->0.0010<!--/v-->;
zero is impossible. Draws that fail to place every matched window are excluded, and
every requested draw must be usable for a unit's p-value to be reported, so an
`R = 1,000` label always means a denominator of 1,000; the per-unit effective counts are
in both the table and the sidecar (`null_usable_draws`, `min_usable_null_draws`). The
20-draw floor governs deliberately shortened diagnostics only.

**The current bundles** were fit at git `110c9ae` on 2026-10-11/12 (logs in
`logs/pfc_refresh2_20261011/`). The estimator's first batch, at `db01df0` (2026-09-16/17),
replaced the maximum-likelihood tree. In the current bundles
(`paper/reported/mpfc.json`), non-converged fits are
<!--v:mpfc.nonconverged_percent-->0.26<!--/v-->% of model fits
(<!--v:mpfc.nonconverged_model_fits-->432<!--/v--> of
<!--v:mpfc.model_fits-->166,968<!--/v-->, over
<!--v:mpfc.nonconverged_neurons-->48<!--/v--> neurons). The null's geometry, by the rules
of `scripts/pfc/measure_pfc_null_geometry.py`: the engagement null's draws land on the
shipped control they are entered beside on a median
<!--v:mpfc.engagement_null_overlap_median-->11 %<!--/v--> of their windows,
<!--v:mpfc.class_called_flat_fraction_min@pctnum-->4<!--/v-->–<!--v:mpfc.class_called_flat_fraction_max-->8 %<!--/v-->
of the units called at raw p < .05 for an event class have no positive held-out
improvement of their own, and
<!--v:mpfc.unmatchable_percent_class_min-->0.27<!--/v-->–<!--v:mpfc.unmatchable_percent_class_max-->0.60<!--/v-->%
of a class's events have no legal within-bout control and enter neither side of its test.
Figure 4 is regenerated from these bundles. The dated audits of the cohort change and of
the canonical-events rerun are listed in
[../history/2026-module-moves.md](../history/2026-module-moves.md).

**The GPU environment.** The batched fitter runs on GPU through the project's `cuda`
extra. These runs installed `jax-cuda12-plugin[with-cuda]==0.8.0`, `dask-cuda==25.10.0`
and `cuda-bindings==13.1.1` by hand, when `uv.lock` had no entries for the extras; it now
locks those versions, so `uv sync --frozen --inexact --extra gpu` installs the same GPU
environment.
