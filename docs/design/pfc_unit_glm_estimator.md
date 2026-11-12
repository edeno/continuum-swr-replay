# The mPFC unit GLM: a prior on the event kernels, scored out of fold

**What this is.** The decision behind how `scripts/pfc/run_pfc_analysis_bandit.py` fits
and tests each mPFC unit, the checks it passed, and its known limits. The settings and
the sidecar fields are described in [../README.md](../README.md) ("Where results live",
the PFC artifacts); the Figure 4 population and its values are question 4.1 in
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
