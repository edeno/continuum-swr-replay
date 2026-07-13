# Phase 10a — Estimability, centered diagnostics, and fit validity

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contract](shared-contracts.md#fit-validity)

Make every actual design matrix prove rank/conditioning before fit, compute conditional diagnostics on
the identifiable within-group design, and replace inconsistent coefficient-threshold filtering with one
fit-validity policy. This phase changes which cells are considered usable but does not yet change SEs.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/fitting.py:37-64](../../../../src/continuum_swr_replay/analysis/replay/fitting.py#L37) — correct within-group rank idea.
- [src/continuum_swr_replay/analysis/replay/fitting.py:437-454](../../../../src/continuum_swr_replay/analysis/replay/fitting.py#L437) — raw, scale-dependent VIF/condition implementation to replace.
- [src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:129-188](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py#L129) — GLM fit without Patsy-design rank preflight.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:104-133](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py#L104) — existing explicit-rank pattern.
- [src/continuum_swr_replay/analysis/replay/_fit_diagnostics.py:25-84](../../../../src/continuum_swr_replay/analysis/replay/_fit_diagnostics.py#L25) — landed convergence/separation helper and raw threshold.
- [src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:79-135](../../../../src/continuum_swr_replay/analysis/replay/contrast_aggregation.py#L79) — term-only `abs(coef)<10` filtering to remove.
- [scripts/rate/fit_uncertainty_rate.py:190-215](../../../../scripts/rate/fit_uncertainty_rate.py#L190) — high-dimensional MU caller that currently drops failed `None` cells.

**Contracts referenced:**

- [Fit validity and diagnostics](shared-contracts.md#fit-validity) — one policy across families.
- [Term diagnostics](shared-contracts.md#term-diagnostics) — failures/exclusions remain visible.
- [Comparison identity](shared-contracts.md#comparison-identity) — unusable fits invalidate comparisons without altering cohort hashes.

## Tasks

- Add `analysis/replay/design_diagnostics.py` with a single SVD implementation. For conditional models,
  demean by likelihood group, drop zero-norm absorbed columns, scale remaining columns to unit RMS, and
  compute rank, singular values, condition number, centered VIF, and named non-estimable terms. Reuse the
  identical transformed matrix for the hard pre-fit rank gate and reported diagnostics.
- Update conditional Poisson, dominant-category conditional logit, and decision conditional logit to
  pass group IDs into diagnostics. Rename output fields to `condition_number_scaled` and
  `vif_centered_*`; schema versioning means ambiguous raw fields are removed rather than aliased.
- For Patsy/statmodels GLMs, construct the model object first, inspect `model.exog`/`exog_names`, and run
  scaled SVD before `.fit()`. Apply to per-bin, uncertainty, SWR-rate contrast, and CV training folds.
  Reject general aliases. A unique constant optional term may be removed only in an explicitly labeled
  estimation sensitivity; it makes the declared selection model invalid.
- Change all fitters/runners from `None` disappearance to one structured result row per attempted cell
  with `fit_status`, reason, rank, column count, condition fields, and named problematic terms. Preserve
  exact cohort/fold identity even for failed cells.
- Extend `_fit_diagnostics.fit_status` to incorporate captured `PerfectSeparationWarning`, non-finite
  Hessian/covariance, convergence, and scaled predictions. Pass captured warnings to GLM fitters as well
  as conditional fitters.
- Replace `contrast_aggregation`'s raw coefficient cutoff with the shared contract: only `ok` cells enter
  headline summaries; separated cells remain in diagnostics and may enter a prespecified penalized
  estimation-only sensitivity. Update `model_comparison` to require both cells `ok`.
- Reclassify the canonical content set in `model_specs.py`: stable M4-style content remains primary;
  sparse last-rewarded M5 is an estimation sensitivity with a weak-ridge/blocked-CV companion; Mvu is
  removed from headline summaries. Mv and Mu are separate construct sensitivities and may not be
  described as independently identifying exploitation versus exploration.
- Add centered diagnostics to SWR-content contrast outputs instead of stripping them. Update docs and
  summary tables to display rank/condition/VIF/fit-status denominators before coefficients.

## Deliberately not in this phase

- Cluster/bootstrap covariance or animal aggregation (Phase 10b).
- Selecting ridge penalty by the same data used for inference; sparse penalized fits are sensitivity
  estimates scored on Phase-9b folds, not ordinary p-value/LRT fits.
- Changing value/uncertainty covariate construction. This phase labels the limits and gates rank.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_centered_diagnostics_invariant_to_group_offsets` | adding any event-level constant leaves diagnostics unchanged |
| `test_scaled_diagnostics_invariant_to_units` | multiplying a predictor by a positive constant leaves scaled condition/VIF unchanged |
| `test_conditional_rank_and_diagnostics_use_same_matrix` | hard gate and reported rank cannot disagree |
| `test_per_bin_patsy_alias_returns_failure_row` | duplicate/constant MU term is named and never pseudo-inverted |
| `test_cv_training_rank_failure_invalidates_metric` | bad fold stays assigned and invalidates whole score |
| `test_separated_fit_excluded_but_audited` | raw result remains, comparison/headline use is false with reason |
| `test_result_grid_retains_failed_cells` | expected cell count equals successes plus explicit failures |
| production diagnostic smoke (slow) | centered VIF/condition distributions reproduce the audit order of magnitude |

## Fixtures

Extend `test_fitting.py`, `test_fit_diagnostics.py`, `test_per_bin_rate_model.py`, and
`test_per_stop_rate_model.py` with group offsets, unit rescaling, exact aliases, constant
`value_diff_patch`, quasi-separated binary terms, and a high-dimensional but full-rank control.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- Every reported diagnostic is computed on the same estimable design the likelihood uses.
- No statsmodels pseudo-inverse can produce an `ok` row for a rank-deficient declared design.
- Failed cells remain visible with cohort/fold identity and do not silently shrink denominators.
- The separation rule is scale-aware and shared; the old raw `abs(coef)<10` exclusion is gone.
- Sparse/value/uncertainty documentation does not outrun what the redesigned estimands identify.
- Every task is implemented, covariance/animal aggregation remains out of scope, and the old raw coefficient exclusion is removed.
- Validation tests pass and exercise actual aliases, scale changes, and failed cells rather than mock status flags.
- Production names/docstrings do not mention this plan or phase numbers; fit-schema docs are updated.
