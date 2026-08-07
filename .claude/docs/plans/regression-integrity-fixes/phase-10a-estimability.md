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

## Locked numerical policy

### Design transform, rank, and condition number

Diagnostics use float64 and the exact design columns/order passed to the likelihood:

1. For an unconditional GLM or OLS, let \(A=X\), including the fitted intercept. For a conditional
   likelihood, within-group demean every slope column and let \(A=C_gX\); a conditional intercept is
   absorbed and is not fitted.
2. For a conditional column \(x_j\), call it absorbed when
   \(\lVert C_gx_j\rVert_2 \leq \max(n,p)\epsilon\lVert x_j\rVert_2\), with an all-zero column absorbed
   when both norms are zero. Record and remove absorbed columns from the diagnostic matrix, but mark a
   declared selection model `rank_deficient`; dropping a declared term never repairs that model.
   Apply the same rule to an all-zero unconditional column. General aliases remain in the matrix for
   SVD to detect.
   A conditional null model with no slope columns is a separate valid `p=0` branch:
   `design_rank=0`, `n_design_columns=0`, no singular values/VIFs,
   `rank_tolerance=0`, `condition_number_scaled=1`, and
   `condition_warning=false`. It proceeds to the conditional null likelihood after the ordinary
   response/group/data-sufficiency checks; no reference to \(s_1\) is made.
3. Scale every remaining column by its RMS
   \(d_j=\sqrt{n^{-1}\sum_i A_{ij}^2}\), giving \(Z_{ij}=A_{ij}/d_j\). Compute
   `np.linalg.svd(Z, full_matrices=False, compute_uv=False)` once and reuse it for the gate and report.
4. With descending singular values \(s_1,\ldots,s_p\), use
   \(\tau=\max(n,p)\epsilon s_1\), where \(\epsilon=\texttt{np.finfo(np.float64).eps}\).
   `design_rank = sum(s > tau)`; a value exactly equal to \(\tau\) is non-estimable. A required
   absorbed term or `design_rank < p` gives `fit_status=rank_deficient` before `.fit()`.
5. Except for the valid conditional-null `p=0` branch above, a full-rank matrix has
   `condition_number_scaled = s_1 / s_p`; a rank-deficient matrix—or a matrix left empty only because
   declared columns were absorbed—has infinity. Set
   `condition_warning = (condition_number_scaled >= 1e8)`. This warning is reported but does not by
   itself change an otherwise valid cell or term status. VIF uses this same \(Z\), never the raw or
   separately transformed matrix.

### Convergence and separation

Capture estimator warnings and exceptions. A cell has a separation signal when any of the following
is true:

- statsmodels raises `PerfectSeparationError` or emits `PerfectSeparationWarning`;
- a binomial/logit fit has a fitted probability at or below \(10^{-10}\) for an observed zero, or at
  or above \(1-10^{-10}\) for an observed one;
- a Poisson fit has a fitted mean at or below \(10^{-10}\) for an observed zero, or at or above
  \(10^{10}\).

Conditional-logit probabilities are recomputed by stable within-group softmax when the result object
does not expose them. Tests use the actual fitted response rows. Large raw or standardized
coefficients alone never set status; prediction-boundary checks are invariant to predictor units.
For status precedence, insufficient-data checks occur first and the SVD gate occurs before fitting.
After a fit attempt: an explicit or prediction-based separation signal gives `separated`; absent that,
an optimizer that did not converge gives `nonconverged`; absent both, a non-finite parameter,
log-likelihood, fitted value, model-based Hessian, or model-based covariance gives `fit_error`;
otherwise the cell is `ok`. Thus separation remains the recorded cause when it also produces
nonconvergence or a non-finite Hessian. OLS has no separation state.

### Cell validity versus term validity

Every attempted cell carries `selection_eligible`; every emitted term carries `term_status`,
`inference_eligible`, and `headline_eligible`. `term_status` is exactly one of `ok`,
`non_estimable`, `cell_invalid`, `robust_covariance_unavailable`, or
`penalized_estimation_only`:

| Condition | Cell/comparison consequence | Term consequence |
| --- | --- | --- |
| `rank_deficient`, `nonconverged`, `separated`, or `fit_error` | `selection_eligible=false`; no AIC/BIC/LRT/CV delta | every term `inference_eligible=false` and `headline_eligible=false`; raw values remain diagnostic |
| full rank with `condition_warning=true` | cell remains eligible and warning is mandatory | an otherwise valid term remains eligible and carries the warning |
| model-based Hessian/covariance non-finite | cell is `fit_error` | all terms invalid |
| cluster/bootstrap covariance unavailable for one term, with finite valid likelihood fit | likelihood/CV selection remains eligible | that term is `robust_covariance_unavailable`, inference/headline false; unaffected terms retain their own status |
| declared absorbed/aliased term | selection cell is `rank_deficient` | named term is `non_estimable`; no silent term deletion |
| penalized sensitivity | distinct estimator and comparison set | coefficient may be estimation-only; no ordinary p-value, AIC, or LRT |

`headline_eligible` can become true only when the cell is `ok`, the term is `ok`, the required robust
covariance exists, and the later Phase-10b dependence/multiplicity and animal-coverage checks pass.
Phase 10a initializes it false rather than anticipating those later checks.

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
  Hessian/covariance, convergence, and the locked prediction-boundary checks. Pass captured warnings to
  GLM fitters as well as conditional fitters.
- Replace `contrast_aggregation`'s raw coefficient cutoff with the shared contract: only `ok` cells with
  eligible terms can enter headline summaries; separated cells remain in diagnostics and may enter a
  prespecified penalized estimation-only sensitivity. Update `model_comparison` to require both cells
  `ok` and `selection_eligible`.
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
| `test_absorbed_conditional_term_invalidates_declared_cell` | an at/below-tolerance within-group column is named and cannot be repaired by dropping it |
| `test_svd_rank_tolerance_boundary_is_locked` | singular values above versus exactly at `max(n,p) * eps * s1` give the specified rank |
| `test_conditional_null_has_valid_empty_design` | a predictor-free conditional null uses the explicit p=0 diagnostics branch and remains fit-eligible |
| `test_condition_warning_boundary_is_not_failure` | values immediately below versus at `1e8` toggle only the warning |
| `test_per_bin_patsy_alias_returns_failure_row` | duplicate/constant MU term is named and never pseudo-inverted |
| `test_cv_training_rank_failure_invalidates_metric` | bad fold stays assigned and invalidates whole score |
| `test_separation_status_precedence_and_scale_invariance` | warnings/boundary predictions remain `separated` after predictor rescaling even when convergence/Hessian also fail |
| `test_separated_fit_excluded_but_audited` | one separated cell blocks comparison and all of its terms while retaining raw diagnostics |
| `test_robust_covariance_failure_is_term_scoped` | a valid likelihood remains selectable but the affected term cannot enter inference/headline output |
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
