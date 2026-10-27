# Phase 1 (A) — When does the mPFC response peak, and does it start before onset?

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

Why: question 4. Everything needed is in the canonical bundles: per-unit ridge curves on ±200 ms for SWR-coupled replay, non-SWR replay and all SWRs, and each class's held-out p-value. No refit. The one new computation is a reference for how far the B-spline basis (knots every 50 ms) and the σ = 3 prior smear a response that starts exactly at onset, because without it a negative onset cannot be read.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/load_population.py:275](../../../../src/continuum_swr_replay/analysis/pfc/load_population.py#L275) — returns `(units, curves, lag_time)`; curves `swr`, `non_local_swr`, `non_local_no_swr`, `matched_control`, each `(201, n_units)`.
- [src/continuum_swr_replay/analysis/pfc/modulation.py:101](../../../../src/continuum_swr_replay/analysis/pfc/modulation.py#L101) `peak_log_modulation` (the peak convention to keep) and `:240` `CURVE_SOURCES` (which model each curve comes from).
- [src/continuum_swr_replay/analysis/pfc/design.py:133-252](../../../../src/continuum_swr_replay/analysis/pfc/design.py#L133) — `symmetric_lags`, `event_basis_functions`, `convolve_event_basis`: the calibration builds its design with these.
- [src/continuum_swr_replay/analysis/pfc/batched_glm.py](../../../../src/continuum_swr_replay/analysis/pfc/batched_glm.py) `make_ridge_penalty`, `fit_poisson_glms` — the shipped fitter (JAX; CPU is enough for the calibration).
- [src/continuum_swr_replay/reported/part4.py:809](../../../../src/continuum_swr_replay/reported/part4.py#L809) `overlap_and_effect_values` and `:1054` `summarize` — the pattern for the new value function.

**Contracts referenced:**

- [Event classes and curves](shared-contracts.md#event-classes-and-curves), [Response window and unit sets](shared-contracts.md#response-window-and-unit-sets) — the peak and onset rules; do not weaken.
- [Bundle trees](shared-contracts.md#bundle-trees) — where the calibration file goes.
- [Reported-value keys](shared-contracts.md#reported-value-keys).

## Tasks

- **`src/continuum_swr_replay/analysis/pfc/timing.py`** (new, pure NumPy):

  ```python
  class PeakOnset(NamedTuple):
      peak: NDArray        # (n_units,) signed log-rate value at the largest |curve|
      peak_lag: NDArray    # (n_units,) s
      onset_lag: NDArray   # (n_units,) s; earliest lag of the half-max run around the peak
      onset_at_edge: NDArray  # (n_units,) bool; the run reaches the first lag

  def peak_and_onset(curves: NDArray, lag_time: NDArray, *, fraction: float = 0.5) -> PeakOnset:
      """curves (n_lags, n_units) on the report window; NaN columns give NaN."""
      c = np.asarray(curves, float).T                      # (n_units, n_lags)
      ok = np.isfinite(c).all(axis=1)
      at = np.argmax(np.abs(np.nan_to_num(c)), axis=1)
      peak = c[np.arange(len(c)), at]
      above = (np.sign(c) == np.sign(peak)[:, None]) & (np.abs(c) >= fraction * np.abs(peak)[:, None])
      lag_index = np.arange(c.shape[1])
      # last lag before the peak that is NOT above; the onset is the next one
      below_before = np.where(~above & (lag_index[None, :] < at[:, None]), lag_index[None, :], -1).max(axis=1)
      onset_index = below_before + 1
      ...  # NaN where not ok; onset_at_edge = onset_index == 0
  ```

  plus `sign_aligned_mean(curves, peak) -> NDArray (n_lags,)` (mean of `curve * sign(peak)` over units) and `population_timing(curves, peak, groups, *, n_boot, seed) -> dict` returning the aligned mean curve's peak lag and onset lag (by `peak_and_onset` on the one mean curve) with a percentile interval from resampling `groups` (session identifiers) with replacement, `np.random.default_rng(seed)`. NumPy docstrings with shapes.

- **Calibration script `scripts/pfc/calibrate_pfc_kernel_timing.py`** (new, CPU, `JAX_PLATFORMS=cpu`): simulate what the shipped estimator does to a response of known onset.
  - Time axis 500 Hz, `n_bins = 267_000` (the median canonical session's fitted bins, from the sidecars' `cohort_counts.n_bins_fit`), one contiguous fit mask; 100 onsets (median per-class count) placed uniformly with at least 0.6 s between them, seed fixed.
  - Truth: baseline 5 Hz; log-rate gain `g * h(t - t0)` with `h` a boxcar of width 100 ms (and a second shape, a 30 ms-rise / 100 ms-decay alpha function), `g in {+0.7, -0.7}` (about the called units' median absolute peak, `part4.effect_abs_peak_median_called_*`), true onsets `t0 in {-50, 0, +25, +50}` ms; 200 simulated units per scenario (independent Poisson draws on one design).
  - Fit: intercept + `convolve_event_basis(onsets, event_basis_functions(symmetric_lags(0.25, 500), 500)...)`, `make_ridge_penalty(width, 3.0)`, `fit_poisson_glms`; curve = basis @ block, trimmed to ±200 ms; `peak_and_onset`.
  - Writes `Processed-Data/pfc_response_followups/timing_calibration.json`: per scenario the true onset, median and 2.5/97.5 percentiles of the apparent onset and of the apparent peak lag, and the git commit; plus the arguments. A `--quick` flag (20 units, 2 scenarios) for the test.
- **`timing_values(population, curves, lag_time, calibration, make)` in `part4.py`**, added to `summarize` after `overlap_and_effect_values` (the summary now keeps the curves `load_population_data` returns). For each class word `swr`, `swr_replay`, `non_swr_replay` (curves `swr`, `non_local_swr`, `non_local_no_swr`) and each direction `enhanced` / `suppressed` (peak sign), over the units called for that class:
  - `timing_units_<cls>_<dir>` (count);
  - `timing_peak_lag_median_ms_<cls>_<dir>`, `..._quartile_low/high`, `timing_peak_before_onset_fraction_<cls>_<dir>` (peak lag < 0);
  - `timing_onset_median_ms_<cls>_<dir>`, `timing_onset_before_onset_fraction_<cls>_<dir>` (onset < 0), `timing_onset_at_edge_fraction_<cls>_<dir>`;
  - `timing_population_peak_ms_<cls>_<dir>`, `timing_population_onset_ms_<cls>_<dir>` with `_ci_low/_ci_high` (1,000 session resamples, seed 0), and `timing_population_onset_animals_negative_<cls>_<dir>` (animals whose own aligned mean curve has onset < 0).
  - Paired, units called for both SWR-coupled and non-SWR replay: `timing_peak_lag_difference_median_ms` (non-SWR minus SWR-coupled) and `timing_non_swr_later_fraction`.
  - Floor: `timing_matched_control_peak_before_onset_fraction` (all units' matched-control curves).
  - Calibration: `timing_calibration_onset_bias_ms_<shape>_<dir>` (median apparent minus true onset at true onset 0), and the same at +50 ms. The calibration file is added to `summary.inputs` with `file_record`.
  - Descriptions say "called (raw p < 0.05)" and name the onset rule; all `pending=None` unless the bundles are stale (the existing `staleness` check).
- **Run** the calibration, then `uv run python scripts/reported/summarize_part.py --part 4` and `uv run python scripts/reported/emit_reported_values.py`; check `changes.md` lists no change to existing Part 4 values.
- **`paper/reported/overview.md` Part 4**: one **When?** bullet with markers only (peak lag and onset per class, the calibration's bias beside them, the floor). Numbers never typed.
- **Docs.** `scripts/README.md`: the calibration script. `docs/README.md` needs nothing (no new design doc); the timing rule is documented in `timing.py`'s module docstring.

## Deliberately not in this phase

- No refit, no new bundle, no change to the canonical runner or loader.
- No timing for SWRs without replay (phase 2 adds it with the same functions) and no timing of marked kernels.
- No inference on timing beyond the session bootstrap; unit fractions stay descriptive.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_peak_and_onset_on_planted_curves` | hand-built curves: a boxcar from +20 ms gives onset +20 ms and peak inside the box; a negative bump gives a negative peak and its own onset; a curve above half-max from −200 ms has `onset_at_edge`; a NaN column gives NaN |
| `test_onset_ignores_a_same_sign_bump_before_a_gap` | two same-sign bumps separated by a sub-half-max gap: the onset is that of the bump holding the peak |
| `test_sign_aligned_mean_flips_suppressed_units` | an enhanced and a suppressed copy of one curve average to that curve |
| `test_population_timing_bootstrap_is_seeded` | same seed → same interval; the interval contains the point estimate |
| `test_calibration_quick_recovers_a_late_onset` (`slow`) | `--quick` with true onset +50 ms: the apparent onset is later than the apparent onset of the 0 ms scenario, and the file has the documented keys |
| `test_timing_values_keys_and_kinds` | on a synthetic population and curves the function returns the documented keys, no key holds a digit, fractions lie in [0, 1] |

## Fixtures

Synthetic curves built in the test module on the real 201-lag axis; the calibration's own simulator. Real data: the 75 canonical bundles, read by the summary step only.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
