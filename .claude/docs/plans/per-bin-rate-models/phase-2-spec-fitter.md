# Phase 2 — MODEL_TIMERESOLVED spec + fit/CV primitives

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/model_specs.py:113](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — `ModelSpec` (the dataclass the new spec parallels) and `:437`-`:484` `MODEL_RATE`/`Rlag`/`N_REWARD_LAGS` (append after these).
- [src/continuum_swr_replay/analysis/replay/rate_model.py:43](../../../../src/continuum_swr_replay/analysis/replay/rate_model.py) — `fit_rate_model` return-dict shape (`coef_*`/`bse_*`/`pval_*`/`rr_*` + stats) to mirror; `:160` `cross_validate_rate_model` (the stop-fold CV pattern to mirror).

**Contracts referenced:**

- [TimeResolvedModelSpec + MODEL_TIMERESOLVED](shared-contracts.md#timeresolvedmodelspec) — implement exactly; the toggle fields (`time_spline`, `reward_lags`, `reward_rate`, `self_history_lags`, `switch`, `reward_time_interaction`) live on the spec, while `BIN_S`/`SPLINE_DF`/`N_HISTORY_LAGS`/`TMAX` are **module constants** (one ladder-wide bin width so M0–M9 are nested/comparable — overview Open Question 1).
- [fit_timeresolved_model / cross_validate_timeresolved_model](shared-contracts.md#fit_timeresolved_model--cross_validate_timeresolved_model) — implement exactly; perfect-separation → `None`, not a crash; fitter returns `predicted` (per-bin intensity) + `dispersion`; CV is 10-fold and returns `cv_bits_per_event`.
- [timeresolved_gof](shared-contracts.md#timeresolved_gof) — implement here too; reuses `non_local_detector.model_checking.sorted_spikes`, does not re-implement time-rescaling.
- [aggregate_to_timebins](shared-contracts.md#aggregate_to_timebins) — the fitter consumes its output (one session's design).

## Tasks

- Add `TimeResolvedModelSpec` (frozen dataclass with `.formula()`), the module constants `BIN_S=0.015`/`TMAX=4.0`/`SPLINE_DF=4`/`N_HISTORY_LAGS=10`, and `MODEL_TIMERESOLVED` (the M0_const…M9_rew_x_time ladder) to `model_specs.py` after line 484, reusing `N_REWARD_LAGS`. Add to `__all__` if present.
- Create `src/continuum_swr_replay/analysis/replay/timeresolved_model.py` with `fit_timeresolved_model(design, spec, *, min_bins=200, min_events=20)`, `cross_validate_timeresolved_model(design, spec, *, n_folds=10, min_bins=200, seed=0)`, and `timeresolved_gof(design, fitted, *, adjust_for_short_trials=True)` per the contracts. Fit with `statsmodels.formula.api.glm(spec.formula(), data=design, family=Poisson(), offset=np.log(design['exposure']))`; wrap in try/except returning `None` (catch `PerfectSeparationError`/`LinAlgError` — the self-history short lags hit this). The fitter records `predicted` (=`results.predict()`), `dispersion` (=`deviance/df_resid`). CV folds split on unique `stop_id`, score per-bin LL and `cv_bits_per_event` vs `M0_const` refit on the same train fold. `timeresolved_gof` imports `TimeRescaling`/`point_process_residuals` — no local time-rescaling math.
- Export `fit_timeresolved_model`, `cross_validate_timeresolved_model`, `timeresolved_gof`, `MODEL_TIMERESOLVED`, `TimeResolvedModelSpec` from `analysis/replay/__init__.py`.
- Docstring on `timeresolved_model.py` states the estimator (per-bin Poisson, equal-exposure offset, spline time course) and how it differs from `rate_model.py` (per-stop, dwell offset) — this is the in-code expression of the "two families" distinction.

## Deliberately not in this phase

- No orchestration script and no `models_timeresolved/` output — Phase 3.
- No edits to the scratch scripts — Phase 4 points them at the orchestrator's output; here they still self-fit.
- No within-animal aggregation — that is a read-time operation (Phase 4 / [tidy schema](shared-contracts.md#tidy-output-schema)).
- No orchestration sweep or GOF *figures* — Phase 3 calls these primitives per session; Phase 5 plots the GOF story. (The interaction `M9` and switch `M8` ARE in the ladder now — not deferred.)

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_timeresolved_model.py::test_formula_strings` | `MODEL_TIMERESOLVED['M0_const'].formula() == 'count ~ 1'`; `M5_time` == `'count ~ bs(tc, df=4)'`; `M2_rewhist` includes `reward_lag1..reward_lag10` with no spline; `M7_time_hist_rew` includes the spline + `reward_lag1..10` + `onset_lag1..10`; `M9_rew_x_time` includes `reward_lag1:bs(tc, df=4)`. |
| `tests/test_timeresolved_model.py::test_gof_reuses_package` | `timeresolved_gof` returns `ks_statistic`/`ks_within_band`/`dispersion`; a planted unit-rate Poisson design gives `ks_within_band == True`; import path is `non_local_detector.model_checking.sorted_spikes`. |
| `tests/test_timeresolved_model.py::test_recovers_known_reward_effect` | Simulated design where rewarded bins have 2× the onset rate at matched `tc` → fitted `rr_reward_lag1` ≈ 2.0 within 10%. |
| `tests/test_timeresolved_model.py::test_recovers_flat_time_course` | Simulated design with a known peaked `tc` profile and no reward effect → `rr_reward_lag1` ≈ 1.0; spline captures the peak (predicted rate at the peak `tc` > at the edges). |
| `tests/test_timeresolved_model.py::test_separation_returns_none` | A design where `onset_lag1>0` perfectly predicts `count==0` → `fit_timeresolved_model` returns `None` (or a finite fit), never raises. |
| `tests/test_timeresolved_model.py::test_cv_folds_split_on_stop` | `cross_validate_timeresolved_model` never places the same `stop_id` in train and test; returns finite `cv_ll_per_bin_mean`. |

## Fixtures

- **Synthesized** designs in `conftest.py`: a `make_timeresolved_design(n_stops, reward_effect, time_profile, seed)` factory producing `aggregate_to_timebins`-shaped frames with a planted reward RR and `tc` profile — used by the recovery tests (the data-processing idiom: known ground truth).
- No real-data fixture needed at the unit level; real-data smoke is Phase 3.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Spec + fitter + CV match [shared-contracts.md](shared-contracts.md); return-dict keys mirror `fit_rate_model`.
- Recovery tests recover the planted effects within tolerance (not `assert True`); separation test proves no-crash.
- "Deliberately not in this phase" honored — no orchestrator, no script edits, no interaction model.
- Docstrings/test/module names free of plan references.
- `__all__`/`__init__` exports updated.
