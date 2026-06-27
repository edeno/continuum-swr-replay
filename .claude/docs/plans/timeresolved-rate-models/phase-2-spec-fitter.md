# Phase 2 — MODEL_TIMERESOLVED spec + fit/CV primitives

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/model_specs.py:113](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — `ModelSpec` (the dataclass the new spec parallels) and `:437`-`:484` `MODEL_RATE`/`Rlag`/`N_REWARD_LAGS` (append after these).
- [src/continuum_swr_replay/analysis/replay/rate_model.py:43](../../../../src/continuum_swr_replay/analysis/replay/rate_model.py) — `fit_rate_model` return-dict shape (`coef_*`/`bse_*`/`pval_*`/`rr_*` + stats) to mirror; `:160` `cross_validate_rate_model` (the stop-fold CV pattern to mirror).

**Contracts referenced:**

- [TimeResolvedModelSpec + MODEL_TIMERESOLVED](shared-contracts.md#timeresolvedmodelspec) — implement exactly; `bin_s`/`reward_lags`/`self_history_lags`/`spline_df` live on the spec.
- [fit_timeresolved_model / cross_validate_timeresolved_model](shared-contracts.md#fit_timeresolved_model--cross_validate_timeresolved_model) — implement exactly; perfect-separation → `None`, not a crash.
- [aggregate_to_timebins](shared-contracts.md#aggregate_to_timebins) — the fitter consumes its output (one session's design).

## Tasks

- Add `TimeResolvedModelSpec` (frozen dataclass with `.formula()`) and `MODEL_TIMERESOLVED` (T0/T1/Tlag/Thist) to `model_specs.py` after line 484, reusing `N_REWARD_LAGS`. Add both to the module's `__all__` if it has one.
- Create `src/continuum_swr_replay/analysis/replay/timeresolved_model.py` with `fit_timeresolved_model(design, spec, *, min_bins=200, min_events=20)` and `cross_validate_timeresolved_model(design, spec, *, n_folds=5, min_bins=200, seed=0)` per the contract. Fit with `statsmodels.formula.api.glm(spec.formula(), data=design, family=Poisson(), offset=np.log(design['exposure']))`; wrap in try/except returning `None` (catch `PerfectSeparationError`/`LinAlgError` — the `Thist` short lags hit this). CV folds split on unique `stop_id`, not rows.
- Export `fit_timeresolved_model`, `cross_validate_timeresolved_model`, `MODEL_TIMERESOLVED`, `TimeResolvedModelSpec` from `analysis/replay/__init__.py`.
- Docstring on `timeresolved_model.py` states the estimator (per-bin Poisson, equal-exposure offset, spline time course) and how it differs from `rate_model.py` (per-stop, dwell offset) — this is the in-code expression of the "two families" distinction.

## Deliberately not in this phase

- No orchestration script and no `models_timeresolved/` output — Phase 3.
- No edits to the scratch scripts — Phase 4 points them at the orchestrator's output; here they still self-fit.
- No within-animal aggregation — that is a read-time operation (Phase 4 / [tidy schema](shared-contracts.md#tidy-output-schema)).
- Do not add the `reward_lag1 × bs(t)` interaction as a 5th model (deferred — overview Open Question 3).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_timeresolved_model.py::test_formula_strings` | `MODEL_TIMERESOLVED['T0'].formula() == 'count ~ bs(tc, df=4)'`; `Tlag` includes `reward_lag1..reward_lag10`; `Thist` includes `onset_lag1..onset_lag10`. |
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
