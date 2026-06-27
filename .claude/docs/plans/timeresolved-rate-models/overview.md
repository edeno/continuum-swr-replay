# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

**Family A (the template to mirror — left UNTOUCHED by this work):**
- `src/continuum_swr_replay/analysis/replay/model_specs.py:113` — `ModelSpec` dataclass (`name, predictors, description, parent`). The T-series spec is a *parallel* dataclass, not a change to this one.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:437` — `MODEL_RATE` (R0/R1/R2) and `:478` `Rlag`; `:476` `N_REWARD_LAGS = 10`. New `MODEL_TIMERESOLVED` is appended after these.
- `src/continuum_swr_replay/analysis/replay/rate_model.py:43` `fit_rate_model`, `:160` `cross_validate_rate_model` — the signature + return-dict shape the T-fitter mirrors.
- `scripts/fit_rate_models.py:50` `RESPONSE_COLS`, `:58` `fit_rate_hierarchy`, `:85` `main` → `Processed-Data/replay_content/models_rate/all_models.pkl` — the orchestration + tidy-output pattern `fit_timeresolved_models.py` mirrors.

**Family B sources to consolidate (all under `scripts/`, all scratch, nothing imports them):**
- `_timeresolved_psth.py` — plot only (rate vs `t_since_pump`).
- `_reward_glm.py` + `_reward_glm_plot.py` — per-(session, reward, tbin) cells, `count ~ bs(tc, df=4) + reward`.
- `_timeresolved_kernel_glm.py` — per-(stop, tbin) cells, `count ~ bs(tc, df=4) + reward_lag1..K`.
- `_selfhistory_glm.py` — per-(stop, 15 ms bin), `count ~ bs(tc, df=4) + reward_lag1..K + onset_lag1..L`.
- Each independently re-implements: the `stop_id → reward_lag` map, the **non-local `is_swr` attach via within-stop time-rank**, and a `cells()`/`build_bins()` aggregation. These are the duplications Phase 1 removes.

**Data tables consumed (read-only; not modified by any phase):**
- `src/.../analysis/replay/_helpers.py:154` `immobility_bout_intervals`, `:356` `assign_bout`, `:757` `lagged_trial_rewards` — existing primitives the new helpers build on.
- Artifacts under `Processed-Data/replay_content/`: `stop_table.pkl` (per-stop, has `reward_lag1..10`, `n_SWR`, `n_non_local*`), `event_onset_table.pkl` (sparse onsets, has `t_since_stop_onset`, `t_since_pump`), `stop_timebin_table.parquet` (dense 4 ms bins), `event_table.pkl` (per-event `is_swr`, `event_start`, `stop_id`).
- `scripts/prepare_replay_content_data.py` `find_sessions` — session discovery the new orchestrator reuses.

## Scope and dependency policy

### Goals
- One reusable home for the time-resolved estimator: helpers in `_helpers.py`, spec in `model_specs.py`, fitter in `timeresolved_model.py`, orchestrator `fit_timeresolved_models.py`, tidy `models_timeresolved/all_models.pkl`.
- T-series hierarchy mirroring R-series: **T0** baseline time course, **T1** + immediate reward, **Tlag** + reward kernel, **Thist** + self-history (autocorrelation).
- Four populations as outcome tags (like `RESPONSE_COLS`): `swr`, `non_local`, `non_local_swr`, `non_local_no_swr`.
- The scratch scripts become thin plotters reading the tidy output, with their statistical content moved into the fitter/orchestrator.
- A `docs/rate_models.md` stating which family to cite for what (A = literature comparison + the Ambrose positive control; B = confound-free dissociation).

### Non-Goals
- **Do not touch Family A** (`MODEL_RATE`, `rate_model.py`, `fit_rate_models.py`, `models_rate/`). It stays as the simple/literature layer.
- Do not re-derive scientific conclusions or change `#19c` — this is an engineering consolidation; numbers must reproduce, not change.
- Do not modify the canonical table builders (`tables.py`, `timebin_tables.py`) or regenerate any data table.
- No new modeling features beyond what the scratch scripts already do (no reward×time interaction promotion, no cross-coupling, no hazard model — those are noted future work, not this plan).
- Not consolidating `_lagged_reward_kernel.py` or `_periswitch_rate.py` (per-stop GEE exploratory scripts — they belong to Family A's exploratory edge, out of scope here).

### Dependency policy
No new third-party dependencies. Uses the already-present `statsmodels` (incl. `statsmodels.formula.api` + patsy `bs()`), `scipy.stats.wilcoxon`, pandas/numpy/pyarrow.

## Metrics
- **Parity (the core success test):** after Phase 3, the tidy output reproduces the scratch-script headline numbers within floating tolerance — Tlag lag-1 RR `1.44 / 3.10 / 1.08` (SWR / NL-w-SWR / NL-no-SWR), Thist self-history kernel (refractory <60 ms, bursty ~100 ms), and the lag-1 reward dissociation p ≈ 6.5e-8 (4/5 animals). These are the regression targets.
- **De-duplication:** the `is_swr`-rank-match and the cells/build-bins aggregation each exist in exactly one place after Phase 1 (grep finds one definition).
- **Reproducibility:** `uv run python scripts/fit_timeresolved_models.py` regenerates `models_timeresolved/all_models.pkl` from the canonical tables with no manual steps.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Refactor silently changes a scratch-script number | Phase 1 captures baseline outputs *before* refactoring and asserts byte/numeric equality after (the optimization-task idiom). |
| `aggregate_to_timebins` over-generalizes and becomes a leaky abstraction covering 3 incompatible cases | Phase 1 fixes the contract to the (stop × bin) design with optional self-history lags; the per-(session, reward, tbin) reward-GLM view is a downstream `groupby`, not a separate mode. |
| 15 ms self-history bins are not a clean multiple of the 4 ms dense grid | Family B builds its own bins from `stop_table.immobility_time` + onset times (continuous), independent of the 4 ms dense table — already how `_selfhistory_glm.build_bins` works; preserve it. |
| `Thist` short-lag refractoriness causes GLM perfect-separation (coef → −∞, RR=0) | Carry it as a known property (it is real refractoriness); the fitter must not crash on it — catch/record `None` per session as `fit_rate_model` does, and the docs note flags the boundary value. |
| Per-bin design is large (≈8.7 M rows at 15 ms, ≈32.5 M at 4 ms) | Orchestrator fits per session (each ≤ ~150 k bins), never holding all sessions' bins at once; Phase 3 smoke-tests timing before the full run. |

## Rollout Strategy
All-at-once, additive. Family A is unchanged, so existing analyses and `models_rate/` are unaffected. Family B is new code + a new output directory (`models_timeresolved/`). The scratch scripts are replaced in place (Phase 1 refactors them; Phase 4 reduces them to plotters) — no parallel v1/v2 left behind; Phase 4 names every scratch file whose statistical body is removed.

## Open Questions
1. **Spline df and bin widths as spec fields vs constants?** Best answer: put `bin_s`, `tmax`, `df`, `self_history_lags` on the `TimeResolvedModelSpec` (so T0–Thist can differ — Thist uses 15 ms bins, the others 250 ms cells) rather than module constants. Settled in [shared-contracts.md](shared-contracts.md#timeresolvedmodelspec).
2. **CV unit for the time-resolved fitter?** Best answer: K-fold over **stops** (not bins), mirroring `cross_validate_rate_model`'s stop-based folds, scoring held-out per-bin Poisson log-likelihood. Settled in [shared-contracts.md](shared-contracts.md#fit_timeresolved_model--cross_validate_timeresolved_model).
3. **Should `reward_lag1 × bs(t)` interaction be a 5th T-model?** Deferred — the `interaction=True` path exists in `_timeresolved_kernel_glm.fit_pop` but the per-session main-effect models already resolve the pooled-vs-per-session concern; leave as a noted extension in `docs/rate_models.md`, not a shipped spec.

## Estimated Effort
- Phase 1: ~150 LOC new (`_helpers.py`), ~−250 LOC across the 5 scripts (net shrink).
- Phase 2: ~120 LOC (`timeresolved_model.py`) + ~40 LOC (`model_specs.py`) + ~80 LOC tests.
- Phase 3: ~120 LOC (`fit_timeresolved_models.py`).
- Phase 4: ~−200 LOC (scripts reduced to plotters) + ~80 lines `docs/rate_models.md`.
