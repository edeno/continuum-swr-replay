# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

**Family A (the template to mirror — left UNTOUCHED by this work):**
- `src/continuum_swr_replay/analysis/replay/model_specs.py:113` — `ModelSpec` dataclass (`name, predictors, description, parent`). The T-series spec is a *parallel* dataclass, not a change to this one.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:437` — `MODEL_PER_STOP_RATE` (R0/R1/R2) and `:478` `Rlag`; `:476` `N_REWARD_LAGS = 10`. New `MODEL_PER_BIN_RATE` is appended after these.
- `src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:43` `fit_per_stop_rate_model`, `:160` `cross_validate_per_stop_rate_model` — the signature + return-dict shape the T-fitter mirrors.
- `scripts/fit_per_stop_rate_models.py:50` `RESPONSE_COLS`, `:58` `fit_per_stop_rate_hierarchy`, `:85` `main` → `Processed-Data/replay_content/models_per_stop_rate/all_models.pkl` — the orchestration + tidy-output pattern `fit_per_bin_rate_models.py` mirrors.

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
- One reusable home for the per-bin estimator: helpers in `_helpers.py`, spec in `model_specs.py`, fitter in `per_bin_rate_model.py`, orchestrator `fit_per_bin_rate_models.py`, tidy `models_per_bin_rate/all_models.pkl`.
- **A point-process GLM conditional-intensity ladder** (Truccolo–Eden–Brown 2005), `MODEL_PER_BIN_RATE` = M0_const … M9_rew_x_time: the four blocks (time-since-stop spline, self-history/autoregressive, reward [lag-1 / kernel / continuous rate], switch) toggled independently so each block's contribution is readable with and without the time course, plus the reward×time interaction. All at one bin width so the ladder is nested and AIC/CV/deviance/GOF are comparable. See [shared-contracts.md](shared-contracts.md#perbinratemodelspec).
- **Treat the events as a point process and check it as one:** time-rescaling KS, rescaled-ISI independence, point-process residuals, overdispersion — reusing `non_local_detector.model_checking.sorted_spikes` (Phase 5). The constant model is *expected* to fail the KS band; that failure motivates the history block.
- Four populations as outcome tags (like `RESPONSE_COLS`): `swr`, `non_local`, `non_local_swr`, `non_local_no_swr` — where `non_local` (pooled) is the one-population null tested against its split, not just a foil.
- The scratch scripts become thin plotters reading the tidy output, with their statistical content moved into the fitter/orchestrator.
- A `docs/rate_models.md` stating which family to cite for what (A = literature comparison + the Ambrose positive control; B = confound-free dissociation + the point-process model-checking).

### Non-Goals
- **Do not touch Family A** (`MODEL_PER_STOP_RATE`, `per_stop_rate_model.py`, `fit_per_stop_rate_models.py`, `models_per_stop_rate/`). It stays as the simple/literature layer.
- Do not re-derive scientific conclusions or change `#19c` — this is an engineering consolidation; numbers must reproduce, not change.
- Do not modify the canonical table builders (`tables.py`, `timebin_tables.py`) or regenerate any data table.
- **Deferred by decision (not in this plan):** arm/well/patch identity covariates (hard to interpret); slow session drift (belongs in the **state-space model**, not as a fixed covariate here); cross-/ensemble-history between event types (stretch goal). These are named in `docs/rate_models.md` as future work, not built.
- **Scope limit — 4 s truncation:** `TMAX = 4.0` means Family B models only the first 4 s of each immobility stop. ~20% of onsets occur after 4 s (median onset ≈2.4 s, long tail); those events are right-truncated out of `count`, the GOF `n_events`, and `cv_bits_per_event`. This matches the scratch scripts (so parity is preserved, not regressed), but every Family-B conclusion is *conditional on the first 4 s*. State this in `docs/rate_models.md`; the long tail is future work (raise `TMAX` or add a censoring term).
- Not consolidating `_lagged_reward_kernel.py` or `_periswitch_rate.py` (per-stop GEE exploratory scripts — they belong to Family A's exploratory edge, out of scope here).
- The switch covariate and the reward×time interaction **are** in scope now (M8/M9 in the ladder) — this supersedes the earlier deferral of the interaction.

### Dependency policy
No new third-party dependencies. Uses the already-present `statsmodels` (incl. `statsmodels.formula.api` + patsy `bs()`), `scipy.stats.wilcoxon`, pandas/numpy/pyarrow.

## Metrics
- **Qualitative reproduction (the core success test):** at the unified 15 ms bins, the headline conclusions survive — lag-1 reward effect larger for SWR-coupled than for NL-no-SWR (RR ordering, not the exact `1.44/3.10/1.08`), the dissociation sign holds per-animal (n/5), and the self-history kernel is refractory <60 ms / bursty ~100 ms. The exact old numbers are *not* the target (different binning); a changed *conclusion* is a failure, a shifted third decimal is not. (See Open Question 4.)
- **Model-checking passes the literature bar:** time-rescaling KS for the full models (M7/M8) lands inside the 1.36/√n band on most sessions, while `M0_const` lands outside it (the expected non-Poisson failure that motivates the history block); dispersion → ~1 once the history block is in. The `cv_bits_per_event` is positive and increases monotonically up the ladder where blocks add real signal.
- **De-duplication:** the `is_swr`-rank-match and the cells/build-bins aggregation each exist in exactly one place after Phase 1 (grep finds one definition).
- **Reproducibility:** `uv run python scripts/fit_per_bin_rate_models.py` regenerates `models_per_bin_rate/all_models.pkl` from the canonical tables with no manual steps.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Refactor silently changes a scratch-script number | Phase 1 captures baseline outputs *before* refactoring and asserts byte/numeric equality after (the optimization-task idiom). |
| `aggregate_to_timebins` over-generalizes and becomes a leaky abstraction covering 3 incompatible cases | Phase 1 fixes the contract to the (stop × bin) design with optional self-history lags; the per-(session, reward, tbin) reward-GLM view is a downstream `groupby`, not a separate mode. |
| 15 ms self-history bins are not a clean multiple of the 4 ms dense grid | Family B builds its own bins from `stop_table.immobility_time` + onset times (continuous), independent of the 4 ms dense table — already how `_selfhistory_glm.build_bins` works; preserve it. |
| Short-lag refractoriness causes GLM perfect-separation (coef → −∞, RR=0) in the history models (M3/M4/M6/M7/M8/M9) | Carry it as a known property (real refractoriness); the fitter catches it → `None` (never crashes). **But `None` is non-random** — sparser sessions separate more easily, so per-animal medians/the n/5 dissociation could be read over a biased survivor set. Mitigation: Phase 3 **logs the count of `None` per (model, outcome)** so the qualitative-reproduction test isn't read over a silently filtered set; if drops are material, an L2-penalized GLM (`fit_regularized`) or Firth is the noted fallback. |
| Per-bin design is large (≈8.7 M rows at 15 ms) — and now the *whole ladder* runs at 15 ms, not just `Thist` | Orchestrator fits per session (each ≤ ~150 k bins), never holding all sessions' bins at once; build the design once per session and fit all M0–M9 against it (same bins); Phase 3 smoke-tests timing before the full run. |
| Models compared on different binnings would make AIC/CV meaningless | One ladder-wide `BIN_S` (Open Question 1) — all of M0–M9 share the 15 ms grid so AIC/deviance/CV/GOF are directly comparable; the spec carries no per-model `bin_s`. |
| Time-rescaling invalid for short/variable-length stops (right-censoring) | Reuse `TimeRescaling(..., trial_id=stop_id, adjust_for_short_trials=True)` — the Wiener (2003) censoring adjustment is built into `non_local_detector`; do not roll our own. |

## Rollout Strategy
All-at-once, additive. Family A is unchanged, so existing analyses and `models_per_stop_rate/` are unaffected. Family B is new code + a new output directory (`models_per_bin_rate/`). The scratch scripts are replaced in place (Phase 1 refactors them; Phase 4 reduces them to plotters) — no parallel v1/v2 left behind; Phase 4 names every scratch file whose statistical body is removed.

## Open Questions
1. **One ladder-wide bin width (decided: yes).** For a *nested, AIC/CV/deviance/GOF-comparable* ladder, every model must be fit on the same bins — you cannot compare models fit on different binnings. So `BIN_S = 0.015` is a module constant shared by all of M0…M9 (fine bins are needed anyway for the self-history block and for time-rescaling GOF), not a per-spec field. `tmax`, `spline_df`, `N_HISTORY_LAGS` are likewise ladder-wide. Settled in [shared-contracts.md](shared-contracts.md#perbinratemodelspec).
2. **CV unit + folds (decided).** 10-fold over **stops** (not bins), mirroring `cross_validate_per_stop_rate_model`'s stop-based folds; score held-out per-bin Poisson log-likelihood **and** `cv_bits_per_event` vs `M0_const` (the interpretable point-process measure). Settled in [shared-contracts.md](shared-contracts.md#fit_per_bin_rate_model--cross_validate_per_bin_rate_model).
3. **reward×time interaction + switch (decided: in).** `M9_rew_x_time` (reward_lag1 × spline) and `M8_full_switch` are shipped specs, not deferred — they test whether reward modulates the time course and whether patch-switch adds beyond the kernel. Supersedes the earlier deferral.
4. **Parity targets must be re-baselined at the unified bins.** The old `Tlag` lag-1 RR `1.44/3.10/1.08` was measured at the scratch scripts' 0.25 s / 15 ms mixed bins. Re-fitting the equivalent reward terms at the uniform 15 ms grid will shift the *absolute* coefficients slightly. Two separate guards: **Phase 1** preserves the *scratch scripts'* exact numbers (behavior-preserving refactor — byte/numeric equality against its own pre-refactor baseline, at the scripts' existing binning); **Phase 3** establishes the new ladder's baseline at uniform 15 ms bins, where the guard is **qualitative reproduction** (lag-1 reward RR ordering SWR > NL-no-SWR, dissociation sign + per-animal n/5, refractory<60 ms / bursty~100 ms history kernel), not the exact 1.44/3.10/1.08 values. Flag any model whose conclusion (not just its third decimal) changes.

## Estimated Effort
- Phase 1: ~150 LOC new (`_helpers.py`), ~−250 LOC across the 5 scripts (net shrink).
- Phase 2: ~120 LOC (`per_bin_rate_model.py`) + ~40 LOC (`model_specs.py`) + ~80 LOC tests.
- Phase 3: ~120 LOC (`fit_per_bin_rate_models.py`).
- Phase 4: ~−200 LOC (scripts reduced to plotters) + ~80 lines `docs/rate_models.md`.
