# Phase 3 — fit_timeresolved_models.py → tidy models_timeresolved/all_models.pkl

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [scripts/fit_rate_models.py:50](../../../../scripts/fit_rate_models.py) — `RESPONSE_COLS`, `:58` `fit_rate_hierarchy`, `:85` `main` → `models_rate/all_models.pkl`: the orchestration + tidy-output pattern to mirror exactly.
- [scripts/build_timebin_tables.py](../../../../scripts/build_timebin_tables.py) — how the Part B tables are loaded (`event_onset_table.pkl`, `stop_timebin_table.parquet`) and validated; reuse the load idiom.
- [scripts/_selfhistory_glm.py](../../../../scripts/_selfhistory_glm.py) — the current per-session × population fit loop whose statistical content moves here (after Phase 1 it already uses the shared helpers).

**Contracts referenced:**

- [POPULATIONS](shared-contracts.md#populations), [aggregate_to_timebins](shared-contracts.md#aggregate_to_timebins), [fit/CV](shared-contracts.md#fit_timeresolved_model--cross_validate_timeresolved_model), [tidy schema](shared-contracts.md#tidy-output-schema) — all consumed here.

## Tasks

- Write `scripts/fit_timeresolved_models.py` mirroring `fit_rate_models.py`'s structure:
  - Load `stop_table.pkl`, `event_onset_table.pkl`, `event_table.pkl` once; `attach_is_swr_to_onsets` once.
  - For each `outcome` in `POPULATIONS`, **build the design ONCE per session** via `aggregate_to_timebins(stop_table, POPULATIONS[outcome](onsets), bin_s=BIN_S, tmax=TMAX, reward_lags=N_REWARD_LAGS, self_history_lags=N_HISTORY_LAGS, stop_covariates=("recent_trial_reward_rate", "is_after_switch"))` — the full column set including M2r's and M8's predictors. Every one of the **11 models** in `MODEL_TIMERESOLVED` (M0_const…M9_rew_x_time) is fit against that *same* design (its `formula()` selects its subset of columns), so all share identical bins and are AIC/CV/GOF-comparable. Per (session, model) call `fit_timeresolved_model` + `cross_validate_timeresolved_model` + `timeresolved_gof`, collecting one tidy row per `(outcome, model, animal, day, epoch)` with coef/CV/GOF columns.
  - Build per session (group by SESSION_COLS) so all sessions' bins are never resident at once (memory: overview Risk row).
  - Save to `Processed-Data/replay_content/models_timeresolved/all_models.pkl`. Log rows/sessions/outcomes like `fit_rate_models.main`.
- Note in the script docstring: the **whole ladder runs at `BIN_S = 0.015`** (one design per session-population, 11 fits against it) — no per-model rebinning; this is what makes the nested comparison + GOF valid.
- Add a `--max-per-animal` smoke flag (like `regenerate_stop_table.py`) for the smoke-test task below.

## Deliberately not in this phase

- No changes to the scratch scripts (Phase 4 repoints them).
- No within-animal aggregation in the saved file — left to read time per the [tidy schema](shared-contracts.md#tidy-output-schema).
- No new models beyond the 11 `MODEL_TIMERESOLVED` keys (M0_const…M9_rew_x_time) defined in Phase 2.

## Validation slice

| Test | Asserts |
| --- | --- |
| smoke: `uv run python scripts/fit_timeresolved_models.py --max-per-animal 1` (slow) | Completes on 5 sessions; prints per-session timing; produces rows for all 4 outcomes × 11 models (and asserts M2r_rewrate and M8_full_switch are present, not silently `None` — the C1 regression guard); record wall-clock to extrapolate the full run (overview Metrics). |
| `tests/test_fit_timeresolved_models.py::test_tidy_schema` | Output has columns `[outcome, model, animal, day, epoch, rr_reward_lag1, cv_ll_per_bin_mean, cv_bits_per_event, dispersion, ks_statistic, ks_within_band, ...]`; `outcome` values ⊆ POPULATIONS keys; `model` values ⊆ MODEL_TIMERESOLVED keys (M0_const…M9). |
| **QUALITATIVE REPRODUCTION** (slow, the core success test) | From the full `all_models.pkl` at the unified 15 ms bins: per-session-median `rr_reward_lag1` is **larger for SWR-coupled than for non_local_no_swr** (the dissociation ordering, not the old `1.44/3.10/1.08` values — those were at mixed binning); the self-history kernel is refractory (RR≈0) <60 ms and bursty (RR>1) ~100 ms; paired SWR-vs-non_local_no_swr `reward_lag1` dissociation holds with the same sign in **4/5 animals**. GOF: `M0_const` `ks_within_band` fraction ≪ `M7`'s; `dispersion` drops toward ~1 once the history block enters. A changed *conclusion* fails; a shifted absolute coefficient does not (overview Open Question 4 / Metrics). |

## Fixtures

- Real-data: the canonical Part B tables already on disk (`Processed-Data/replay_content/`). The smoke uses `find_sessions(max_per_animal=1)`. No synthesized fixture — parity is the test, against the Phase-1 baselines.
- **Long-running**: full run ≈ the per-stop fit-models scale × 4 models × 4 outcomes; smoke-test timing first (the long-running idiom), extrapolate, and run the full job in `tmux` if > 5 min (document the session name in the PR).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Orchestrator mirrors `fit_rate_models.py`; output matches the [tidy schema](shared-contracts.md#tidy-output-schema).
- **Qualitative reproduction holds** (overview Metrics / Open Question 4) — the dissociation ordering + sign + per-animal n/5 + the refractory/bursty history kernel survive at the unified bins; **report the new absolute lag-1 RRs in the PR** (so there's a concrete number to regress against next time), plus the per-(model,outcome) `None` count (the separation survivorship guard). If a *conclusion* changed (not just a decimal), reconcile before merge.
- Per-session design build (no all-sessions-resident memory blowup); smoke timing recorded; full run done in tmux if needed.
- "Deliberately not in this phase" honored.
- Docstrings/names free of plan references.
