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
  - For each `outcome` in `POPULATIONS` × each `model` in `MODEL_TIMERESOLVED`: build the session designs via `aggregate_to_timebins(stop_table, POPULATIONS[outcome](onsets), bin_s=spec.bin_s, tmax=spec.tmax, reward_lags=spec.reward_lags, self_history_lags=spec.self_history_lags)`, then per session call `fit_timeresolved_model` (+ `cross_validate_timeresolved_model`), collecting one tidy row per `(outcome, model, animal, day, epoch)`.
  - Build the design **per session** (group by SESSION_COLS) so all sessions' bins are never resident at once (memory: overview Risk row).
  - Save to `Processed-Data/replay_content/models_timeresolved/all_models.pkl`. Log rows/sessions/outcomes like `fit_rate_models.main`.
- Note in the script docstring: `Thist` uses 15 ms bins (heavier); the design for `Thist` is rebuilt at its own `bin_s` (the loop keys aggregation off `spec.bin_s`, so cells and 15 ms bins coexist without special-casing).
- Add a `--max-per-animal` smoke flag (like `regenerate_stop_table.py`) for the smoke-test task below.

## Deliberately not in this phase

- No changes to the scratch scripts (Phase 4 repoints them).
- No within-animal aggregation in the saved file — left to read time per the [tidy schema](shared-contracts.md#tidy-output-schema).
- No new models beyond T0/T1/Tlag/Thist.

## Validation slice

| Test | Asserts |
| --- | --- |
| smoke: `uv run python scripts/fit_timeresolved_models.py --max-per-animal 1` (slow) | Completes on 5 sessions; prints per-session timing; produces rows for all 4 outcomes × 4 models; record wall-clock to extrapolate the full run (overview Metrics). |
| `tests/test_fit_timeresolved_models.py::test_tidy_schema` | Output has columns `[outcome, model, animal, day, epoch, rr_reward_lag1, cv_ll_per_bin_mean, ...]`; `outcome` values ⊆ POPULATIONS keys; `model` values ⊆ MODEL_TIMERESOLVED keys. |
| **PARITY** (slow, the core success test) | From the full `all_models.pkl`: per-session-median `rr_reward_lag1` for `Tlag` = **1.44 / 3.10 / 1.08** (swr / non_local_swr / non_local_no_swr) ±0.03; `Thist` self-history kernel reproduces refractory (RR≈0) <60 ms and bursty (RR>1) ~100 ms; paired SWR-vs-non_local_no_swr `reward_lag1` dissociation **p ≈ 6.5e-8**, 4/5 animals. Compare against `baselines/` from Phase 1. |

## Fixtures

- Real-data: the canonical Part B tables already on disk (`Processed-Data/replay_content/`). The smoke uses `find_sessions(max_per_animal=1)`. No synthesized fixture — parity is the test, against the Phase-1 baselines.
- **Long-running**: full run ≈ the per-stop fit-models scale × 4 models × 4 outcomes; smoke-test timing first (the long-running idiom), extrapolate, and run the full job in `tmux` if > 5 min (document the session name in the PR).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Orchestrator mirrors `fit_rate_models.py`; output matches the [tidy schema](shared-contracts.md#tidy-output-schema).
- **Parity holds** — the headline numbers reproduce the scratch-script baselines within tolerance (this is the whole point; if they don't, the consolidation changed the science and must be reconciled before merge).
- Per-session design build (no all-sessions-resident memory blowup); smoke timing recorded; full run done in tmux if needed.
- "Deliberately not in this phase" honored.
- Docstrings/names free of plan references.
