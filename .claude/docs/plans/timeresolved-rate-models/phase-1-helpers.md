# Phase 1 — Extract shared helpers; refactor scratch scripts (behavior-preserving)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/_helpers.py:757](../../../../src/continuum_swr_replay/analysis/replay/_helpers.py) — `lagged_trial_rewards` (style/placement model for the new helpers; add the new functions nearby).
- [scripts/_selfhistory_glm.py](../../../../scripts/_selfhistory_glm.py) — `build_bins` (lines ~37-49) and the non-local `is_swr` rank-match in `main` (~57-61): the logic being lifted.
- [scripts/_reward_glm.py](../../../../scripts/_reward_glm.py) — `cells` (~33-39) + the rank-match (~44-48): second copy.
- [scripts/_timeresolved_kernel_glm.py](../../../../scripts/_timeresolved_kernel_glm.py) — `onset_cells` (~38-41) + rank-match (~81-85): third copy.
- [scripts/_reward_glm_plot.py](../../../../scripts/_reward_glm_plot.py) — `cells` + rank-match: fourth copy.

**Contracts referenced:**

- [attach_is_swr_to_onsets](shared-contracts.md#attach_is_swr_to_onsets) — define exactly this signature; do not weaken to `merge_asof`.
- [aggregate_to_timebins](shared-contracts.md#aggregate_to_timebins) — define exactly this; the (stop × bin) design is canonical.
- [POPULATIONS](shared-contracts.md#populations) — define the selector here.

## Tasks

- **Capture baselines BEFORE touching anything.** Run each scratch script and save its printed numeric output to `.claude/docs/plans/timeresolved-rate-models/baselines/<script>.txt` (the lag kernels, the self-history kernel, the reward RRs, the dissociation p-values). These are the Phase-1 regression oracle — the refactor must reproduce them.
- Add `attach_is_swr_to_onsets`, `aggregate_to_timebins`, and `POPULATIONS` to `_helpers.py` per [shared-contracts.md](shared-contracts.md). Export `attach_is_swr_to_onsets` and `aggregate_to_timebins` from `analysis/replay/__init__.py` alongside the existing helper exports.
- Refactor the five scratch scripts to call the new helpers instead of their inline copies, **changing no outputs**: `_reward_glm.py`, `_reward_glm_plot.py`, `_timeresolved_kernel_glm.py`, `_selfhistory_glm.py`, `_timeresolved_psth.py`. Each loses its private `cells`/`build_bins`/`onset_cells` and its rank-match block. For the reward-GLM per-(session, reward, tbin) view, derive it as a `groupby` on the `aggregate_to_timebins` (stop × bin) output (carry `reward_lag1` as the `reward` grouping column).
- Keep the scripts' formulas, bin widths, `TMAX`, `K`, `L`, populations, and printed/plotted outputs identical — this phase is pure extraction.

## Deliberately not in this phase

- No `MODEL_TIMERESOLVED` spec, no `fit_timeresolved_model`, no persisted `models_timeresolved/` output — that is Phase 2/3. The scripts still hand-roll their `smf.glm` calls here.
- No reduction of the scripts to "plotters" — they keep fitting inline for now (Phase 4 strips that once the orchestrator exists).
- Do not touch `MODEL_RATE` / `rate_model.py` / `fit_rate_models.py`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_timeresolved_helpers.py::test_attach_is_swr_match_rate` | On real `event_onset_table` + `event_table` (one cached session slice), `is_swr.notna()` ≥ 0.999 on non-local rows; counts of `is_swr==True/False` match `stop_table.n_non_local_SWR/_no_SWR` sums for that session. |
| `tests/test_timeresolved_helpers.py::test_aggregate_exposure_and_lags` | On a synthesized 2-stop fixture: `exposure` sums to total immobility; `onset_lag1` equals `count` shifted one bin within stop and is 0 at each stop's first bin; bins capped at `tmax`. |
| `scripts/_selfhistory_glm.py` re-run (slow) | Printed self-history kernel + reward RRs + dissociation p match `baselines/_selfhistory_glm.txt` to ≥3 sig figs. Mark slow. |
| `scripts/_timeresolved_kernel_glm.py` re-run (slow) | lag kernel + lag-1 dissociation match `baselines/_timeresolved_kernel_glm.txt`. Mark slow. |

## Fixtures

- **Synthesized** (`tests/conftest.py` or local): a 2-stop onset+stop frame with known durations and onset times, covering an empty bin, a stop boundary, and `tmax` truncation — for `test_aggregate_exposure_and_lags`.
- **Real slice**: one session's `event_onset_table`/`event_table`/`stop_table` rows (pull via `find_sessions(max_per_animal=1)[0]` keys, or check a small slice into fixtures) — for the match-rate test.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Every task implemented as specified; the five scripts call the shared helpers and define no private `cells`/`build_bins`/rank-match.
- "Deliberately not in this phase" honored — no spec/fitter/orchestrator added.
- Validation tests pass; slow re-runs marked; baselines reproduced.
- Tests exercise behavior (match rate, lag shift, exposure), not tautologies; fixtures shared, not copy-pasted.
- Docstrings/test/module names don't reference this plan or "Phase 1".
- The inline `cells`/`build_bins`/rank-match copies are actually deleted (no orphans).
