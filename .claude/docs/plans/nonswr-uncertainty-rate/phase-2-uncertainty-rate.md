# Phase 2 — Non-local rate vs uncertainty/value (SWR vs non-SWR)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Fit the Family B per-bin rate model **per population** (`swr` /
`non_local_swr` / `non_local_no_swr`) with behavioral-uncertainty and
task-grounded value covariates, and test whether they elevate the non-SWR rate
more than the SWR rate — controlling for the within-stop time course,
self-history, the reward kernel, and `recent_trial_reward_rate`.

**Inputs to read first:**

- [shared-contracts.md#contingency-table-schema](shared-contracts.md#contingency-table-schema) — the Phase-1 columns this phase merges by `stop_id` (nominal value, `rpe`, `value_diff_patch`, `block_number`, `trials_since_block_change`).
- `src/continuum_swr_replay/analysis/replay/_helpers.py:757` `lagged_trial_rewards` — placement/style for the new covariate helper; `:880` `aggregate_to_timebins` (`stop_covariates` at `:889`).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:529` `PerBinRateModelSpec` (`formula()` `:580`, `reward_rate` toggle `:586`); `:525` `STOP_COVARIATES`.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:39` `_design_columns`; `:58` `fit_per_bin_rate_model`.
- `scripts/_drift_task_correlation.py` — structure to mirror for the per-session runner (BLAS pinned, per-session fit, tidy save).

**Contracts referenced:**

- [contingency-table-schema](shared-contracts.md#contingency-table-schema) — read-only; do not recompute the parse here.

## Tasks

- **`add_uncertainty_covariates(stop_table, *, switch_window=5, q_alpha=0.3)`**
  in `_helpers.py` (exported), computing four model-free signals per stop from
  existing columns (group by `(animal, day, epoch)`, sort by `stop_idx`,
  preserve index):
  - `omission_streak`: leading consecutive omissions = count of leading 0s in
    `reward_lag1, reward_lag2, …` (stop at first 1; NaN lag terminates).
  - `recent_switch_rate`: `(1 - is_stay).shift(1).rolling(switch_window,
    min_periods=1).mean()` within session (prior stops only — no leakage).
  - `wsls_deviation`: 1.0 unless `(is_after_reward & is_stay)` or
    `(is_after_omission & ~is_stay)`, else 0.0.
  - `leaky_q_value_diff`: per-arm leaky reward-rate `Q_a ← (1-q_alpha)Q_a +
    q_alpha·reward_outcome` updated in stop order (init 0.5); covariate =
    `sort(Q)[-1] - sort(Q)[-2]` evaluated **before** the stop's update
    (experienced-value choice margin). `q_alpha`/`switch_window` are free params
    (the RL model would fit them) — document as such.

- **Generalize `PerBinRateModelSpec` with `extra_covariates: tuple[str,...] = ()`**
  (add after `self_history_basis`; default `()` → shipped specs byte-identical).
  In `formula()` append `" + ".join(extra_covariates)` as plain linear terms;
  add them to `_design_columns`. Leave the `reward_rate` toggle as-is.

- **Runner `scripts/fit_uncertainty_rate.py`** (mirror
  `_drift_task_correlation.py`; BLAS pinned; `--max-per-animal` smoke first).
  Per session: load `stop_table`; `add_uncertainty_covariates`; left-merge
  `contingency_table.pkl` on `stop_id`; **drop the first session per animal**
  (exposure); `attach_is_swr_to_onsets`. Per population in
  `{swr, non_local_swr, non_local_no_swr}`: build the design via
  `aggregate_to_timebins(..., stop_covariates=(*STOP_COVARIATES,
  *UNCERTAINTY_COVARIATES, *VALUE_COVARIATES), self_history_basis=N_HISTORY_BASIS)`
  and fit a spec instantiated in the runner (not added to `MODEL_PER_BIN_RATE`):

  ```python
  UNCERTAINTY_COVARIATES = ("omission_streak", "recent_switch_rate",
                            "wsls_deviation", "leaky_q_value_diff")
  VALUE_COVARIATES = ("p_reward_current", "rpe", "value_diff_patch",
                      "trials_since_block_change")
  MU = PerBinRateModelSpec(
      "MU_uncertainty", True, N_REWARD_LAGS, True, N_HISTORY_LAGS, False, False,
      "uncertainty/value beyond time/reward/history/reward-rate",
      "M7r_time_histcos_rew", self_history_basis=N_HISTORY_BASIS,
      extra_covariates=(*UNCERTAINTY_COVARIATES, *VALUE_COVARIATES))
  ```

  Collect per-(session, population) the `coef_*` for every uncertainty/value
  covariate → `Processed-Data/replay_content/models_per_bin_rate/uncertainty_rate.pkl`.

- **Report + figure.** Per covariate: per-population median RR, %-positive,
  per-animal n/5, and the **paired** `non_local_no_swr` vs `swr` and vs
  `non_local_swr` Wilcoxon. Figure `paper/figures/uncertainty_rate_by_population.png`:
  one panel per covariate, three populations' per-session coefficient boxplots
  (RR scale), dissociation p annotated. Report results honestly — a null is a
  valid outcome; do not assert a magnitude.

- **Peri-block-change latency (anchor).** Using `trials_since_block_change`, plot
  per-population mean rate (or residual) vs trials-since-change (0..~15), to see
  whether non-SWR rate transiently rises after a contingency change and decays
  while SWR doesn't — the SWR-resolved Sasaki test. Note (overview Risks) that
  changes are uncued, so expect a *lagged* response if any.

## Deliberately not in this phase

- **No re-parsing of contingencies** — consume `contingency_table.pkl` (Phase 1).
- **No replay-direction split** — rate-only.
- **No promotion into `MODEL_PER_BIN_RATE` / `all_models.pkl` regen** — the spec
  lives in the runner; promote only if the dissociation replicates.
- **No fitting the real RL model** — `leaky_q_value_diff` + nominal `p(R)` stand in.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_uncertainty::test_omission_streak` | `reward_lag1..3=[0,0,1]`→2; `[1,…]`→0; NaN-terminated. |
| `test_uncertainty::test_recent_switch_rate` | Uses only prior stops (shifted); known stay/switch sequence → hand-computed rolling fraction. |
| `test_uncertainty::test_wsls_deviation` | Truth table: win-stay/lose-shift→0; win-shift/lose-stay→1. |
| `test_uncertainty::test_leaky_q_value_diff` | 2-arm hand-worked sequence matches the recurrence; pre-update margin. |
| `test_uncertainty::test_no_mutation` | helper returns same index/rowcount; input unmodified. |
| `test_per_bin_rate_model::test_extra_covariates_formula` | `extra_covariates=("a","b")` → `+ a + b` in `formula()` and `_design_columns`; default `()` leaves shipped specs unchanged. |
| `scripts/fit_uncertainty_rate.py --max-per-animal 1` (slow) | Runs end-to-end across the three populations; writes the tidy table with all `coef_*`. |

Mark the smoke run slow.

## Fixtures

- **Synthetic**: a small per-session stop frame with known `current_arm` /
  `reward_outcome` / `is_stay` / `reward_lag*` covering win-stay/lose-shift/
  win-shift/lose-stay, an omission streak, and a two-arm value divergence —
  drives the covariate unit tests. A tiny synthetic `contingency_table` slice
  for the merge.
- **Real-data smoke**: one session via `--max-per-animal 1` (needs Phase-1
  `contingency_table.pkl`).

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- Covariate definitions match the docstrings exactly (`leaky_q` pre-update
  margin; `recent_switch_rate` prior-stops-only — no current-stop leakage).
- `extra_covariates` defaults to `()`; shipped `MODEL_PER_BIN_RATE` /
  `all_models.pkl` schema unchanged; the uncertainty spec lives in the runner.
- First session is excluded; value covariates come from the merged
  `contingency_table` (schema-bound), not recomputed.
- Dissociation is *reported*, not asserted to a magnitude; tests exercise
  hand-computed values, not tautologies; fixtures shared.
- Docstrings/test/module names don't reference this plan.
