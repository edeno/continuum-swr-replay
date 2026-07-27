# Phase 1 — Beta covariate engineering

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make every beta replacement covariate *exist and standardized* in the source
tables, without touching any model rung yet. At the end of this phase the tables
carry the new columns; the ladders still name HMM covariates (swapped in Phase 2).

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:85-125](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py) — `QVALS_STANDARDIZED_COLS` (`:85`) and `COVARIATES` (`:96`), where beta DV columns (`rpe`, `switch_advantage_mu_preoutcome_sess`, `global_mu_update_abs_postoutcome_sess`, `betadist_var{i}`) already arrive.
- [src/continuum_swr_replay/analysis/behavior/qvals_table.py:240-314](../../../../src/continuum_swr_replay/analysis/behavior/qvals_table.py) — `add_experienced_value_columns` (`rpe` at `:245`) and `add_hmm_decision_variables` (the pattern to mirror; the leaf value/variance columns `Q{i+1}`/`betadist_var{i}` are the entropy inputs).
- [src/continuum_swr_replay/analysis/replay/covariate_timing.py:28-75](../../../../src/continuum_swr_replay/analysis/replay/covariate_timing.py) — leakage registry sets.

**Contracts referenced:**

- [Contract 1 (mapping)](shared-contracts.md#contract-1--the-covariate-mapping), [Contract 2 (beta entropy)](shared-contracts.md#contract-2--beta_reward_entropy-definition), [Contract 3 (timing)](shared-contracts.md#contract-3--timing-class-of-each-replacement) — this phase materializes all three. Do not weaken the timing classifications.

## Tasks

- **Add `beta_reward_entropy` to `qvals_table.py`.** Insert a function that
  computes the per-stop scalar from the per-leaf `Q{i+1}` (mean) and
  `betadist_var{i}` (variance) columns using the Contract 2 formula, called in
  the same place the DV columns are assembled (near
  `add_experienced_value_columns`, `qvals_table.py:244-246`). Add the column name
  to `COVARIATES` (`qvals_table.py:96`).
- **Standardize the beta DV columns.** Add `rpe`,
  `switch_advantage_mu_preoutcome_sess`, `global_mu_update_abs_postoutcome_sess`,
  and `beta_reward_entropy` to `QVALS_STANDARDIZED_COLS`
  (`qvals_table.py:85-88`) so their `_z` forms are produced by the existing
  `attach_standardized` call (`:482`). Do **not** remove the HMM names from this
  list yet (Phase 2 does that) — both live here transiently so the table can be
  regenerated once.
- **Register beta names in `covariate_timing.py`** per Contract 3: add
  `switch_advantage_mu_preoutcome_sess` and `beta_reward_entropy` to
  `PRE_OUTCOME_SAFE` (`covariate_timing.py:37`); add
  `global_mu_update_abs_postoutcome_sess` to `POST_OUTCOME` (`:28`). `rpe` is
  already in `POST_OUTCOME` (`:30`). Leave the HMM entries in place (Phase 2
  removes them).
- **Regenerate `qvals_table` only** (no re-decode, no `load_data`): rerun the
  `qvals_table` build (`python -m ...qvals_table --write`). Output:
  `qvals_table.pkl` with the new `_z` columns.
  **EXECUTION FINDING (2026-07-13):** `event_arm_table` does **not** need
  regenerating here. The fit runners (`fit_swr_contrast_content.py:127-148`,
  and the rate runner) read `qvals_table` directly and merge/standardize the
  scalar interaction bases themselves — the `_z` scalars are *not* stored in
  `event_arm_table`. The per-arm betas (`q_arm`, `arm_uncertainty`, `swr_x_q_arm`)
  already exist in `event_arm_table`. So the `event_arm_table` regen (to *drop*
  the now-dead `q_arm_hmm` per-arm column) is deferred to Phase 2, where the HMM
  attach is removed. The zero-drift check was therefore run on `qvals_table`
  (96/96 shared cols identical, 5 new cols: `beta_reward_entropy` + 4 `_z`).
- **No model_specs, runner, plot, or doc edits in this phase.**

## Deliberately not in this phase

- Swapping any rung name or deleting HMM columns/functions — Phase 2 (keeps the
  table regen a single zero-drift step, decoupled from the fit).
- Building `_x_current` / `swr_x_` interaction columns for the beta names — those
  are constructed in the fit runners (Phase 2).
- Deriving `chosen_leaf_mu_update` as an alternative — only the chosen
  `global_mu_update_abs` variant (overview Open Question 1).

## Validation slice

| Test | Asserts |
| --- | --- |
| `beta_reward_entropy` finite | on the full `qvals_table`, column is finite (no NaN/inf) for all stops with defined leaf beliefs; monotone check: entropy decreases as mean `betadist_var` decreases (spot-check 3 stops). |
| new `_z` columns present | `event_arm_table` / `qvals_table` contain `rpe_z`, `switch_advantage_mu_preoutcome_sess_z`, `global_mu_update_abs_postoutcome_sess_z`, `beta_reward_entropy_z`, each with mean≈0, sd≈1. |
| **zero-drift on unchanged columns** | every non-new, non-HMM column of the regenerated `event_arm_table` is bit-identical (`pd.testing.assert_frame_equal`) to a pre-change snapshot (row count 593217 baseline; only the new `_z` cols added). |
| per-session conditioning (smoke) | on one session, a within-arm design matrix with the 4 standardized beta covariates has condition number < 30 (guards the singular-covariance risk). |

Mark the full-table regeneration slow.

## Fixtures

- Pre-change snapshot of `event_arm_table.pkl` (copy before regenerating) for the
  zero-drift comparison.
- One real session already in `_session_cache/` (e.g.
  `chimi20200213_02_r1_HPC`) for the conditioning smoke test.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- Every task implemented as specified; "Deliberately not in this phase" honored.
- The zero-drift test passes and is a real `assert_frame_equal`, not a shape check.
- `beta_reward_entropy` matches Contract 2 exactly (clipping, mean over leaves).
- Timing classifications match Contract 3; no HMM entry removed yet.
- No model_specs / runner / plot / doc file is touched.
- Docstrings/column names don't reference this plan or phase numbers.
