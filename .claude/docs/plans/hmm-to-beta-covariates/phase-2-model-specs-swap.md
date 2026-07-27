# Phase 2 — Model-specs swap, HMM deletion, and re-fit

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Repoint every rate and content rung from HMM covariates to their beta
replacements, delete the now-dead HMM creation/attach/fetch code, and re-fit the
ladders. Phase 1 guarantees the beta `_z` columns already exist in the tables.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/model_specs.py:834-959](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — rate `_BLOCK_F`/`_BLOCK_COLS` (`:838-841`), G/S rungs (`:877-878`, `:895-898`), content ladder + key-term map (`:912-959`); and `Mv` at `:226`.
- [src/continuum_swr_replay/analysis/replay/tables.py:163-165,1315-1377,1720](../../../../src/continuum_swr_replay/analysis/replay/tables.py) — `EVENT_ARM_STANDARDIZED_COLS`, the `q_arm_hmm`/`swr_x_q_arm_hmm` attach block (`q_arm`/`swr_x_q_arm` already alongside), schema list.
- [scripts/content/fit_swr_contrast_content.py:152-155](../../../../scripts/content/fit_swr_contrast_content.py) and [scripts/rate/fit_swr_contrast_rate.py:115-129](../../../../scripts/rate/fit_swr_contrast_rate.py) — where `_x_current` interactions are built from `_z` scalars.
- [src/continuum_swr_replay/analysis/replay/cohort_audit.py:64-68,229](../../../../src/continuum_swr_replay/analysis/replay/cohort_audit.py) — audit covariate list.

**Contracts referenced:**

- [Contract 1](shared-contracts.md#contract-1--the-covariate-mapping) — the exact name substitutions; [Contract 1 notes](shared-contracts.md#contract-1--the-covariate-mapping) — `q_arm` is the sole value construct, `swr_x_q_arm` already exists, content-entropy maps to `arm_uncertainty_z`.

## Tasks

- **Rate ladder (`model_specs.py:838-841`).** In `_BLOCK_F`/`_BLOCK_COLS`:
  `VALUE`: `hmm_relative_value → switch_advantage_mu_preoutcome_sess`;
  `RPE`: `hmm_rpe → rpe`; `UPDATE`: `hmm_value_update → global_mu_update_abs_postoutcome_sess`;
  `UNCERT`: `reward_entropy → beta_reward_entropy`. G/S rung wiring (`:871-901`)
  is name-agnostic — no rung-list edit needed.
- **Content ladder (`model_specs.py:912-959`).** `C4_armvalue`/`CS5_armvalue`:
  `q_arm_hmm → q_arm`, `swr_x_q_arm_hmm → swr_x_q_arm`. `C6_rpe`
  (`rpe_x_current`), `C7_value` (`relval_x_current`), `C9_update`
  (`update_x_current`) keep their interaction *names* but repoint their sources
  in the runner (below). `C8_uncert` (`entropy_x_current`): resolve overview Open
  Question 2 — map to `arm_uncertainty_z` (drop `C8` if it duplicates `C13`, else
  keep as the `×is_current` form). Update `SWR_CONTRAST_CONTENT_ARM_KEY_TERM`
  (`:947-959`) to match. `Mv` (`:226`): `q_arm_hmm → q_arm`.
- **Interaction construction in runners** (`fit_swr_contrast_content.py:152-155`,
  `fit_swr_contrast_rate.py:115-129`): build `rpe_x_current = cur * rpe_z`,
  `relval_x_current = cur * switch_advantage_mu_preoutcome_sess_z`,
  `update_x_current = cur * global_mu_update_abs_postoutcome_sess_z`,
  `entropy_x_current = cur * arm_uncertainty_z`.
- **`tables.py`:** drop `q_arm_hmm` from `EVENT_ARM_STANDARDIZED_COLS` (`:163`),
  remove the `q_arm_hmm`/`swr_x_q_arm_hmm` attach lines (`:1321-1323`, `:1364`)
  and the schema entry (`:1720`). `q_arm`/`swr_x_q_arm` already attached — verify
  they remain.
- **Delete the HMM code paths (name them explicitly):**
  - `attach_hmm_arm_value` and `_Q_PORT_HMM_COLS` — `covariates.py:36,773-825`.
  - `add_hmm_decision_variables` — `qvals_table.py:250-314`; and `HMM_KEY`,
    `HMM_COLS`, `HMM_DV_COLS`, `HMM_Q_PORT_COLS` config (`:66-88`) + their fetch/merge
    (`:366-367`, `:473-474`) + the HMM entries in `QVALS_STANDARDIZED_COLS` (`:86-87`).
  - HMM names in `covariate_timing.py` (`hmm_rpe`, `hmm_value_update`,
    `hmm_relative_value`, `q_arm_hmm`, `reward_entropy`, `state_entropy`,
    `swr_x_q_arm_hmm`, and the HMM sources in `DERIVED_COVARIATE_SOURCES:51-68`).
  - HMM names in `cohort_audit.py:64-68,229`.
- **Regenerate `event_arm_table`/`qvals_table` once more** (from cache) so the
  dropped HMM columns disappear and the schema is beta-only.
- **Re-fit** (smoke → full, see Validation): `fit_swr_contrast_rate.py`,
  `fit_swr_contrast_content.py`, and the arm hierarchy `Mv`/`Mu`/`Mvu`
  (`fit_replay_content_models.py`). Regenerate `swr_contrast_*_term_diagnostics.csv`.

## Deliberately not in this phase

- Editing `plot_swr_contrast_coefs.py` — Phase 3 (needs these refit pickles first).
- Any docs / synthesis / memory edits — Phase 4.
- Changing the rate baseline or spatial content base rungs.

## Validation slice

| Test | Asserts |
| --- | --- |
| grep clean | no `hmm_`, `q_arm_hmm`, `reward_entropy`, `state_entropy` in `src/` or `scripts/` (docs deferred to Phase 4). |
| specs resolve | every rung in `SWR_CONTRAST_RATE_*`, `SWR_CONTRAST_CONTENT_ARM_LADDER`, and the arm hierarchy references only columns present in the regenerated tables (assert each predictor ∈ table columns). |
| fit smoke (1 session) | rate + content ladders fit on one session; all beta rungs converge; per-term VIF < 10; record wall-clock + peak memory, extrapolate to 218. |
| full re-fit | ≥213/218 sessions converge for beta value rungs; no new quasi-separated sessions vs the HMM term-diagnostics baseline; `n_animals_used == 5`. (slow, tmux) |
| value sign | `swr_x_q_arm` (CS5) median is **negative** across sessions (the beta value-avoidance; a positive median means the wrong column was wired). |

Mark fit smoke and full re-fit slow/integration.

## Fixtures

- One `_session_cache/` session for the fit smoke test.
- The Phase-1 regenerated tables (beta `_z` columns present).
- Prior HMM term-diagnostics CSV as the convergence-comparison baseline.

## Review

Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- All name substitutions match shared-contracts Contract 1 exactly.
- Every listed HMM code path is actually deleted — no orphaned
  `attach_hmm_arm_value`, `add_hmm_decision_variables`, or dead `HMM_*` config.
- `swr_x_q_arm` sign is negative (beta value-avoidance) — flag if it reproduces
  the retired HMM `+1.06`.
- Fit convergence didn't regress vs the HMM baseline.
- No plot or doc file touched; no docstring references this plan.
