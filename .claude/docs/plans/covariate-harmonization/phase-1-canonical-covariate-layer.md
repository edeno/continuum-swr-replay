# Phase 1 — Canonical covariate layer + consume it everywhere

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Build the single canonical covariate source and point every model family at it, with
the renames/additions/removals from [shared-contracts.md](shared-contracts.md). No
re-fitting of the full dataset here (that's phase 2) — this phase ends with the code
coherent, unit tests green, and a one-session smoke fit per family showing the
canonical covariates flow.

**Inputs to read first:**
- [docs/covariate_dictionary.md](../../../../docs/covariate_dictionary.md) — the spec (definitions, windows, entry-rule).
- [shared-contracts.md](shared-contracts.md) — canonical names + `canonical_covariates*.pkl` schema.
- `_helpers.py:618` `recent_reward_rate_all_events`, `:704` `recent_trial_reward_rate` — reuse with k=10.
- `scripts/_bandit_csv_contingency.py:58-116` — the fitted-α leaky-Q (`_q_chosen`/`_fit_alpha`) + decoded per-port `p_reward`; extend to per-arm.
- `model_specs.py:194,205,310` + `MODEL_CHOICE` — predictor lists to rename/extend.
- `scripts/fit_uncertainty_rate.py:86,198`, `fit_swr_power_model.py:41`, `compare_event_source_models.py` + `choice_prediction.py` — consumers.

**Contracts referenced:** [canonical-covariate-names](shared-contracts.md#canonical-covariate-names), [canonical-covariates-pkl](shared-contracts.md#canonical-covariates-pkl) — do not weaken (names/schema are what phase 2 consumes).

## Tasks

- **Canonical covariate module.** Add `src/continuum_swr_replay/analysis/replay/canonical_covariates.py` (or a section in `_helpers.py`) with one function that, given a session's `trials` + parsed contingency (`p_reward_port{0..5}`) + stop/event tables, returns the two frames in [the schema](shared-contracts.md#canonical-covariates-pkl): per-arm (`reward_rate_arm_k10` via `recent_reward_rate_all_events(..., k=10)`; `p_reward_arm` = decoded port prob per arm; `q_arm` = per-leaf fitted-α leaky-Q) and per-stop (`reward_rate_trial_k10`, `q_current`, re-exported scalars). Promote `_q_chosen`/`_fit_alpha` out of `_bandit_csv_contingency.py` into this module and **extend the Q rollout to return all-arm Q** (not just the chosen arm), so `q_arm`/`q_current` come from one fit (α per animal, recorded per [contract invariant](shared-contracts.md#canonical-covariates-pkl)).
- **Precompute script.** `scripts/build_canonical_covariates.py` → writes `canonical_covariates_arm.pkl` + `canonical_covariates_stop.pkl` over all sessions. Smoke-test on one session first (Open Question 1), print per-session timing, extrapolate before the full precompute.
- **Spec renames/additions** (`model_specs.py`): `recent_reward_rate` → `reward_rate_arm_k10` in `_M3` (`:194`) and everywhere it appears (incl. the `swr_x_recent_reward_rate` interaction → `swr_x_reward_rate_arm_k10`), patch analogues. Add a **canonical value block** as additive rungs (do **not** touch M0–M7 / P0–P4 coefficients): e.g. `Mv_value = M5 + (p_reward_arm, q_arm)` for the arm hierarchy and the patch analogue, so the value covariates are present without perturbing the re-verified ladder.
- **Runner updates:** `fit_uncertainty_rate.py` — `CURATED_COVARIATES` swap `leaky_q_value_diff`→`q_current` (`:86`), drop `--q-alpha` (`:198`); merge `canonical_covariates_stop.pkl` instead of (or in addition to) `contingency_table_csv.pkl`. `fit_swr_power_model.py:41` — add `q_current` to `PREDICTORS`. `compare_event_source_models.py`/`choice_prediction.py` — rename + add `p_reward_arm`,`q_arm` to the choice dataset (merge `canonical_covariates_arm.pkl` by `(event_id, arm)`). Content runner (`fit_replay_content_models.py`) — merge the arm precompute and use canonical names.
- **Stop emitting `leaky_q_value_diff`** from `add_uncertainty_covariates` (`_helpers.py:883`) — or leave the function but remove it from all consumers; name the removal here so no orphan consumer remains.
- **Docstring/doc updates:** mark `docs/covariate_dictionary.md` §"implemented in code"; update the `fit_uncertainty_rate.py` module docstring (merge source + α-fit note).

## Deliberately not in this phase
- **No full-dataset re-fit / no parity check** — phase 2.
- **No rollup-table (`event_arm_table`) regen** — the stale 5-visit `recent_reward_rate` column stays baked but **unreferenced** (every consumer now reads `reward_rate_arm_k10` from the precompute); its removal is a follow-up at the next regen.
- No spline/self-history/spatial changes.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_canonical::test_arm_reward_window` | On a synthetic trial sequence, `reward_rate_arm_k10` = rewarded fraction of the last 10 visits to that arm; trunk arms 0.0; never-visited NaN. |
| `test_canonical::test_q_arm_matches_q_current` | `q_arm` at the current arm == `q_current` for that stop (same Q rollout); α recorded in attrs. |
| `test_canonical::test_p_reward_arm` | `p_reward_arm` equals the decoded contingency port prob per arm; `p_reward_arm` at current arm == `p_reward_current`. |
| `test_specs::test_predictors_resolve` | Every model spec's predictor list resolves to columns present in the canonical source (no stale `recent_reward_rate`/`leaky_q_value_diff`). |
| grep gate (CI/manual) | No `recent_reward_rate`, `recent_trial_reward_rate`, `leaky_q_value_diff` references remain in `src/`+`scripts/` consumers (only the canonical names). |
| smoke fit (slow) | One session per family fits with canonical covariates; coefficients finite; M0–M7 unchanged vs a one-session pre-refactor fit (the value block is additive). |

## Fixtures
Synthetic per-arm trial/visit sequences (known reward pattern) for the window/Q/value
unit tests; one real session (`chimi20200213_02_r1_HPC`) for the smoke fits.

## Review
Dispatch `code-reviewer` against the diff. Confirm: canonical names/schema match
[shared-contracts.md](shared-contracts.md); M0–M7 / P0–P4 predictor lists are
bit-identical (value covariates added only as new rungs); `leaky_q` has no remaining
consumer; the grep gate is clean; α-per-animal recorded; tests exercise the window/Q/
value definitions (not tautologies); no plan/phase references in code or test names.
