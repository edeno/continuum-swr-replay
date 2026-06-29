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

- **Canonical covariate function (one source for builder + backfill).** Add `src/continuum_swr_replay/analysis/replay/canonical_covariates.py` with one function that, given a session's CSV-derived trial sequence (per-arm visit+reward) + parsed contingency (`p_reward_port{0..5}`) + Q-learner, returns the per-arm covariates (`reward_rate_arm_k10` via `recent_reward_rate_all_events(..., k=10)`; `p_reward_arm`; `q_arm`) keyed `(event_id, arm)` and the per-stop covariates (`reward_rate_trial_k10`, `q_current`) keyed `stop_id`, per [the schema](shared-contracts.md#where-the-covariates-live). Promote `_q_chosen`/`_fit_alpha` out of `_bandit_csv_contingency.py` into this module and **extend the Q rollout to return all-arm Q** (not just the chosen arm), one fit (α per animal, recorded).
- **Update the table builders** to call this function so future regens are canonical: `tables.py:1247-1252` (`build_event_arm_table`) emits `reward_rate_arm_k10`/`p_reward_arm`/`q_arm` instead of the 5-visit `recent_reward_rate`; `build_event_patch_table` aggregates them; the stop-table builder emits `reward_rate_trial_k10`/`q_current`.
- **Backfill the existing pickles in place** (no regen): `scripts/backfill_canonical_covariates.py` loads `event_arm_table.pkl` / `event_patch_table.pkl` / `stop_table.pkl`, recomputes **only** the covariate columns via the same function, **asserts the response/posterior columns and keys are byte-identical**, writes to a temp, validates, swaps (keep a backup). Cheap — CSV/contingency/Q inputs, no `load_data`, no posterior rollup; smoke on one session first, then all.
- **Spec renames/additions** (`model_specs.py`): `recent_reward_rate` → `reward_rate_arm_k10` in `_M3` (`:194`) and everywhere it appears (incl. the `swr_x_recent_reward_rate` interaction → `swr_x_reward_rate_arm_k10`), patch analogues. Add a **canonical value block** as additive rungs (do **not** touch M0–M7 / P0–P4 coefficients): e.g. `Mv_value = M5 + (p_reward_arm, q_arm)` for the arm hierarchy and the patch analogue, so the value covariates are present without perturbing the re-verified ladder.
- **Runner updates (now just read the updated tables — no sidecar merge):** `fit_uncertainty_rate.py` — `CURATED_COVARIATES` swap `leaky_q_value_diff`→`q_current` (`:86`), drop `--q-alpha` (`:198`); `q_current`/`reward_rate_trial_k10` now come from `stop_table`. `fit_swr_power_model.py:41` — add `q_current` to `PREDICTORS`. `compare_event_source_models.py`/`choice_prediction.py` — use canonical names; `p_reward_arm`/`q_arm`/`reward_rate_arm_k10` are in `event_arm_table`. Content runner (`fit_replay_content_models.py`) — use canonical names (columns already present).
- **Stop emitting `leaky_q_value_diff`** from `add_uncertainty_covariates` (`_helpers.py:883`) — or leave the function but remove it from all consumers; name the removal here so no orphan consumer remains.
- **Docstring/doc updates:** mark `docs/covariate_dictionary.md` §"implemented in code"; update the `fit_uncertainty_rate.py` module docstring (merge source + α-fit note).

## Deliberately not in this phase
- **No full-dataset re-fit / no parity check** — phase 2.
- **No posterior re-rollup / no `load_data`** — the backfill recomputes *only* the behavioral covariate columns; the response/posterior columns are preserved byte-identical. (The old 5-visit `recent_reward_rate` column is *replaced* in place, not left stale.)
- No spline/self-history/spatial changes.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_canonical::test_arm_reward_window` | On a synthetic trial sequence, `reward_rate_arm_k10` = rewarded fraction of the last 10 visits to that arm; trunk arms 0.0; never-visited NaN. |
| `test_canonical::test_q_arm_matches_q_current` | `q_arm` at the current arm == `q_current` for that stop (same Q rollout); α recorded in attrs. |
| `test_canonical::test_p_reward_arm` | `p_reward_arm` equals the decoded contingency port prob per arm; `p_reward_arm` at current arm == `p_reward_current`. |
| `test_specs::test_predictors_resolve` | Every model spec's predictor list resolves to columns present in the canonical tables (no stale `recent_reward_rate`/`leaky_q_value_diff`). |
| `test_backfill::test_response_preserved` | After backfill, `arm_count_hard/soft`, `arm_posterior_mass`, offsets, keys, `is_swr` are byte-identical to the pre-backfill table (only covariate columns changed). |
| `test_backfill::test_builder_equals_backfill` | On a sample session, the updated `build_event_arm_table` covariate columns equal the backfill output (same function, same inputs → no divergence). |
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
