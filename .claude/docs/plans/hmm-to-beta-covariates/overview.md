# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `src/continuum_swr_replay/analysis/behavior/qvals_table.py:66-88` — HMM fetch
  config (`HMM_KEY`, `HMM_COLS`, `HMM_DV_COLS`, `HMM_Q_PORT_COLS`) and
  `QVALS_STANDARDIZED_COLS`. Beta DV columns already fetched into `COVARIATES`
  (`:97-110`): `rpe`, `switch_advantage_mu_preoutcome_sess`,
  `global_mu_update_abs_postoutcome_sess`, `betadist_var{1..6}`. **Changed** in
  Phase 1 (add beta `_z` cols + `beta_reward_entropy`) and Phase 2 (delete HMM
  fetch).
- `src/continuum_swr_replay/analysis/behavior/qvals_table.py:250-314`
  (`add_hmm_decision_variables`) — creates `hmm_q_port{i}`, `hmm_rpe`,
  `hmm_value_update`, `hmm_relative_value`. **Deleted** in Phase 2.
- `src/continuum_swr_replay/analysis/replay/covariates.py:773-825`
  (`attach_hmm_arm_value` → `q_arm_hmm`). Beta twins already present:
  `attach_contingency_value_block` → `q_arm` (`:688-770`), `attach_arm_uncertainty`
  → `arm_uncertainty` (`:828-881`). **`attach_hmm_arm_value` deleted** in Phase 2.
- `src/continuum_swr_replay/analysis/replay/tables.py:163-165`
  (`EVENT_ARM_STANDARDIZED_COLS`, currently includes `q_arm_hmm`),
  `:1321-1323` (`q_arm_hmm` attach), `:1364` (`swr_x_q_arm_hmm`), `:1720` (schema).
  **Changed** in Phase 2 (drop `q_arm_hmm`; `q_arm`/`swr_x_q_arm` already present).
- `src/continuum_swr_replay/analysis/replay/covariate_timing.py:28-75` — leakage
  registry. **Changed** in Phase 1 (register beta names per shared-contracts
  Contract 3) and Phase 2 (drop HMM names).
- `src/continuum_swr_replay/analysis/replay/model_specs.py:834-946` — the rate
  `_BLOCK_F`/`_BLOCK_COLS` (`:838-841`), G/S rungs (`:877-878`, `:895-898`),
  content ladder (`C4`/`C6`/`C7`/`C8`/`C9`/`CS5`, `:917-938`), and `Mv`
  (`:226`). **Changed** in Phase 2 — the single source of truth for rungs.
- `src/continuum_swr_replay/analysis/replay/cohort_audit.py:64-68,229` — audit
  covariate list with HMM names. **Changed** in Phase 2.
- `scripts/content/fit_swr_contrast_content.py:152-155` and
  `scripts/rate/fit_swr_contrast_rate.py:115-129` — build the `_x_current`
  interaction columns from `_z` scalars. **Changed** in Phase 2.
- `scripts/figures/plot_swr_contrast_coefs.py:52-56` (`SD` list), `:140-180`
  (hardcoded `rate_rows`/`mod_rows`/`con_rows` term tuples + labels). **Changed**
  in Phase 3. The plot engine reads term names, not specs — mechanical literal swap.
- `docs/covariate_dictionary.md`, `docs/behavior_models.md`,
  `docs/replay_rate_content_swr_synthesis.md`, `docs/models_table.md`,
  `docs/design/content_model_design.md`, and memory notes
  `project_swr_value_content_construct_dependence`, `project_arm_uncertainty_content`.
  **Changed** in Phase 4.

## Scope and dependency policy

### Goals

- Remove all HMM behavioral covariates from active model rungs and figures.
- Replace value / relative-value / RPE / value-update / reward-entropy with beta
  equivalents (shared-contracts Contract 1); derive `beta_reward_entropy`.
- Re-fit rate + content ladders and arm hierarchy; regenerate the coefficient
  forest plot; reconcile docs/memory to the beta value sign.

### Non-Goals

- **No re-ingest of the behavior model** — use the beta fit already feeding
  `q_arm`/`arm_uncertainty` (confirmed with user).
- **No new decoding / re-decode.** Uses existing `event_arm_table` / `stop_table`
  substrate; only behavioral covariate columns change.
- **No change to the rate baseline (time-course / reward-history / self-history)
  or the spatial content base** — only the value/RPE/update/entropy covariates.
- **No new scientific analyses** (the within-arm horse-race / uncertainty×value
  exploratory work is separate; not part of this migration).

## Metrics

- **Zero-drift on unchanged columns:** after Phase 1 regeneration, every column
  of `event_arm_table`/`qvals_table` that is not new or HMM must be bit-identical
  to the pre-change table (the `arm_uncertainty` precedent — memory
  `project_arm_uncertainty_content` regenerated with "only 3 new cols").
- **No HMM names remain** in `src/`, `scripts/`, active model rungs, or figures
  (grep clean except Phase-4-pending docs and archived reviews).
- **Refit sanity:** beta value rungs converge on ≥213/218 sessions with no new
  quasi-separation vs the HMM fit; term-diagnostics CSV regenerated.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Beta `switch_advantage`/`global_mu_update` on a different scale than HMM → singular per-session covariance (the `arm_uncertainty` SD-0.03 problem) | Standardize (`_z`) before use, as shared-contracts Contract 1 requires; Phase 1 validation checks per-session VIF/condition number on a smoke session. |
| `beta_reward_entropy` derivation ill-defined when `betadist_var → 0` (μ near 0/1) | Contract 2 clips μ and var; Phase 1 validates finite range on real data. |
| Value sign flip silently breaks a downstream claim/figure that assumed HMM `+` | Phase 4 explicitly rewrites the value narrative; grep for `+1.06`/"prefers high value" in docs. |
| Re-fit is long-running (218 sessions × ladder) | Phase 2 smoke-tests one session first, extrapolates, then runs full in tmux. |

## Rollout Strategy

Replace-in-place — this is a research repo with no external consumers (paper text
has zero HMM mentions). Fitted pickles and figures are regenerated, not
versioned in parallel. No feature flag, no deprecation window. The migration is
not "done" until Phase 4 reconciles the docs, so the ladders and docs never
disagree in a shipped state (each phase is internally consistent: Phase 2 leaves
beta-only ladders + refit pickles; Phase 3 matches figures; Phase 4 matches prose).

## Open Questions

1. **Value-update variant** — `global_mu_update_abs` vs `chosen_leaf_mu_update_abs`
   (both exist, `qvals_table.py:108-109`). Best answer: `global_mu_update_abs`
   (matches `hmm_value_update`'s Σ|ΔQ| global scope). Revisit if the refit shows
   it degenerate; the chosen-leaf variant is the fallback.
2. **Content entropy rung** — map `entropy_x_current` onto existing
   `arm_uncertainty_z` (per-arm) rather than the scalar `beta_reward_entropy`.
   Best answer: yes (Contract 1 note) — but this makes `C8_uncert` nearly a
   duplicate of `C13_armuncert`; Phase 2 should either drop `C8` as redundant or
   keep it as the `×is_current` interaction form. Decide at Phase 2.

## Estimated Effort

Phase 1 ~150 LOC (qvals cols + timing reg + regen script). Phase 2 ~200 LOC of
edits across specs/runners + deletions (net negative after HMM removal). Phase 3
~40 LOC of literal edits. Phase 4 docs-only. Dominated by re-fit wall-clock, not LOC.
