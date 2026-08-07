# Phase 4 — Promote exploratory content models into the audited pipeline

> **SUPERSEDED (2026-07-29):** do not execute this phase. Its generic
> cohort/diagnostic/refit work and content-model promotion are owned by
> [`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md), principally
> Phases 8–10 and 13. Retain this file only as historical rationale; reconsider a
> still-novel destination estimand as a new downstream question after Phase 13.

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **HISTORICAL STATUS:** it was not started before being superseded. The content
> scripts named below are not an active implementation queue.

Two of this session's content findings live in scratch scripts that fit per-session
and save **only coefficient medians** — dropping `n_events`, standard errors,
p-values, within-group-centered VIF/condition number, FDR across the family, and
model provenance. Until they carry those diagnostics and live in the audited
model-comparison pipeline, the findings are **exploratory** and cannot be paper
claims. This phase promotes them:

- **`_content_reward_interactions.py`** → the `is_current ×` {reward outcome, value, exploration} content modulation (claim **#19f**, the "reward pushes non-SWR content *off* the current arm" result, duration-controlled by `tsc_x_current`).
- **`_content_patch_destination.py`** → the patch-destination pattern (off-current content samples the third/novel patch, avoids prev/last-rewarded).

**Depends on phase 2** (the harmonized covariates + the `p_reward_arm`/`q_*` value
block must exist so the promoted models enter the canonical terms). Benefits from
phase 3 but does not require it — the SWR split here reads the **already-canonical**
`event_arm_table.is_swr` (merged in the replay builder), not an ancillary re-derivation.

**Inputs to read first:**
- `scripts/_content_reward_interactions.py` (whole file): `:42-44` SCALARS/INTERACTIONS, `:59-66` the `is_current ×` terms + the z-scored-**linear** `tsc_x_current` duration control, `:70-83` `_fit` (saves only `coef_*` — the diagnostics gap), `:100-121` the one-at-a-time-on-M5 design.
- `scripts/_content_patch_destination.py` (whole file): `:21-39` hard-count patch fractions / role means, descriptive only (no inference/uncertainty).
- Audited pipeline to fold into: `analysis/replay/model_specs.py:122` `ModelSpec`, `:196` `_M5`, `:205` `MODEL_HIERARCHY`, the scalar→role-interaction rule `:171-188`; `analysis/replay/fitting.py:31`/`:114`/`:451` (within-group rank + complete-case guards), `:420` (condition-number/VIF on the **raw** matrix — to be centered); runner `scripts/fit_replay_content_models.py`; verifier `scripts/_verify_content_models.py`.
- The audit findings this phase closes: [04 (#4 reward/value/switch interactions exploratory; #6 patch-destination descriptive; #7 centered VIF; Next Checks 3/6)](../../reviews/04_content_behavior_covariates_review.md).

## Tasks

- **Add an interaction family as proper `ModelSpec` rungs** in `model_specs.py`: the `is_current ×` {`reward_lag1` (within-well outcome contrast), `p_reward_arm`/`p_reward_current` (value), `recent_switch_rate` (exploration), `q_current` (learned value)} interactions plus the `is_current ×` time-since-stop duration control, layered on M5 (e.g. `Mi_rew`, `Mi_value`, `Mi_switch`, `Mi_q`, each `M5 + tsc_x_current + <one interaction>`). The interactions are mutually collinear (each is `is_current ×` an event-constant scalar), so they stay **one-at-a-time** rungs, not a combined spec — encode that in the ladder, not in a scratch loop. Use the **canonical** covariate names from phase 1.
- **Replace the z-scored-linear duration control with a nonlinear one** and report both: a spline basis on `log1p(time_since_stop)` × `is_current` as the primary `tsc` control (audit 04 #4 "sensitivity to nonlinear time-since-stop"), keeping the linear z-scored term as a reported sensitivity.
- **Emit full diagnostics from the audited runner** (`fit_replay_content_models.py`, extended — not a scratch script): per-session `n_events`, coefficient SE, p-value, **within-group-centered** VIF/condition number, plus family-level FDR across the interaction rungs and per-animal sign consistency, and model provenance (git hash + params). Fit the compared rungs on a **common complete-case cohort** so coefficients are comparable across the family.
- **Center the conditional VIF/condition-number diagnostic** (`fitting.py:420`): compute collinearity diagnostics on the **within-group-centered** design (matching the conditional estimand and the rank screen at `:31`), not the raw predictor matrix. This corrects the diagnostic for every conditional content fit, not just the interactions — flag the (reported-diagnostics-only, coefficients-unchanged) effect.
- **Promote patch destination to a conditional model**: a multinomial-over-patches (current/prev/other) conditional fit on switch events with role contrasts, replacing the hard-count means, carrying session/animal-level inference, a **posterior-mass (soft)** response sensitivity alongside the hard counts, and uncertainty estimates. Add it to the audited runner + verifier.
- **Retire the scratch scripts**: delete `scripts/_content_reward_interactions.py` and `scripts/_content_patch_destination.py` once the audited rungs reproduce their headline numbers (the promotion replaces them — no parallel scratch path). Record the headline reproduction in the PR description.
- **Docs:** `paper/claims_table.md` — move #19f and the patch-destination pattern from **exploratory** to supported, citing the audited rungs + the diagnostics (n_events/SE/p/FDR/sign-consistency); note that full hierarchical/cluster-robust inference remains the deferred cross-cutting item ([overview](overview.md)). Extend `_verify_content_models.py` to cover the new rungs.

## Deliberately not in this phase
- **No full hierarchical / cluster-robust inference** across all claims — that methodology change spans every family and stays the deferred cross-cutting workstream ([overview](overview.md#folded-in-audit-findings-claudereview)). This phase reports session-level stats + per-animal sign consistency (what the audit asks for #19f), not a mixed model.
- No re-decode; no new behavioral covariate definitions (phase 1 owns those).
- No change to the SWR-split source — it reads the already-canonical `event_arm_table.is_swr` (phase 3 fixes the *ancillary* scripts, not this one).

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_content_interactions::test_rungs_resolve` | each `Mi_*` rung's predictors resolve to present canonical columns; the interactions are built as `is_current ×` the named scalar. |
| `test_content_interactions::test_one_at_a_time` | the mutually-collinear interactions are separate rungs (a combined fit is rank-deficient and rejected by the within-group screen). |
| `test_content_interactions::test_diagnostics_emitted` | promoted fits carry `n_events`, SE, p, centered VIF/condition number, FDR, per-animal sign — none NaN-by-omission. |
| `test_fitting::test_centered_vif` | VIF/condition number computed on the within-group-centered design (a between-event-only collinear pair that cancels conditionally is *not* flagged). |
| `test_patch_destination::test_conditional_multinomial` | patch-destination rung fits current/prev/other contrasts; emits soft-response sensitivity + per-session inference. |
| reproduction (slow) | audited `Mi_rew`/patch rungs reproduce the retired scripts' headline numbers (non-SWR `rew_x_current` ≈ −0.76 duration-controlled; off-current → third patch) within noise. |

## Fixtures
The regenerated `event_arm_table.pkl` / `event_patch_table.pkl` + `stop_table.pkl` (post phase-1) for one real session for the rung/diagnostic tests; the full set for the reproduction run.

## Review
Dispatch `code-reviewer`. Confirm: the interactions live as audited `ModelSpec` rungs (not a scratch loop) using canonical covariate names; full diagnostics (n_events/SE/p/centered-VIF/FDR/sign) are emitted and non-trivial; the VIF centering matches the conditional estimand and only changes reported diagnostics (coefficients unchanged for existing fits); patch destination is a conditional model with inference + soft-response sensitivity; the two scratch scripts are **deleted** (no parallel path) after headline reproduction is recorded; claims #19f + patch-destination are re-labelled with the diagnostics and the still-deferred hierarchical-inference caveat; no plan/phase references in code/test names.
