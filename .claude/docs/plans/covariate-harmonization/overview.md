# Overview — Scope, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Covariate-computation layer (functions to reuse with canonical params):
- `src/continuum_swr_replay/analysis/replay/_helpers.py:618` `recent_reward_rate_all_events` — per-arm reward rate; **call with k=10** (currently 5) for `reward_rate_arm_k10`.
- `_helpers.py:704` `recent_trial_reward_rate` — all-arm 10-trial rate; reuse as-is for `reward_rate_trial_k10` (rename at the consumer).
- `_helpers.py:814` `_leaky_q_value_diff`, `_helpers.py:883` `add_uncertainty_covariates` — `leaky_q_value_diff` emitted here; **stop emitting / stop consuming it**.
- `scripts/_bandit_csv_contingency.py:58` `_q_chosen`, `:77` `_fit_alpha`, `:115` per-animal α fit — the fitted-α leaky-Q; **promote to the covariate layer and extend to per-arm `q_arm`** (currently only `q_chosen` for the current arm) and per-arm nominal `p_reward_arm` (from the contingency `p_reward_port{0..5}` already decoded there).

Per-arm covariate computation (the canonical home — updated here):
- `src/.../tables.py:1247-1252` — `visit_recency` / `cumulative_reward` / `recent_reward_rate` batch in `build_event_arm_table`. The current 5-visit `recent_reward_rate` is computed here. This plan **updates the builder** to emit the canonical covariates (via the shared canonical function, k=10 + `p_reward_arm` + `q_arm`) so future regens are canonical, **and backfills the existing pickles in place** (recompute only the covariate columns from the CSV/contingency/Q — no `load_data`, no posterior re-rollup) so we get canonical tables now without the ~4.3 h regen. Builder and backfill call the **same** canonical function → no divergent computation.

Model specs / runners (consumers to update):
- `model_specs.py:194` `_M3 = (*_M2, "recent_reward_rate")` and the M-ladder (`MODEL_HIERARCHY` `:205`), patch ladder (`MODEL_PATCH` `:310`), `MODEL_CHOICE` — rename `recent_reward_rate`→`reward_rate_arm_k10`; add the canonical value block (`p_reward_arm`, `q_arm`) per the dictionary's per-family matrix.
- `scripts/fit_uncertainty_rate.py:86` `CURATED_COVARIATES` — swap `leaky_q_value_diff`→`q_current`; `:198` drop the `--q-alpha=0.3` knob (α now fit, not set).
- `scripts/fit_swr_power_model.py:41` `PREDICTORS` — already canonical (`p_reward_current`, `recent_switch_rate`); add `q_current`.
- `scripts/compare_event_source_models.py` + `choice_prediction.py` `build_choice_dataset` — rename + add `p_reward_arm`/`q_arm` to the choice dataset.
- Untouched: `bs(tc)` spline, reward-lag kernel, `hist_rcos` self-history, spatial-role/`arm_fe`, `arm_dist_rank` — definitions unchanged.

## Scope and dependency policy

### Goals
- One canonical definition per construct, written into the canonical tables every family reads; coefficients comparable within an estimator class.
- **Consolidate covariate definitions into one module** `src/.../analysis/replay/covariates.py` (extracted from the `_helpers.py` grab-bag + the Q-learner promoted from the scratch script), mirroring `docs/covariate_dictionary.md` 1:1 — done behavior-preservingly in **phase-0** (re-export shim keeps the ~16 importers working) before any definition changes.
- Implement D1–D5 exactly (window 10; add `p_reward_arm`; `leaky_q`→fitted Q; `recent_switch_rate` canonical + `is_after_switch` kept; entry-rule contract).
- Re-fit all families on current tables; prove unchanged-definition covariates reproduce.

### Non-Goals
- **No posterior re-rollup, no `load_data`** — only the behavioral covariate columns are recomputed (from the CSV/contingency/Q) and written into the canonical tables; the ~4.3 h decode-rollup that produced the response columns is not redone.
- **No re-decode** — orthogonal (that's the reality-run/W3 workstream).
- No new science questions — this is definitional harmonization only.
- No change to the spline/self-history/spatial definitions.

### Dependency policy
No new dependencies.

## Metrics
- **Parity (the core check):** for every covariate whose definition is unchanged
  (reward kernel, `p_reward_current`, `rpe`, `value_diff_patch`, `recent_switch_rate`,
  `is_after_switch`, spatial role, spline, self-history), the re-fit per-session
  coefficients must match the pre-refactor fits to **floating-point tolerance**
  (same data, same definition → effectively identical; tol e.g. |Δcoef| < 1e-6 median,
  and the signed-rank verdicts unchanged).
- **Changed covariates** (`reward_rate_arm_k10` vs old 5-visit; `q_*` vs `leaky_q`):
  coefficients change by design — **report** old-vs-new, don't assert equality.
- **Coverage:** every family fit on the same session set as before (no shrinkage from
  the merge; coverage gaps surface as NaN→listwise-drop, counted and reported).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| In-place table rewrite corrupts the response/posterior columns | Recompute **only** the covariate columns; assert `arm_count_*` / `arm_posterior_mass` / offsets / keys are byte-identical pre/post; write to a temp, validate, then swap. Keep a backup copy. |
| Builder path and backfill path diverge (compute the covariate two ways) | Both call the **same** canonical function on the **same** inputs (CSV-derived trial sequence + contingency + Q); a test asserts the backfill output equals what the updated builder produces on a sample session. |
| Renames silently miss a consumer → model fits on a wrong/old column | Phase-1 grep for every old name across `src/` + `scripts/`; unit test that each spec's predictor list resolves to columns present in the canonical tables. |
| Adding `p_reward_arm`/`q_arm` to content changes the model ladder | Add them as an explicit canonical **value block** (new rungs), not silently into M0–M7; the existing M0–M7 stay bit-identical so the re-verified content claims are preserved, and the value block is additive. |

## Rollout Strategy
Replace-in-place; internal covariates with no external consumers (per CLAUDE.md
default). Old per-arm columns remain in the rollup table until the next regen but are
unreferenced. No deprecation window.

## Open Questions
1. **Canonical behavioral input.** Standardize the per-arm visit/reward history on the
   trial-level SpatialBanditTask CSV (already adopted for contingency) rather than
   `load_data` trials, so the builder and the backfill share one cheap input and the
   covariates are consistent with the contingency. Best answer: yes — but confirm the
   CSV→event alignment (`_bandit_csv_contingency.py`) gives a per-arm visit sequence at
   event granularity; if a gap, fall back to `load_data` trials for the recompute only.
2. **Value block rungs for content/choice.** Exact rung names/order for the added
   `p_reward_arm` + `q_arm` (e.g., a `Mv_*` value extension on M5). Deferred to
   phase 1; must not perturb M0–M7.

## Estimated Effort
~250–400 LOC: a canonical-covariate module + precompute script (~150), spec/runner
renames + value block (~100), tests (~100). Phase 2 is mostly running the existing
runners + a parity-check script (~80).
