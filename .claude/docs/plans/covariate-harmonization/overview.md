# Overview — Scope, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Covariate-computation layer (functions to reuse with canonical params):
- `src/continuum_swr_replay/analysis/replay/_helpers.py:618` `recent_reward_rate_all_events` — per-arm reward rate; **call with k=10** (currently 5) for `reward_rate_arm_k10`.
- `_helpers.py:704` `recent_trial_reward_rate` — all-arm 10-trial rate; reuse as-is for `reward_rate_trial_k10` (rename at the consumer).
- `_helpers.py:814` `_leaky_q_value_diff`, `_helpers.py:883` `add_uncertainty_covariates` — `leaky_q_value_diff` emitted here; **stop emitting / stop consuming it**.
- `scripts/_bandit_csv_contingency.py:58` `_q_chosen`, `:77` `_fit_alpha`, `:115` per-animal α fit — the fitted-α leaky-Q; **promote to the covariate layer and extend to per-arm `q_arm`** (currently only `q_chosen` for the current arm) and per-arm nominal `p_reward_arm` (from the contingency `p_reward_port{0..5}` already decoded there).

Per-arm covariate baking (the deferred canonical home — *not* rebuilt in this plan):
- `src/.../tables.py:1247-1252` — `visit_recency` / `cumulative_reward` / `recent_reward_rate` batch in `build_event_arm_table`. The current 5-visit `recent_reward_rate` is baked here; this plan **leaves the 304 MB rollup table alone** and supplies canonical covariates via the precompute (see Open Question 1). Folding canonical covariates in here is a follow-up at the next table regen.

Model specs / runners (consumers to update):
- `model_specs.py:194` `_M3 = (*_M2, "recent_reward_rate")` and the M-ladder (`MODEL_HIERARCHY` `:205`), patch ladder (`MODEL_PATCH` `:310`), `MODEL_CHOICE` — rename `recent_reward_rate`→`reward_rate_arm_k10`; add the canonical value block (`p_reward_arm`, `q_arm`) per the dictionary's per-family matrix.
- `scripts/fit_uncertainty_rate.py:86` `CURATED_COVARIATES` — swap `leaky_q_value_diff`→`q_current`; `:198` drop the `--q-alpha=0.3` knob (α now fit, not set).
- `scripts/fit_swr_power_model.py:41` `PREDICTORS` — already canonical (`p_reward_current`, `recent_switch_rate`); add `q_current`.
- `scripts/compare_event_source_models.py` + `choice_prediction.py` `build_choice_dataset` — rename + add `p_reward_arm`/`q_arm` to the choice dataset.
- Untouched: `bs(tc)` spline, reward-lag kernel, `hist_rcos` self-history, spatial-role/`arm_fe`, `arm_dist_rank` — definitions unchanged.

## Scope and dependency policy

### Goals
- One canonical definition per construct, one source (`canonical_covariates*.pkl`) feeding every family; coefficients comparable within an estimator class.
- Implement D1–D5 exactly (window 10; add `p_reward_arm`; `leaky_q`→fitted Q; `recent_switch_rate` canonical + `is_after_switch` kept; entry-rule contract).
- Re-fit all families on current tables; prove unchanged-definition covariates reproduce.

### Non-Goals
- **No `event_arm_table` / rollup regen** (the 4.3 h, 304 MB build) — canonical covariates are supplied by the precompute; baking them into the table builder is a deferred follow-up.
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
| Two sources of truth (canonical precompute vs the 5-visit column still baked in the table) | The precompute is the single source consumers read; the stale baked columns are left **unused** and removed at the next table regen (named in phase 1's "deliberately not"). A grep gate ensures no consumer still references the old names. |
| Renames silently miss a consumer → model fits on a wrong/old column | Phase-1 grep for every old name across `src/` + `scripts/`; unit test that each spec's predictor list resolves to columns present in the canonical source. |
| Adding `p_reward_arm`/`q_arm` to content changes the model ladder | Add them as an explicit canonical **value block** (new rungs), not silently into M0–M7; the existing M0–M7 stay bit-identical so the re-verified content claims are preserved, and the value block is additive. |
| Precompute needs per-session trials (for the 10-visit recompute) → cost | Smoke-test the precompute on one session, measure, extrapolate (Open Question 1). |

## Rollout Strategy
Replace-in-place; internal covariates with no external consumers (per CLAUDE.md
default). Old per-arm columns remain in the rollup table until the next regen but are
unreferenced. No deprecation window.

## Open Questions
1. **Precompute input cost.** `reward_rate_arm_k10` (10-visit per-arm) must be
   recomputed from per-session trial/visit history (`recent_reward_rate_all_events`
   needs `trials`). Best answer: build the precompute once from the same `trials`
   the table builder uses (via `load_data` per session or a cached trials frame),
   smoke-test one session for cost, extrapolate; if prohibitive, fall back to baking
   into the next rollup regen. `p_reward_arm`/`q_arm` are cheap (already computed
   per session in `_bandit_csv_contingency.py`).
2. **Value block rungs for content/choice.** Exact rung names/order for the added
   `p_reward_arm` + `q_arm` (e.g., a `Mv_*` value extension on M5). Deferred to
   phase 1; must not perturb M0–M7.

## Estimated Effort
~250–400 LOC: a canonical-covariate module + precompute script (~150), spec/runner
renames + value block (~100), tests (~100). Phase 2 is mostly running the existing
runners + a parity-check script (~80).
