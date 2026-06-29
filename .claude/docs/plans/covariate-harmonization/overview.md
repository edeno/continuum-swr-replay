# Overview — Scope, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

Covariate-computation layer (extracted to `covariates.py` in phase-0; definitions changed in phase-1):
- `_helpers.py:618` `recent_reward_rate_all_events` — per-arm reward rate (**timestamp-driven**: needs `event_times` + `trials` goal-arrival cutoffs); **call with k=10** for `reward_rate_arm_k10`.
- `_helpers.py:704` `recent_trial_reward_rate` — all-arm 10-trial rate; reuse as-is for `reward_rate_trial_k10` (rename only — assert byte-identical in phase-2).
- `_helpers.py:814` `_leaky_q_value_diff`, `:883` `add_uncertainty_covariates` — `leaky_q_value_diff` (per-stop margin); **kept** as a reported comparison (D3 revised).
- `scripts/_bandit_csv_contingency.py:58` `_q_chosen`, `:77` `_fit_alpha` — fitted-α leaky-Q; **promote to `covariates.py` and extend to all-arm `q_arm`/`q_current`** (today only the chosen arm). Unify the existing `q_chosen` name → `q_current`.

Per-arm covariate computation (the canonical home — updated, then **rebuilt**):
- `tables.py:1247-1252` `build_event_arm_table` covariate batch, and `:1277` `nominal_reward_prob` (deferred NaN → **populated and renamed `p_reward_arm`**; update the `:1646` schema list + the viz registry). Update the builder to emit canonical covariates, then **regenerate the replay canonical tables**. (Rebuild chosen over in-place backfill: the per-arm rate is timestamp-driven and needs `load_data(trials)` regardless — `event_arm_table` does **not** store `event_midpoint_time` — so a rebuild is one clean code path / true single source.)
- `analysis/contfrag/tables.py:36,638,657,680,767` — contfrag builds its **own** per-arm `recent_reward_rate` (5-visit) + `swr_x_recent_reward_rate` + `recent_trial_reward_rate`. **In scope (per decision):** apply the same canonical definitions and regenerate the contfrag tables, else contfrag silently stays 5-visit.

Other consumers to sweep (not just fit runners):
- `model_specs.py:194` `_M3 = (*_M2, "recent_reward_rate")` + M-ladder (`:205`), patch (`:310`), `MODEL_CHOICE` — rename `recent_reward_rate`→`reward_rate_arm_k10`; add the value block (`p_reward_arm`, `q_arm`) as additive rungs.
- `fit_uncertainty_rate.py:86` `CURATED_COVARIATES` — **add** `q_current`, **keep** `leaky_q_value_diff`; `:198` drop `--q-alpha` (α now fit).
- `fit_swr_power_model.py:41` `PREDICTORS` — add `q_current`.
- `compare_event_source_models.py` + `choice_prediction.py` — rename + add `p_reward_arm`/`q_arm`.
- `simulation.py:62,142,144,298`, `learning.py:8,59,90,98,103`, `visualization/session_tables.py` (the `nominal_reward_prob` panel) — reference the renamed names; update or a re-export alias must cover them. Grep gate spans **all** of `src/`+`scripts/`.
- Untouched definitions: `bs(tc)`, reward-lag kernel, `hist_rcos`, spatial-role/`arm_fe`, `arm_dist_rank`.

## Scope and dependency policy

### Goals
- One canonical definition per construct (per `docs/covariate_dictionary.md`), computed by the table **builders** and written into the canonical tables every family reads — single source.
- Consolidate covariate code into `covariates.py` (phase-0, behavior-preserving; re-export shim keeps the ~16 importers working).
- Implement D1–D5 as revised; **harmonize both the replay and contfrag tables**.
- Re-fit all families; prove unchanged-definition covariates reproduce.

### Non-Goals
- **No re-decode** — orthogonal (the reality-run / W3 workstream). The rebuild reuses the existing decodes (re-rolls the posteriors from them; response columns must reproduce).
- No new science questions — definitional harmonization only.
- No change to spline / self-history / spatial / `arm_dist_rank` definitions.

### Dependency policy
No new dependencies.

## Folded-in audit findings (`.claude/review/`)

The independent pipeline audit (2026-06-29) is incorporated where it intersects this
refactor; the rest is tracked as deferred so nothing is lost.

**Incorporated (this plan):**
- *04* — content/choice covariate drift from the dictionary **is** this plan; additionally the **choice runner switches `dist_to_arm`→`arm_dist_rank`** (dictionary §4; phase 1) since it's already a dictionary covariate and `build_choice_dataset` carries only `dist_to_arm`.
- *01* — add a **`validate_dot_event_contract`** (monotonic time within session, single session, positive labels forming one contiguous run, non-negative durations) and **run it across every `data_over_time.pkl` before the phase-1 table regen** (the regen assumes these contracts).
- *03* — fix the **`TMAX=4.0`→3.99 s edge** (`aggregate_to_timebins` uses `int(tmax/bin_s)` dropping 169 onsets): use `ceil` + clip the final bin's exposure to `tmax`, with a non-divisible-`tmax` regression test (done alongside the phase-1 design change since phase-2 re-fits the rate ladder anyway).
- *03/04* — **pre-existing red tests are ours to fix**: `test_uncertainty_covariates` (expects the removed `UNCERTAINTY_COVARIATES`) and `test_compare_event_source_models` (expects `R0/R1/R2`, source set now has `Rlag`). Get the baseline **green in phase-0** (precondition for "suite passes unchanged"); update the uncertainty-covariate test to the canonical set in phase-1.
- *03* — refresh the **stale claim #19c per-bin numbers** (1.42/3.10/1.08 → current) in phase-2's docs pass.

**Deferred (NOT this plan — tracked backlog):**
- Animal/session **hierarchical / cluster-robust inference** for the headline p-values (03 #4, 05) — a methodology change spanning all claims; do as its own workstream.
- Promote the **exploratory content scripts** (`_content_reward_interactions`, `_content_patch_destination`) into the audited model pipeline with diagnostics (SE/p/VIF/FDR/n_events/provenance) so #19f + patch-destination become non-exploratory (04).
- Make **ancillary scripts consume canonical `is_swr`** instead of re-deriving it (spike-evidence 02, occupancy 04, PFC 05).
- Reality **full-dataset run** (gated on the re-decode) and **PFC matched-control + canonical split** (02, 05).

## Metrics
- **Unchanged-definition covariates** (reward kernel, `p_reward_current`, `rpe`, `value_diff_patch`, `recent_switch_rate`, `is_after_switch`, `leaky_q_value_diff`, spatial, spline, self-history, and the rename-only `reward_rate_trial_k10`): assert the **design-matrix columns are identical** to baseline (deterministic, the strong guarantee); coefficients then match within the fitters' convergence tolerance (looser than 1e-6 — don't over-tighten against BLAS/optimizer nondeterminism).
- **Rebuild integrity:** the regenerated tables' **response/posterior columns** (`arm_count_*`, `arm_posterior_mass`, offsets, `is_swr`, keys) reproduce the pre-rebuild tables (the rollup is deterministic on fixed decodes) — confirms the rebuild changed *only* covariates.
- **Changed covariates** (`reward_rate_arm_k10` vs old 5-visit; new `q_*`): coefficients change by design — **report** old→new.
- **Coverage:** every family fit on the same session set as before (counted).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Rebuild silently perturbs the response columns (not just covariates) | Assert the regenerated response/posterior columns reproduce the pre-rebuild tables (deterministic rollup on fixed decodes); diff covariate columns only. Keep a backup of the old tables. |
| Rebuild cost (replay ~4.3 h + contfrag) | Long batch in tmux; smoke one session, measure, extrapolate before the full regen (reuses existing decodes — no re-decode). The 4.3 h is **provisional** (inherited from prior work); contfrag (SWR+MUA × areas) is a second pass needing its **own** extrapolation from the smoke. |
| Concurrent readers see half-old/half-new tables during the in-place regen | These tables are read by active scratch consumers (`_reality_reward_test.py`, `_periswitch_*`, `_univariate_coefs.py`, …) and a possibly-running `reality_sweep` tmux job. **Quiesce or branch** the regen (write to temp, swap atomically when done; pause concurrent jobs); the grep gate also asserts the reality engine does **not** consume `recent_reward_rate`/`reward_rate_arm_k10` (so it's unaffected). |
| contfrag left at 5-visit → reintroduces the inconsistency | contfrag in scope (this plan); grep gate spans `src/`+`scripts/` incl. contfrag/simulation/learning/viz. |
| Renames silently miss a consumer | Grep gate over all of `src/`+`scripts/`; unit test that every spec predictor resolves to a present column; phase-0 re-export alias covers stragglers. |
| **Content ladder M3–M7 move** (`recent_reward_rate` is inside `_M3`/M7) | Scope the "ladder unchanged" claim to **M0–M2 / rungs without the per-arm rate**; explicitly flag M3–M7 and the swr×reward interaction **will change at k=10** and re-verify the affected content claims (incl. the autocorrelation caveats in memory). The added `p_reward_arm`/`q_arm` value block is separately additive. |
| `p_reward_arm` vs existing `nominal_reward_prob` duplicate | Populate + rename the existing `nominal_reward_prob` column (tables schema + viz registry); no parallel column. |

## Rollout Strategy
Replace-in-place; internal covariates, no external consumers. Tables regenerated (decodes reused). No deprecation window.

## Open Questions
1. **Value block rungs for content/choice.** Exact rung names/order for the added `p_reward_arm` + `q_arm` (e.g., a `Mv_*` value extension on M5). Deferred to phase 1; must not perturb M0–M2.
2. **`q_*` cross-animal comparability.** α is fit per animal, so `q_*` is on each animal's own learning timescale — report the per-animal α range and interpret `q_*` coefficients within-animal; a shared-α sensitivity fit is optional.

## Estimated Effort
~300–450 LOC: `covariates.py` extraction (phase-0, mostly moves) + definition changes + builder updates (replay + contfrag) (~150), spec/runner renames + value block (~120), tests + parity script (~150). Plus the table regen wall-clock (batch, not LOC).
