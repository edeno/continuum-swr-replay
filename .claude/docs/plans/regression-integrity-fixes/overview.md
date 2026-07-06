# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Source audit: `.claude/docs/reviews/06_data_provenance_regression_audit.md` +
`06_covariate_provenance.csv`. Read those for the evidence behind each fix.

## Current codebase integration points

Verified against the tree at plan time (read the line before editing — see each phase's *Inputs*).

- `scripts/content/fit_swr_contrast_content.py:78` — blanket `warnings.filterwarnings("ignore")`; phase 2 narrows it.
- `scripts/content/fit_swr_contrast_content.py:302` — saved `q_arm_source`/`deferred` provenance JSON is **stale** (says beta/HMM-deferred while C4 uses `q_arm_hmm`); phase 1 corrects.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:880` — `C4_armvalue` uses `q_arm_hmm`; `:873` `BASEc`; `:485`–`:532` `MODEL_PER_STOP_RATE` R0/R1/R2/Rlag; `:840`–`:859` rate value/RPE rungs (G4/G5/S7…); `:876`–`:901` content value/RPE/update rungs (C4/C6/C9/C10). Phases 1/4 touch.
- `src/continuum_swr_replay/analysis/replay/tables.py:1300` — `reward_rate_arm_k10` attached per (event,arm); `:1296` `visit_recency_demeaned`; `:1116`/`_event_arm_table_columns` canonical column order. Phases 4/6 touch.
- `src/continuum_swr_replay/analysis/replay/covariates.py:179`–`:198` — `recent_reward_rate_all_events`: **deliberately** returns NaN for unvisited leaf arms (trunk arms → `0.0`). Phase 4 **replaces this with encoding** (unvisited leaf → `0.0` + `has_arm_history` indicator) per the resolved mechanism-A decision below.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:124`–`:128` — `reward_outcome`/`is_after_reward`/`is_after_omission` assigned from the **trip-in-progress** trial (indexed by trial-start time, `:83`). Not strictly pre-outcome for approach-phase stops. Phase 4 addresses.
- `src/continuum_swr_replay/analysis/behavior/qvals_table.py:292`–`:294` — `hmm_rpe` (uses current reward), `hmm_value_update` (uses next-trial Q). Post-outcome; phase 4 gates their consuming rungs. `:14` module docstring interval bug (phase 1).
- `src/continuum_swr_replay/analysis/replay/fitting.py:160`,`:380` — conditional-Poisson / conditional-logit `.fit(disp=0)` with **no convergence check** (rank guard exists at `:150`/`:370`). Phase 2 adds convergence.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:153`,`:250`–`:254` — per-bin GLM `.fit()`; `:118` documents that separation yields a large finite coef. Phase 2 flags it.
- `src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:130` — per-stop GLM `.fit(disp=0)` (rank guard `:113`). Phase 2 adds convergence.
- `src/continuum_swr_replay/analysis/replay/choice_prediction.py:492` — choice conditional-logit `.fit(disp=0)` (rank guard `:482`); `:313`–`:340` choice predictor set (no value covariate — phase 1 doc fix / phase 5 value-controlled model, **required if the choice model is paper-facing**).
- `src/continuum_swr_replay/analysis/replay/contrast_aggregation.py:89`–`:113` — per-animal median + n=5 sign consistency. Phase 5 reuses this for the base ladders.
- `scripts/content/fit_replay_content_models.py:241`–`:251` (session Wilcoxon), `:502`–`:510` (union complete-case cohort). Phase 5 promotes per-animal; phase 3 instruments the drops.
- `learning.py:481` — the one existing `mle_retvals["converged"]` check; phase 2 copies this pattern. Stage-6 `is_stay` uses `is_after_omission` (`:173`) but the modeled unit is post-outcome — phase 4 annotates, does not change.
- `scripts/exploratory/_add_standardized_cols.py:27`,`:46` — the in-place `_z` stamping phase 6 retires into the canonical builder.

## Scope and dependency policy

### Goals

- Every saved result carries accurate provenance and an accurate retained-N / drop profile.
- Fit diagnostics (convergence, separation, rank) applied uniformly across content, rate, and choice fitters; no blanket warning suppression hiding them.
- Covariates used as predictors are pre-outcome relative to the modeled unit, or the model is explicitly labeled retrospective.
- Population claims for the base content/rate ladders rest on animal-level (n=5) evidence, not session-level Wilcoxon.
- A clean `prepare_data.py` rebuild reproduces the fitted tables exactly (incl. `_z` and `q_arm_hmm`), with no exploratory-script post-processing required.

### Non-Goals

- **No re-decoding.** The detector outputs (`*_data_over_time.pkl`, posteriors) are unchanged.
- **No change to the relational spine, cardinality logic, or the response-only quarantine** of `next_arm`/`is_next`/`ends_at_next` — the audit confirmed these correct.
- **No full hierarchical (NumPyro) mixed model in this plan.** Phase 5 delivers per-animal aggregation as primary; the mixed conditional-logit with animal random effects stays a documented follow-up (Open Question 2).
- No changes to the `contfrag` sibling pipeline.

### Dependency policy

No new runtime dependencies. Phase 5's optional NumPyro follow-up is out of scope; if pursued later, JAX is already a dependency.

## Metrics

- **Phase 1:** grep of saved `*_meta.json` / provenance strings shows zero "harmonization deferred" / "beta-model per-arm value" for runs that use `q_arm_hmm`; docs match `model_specs.py`.
- **Phase 2:** every fitter returns a `converged` (and `separated`) status; a synthetic separated design is flagged, not silently returned; aggregation drop counts logged.
- **Phase 3:** each fit run writes a cohort table whose `n_in − Σ drops = n_modeled` reconciles exactly; drop profile reports session-time / reward / value-quantile of dropped rows.
- **Phase 4:** rate R1/R2 refit on `reward_lag1` is a sign-flip reparameterization (`≈ 1 − is_after_omission`) — coefficient changes everywhere, causal-vs-leaky *values* disagree only on the ~10% approach stops; value/RPE/update rungs gated to post-current-outcome bins or relabeled descriptive; mechanism-A **encode** implemented (unvisited-leaf `reward_rate_arm_k10=0` + `has_arm_history`), retaining the ~11.6% of events.
- **Phase 5:** base content/rate headline coefficients reported with per-animal `n_animals_pos/n_animals_used` (with a headline warning when `n_animals_used < 5`); common-cohort refits (content **and** rate) show whether cross-rung ΔAIC shifts are cohort-driven; `swr_x_value` reported as two separate checks — construct sensitivity (beta vs HMM) on the *shared* qvals-present cohort, plus a *descriptive* missing-tail profile (the tail is **not** value-imputed; beta cannot restore it — F3).
- **Phase 6:** a one-session rebuild reproduces the current materialized tables' **non-`_z`** columns within `atol=1e-10`, and reproduces the **`_z`** columns within `atol=1e-10` **only when applied via the persisted frozen references** (event_arm: `standardization_reference.json`; qvals: `qvals_standardization_reference.json`) — a per-session recompute of global `_z` cannot and should not match (F5). `_add_standardized_cols.py` deleted; both builders own their standardization.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Mechanism-A encoding changes the content estimand (novel-arm regime enters the model) | **Decided: encode.** Phase 4 reports base rungs both encoded and (once, for comparison) on the old restricted cohort, so the estimand change is measured, not silent; `has_arm_history` lets the novel-arm state be a modeled category rather than a hidden exclusion. |
| Time-gating value/RPE rungs shrinks their N further | Phase 3 cohort tables make the shrink explicit; phase 5 sensitivity confirms sign stability. |
| `swr_x_value` sign is cohort- or construct-driven (known construct-dependence) | Phase 5 reports beta `q_arm` vs HMM `q_arm_hmm` on the *shared* qvals-present cohort (construct sensitivity) and separately characterizes the missing tail descriptively — no value-imputed "restored tail" claim (beta shares HMM's missingness, F3). Restoring the tail needs upstream fit coverage, which is **not pursued** (resolved decision #2 — drop the tail). |
| Builder consolidation (phase 6) drifts table contents | Phase 6 validation refits one session end-to-end and diffs against the current materialized table within float tolerance. |
| Per-animal aggregation with n=5 is low-powered | Documented honestly as bookkeeping, not a power upgrade (mirrors `contrast_aggregation.py` wording); it is the correct unit, not a stronger test. |

## Rollout Strategy

Replace-in-place; no parallel v1/v2 code paths. **Execution order: 1 → 2 → 3 → 6 → 4 → 5.** Phases 1–3
are non-destructive to fitted numbers (labels, diagnostics, reporting) and can merge in any order among
themselves. Then phase 6 (builder consolidation) **before** phase 4 (mechanism-A encoding), because both
regenerate `event_arm_table` and phase 4's `has_arm_history` must land on a builder that already owns
`_z`/`q_arm_hmm` — so the table is regenerated once from the clean builder, then again with the
`reward_rate_arm_k10=0` + `has_arm_history` encoding. **Phase 5 runs last (F4): it depends on both phase 3
(cohort/term-diagnostics tables) and phase 4 (its paper-facing aggregation and `swr_x_value` sensitivity
must fit the phase-4-cleaned designs, not the pre-cleanup covariates).** Each number-changing phase
re-runs the affected ladders with the phase-3 cohort tables attached.

## Resolved decisions

1. **Mechanism A — DECIDED (2026-07-04): ENCODE.** For an unvisited leaf arm, set
   `reward_rate_arm_k10 = 0.0` and add a `has_arm_history` 0/1 indicator (**main effect only** from M3
   onward — no `has_arm_history:reward_rate_arm_k10` interaction; under this encoding it is perfectly
   collinear with the `reward_rate_arm_k10` main effect, see phase 4). This retains the ~11.6% of events (the
   early-session / novel-arm regime this project studies) instead of dropping them. It changes the
   current deliberate NaN-drop behavior at `covariates.py:189`–`:198` and the `event_arm_table` schema,
   so the table is regenerated. Phase 4 reports base rungs both **encoded** and (once, for comparison) on
   the old restricted cohort so the estimand change is measured, not silent. *Restrict was rejected.*
2. **Mechanism B — DECIDED (2026-07-04): DROP THE UNFITTED TAIL (definitional population).** The
   2,056-stop / 896-event tail with no fitted HMM state is **not** value-imputed and the upstream fit is
   **not** extended. Value models are defined on "trials with a fitted HMM state"; the drop is reported
   per-model (phase 3 `n_drop_qvals_gap`) and characterized descriptively (phase 5) so the exclusion is
   visible and shown not to distort. Confirmed empirically that no in-table proxy can restore it (beta
   `q_arm`/`q_arm_hmm`/`p_reward_arm` share identical missingness; 0 beta-present-HMM-missing rows).

## Open Questions

1. **Content outcome-timing merge scope (phase 4).** Gating the *content* value/RPE/update rungs to
   post-current-outcome events requires merging each event's current-trial outcome-known time (pump if
   rewarded, else goal-arrival) onto events (they carry only
   `time_since_stop`). If that merge is judged too invasive, the fallback is to relabel those content
   rungs *descriptive* rather than predictive. Rate rungs are unaffected (their per-bin design already
   has bout timing). Decide during phase-4 execution.
2. **Full hierarchical model.** Deferred — mixed conditional-logit with animal random effects
   (`fit_swr_contrast_content.py:302` already lists it as deferred; needs NumPyro). Not in this plan;
   per-animal aggregation is the phase-5 deliverable.

## Estimated Effort

- Phase 1: ~40 LOC across 3 files + doc edits. Phase 2: ~120 LOC (a shared diagnostics helper + call-site
  wiring). Phase 3: ~150 LOC (audit-table emitter + wiring). Phase 4: ~200 LOC + table regen (schema
  change: encode). Phase 5: ~200 LOC (aggregation reuse + sensitivity runners) + refits. Phase 6:
  ~120 LOC into the builder + delete `_add_standardized_cols.py` + one-session regen validation.
