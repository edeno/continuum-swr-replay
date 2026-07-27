# Overview — the two rate/content residuals

[← back to PLAN.md](PLAN.md)

## Why these two survive

Both parent plans (`nonswr-uncertainty-rate`, `rate-content-dissociation`) are
SUPERSEDED — see their retired `PLAN.md` reconciliations. What's delivered: the
uncertainty→content result (claim 19g, `uncertainty_content` feature), the
reward-gating rate/content double dissociation (claim 19f, the synthesis "covariate
atlas"), the continuous-vs-binary SWR question at the *distribution* level
(`fit_swr_power_model` → unimodal), and the synthesis doc + claim ledger. Two things
were left genuinely un-done, each currently a **hedged** conclusion:

1. **The replay→choice claim (#19d) is soft.** The synthesis explicitly flags it
   `in-sample only — no held-out CV`, and the choice model **drops `dist_to_arm`**
   (quasi-separation: the animal picks the closest arm ~95–99%). So the reported
   `replay_frac_non_swr +0.34` may partly reflect nearest-arm bias.
2. **The SWR/non-SWR continuum verdict is unwritten.** `fit_cross_history_rate.py`
   (does one non-local stream's history drive the other's rate?) was built but never
   run/documented, and the definitive **peri-onset cross-correlogram / peak** test is
   flagged "not yet run" (synthesis Layer 2).

These are unrelated (behavioral relevance vs process identity) — hence one router,
two independent phases.

## Current codebase integration points

- `scripts/content/compare_event_source_models.py:453` (`_fit_choice_block`), `:113`
  / `:121` (`_choice_models_for_source` / `_choice_swr_split_models_for_source`),
  `:220`/`:314` (`_freeze_common_cohort` / `_fit_per_session_common_cohort` — the
  common-cohort infra already exists). **Phase 1** extends this; currently in-sample.
- `src/continuum_swr_replay/analysis/replay/choice_prediction.py:96`
  (`predict_arm_fractions`), `:172` (`build_residualized_choice_dataset` → the
  `replay_frac_residual` column already exists). **Phase 1** uses these.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:66,768-777` — the
  documented rationale for excluding `dist_to_arm` (quasi-separation) and using
  `arm_dist_rank` as the choice baseline. **Phase 1** must respect/re-test this.
- `scripts/rate/fit_cross_history_rate.py:129` (`fit_cross_history`), `:172`
  (`main`/`--write` → `models_per_bin_rate/cross_history_models.pkl` at `:195`),
  `:144` (`_summary`), `:123` (`delta_cv_bits`). **Phase 2** runs + interprets this.
- `Processed-Data/replay_content/event_onset_table.pkl` + `event_table.pkl` — per-event
  onset times + `is_swr`; the substrate for the **Phase 2** cross-correlogram.
- `docs/per_bin_rate_results.md` (§4 cross-history note) and
  `docs/replay_rate_content_swr_synthesis.md` (Layer 2) — where the Phase-2 verdict lands.
- `paper/claims_table.md` claim **#19d** — Phase 1 updates it (firm up or retire).

## Scope and non-goals

### Goals
- Phase 1: a **held-out / cross-session CV** replay→choice test with a **distance
  control that avoids quasi-separation**, on a common complete-case cohort — then
  restate claim 19d as either a controlled effect or a retired one.
- Phase 2: a documented **SWR↔non-SWR continuum verdict** (cross-history GLM) + the
  **peri-onset cross-correlogram / peak** figure.

### Non-goals
- The formal QA/audit harness, machine-readable summary/support/provenance/ledger
  tables from the retired plans — **explicitly dropped** (the synthesis doc + claims
  table already serve the human-readable role; process-only value).
- Re-deriving anything already in claims 19a–g / the synthesis.
- Promoting the graded ripple-power content model (kept exploratory by decision).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A distance control re-introduces the quasi-separation that got `dist_to_arm` dropped | Use `arm_dist_rank` (ordinal, already the choice baseline) or a distance-stratified fit; verify convergence per session before trusting coefficients (Phase 1). |
| Cross-history GLM is underpowered / null per session | Report `delta_cv_bits` with per-animal aggregation (the script already does this); a null is a valid continuum-consistent result — state it, don't chase significance. |
| Held-out CV on per-session clogit is unstable (few choices/session) | Cross-**session** (leave-session-out within animal) rather than within-session folds; report per-animal. |

## Open Questions
1. **Distance control form** — `arm_dist_rank` vs distance-stratified vs `is_same_patch`.
   Best answer: `arm_dist_rank` (already the identifiable baseline). Decide in Phase 1
   from convergence.
2. **CV grain** — leave-session-out (recommended) vs k-fold within session. Phase 2 / 1
   pick from stability.

## Estimated Effort
Phase 1 ~150 LOC (CV wrapper + distance-control refit + claim update). Phase 2 ~120 LOC
(run existing fit + a new ~80-LOC cross-correlogram script/figure + doc verdicts).
