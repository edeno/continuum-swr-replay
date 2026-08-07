# Phase 1 — Held-out CV + distance control for the replay→choice claim

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

> **SUPERSEDED (2026-07-29):** do not execute this stop-grain/cross-session design.
> [`regression-integrity-fixes` Phase 11](../regression-integrity-fixes/phase-11-decision-models.md)
> owns the replacement six-arm, one-row-per-decision analysis and uses the shared
> cohort/fold contracts from Phases 9a/9b. The material below is audit history.

Firm up or retire claim **#19d** ("non-SWR content modestly predicts the upcoming
choice"). It is currently the weakest link: fit **in-sample only** and **not
distance-controlled** (`dist_to_arm` was dropped for quasi-separation — the animal
picks the closest arm ~95–99%), so the reported `replay_frac_non_swr +0.34` may be
nearest-arm bias. This phase adds cross-session CV and a distance control on a common
cohort, then restates the claim.

**Inputs to read first:**
- [scripts/content/compare_event_source_models.py:453](../../../../scripts/content/compare_event_source_models.py) — `_fit_choice_block` (the in-sample choice fit); `:113`/`:121` the choice-model selectors; `:220`/`:314` `_freeze_common_cohort` / `_fit_per_session_common_cohort` (reuse this cohort infra).
- [src/continuum_swr_replay/analysis/replay/choice_prediction.py:96,172](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py) — `predict_arm_fractions` and `build_residualized_choice_dataset` (the `replay_frac_residual` column already exists).
- [src/continuum_swr_replay/analysis/replay/model_specs.py:66,756-777](../../../../src/continuum_swr_replay/analysis/replay/model_specs.py) — `MODEL_CHOICE`, the `dist_to_arm` quasi-separation rationale, and `arm_dist_rank` as the identifiable baseline.
- [paper/claims_table.md](../../../../paper/claims_table.md) claim **#19d**, and the choice/prediction paragraph in [docs/replay_rate_content_swr_synthesis.md](../../../../docs/replay_rate_content_swr_synthesis.md) (the "Prediction models (Stage 5A)" block) — the current hedged wording to update.

## Tasks
- **Add a cross-session held-out score to the choice comparison.** Extend
  `_fit_choice_block` (or a sibling in `compare_event_source_models.py`) with
  **leave-session-out-within-animal** CV: fit `C_behavior` vs `C_behavior + replay_frac`
  on all-but-one session, score held-out choice log-likelihood, aggregate ΔLL
  per animal. Cross-**session** (not within-session k-fold) because per-session
  clogit has few choices. Report per-animal ΔLL sign (n/5) — the population read.
- **Re-introduce a distance control that converges.** Refit the choice models with
  `arm_dist_rank` in the baseline (it is ordinal and already the identifiable choice
  baseline, unlike raw `dist_to_arm`). Confirm per-session convergence (no
  quasi-separation flag) before trusting coefficients; if `arm_dist_rank` still
  separates, fall back to a distance-stratified fit (within near/far strata). The
  test becomes: does `replay_frac` add held-out predictive value **beyond** behavior
  **and** distance rank?
- **Refit on a common complete-case cohort** across the compared models (reuse
  `_freeze_common_cohort`) so the ΔLL is not confounded by differing drop sets.
- **Restate claim #19d** in `paper/claims_table.md` and the synthesis "Stage 5A"
  block from the held-out, distance-controlled result — a controlled effect if it
  survives, a retired/negative claim if it does not. Remove the "in-sample only /
  not distance-controlled" hedge either way (it will have been resolved).

## Deliberately not in this phase
- The stay/leave analysis (#19e) — already null and not distance-sensitive.
- Any rate/content model change — this is choice-model rigor only.
- The formal machine-readable provenance/ledger tables (dropped, see overview).

## Validation slice
| Test | Asserts |
| --- | --- |
| CV wrapper on synthetic data | a planted `replay_frac`→choice signal yields positive held-out ΔLL; pure-noise `replay_frac` yields ΔLL ≈ 0 (not systematically positive). |
| distance-control convergence | with `arm_dist_rank` in the baseline, ≥ (a stated fraction, e.g. 90%) of sessions converge without quasi-separation; the fallback stratified path is exercised if not. |
| common-cohort parity | the compared models share one complete-case cohort (n identical); assert equal row sets. |
| claim provenance | the updated #19d cites the new held-out + distance-controlled artifact, not the in-sample fit. |

Mark the full per-session CV run slow.

## Fixtures
- A small synthetic choice dataset (planted + null `replay_frac`) in the test module.
- One real event-source table slice (`event_source_comparison/*`) for a smoke run.

## Review
Before opening the PR, dispatch `scientific-code-reviewer` against the diff. Confirm:
- CV is leave-session-out (no within-session leakage into the held-out score).
- The distance control actually converges (or the stratified fallback is used and stated), so `replay_frac`'s effect is genuinely distance-adjusted.
- The compared models share one cohort; claim #19d is restated from the controlled result with no residual "in-sample/uncontrolled" hedge.
