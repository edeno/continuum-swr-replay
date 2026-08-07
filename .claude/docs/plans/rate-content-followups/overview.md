# Overview — the surviving rate/content residual

[← back to PLAN.md](PLAN.md)

> **Ownership reconciliation (2026-07-29):** the choice/CV material below is
> historical and superseded by
> [`regression-integrity-fixes` Phase 11](../regression-integrity-fixes/phase-11-decision-models.md).
> Phase 13 owns the cross-history refit. Only the onset cross-correlogram and
> continuum interpretation remain executable from this plan.

## Historical context and surviving work

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
   (does one non-local stream's history drive the other's rate?) is rebuilt and run
   under regression-integrity Phase 13; this plan consumes that valid output. The
   definitive **peri-onset cross-correlogram / peak** test remains unbuilt and is
   flagged "not yet run" (synthesis Layer 2).

The behavioral-relevance item and process-identity item were unrelated. The former
now has a newer owner; this router retains only the latter's downstream work.

## Current codebase integration points

- The former `compare_event_source_models.py`/`choice_prediction.py` stop-grain
  integration is historical; regression-integrity Phase 11 replaces rather than
  extends it.
- `scripts/rate/fit_cross_history_rate.py:129` (`fit_cross_history`), `:172`
  (`main`/legacy `--write`), `:144` (`_summary`), `:123` (`delta_cv_bits`).
  **Regression-integrity Phase 13** owns the canonical refit; **Phase 2** consumes
  its promoted result and interprets it.
- Phase-13 promoted `event_onset_table` + `event_table` — per-event onset times + `is_swr`;
  the manifest-validated substrate for the **Phase 2** cross-correlogram. Fixed
  `Processed-Data/replay_content/*.pkl` paths are historical and forbidden.
- `docs/per_bin_rate_results.md` (§4 cross-history note) and
  `docs/replay_rate_content_swr_synthesis.md` (Layer 2) — where the Phase-2 verdict lands.
- `paper/claims_table.md` claim **#19d** — regression-integrity Phase 11/13 owns its
  replacement and final adjudication.

## Scope and non-goals

### Goals
- Phase 1: **superseded**; regression-integrity Phase 11 owns the replacement
  decision-grain choice analysis.
- Phase 2: a documented **SWR↔non-SWR continuum verdict** (promoted cross-history GLM) + the
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
| Cross-history GLM is underpowered / null per session | Report `delta_cv_bits` with per-animal aggregation; a null is inconclusive about shared versus distinct mechanisms and cannot be counted as continuum support. |
| Apparent onset coupling is driven by slow rate modulation or edge effects. | Use a per-session jitter/shuffle null that preserves event counts and immobility support; report the full lag curve and animal-balanced summaries. |

## Open Questions

None for the surviving slice. Choice controls/folds are locked in regression-integrity
Phase 11; Phase 2 consumes the promoted cross-history result and owns only the
prespecified onset analysis.

## Estimated Effort
Phase 2 ~80-120 LOC for the cross-correlogram script/figure, manifest input gate, and
continuum verdict. Superseded Phase-1 effort is excluded.
