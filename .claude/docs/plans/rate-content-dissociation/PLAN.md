# Rate-content dissociation implementation plan

**Status: SUPERSEDED (2026-07-13).** The scientific spine was delivered by work
that post-dates this plan; one residual extracted to
[`rate-content-followups`](../rate-content-followups/PLAN.md). Phase files below are
retained as the historical record.

**Reconciliation (what actually happened):**

- **Phase 1 (content refit + audit tooling) — DONE (refit) / obsolete (audit harness).**
  Content models were re-fit 2026-06-28 and again 2026-07-13 (HMM→beta); the
  `recent_reward_rate`→`reward_rate_arm_k10` alias is migrated. The formal
  `audit_replay_content_artifacts.py` QA harness was never built and is now low-value
  (the refits already happened; the synthesis interprets the current fits).
- **Phase 3 (joint double dissociation + continuous-SWR) — DONE.** This is the
  synthesis headline and claim **#19f** ("reward acts on with-SWR *occurrence* but
  non-SWR *content*"); the continuous-SWR question is answered by `fit_swr_power_model.py`
  (`ripple_power_z` unimodal → "cut on a continuum"; graded > binary but binary retained).
- **Phase 5 (synthesis doc + claim ledger) — DONE.** `docs/replay_rate_content_swr_synthesis.md`
  and `paper/claims_table.md` (claims 19a–g). Machine-readable hashed-artifact ledger unbuilt (low value).
- **Phases 2 & 5 formal tables (summary/support-classification/provenance manifests) —
  REMAIN but low value.** The synthesis "covariate atlas" already serves as the
  human-readable cross-family summary.
- **Phase 4 (choice prediction / mediation) — SUPERSEDED.** Fits exist
  (`compare_event_source_models.py`, claims #19d/#19e) but are **in-sample-only (no
  held-out CV)** and **not distance-controlled** (`dist_to_arm` dropped for
  quasi-separation). The residual first moved to `rate-content-followups` phase-1 and
  is now replaced by
  [`regression-integrity-fixes` Phase 11](../regression-integrity-fixes/phase-11-decision-models.md).
  Do not execute either historical choice phase.
- **HMM-era falsehoods in this plan's `overview.md`/`designs.md`:** `entropy_x_current`
  is now **null** and `q_arm`'s content effect is *avoidance* (`swr_x_q_arm −2.45`), not
  the ambiguous "per-arm value contrast" the overview assumes.

---

The current coefficient readout points to a strong rate/content split: reward
appears to gate **how often** SWR-overlapping non-local events occur, while the
content models ask **where replay is allocated** within an event and show strong
current-arm/current-patch structure. Some content artifacts are stale relative to
the current covariate names and missing-data contract, so this plan first audits
and regenerates the model artifacts before building joint rate-content tests.

The goal is to make the interpretation reproducible enough for paper-facing
figures: decide which effects are rate effects, which are content effects, which
are SWR-specific, and whether replay content predicts future choice beyond a
behavioral baseline.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a phase?** Open its phase file. Each phase is self-contained.
2. **Need shared schema or estimand definitions?** [shared-contracts.md](shared-contracts.md).
3. **Need algorithm sketches?** [designs.md](designs.md).
4. **Need motivation, model questions, integration points, risks?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) - motivation, questions answered, code integration points, gates, risks, metrics.
- [shared-contracts.md](shared-contracts.md) - shared summary schema, complete-case policy, artifact freshness rules, estimand labels.
- [designs.md](designs.md) - audit, summary-builder, dissociation-test, continuous-SWR, and choice-feature designs.
- Phases:
  - [phase-1-artifact-audit-refit.md](phase-1-artifact-audit-refit.md) - audit and regenerate stale content artifacts before interpreting content coefficients.
  - [phase-2-summary-tables.md](phase-2-summary-tables.md) - build one canonical rate/content coefficient-summary table with provenance.
  - [phase-3-joint-dissociation.md](phase-3-joint-dissociation.md) - test rate-vs-content dissociations and binary-vs-continuous SWR coupling.
  - [phase-4-choice-mediation.md](phase-4-choice-mediation.md) - test whether replay content predicts next choice beyond behavior and whether that mediates rate/content effects.
  - [phase-5-figures-docs.md](phase-5-figures-docs.md) - produce paper-facing figures, claim ledger, and docs from audited outputs only.
