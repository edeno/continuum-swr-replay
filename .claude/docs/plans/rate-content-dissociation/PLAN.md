# Rate-content dissociation implementation plan

**Status:** Not started.

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
