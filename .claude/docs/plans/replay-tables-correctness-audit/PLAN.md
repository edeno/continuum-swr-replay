# Replay Tables Correctness Audit Implementation Plan

**Status:** Not started.

This plan builds a fail-closed correctness case for the five replay analysis tables: formal input/output contracts and an independent hand-computable reference implementation, a read-only audit of every materialized session with deterministic manual QC, and a generation-coherent candidate rebuild plus claim-impact review using the corrected run/cohort/inference infrastructure already owned by the regression-integrity plan. It treats unexplained table drift as a scientific blocker and never overwrites or silently promotes canonical artifacts.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

Execute the phases in order. Each phase is a separate PR. If Phase 1 or Phase 2 discovers a production defect, land that correction in a narrowly scoped table-fix PR, rerun all earlier gates, and only then continue. Phase 3 cannot begin with an unresolved hard/identity finding or before regression-integrity Phases 7-12 supply source-valid run artifacts, exact cohorts/folds, fit validity, robust inference, decision-grain behavior, and identified SWR-power APIs. It consumes those APIs; it does not create a parallel pipeline.

## Files

- [overview.md](overview.md) — scope, live integration points, success metrics, rollout, risks, and unresolved scientific semantics
- [shared-contracts.md](shared-contracts.md) — table grains, keys, input/output invariants, comparison policy, artifact isolation, and verdict rules
- [designs.md](designs.md) — independent oracle, corpus audit, QC selection, isolated rebuild, keyed comparison, and claim-impact algorithms
- Phases (each ships as a separable PR):
  - [phase-1-contracts-reference-oracle.md](phase-1-contracts-reference-oracle.md) — formalize public boundaries and verify core responses against independent hand oracles
  - [phase-2-full-corpus-audit.md](phase-2-full-corpus-audit.md) — audit all materialized sessions read-only and complete a deterministic source-to-table QC catalog
  - [phase-3-scientific-sensitivity-verdict.md](phase-3-scientific-sensitivity-verdict.md) — audit a coherent noncanonical run, quantify table-dependent claim impact with corrected estimators, and issue a provenance-backed verdict
