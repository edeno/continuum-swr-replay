# Package Readability Refactor Implementation Plan

**Status:** Phases 0-4 done and committed (green). Phase 5 found already
satisfied on execution — the covariate builders are already I/O-pure; a
regression guard test was added. Phase 4's confirming regeneration byte-diff runs
once the rebuilt behavioral cache is ready.

Restructure the `continuum_swr_replay` package so a scientist with no prior context can trace how data was processed — raw load → behavioral parse → canonical tables → covariates → fits — as a straight line, with each stage in one obvious place. The work makes the code self-documenting (single session-key definition, one publish path, per-grain table builders, I/O-pure covariate builders, a terse model catalog) without changing any scientific output or any published file path. Every numerical-path change is proven identical against a frozen v5 baseline by byte-diff.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / verification policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, verification methodology, risks.
- [shared-contracts.md](shared-contracts.md) — `session_keys` API and `publish_canonical()` signature.
- Phases (each ships as a separable PR):
  - [phase-0-baseline.md](phase-0-baseline.md) — freeze the v5 golden baseline; move the mutation harness into the repo.
  - [phase-1-builder-tests.md](phase-1-builder-tests.md) — close the 36 builder value-level test gaps (searchsorted `side=` first).
  - [phase-2-cleanups.md](phase-2-cleanups.md) — one `session_keys` module, centralized version literals, model_specs prose → docs.
  - [phase-3-publish-chokepoint.md](phase-3-publish-chokepoint.md) — single `publish_canonical()` for every canonical writer.
  - [phase-4-tables-split.md](phase-4-tables-split.md) — split `tables.py` by grain with thin re-export.
  - [phase-5-builder-purity.md](phase-5-builder-purity.md) — make covariate builders I/O-pure.
