# Rate/content follow-ups — one surviving downstream residual

**Status:** Partially superseded (2026-07-29). Phase 1 is superseded by the newer
decision-grain implementation in
[`regression-integrity-fixes` Phase 11](../regression-integrity-fixes/phase-11-decision-models.md);
do not execute the older stop-grain choice-CV phase. Phase 2 remains not started,
is independently scoped, and must consume the corrected run promoted by Phase 13.
Phase 13 owns the cross-history model refit; Phase 2 owns only the new onset
cross-correlogram and the continuum interpretation of both outputs.

Historically collected two residuals left after `nonswr-uncertainty-rate` and
`rate-content-dissociation` were superseded (2026-07-13). The choice residual is
now owned by regression-integrity Phase 11. Only the downstream SWR/non-SWR onset
cross-correlogram and continuum interpretation remain here.

## Reading order

1. **Working a phase?** Open its file — each is self-contained.
2. **Cross-phase context / ownership history?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, and ownership boundaries
- Phase files:
  - [phase-1-choice-mediation-cv.md](phase-1-choice-mediation-cv.md) — **superseded historical design**; replaced by the six-arm, one-row-per-decision model in regression-integrity Phase 11
  - [phase-2-swr-continuum.md](phase-2-swr-continuum.md) — consume Phase-13's valid cross-history output, build the peri-onset cross-correlogram, and write the continuum verdict
