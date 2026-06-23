# Replay Trajectory Metrics Implementation Plan

**Status:** Not started.

Add ~30 per-event summary statistics that characterize replay trajectories —
speed, displacement, path length, spatial extent, continuity, direction,
brain-state, and decode-uncertainty measures — to both the Non-Local and the
ContFrag event tables using shared definitions, so the two scenarios are
directly comparable. Per-timebin signals are computed once into
`data_over_time`; per-event scalar rollups are merged into the event tables.

The design rationale and full metric catalog live in the spec:
[docs/superpowers/specs/2026-06-23-replay-trajectory-metrics-design.md](../../../../docs/superpowers/specs/2026-06-23-replay-trajectory-metrics-design.md).
This plan covers *how* to build it.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is
   self-contained: upstream files to read, designs it depends on, tasks,
   validation slice, fixtures.
2. **Need a per-component design?** [designs.md](designs.md).
3. **Need broader scope / risks / metrics?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, risks, rollout, open questions
- [designs.md](designs.md) — the shared rollup module: signatures and non-obvious metric implementations
- Phases (each ships as a separable PR):
  - [phase-1-tier1-signals.md](phase-1-tier1-signals.md) — new per-timebin signals in `data_over_time` + regeneration
  - [phase-2-event-rollups.md](phase-2-event-rollups.md) — rollup module wired into both event tables, with merged path-continuity + retrospective metrics and a cross-scenario parity test
