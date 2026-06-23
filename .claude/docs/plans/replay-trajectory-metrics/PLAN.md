# Replay Trajectory Metrics Implementation Plan

**Status:** Phase 1 complete + PR-reviewed + revised (8e13682, 7e3f0d8,
e3fc792). data_over_time augmentation uses the **exact posterior** path for
`step_distance` (Option B — cross-animal validation found the map_estimate
reconstruction diverged by ≤1 place-bin on 2/5 animals). Full augmentation
running in tmux `augment_dot` (log: `logs/augment_data_over_time.log`),
idempotent/resumable, atomic in-place writes; the reindex guard now tolerates
real LFP coverage gaps (a201f60). Phase 2 LARGELY DONE: `trajectory.py` has all
4 rollup families + the `compute_trajectory_rollups` orchestrator, wired into
BOTH `build_event_table` and `build_contfrag_event_table` (gated on the
augmented columns), with `path_continuity` merged and a cross-scenario parity
test (157b966, 27e103a, d55be8d, 0949748, a201f60). REMAINING: merge the
existing retrospective metrics into the canonical event table + add the
`prev_arm` mirror (`ends_at_prev` / `contains_path_from_prev` /
`chosen_arm_fraction_prev`).

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

## Deferred milestone — full re-decode

The saved decodes (2026-04-01) are stale relative to the current pipeline:
`get_position_info` now trims post-trial padding (e5a0def, 2026-04-24) and the
decoding params drifted, so reloading `load_data` no longer reproduces the
decode axis (~76 s LFP tail for some epochs). The trajectory work sidesteps
this (z-scored signals from the pkls; augmentation needs no `load_data`). A
full re-decode on current code — to make position/LFP/decode-axis/signals
mutually consistent — is deliberately deferred (days of GPU, would change the
decodes and require regenerating all downstream figures/models). See memory
`project-decode-data-drift-redecode`.

## Files

- [overview.md](overview.md) — scope, integration points, risks, rollout, open questions
- [designs.md](designs.md) — the shared rollup module: signatures and non-obvious metric implementations
- Phases (each ships as a separable PR):
  - [phase-1-tier1-signals.md](phase-1-tier1-signals.md) — new per-timebin signals in `data_over_time` + regeneration
  - [phase-2-event-rollups.md](phase-2-event-rollups.md) — rollup module wired into both event tables, with merged path-continuity + retrospective metrics and a cross-scenario parity test
