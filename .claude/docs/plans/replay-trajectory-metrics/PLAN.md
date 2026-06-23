# Replay Trajectory Metrics Implementation Plan

**Status:** Phase 1 + Phase 2 code COMPLETE and committed. The augmentation
was reworked to need no `load_data` (neural-signal rollups use the z-scores
stored in each pkl; `step_distance` is exact from `results.nc`) after
root-causing a `load_data` time-axis drift (e5a0def trims post-trial padding,
post-dating the 2026-04-01 decodes — see the deferred milestone). Full
augmentation running in tmux `augment_dot`, idempotent/resumable, atomic
in-place writes. Phase 2: `trajectory.py` (4 rollup families +
`compute_trajectory_rollups` orchestrator) wired into BOTH `build_event_table`
and `build_contfrag_event_table` with `path_continuity` merged + cross-scenario
parity test; retrospective metrics + `prev_arm` mirror merged into the canonical
event table at assembly (`prepare_replay_content_data.py`). REMAINING: real-data
validation of the event tables once the augmentation batch finishes.


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
