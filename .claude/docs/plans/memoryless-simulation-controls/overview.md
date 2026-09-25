# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `scripts/qa/sweep_memoryless_detection.py:242` — `memoryless_outcome`: split into a
  run step (decode, detect, shuffle; returns reported events) and a score step. Scored
  columns must be unchanged.
- `scripts/qa/sweep_memoryless_detection.py:420` — `state_space_outcome`: same split.
  The run step also returns the continuous-only intervals and the per-state
  probabilities the controls need.
- `scripts/qa/sweep_memoryless_detection.py:405` — `non_local_intervals`: reused for the
  continuous-only definition by passing the continuous probability alone.
- `scripts/qa/sweep_memoryless_detection.py:159` — `build_recording`: gains a `cells`
  argument (phase 4) so a fold can fit the detector on discovery cells only.
- `scripts/qa/sweep_memoryless_detection.py:620` — `summarize`: gains the paired
  onset/offset rows (phase 1); every existing row unchanged.
- `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/simulation.py:87` —
  `inject_replay`: rebuilt on a general injector, same signature, identical output for a
  fixed seed.
- `src/continuum_swr_replay/analysis/hippocampus/memoryless_sequence/scoring.py` —
  `score_against_injection`: accepts a true path that is NaN in gaps; new part-recovery
  and segmentation functions.
- `src/continuum_swr_replay/analysis/inventory_comparison/` — new `heldout.py` (phase 4).
- `scripts/figures/generate_memoryless_comparison.py:975` — `make_figure`: `GridSpec(4, 8)`
  becomes six rows; `ROW_HEIGHTS` (line 92) and the figure height (line 986) grow;
  `plot_duration_bias` (line 851) is removed with panel (h); `write_caption` (line 1162)
  and `check_summary_claims` (line 1064) gain paragraphs and checks.
- `scripts/qa/compare_memoryless_to_reference.py` — imports from the sweep; must keep
  working after the split (it calls `sweep.build_recording`, `sweep.CONFIG`, and its own
  decode helpers).
- `docs/design/replay_detection_comparison/` — two new summary CSVs; spec 02's
  sweep section gains the new results.

## Scope and dependency policy

### Goals

- Every item of the approved design, on sorted spikes on the simulated linear track.
- Numbers in the caption and spec come from committed summary CSVs, each with
  provenance rows, produced on a clean tree.

### Non-Goals

- Recorded data, clusterless decoding, the encoding-pair shuffle, branched tracks.
- Tuning any detector threshold or held-out parameter against these results.
- Changing the injected-replay sweep's grid or its existing summary rows.

## Metrics

- Refactor: the sweep's existing summary rows value-for-value identical before and
  after phase 1; `inject_replay` output identical for fixed seeds after phase 2.
- Each script's one-seed smoke run completes and is timed before its full run.
- The figure builds, and its caption checks pass, from the committed summaries.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| The run/score split silently changes a scored column | Diff the full summary CSV before and after phase 1; must be identical apart from provenance |
| A condition cannot be placed in the recording's immobile stretches (500 ms chains need ~0.9 s plus margins) | Placement raises, never drops; smoke run surfaces it before the full run |
| A six-row figure becomes unreadable | Rows 5 and 6 use compact bar layouts; render and look at the PNG before committing |
| Held-out refits per fold change the detected population | Expected and reported (spec 05); every comparison is within a fold |

## Rollout Strategy

All at once on the unpushed `memoryless-sequence-detection` branch; no backwards
compatibility (confirmed at planning time). Each phase regenerates the outputs it
changes and commits them with the code.

## Open Questions

None open; the design's decisions C1–C10 are settled.

## Estimated Effort

About 1500 lines including tests: phase 1 ~250, phase 2 ~450, phase 3 ~400, phase 4 ~500.
