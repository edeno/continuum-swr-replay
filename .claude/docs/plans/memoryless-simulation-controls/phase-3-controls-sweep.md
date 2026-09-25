# Phase 3 — Controls sweep and figure row 5

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

**Inputs to read first:**

- The design's "Simulated inputs" and "Scoring" sections:
  `docs/design/replay_detection_comparison/memoryless_simulation_controls.md`.
- `scripts/qa/sweep_memoryless_detection.py` — `build_recording`, the run functions from
  phase 1, `main` (placement loop).
- The builders and scoring functions from phase 2.

**Contracts referenced:** [Reported events](shared-contracts.md#reported-events),
[Truth labels](shared-contracts.md#truth-labels),
[Detector outputs](shared-contracts.md#detector-outputs),
[Summary CSV layouts](shared-contracts.md#summary-csv-layouts).

## Tasks

- **`scripts/qa/sweep_memoryless_controls.py`**:
  - Conditions and grids exactly as the design's table: no replay (every usable
    immobile stretch, decoded whole); place-less and jumping bursts at 80, 200, 500 ms;
    jumps J = 0, 20, 40, 80 cm with 120 or 200 ms segments of 60 cm; chains of two
    150 ms, 60 cm replays with gaps 10, 30, 60, 100, 200 ms, continuing or starting
    >= 40 cm away; isolated parts of 120, 150, 200 ms. Peak rates 20, 60, 160 Hz; 8
    seeds; placements from the same immobile intervals; a condition that cannot be
    placed raises.
  - Midpoints: bursts anywhere; jumps and chains centered so every part lies on the
    track; the "elsewhere" chain's second part starts >= 40 cm from the first's end.
  - Per trial, both run functions, then the four detector outputs scored: for the
    no-replay control, events per minute and time share; for bursts, called or not and
    (state-space) the share of burst time in each non-local state; for multi-part and
    isolated parts, per-part recovery, segmentation, onset/offset errors.
  - Raw table to `Processed-Data/replay_detection_comparison/memoryless_controls.parquet`,
    summary to `docs/design/replay_detection_comparison/detection_controls_summary.csv`
    with provenance.
- **Smoke run** on one seed, timed; extrapolate; then the full run on a clean tree.
- **Figure row 5**: (k) controls, (l) jumps, (m) chains, per the design; the figure
  reads the controls summary too (`--controls-summary` argument, default path).
- **Caption** paragraph for (k–m) with checks for each factual sentence; read against
  the two truth targets (a jumping burst called non-local by the canonical definition
  is not a false positive).
- **Spec**: spec 02 gains a short section with the controls' results, pointing at the
  summary.

## Deliberately not in this phase

- Held-out anything (phase 4).
- Changing the injected-replay sweep's grid or rows.

## Validation slice

| Test | Asserts |
| --- | --- |
| Unit tests for any pure helper added to the script (placement of multi-part events, per-trial scoring from `ReportedEvent`s) | Parts land inside the track and inside one immobile stretch; scoring labels on hand-built events |
| Smoke run (slow, manual) | Completes; runtime recorded in the commit message |
| Figure build | Caption checks pass; PNG inspected |
| Full suite | Passes, including the pipeline map |

## Fixtures

The simulated recording from `build_recording`.

## Review

Before merging, dispatch `code-reviewer` (read-only) against the diff. Confirm the grids
match the design, truth labels follow the two targets, and the summary has provenance
from a clean commit.
