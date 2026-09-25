# Memoryless simulation controls — Implementation Plan

**Status:** Phases 1-4 implemented (4a5d519 to b568cea); phase 4 review pending.

Extends the simulated comparison between the memoryless sliding-window detector and our
state-space detector, as designed in
[memoryless_simulation_controls.md](../../../../docs/design/replay_detection_comparison/memoryless_simulation_controls.md):
signed onset and offset errors; negative controls (no replay, place-less and jumping
bursts); mid-event jumps and replay chains scored per part and by segmentation against
isolated-part baselines; and a held-out test that detects on discovery cells and
assesses content with evaluation cells under both detectors' content estimates. The
supplemental comparison figure grows from four rows to six.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals, non-goals, risks
- [shared-contracts.md](shared-contracts.md) — run outputs, truth labels, summary CSV layouts
- [designs.md](designs.md) — injector, builders, multi-part scoring, held-out rule
- Phases (each ships as a separable PR):
  - [phase-1-timing.md](phase-1-timing.md) — split the sweep into run/score; onset/offset rows; figure row 4
  - [phase-2-injection-and-scoring.md](phase-2-injection-and-scoring.md) — general injector, builders, multi-part scoring (src, tested)
  - [phase-3-controls-sweep.md](phase-3-controls-sweep.md) — controls sweep, its summary, figure row 5
  - [phase-4-heldout.md](phase-4-heldout.md) — held-out rule and folds, held-out sweep, figure row 6
