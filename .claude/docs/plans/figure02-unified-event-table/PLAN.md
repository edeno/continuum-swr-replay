# Unified event-detection pipeline for Figure 2

**Status:** Revised after review; not started.

Figure 2's four panels (a/b/c/d) currently each derive non-local-event start/end and SWR-overlap classification independently — three different overlap rules and two different immobility filters across the figure. This plan introduces a single canonical per-non-local-event table that holds timing, overlap classification, peak non-local probability, and per-event speed summaries. Panels a/b/d read directly from it; panel c keeps its existing triggered-average helper for this phase but gains a parity test against the table. The replay-content models (`build_event_table` in `analysis/replay/tables.py`) consume the same canonical base without changing their legacy population/timing columns.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts referenced, tasks, validation slice, and fixtures.
2. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — Goals, integration points, risks, rollout, and the regen story for `event_table.pkl`.
- Phases (each ships as a separable PR):
  - [phase-1-classified-event-table.md](phase-1-classified-event-table.md) — New `analysis/hippocampus/event_table.py`; refactor `analysis/replay/tables.py:build_event_table` to compose on top of it; align `is_swr` semantics while preserving replay population/timing; regenerate caches.
  - [phase-2-figure02-rewire.md](phase-2-figure02-rewire.md) — Rewire `generate_figure02.py` panels a/b/d to read from the classified table; parity-test panel c; add example-event class assertion; delete the now-duplicate helpers.
