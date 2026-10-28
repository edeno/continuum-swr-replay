# mPFC Response Follow-ups Implementation Plan

**Status:** Phases 0, 1 (A, with decreases) and 2 (B) done and in `part4.json`; phase 3 (C, revised by the lead: no reference, no context models) batch running since 2026-10-27 18:33 (tmux `pfc_c_marked_family`); Q8 stop-level design waiting for the lead; phase 4 for Phase C after the batch. See ledger.md.

Answers five questions about how mPFC units respond to hippocampal non-local events, comparing SWR-coupled replay (non-local events that overlap a Kay SWR) with non-SWR replay: (4) when the response peaks and whether it starts before event onset; (2) whether mPFC responds to Kay SWRs that overlap no non-local event as it does to SWR-coupled replay; (3) whether the response depends on what is replayed; (5) whether it scales with event strength; (8) whether it differs with the stop's reward outcome and in the trials before a patch switch. Timing is read from the canonical Figure 4 bundles. The other questions need new fits, which write to their own bundle trees. Every reported number goes through `paper/reported/part4.json`.

The lead's phase letters map to files as A → phase 1, 0 → phase 0, B → phase 2, C → phases 3a/3b, D → phase 4. Phases 0 and 1 have no GPU work and run first.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need broader scope / risks / open questions?** [overview.md](overview.md).
4. **What ran, where, and what came out?** [ledger.md](ledger.md).

## Files

- [overview.md](overview.md) — integration points, goals, non-goals, metrics, risks, open questions for the lead.
- [shared-contracts.md](shared-contracts.md) — event-class names, bundle trees, response windows and unit sets, reported-value key scheme.
- [ledger.md](ledger.md) — runs, commits, run ids/paths, runtimes, results as summary keys.
- Phases (each ships as a separable PR):
  - [phase-0-port-marked-family.md](phase-0-port-marked-family.md) — (0) bring the content-marked family onto this branch: new module layout, current session assembly, tests green.
  - [phase-1-response-timing.md](phase-1-response-timing.md) — (A) peak lag and onset from the canonical bundles, a smoothing calibration, values into `part4.json`.
  - [phase-2-swr-without-replay.md](phase-2-swr-without-replay.md) — (B) the `SWR_no_non_local` class, its matched null, smoke, 75-session refit into a new tree.
  - [phase-3a-marked-family-v2.md](phase-3a-marked-family-v2.md) — (C, code) per-stop baseline terms, content/strength/context models, marginal tier, shuffled-mark reference.
  - [phase-3b-marked-family-refit.md](phase-3b-marked-family-refit.md) — (C, runs) smoke, corpus refit, unit → session → animal summary.
  - [phase-4-reported-values.md](phase-4-reported-values.md) — (D) phases 2–3 results into `part4.json` and `paper/reported/overview.md` (markers only).
