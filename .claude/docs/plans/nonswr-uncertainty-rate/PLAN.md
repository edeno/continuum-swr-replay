# Non-local replay rate vs uncertainty/value (SWR vs non-SWR) Implementation Plan

**Status:** Not started.

Test whether the *rate* of non-local replay is differentially modulated by the
animal's behavioral uncertainty and the task's value structure across
SWR-coupled vs non-SWR non-local events, in the Spatial Bandit task. First parse
the contingency structure (nominal per-port `p(R)`, block index, change times)
out of the StateScript log (a temporary, exploratory adapter — to be superseded
by a canonical parsed-events table), then fit the Family B per-bin rate model
per population with task-grounded value/RPE/block covariates plus model-free
uncertainty signals, controlling for the within-stop time course, self-history,
reward kernel, and reward rate. Extends the existing per-population dissociations
(recent-reward-rate NL-w/SWR 1.28 vs NL-no-SWR 0.89, p=2e-6; patch-switch 0.76
vs 1.16, p=6e-8). Rate-only — replay direction is out of scope.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — each is
   self-contained (upstream files, contracts, tasks, validation, fixtures).
2. **Need the shared table schema?** [shared-contracts.md](shared-contracts.md).
3. **Need scope / dependency policy / risks / open questions?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, dependency policy (no `trodes_to_nwb` import; temporary/exploratory), risks, metrics, open questions.
- [shared-contracts.md](shared-contracts.md) — the `contingency_table.pkl` column schema that couples the two phases.
- Phases (each ships as a separable PR):
  - [phase-1-contingency-parser.md](phase-1-contingency-parser.md) — temporary StateScript→contingency parser (nominal `p(R)`, block index, change times), DIO-aligned, keyed by `stop_id`.
  - [phase-2-uncertainty-rate.md](phase-2-uncertainty-rate.md) — per-population per-bin rate fit with uncertainty/value covariates; SWR-vs-non-SWR dissociation + peri-block latency.
  - [phase-3-continuum-test.md](phase-3-continuum-test.md) — follow-up: is non-SWR non-local a separate less-bursty mode or part of one bursty continuum with SWR? Cross-correlogram + cross-type ensemble-history GLM.
