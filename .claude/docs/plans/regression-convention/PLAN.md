# Regression convention and Figures 2 to 5 on one data subset: Implementation Plan

**Status:** Not started.

This plan builds the approved design `docs/design/2026-11-20-regression-convention-and-figure04.md`
(read all of it; its decision rounds in section 9 override earlier text). It does two things, in the
lead's order: Figures 2 and 3 and the events summary move onto the common data subset (events inside a
well visit, chosen by one selection function), and every regression model of Figures 4 and 5 is rebuilt
on one convention, with one shared penalized fitter inside this repository. It ends with Figure 4 drawn
and the replaced models removed.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists
   upstream files to read, contracts and designs it depends on, tasks, validation slice and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md): the convention's rules, the
   selection function, the fit-row record, the declared tests, how to run code and rebuild values, the GPU
   and coordination rules, and C13 (no legacy left behind).
3. **Need a per-component design (math, code)?** [designs.md](designs.md).
4. **Need broader scope, dependencies, risks, decisions?** [overview.md](overview.md).
   **What to remove, and when?** [legacy-inventory.md](legacy-inventory.md).
5. **Need the scratch prototypes and outside references?** [appendix.md](appendix.md).

## Files

- [overview.md](overview.md) — goals, non-goals, phase graph, integration points, risks, rollout,
  decisions for the lead, effort.
- [shared-contracts.md](shared-contracts.md) — the rules every phase obeys.
- [designs.md](designs.md) — the fitter's math (penalized Newton, exact REML derivatives, pooled REML),
  bases, collinearity, support, folds, bands, and the rate and content designs.
- Phases (each ships as a separable PR; branch names are suggestions):
  - [phase-1-shared-fitter.md](phase-1-shared-fitter.md) — `analysis/regression/`: likelihoods,
    penalized Newton, REML with a grid start and exact derivatives, checks 1 to 5, the mgcv script;
    the older all-bins history model deleted first.
  - [phase-2-convention-tools.md](phase-2-convention-tools.md) — bases, collinearity, support rule,
    fit-row record, trial folds, simultaneous bands, the session to day to animal reduction.
  - [phase-3-visit-tables.md](phase-3-visit-tables.md) — the well visit as the unit: visit clocks and
    stop-end kinds in the tables, the visit at-risk bin design, the two new uncertainty covariates;
    one table rebuild.
  - [phase-4-figures-2-3-common-subset.md](phase-4-figures-2-3-common-subset.md) — Figures 2 and 3 and
    `paper/reported/events.json` on the common subset (after the other session's Figure 2 branch merges).
  - [phase-5a-rate-models.md](phase-5a-rate-models.md) — rate on the visit: three clocks, models A and D
    for reward history, both forms of the reward effect, Q1, Q2, Q7.
  - [phase-5b-rate-families-moved.md](phase-5b-rate-families-moved.md) — the per-stop (total) family,
    the SWR-contrast covariates and the uncertainty rate model moved to the convention.
  - [phase-6a-content-models.md](phase-6a-content-models.md) — content on nine arms by arm role, smooth
    uncentered recency, the three uncertainty measures, current-arm share against time until leaving;
    Q3, Q4, Q5, Q7.
  - [phase-6b-content-robustness.md](phase-6b-content-robustness.md) — count and Dirichlet-multinomial
    estimators on the new design; the within-arm comparison deleted.
  - [phase-7-choice-stay-power.md](phase-7-choice-stay-power.md) — Q6 (content and the next choice),
    stay or leave per visit, ripple power, on the convention.
  - [phase-8-mpfc.md](phase-8-mpfc.md) — the mPFC unit models on the convention (JAX batch fitter,
    REML shared over a session's units, trial folds); Figure 5 redrawn.
  - [phase-9-inference-family.md](phase-9-inference-family.md) — the declared family of tests with FDR,
    the simultaneous bands, the inference values in the behavior summary.
  - [phase-10-figure04.md](phase-10-figure04.md) — Figure 4 drawn to the standards of Figures 1 to 3
    and 5.
  - [phase-11-history-models.md](phase-11-history-models.md) — the two event-history models moved to
    the shared fitter (after Figure 4, as the design orders).
  - [phase-12-legacy-sweep.md](phase-12-legacy-sweep.md) — the final sweep: cross-cutting leftovers, the
    ASK list answered, and audits that prove nothing replaced is left (each phase removes what it
    replaced).
- [legacy-inventory.md](legacy-inventory.md) — every piece of legacy in the touched areas, by the phase that
  removes it, classed REMOVE, ASK or KEEP, with the ASK list for the lead.
- [appendix.md](appendix.md) — map of the scratch prototypes to the phase, test or stored analysis that
  replaces each; outside references with commits.
