# Non-local replay rate vs uncertainty/value (SWR vs non-SWR) Implementation Plan

**Status: SUPERSEDED (2026-07-13).** Delivered by work that post-dates this plan;
one residual extracted to [`rate-content-followups`](../rate-content-followups/PLAN.md).
The phase files below are retained as the historical record of what was planned.

**Reconciliation (what actually happened):**

- **Phase 1 (StateScript contingency parser) — OBSOLETE.** The StateScript adapter
  was built and then *retired* in favor of the canonical LorenFrankLab/SpatialBanditTask
  parsed CSVs (commit `96dfed1`, ~96% agreement), mapped to `stop_id`. The deliverable
  table exists: `Processed-Data/replay_content/models_per_bin_rate/contingency_table.pkl`.
- **Phase 2 (per-population uncertainty/value rate fit + dissociation hypothesis) — DONE,
  and the hypothesis is answered.** `scripts/rate/fit_uncertainty_rate.py` + the
  `is_swr`-contrast G/S ladder (`fit_swr_contrast_rate.py`) deliver it. Result:
  non-SWR events are broadly task-modulated, SWR narrowly reward-gated — but
  **"uncertainty gates rate" is NULL**: the HMM→beta migration re-fit belief entropy
  (`beta_reward_entropy`) on the rate at RR≈1.2, p>0.08. The model-free `leaky_q`/nominal-p(R)
  proxies and the pre-beta motivating numbers in this plan's prose are HMM-era and no
  longer the framing (headline is reward-gating RR 3.09).
- **Phase 3 (cross-type ensemble-history GLM + onset cross-correlogram) — the RESIDUAL.**
  `scripts/rate/fit_cross_history_rate.py` exists but was never *run/documented* (no
  continuum verdict), and the peri-onset-peak cross-correlogram is unbuilt. → moved to
  `rate-content-followups` phase-2.

---

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
