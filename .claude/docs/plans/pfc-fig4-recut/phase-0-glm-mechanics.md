# Phase 0 — PFC spike-bin and GLM mechanical validity

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contract](shared-contracts.md#pfc-fit-validity)

Correct the final spike-bin edge, preflight every actual design, persist fit validity,
and remove the effectively unregularized theta workaround. This is behavior-preserving
where the old design was estimable, except for spikes in the formerly zero-width final bin.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/glm.py:25-60](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L25) — current left-edge binning and last-edge inclusion.
- [src/continuum_swr_replay/analysis/pfc/glm.py:167-245](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L167) — theta GLM and `alpha=1e-20` fit.
- [src/continuum_swr_replay/analysis/pfc/glm.py:446-480](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L446) — blind immobility model fitting.
- [src/continuum_swr_replay/analysis/pfc/glm.py:680-735](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L680) — rank detection after fitting and LR routing.
- [tests/unit/test_pfc_spike_binning_last_bin.py:1](../../../../tests/unit/test_pfc_spike_binning_last_bin.py#L1) — test currently enshrining a zero-width endpoint bin.

**Contracts referenced:** [PFC fit validity](shared-contracts.md#pfc-fit-validity).

## Tasks

- Rewrite `get_spikecount_per_time_bin` around explicit edges: validate increasing,
  near-regular left edges; infer final right edge as `time[-1] + median(diff(time))`;
  use half-open bins; return exactly `len(time)` counts. Update all docstrings from
  the contradictory center/left-edge language.
- Add a PFC design-preflight helper using scaled SVD on the actual immobility rows.
  `_fit_immobility_models` returns structured success/failure entries and captures
  convergence/separation warnings; it does not fit a known rank-deficient design.
- Route LR, AIC, modulation, and saved arrays through the same validity status. Remove
  the later duplicate rank check and any finite-degenerate fallback.
- Fit the well-posed three-column theta model unregularized for coefficients and LR,
  with the same rank/convergence guard. Remove the claim that `alpha=1e-20` provides
  meaningful regularization.
- Thread validity fields into per-neuron artifacts and provenance. Update loader tests
  to distinguish unavailable from estimated-zero effects.
- Document the bin convention and fit-status schema in public docstrings.

## Deliberately not in this phase

- Canonical event labels, matched controls, spike-history/drift, or temporal nulls.
- Batch regeneration; validate with synthetic data and 2–3 neurons from one session.

## Validation slice

| Test | Asserts |
| --- | --- |
| spike-bin boundary tests | every spike in `[first, inferred final edge)` is counted once; final-right-edge spike excluded |
| irregular/nonmonotone time test | malformed axes fail with a clear error |
| absent-event rank fixture | model is not fit and all dependent outputs are unavailable with reason |
| nonconverged/NaN fixture | cannot produce finite LR/AIC/modulation |
| theta planted/null fixtures | well-posed fit recovers phase; degenerate phase design fails status |
| one-neuron smoke (slow) | current artifact schema includes validity and finite outputs for estimable models |

## Fixtures

Extend existing 500-Hz synthetic time/spike/event fixtures with boundary spikes,
empty event class, duplicate design column, and constant theta phase.

## Review

Before opening the PR, dispatch `scientific-code-reviewer`. Confirm the old zero-width
last-bin behavior and post-fit-only rank route are deleted; every downstream statistic
uses one validity status; and no event-definition/inference change entered this PR.

Also confirm every task is complete, deliberately deferred event/inference work stayed out of scope,
validation tests pass and are nontrivial, old paths are removed, public docstrings are updated, and no
production name references this plan or its phase number.
