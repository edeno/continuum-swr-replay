# Phase 2b — PFC dependence-aware event inference

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [shared contract](shared-contracts.md#pfc-temporal-inference)

Add spike-history and slow-drift nuisance structure, autocorrelation-preserving event
nulls, and contiguous held-out scoring before the batch rerun. Symmetric peri-event
kernels remain descriptive; directional claims use post-event effects with pre-event placebo checks.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/pfc/glm.py:347-443](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L347) — current intercept + event-kernel designs.
- [src/continuum_swr_replay/analysis/pfc/glm.py:538-760](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L538) — neuron statistics and nominal LR/AIC path.
- [tests/analysis/pfc/test_glm_pipeline.py:30](../../../../tests/analysis/pfc/test_glm_pipeline.py#L30) — synthetic 500-Hz fixtures to extend.
- [phase-2-matched-control.md](phase-2-matched-control.md) — matched-control event class available before this phase.

**Contracts referenced:** [PFC fit validity](shared-contracts.md#pfc-fit-validity) and [temporal inference](shared-contracts.md#pfc-temporal-inference).

## Tasks

- Add a prespecified raised-cosine spike-history basis and slow-time spline/session-drift
  columns to every event candidate, including constant and matched-control models.
  Construct nuisance columns once so model comparisons differ only in event terms.
- Add deterministic contiguous immobility-block assignments and identical held-out
  Poisson deviance scoring across candidate models. Retain AIC under descriptive names only.
- Implement seeded circular time-shift or whole-block permutation nulls that preserve
  each neuron's spike train/autocorrelation and event counts. Estimate event-term and
  joint-kernel p-values from the empirical null; record valid permutation fraction.
- Make post-event kernel/joint effect primary for directional response language; use the
  pre-event component as a placebo and keep the symmetric curve as visualization.
- Define the multiplicity family across neuron × prespecified contrast and write raw plus
  corrected values. Population consumers aggregate neuron→session→animal in Phase 4.
- Add fixed-seed null-calibration and planted-response simulations. Gate the real rerun on
  acceptable false-positive calibration and recovery, not merely finite output.
- Extend provenance with nuisance basis, block/fold hash, null method/seed/count, held-out
  scores, and calibration version.

## Deliberately not in this phase

- Changing canonical event/control definitions (Phases 1/2).
- Full batch compute (Phase 3) or final population figure (Phase 4).
- Claiming SWR/non-SWR equivalence; engagement remains the target.

## Validation slice

| Test | Asserts |
| --- | --- |
| spike-history planted fixture | refractory/bursty dependence is captured without false event effect |
| slow-drift fixture | drift does not become peri-event modulation |
| block-assignment test | no temporal interleaving; all candidate models share fold hash |
| circular/block null invariants | spike autocorrelation and event count preserved; seed deterministic |
| null calibration simulation | false-positive rate lies within prespecified Monte Carlo interval |
| planted post-event response | held-out score improves and post-event empirical-null test detects effect |
| one-session smoke (slow) | compute/memory and valid-permutation fraction are recorded before batch |

## Fixtures

Extend PFC pipeline fixtures with autoregressive/refractory spikes, slow rate drift,
null events, planted post-event response, and multiple immobility blocks. Use one cached
real session only for the marked slow smoke.

## Review

Before opening the PR, dispatch `scientific-code-reviewer`. Confirm all models share
identical nuisance columns/folds, empirical nulls preserve the intended dependence,
nominal chi-square/AIC are not headline inference, and calibration tests are nontrivial.

Also confirm every task is complete, batch/figure work stayed out of scope, validation tests pass with
slow tests marked, old nominal-headline paths are removed, docs are updated, and no production name
references this plan or its phase number.
