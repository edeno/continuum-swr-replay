# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Background

Figure 4 supports claim #20 ("non-SWR replay significantly modulates mPFC
activity"). A code/methods review
([.claude/docs/reviews/05_pfc_cross_area_review.md](../../reviews/05_pfc_cross_area_review.md))
returned **changes requested**: the PFC pipeline defines its own non-canonical
`is_swr` split, the saved population artifacts look stale, there is no matched
non-event control, and inference pools ~10,980 neurons across 5 animals with no
animal-level accounting. This plan implements the review's five "Next Checks".
The 2026-07-10 [cross-family regression audit](../../reviews/07_comprehensive_regression_model_statistical_audit.md)
adds the spike-bin, estimability, spike-history/drift, and temporal-calibration requirements in
Phases 0 and 2b.

## Decisions locked at planning time

1. **Scope: full recut** — GLM mechanics, canonical events, matched control,
   dependence-aware inference, animal-level inference, and Fig 4 regen.
2. **Matched control: time-shifted onsets** — draw control onsets from the same
   immobility bouts, offset from real events, matching count and duration.
3. **Inference: per-animal-then-combine** — report per-animal fractions/effects
   and a simple animal-level test; no mixed model, no cluster bootstrap.
4. **Claim framing: engagement only** — drop equivalence/"regardless". No powered
   TOST. Fig 4 is retitled to an engagement claim.

## Current codebase integration points

- [src/continuum_swr_replay/analysis/pfc/glm.py:1002-1046](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L1002-L1046) — `run_session` event construction. **Replaced** in Phase 1: currently builds non-local events with `get_confident_times` (line 1007) then splits SWR/non-SWR via `get_overlap_times(non_local_times, data["ripple_times"])` (line 1017). Immobility filter uses `< immobility_threshold` (lines 1014, 1042).
- [src/continuum_swr_replay/analysis/pfc/glm.py:347-441](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L347) — `_make_design_matrices`: model dict at lines 435-439. **Extended** in Phase 2 with a `matched_control` design matrix.
- [src/continuum_swr_replay/analysis/pfc/glm.py:25-60](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L25) — spike-bin left-edge contract and zero-width final-bin behavior. **Corrected** in Phase 0.
- [src/continuum_swr_replay/analysis/pfc/glm.py:446-480](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L446) — blind per-model Poisson fitting. **Guarded** in Phase 0 and extended with history/drift in Phase 2b.
- [src/continuum_swr_replay/analysis/pfc/glm.py:538-929](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L538) — `process_neuron`: LR-test pairs listed at lines 733-743. **Extended** in Phase 2.
- [src/continuum_swr_replay/analysis/pfc/glm.py:930-938](../../../../src/continuum_swr_replay/analysis/pfc/glm.py#L930) — `run_session` signature returning a 5-tuple. **Extended** in Phase 2 (adds matched-control modulation array + provenance dict).
- [src/continuum_swr_replay/analysis/hippocampus/event_table.py:206](../../../../src/continuum_swr_replay/analysis/hippocampus/event_table.py#L206) — `build_classified_nonlocal_event_table`, the canonical source. Emits `start_time`/`end_time` (renamed from `event_start`/`event_end` at line 318) and `is_swr = overlap_class != "non_swr"` (line 324). **Read, not modified.**
- [scripts/pfc/run_pfc_analysis_bandit.py:86-95](../../../../scripts/pfc/run_pfc_analysis_bandit.py#L86-L95) — `results_exist` file set; saves at lines 131, 168-183; `git_hash` already threaded (line 105). **Extended** in Phases 2/3 (new artifact files + provenance sidecar).
- [src/continuum_swr_replay/analysis/pfc/load_population.py:55](../../../../src/continuum_swr_replay/analysis/pfc/load_population.py#L55) — mPFC-only glob; animal column added at line 139; FDR pooled across all neurons at lines 161-213. **Extended** in Phase 4 (per-animal aggregation) with a stale-column guard.
- [scripts/figures/generate_figure04.py:2](../../../../scripts/figures/generate_figure04.py#L2) — title "respond to replay regardless of SWR detection"; duplicate non-canonical event construction at lines 330-352; immobility `<=` at line 357; `% significant` computed as `df[col] < 0.05` pooled across all neurons at line 261; output dir `paper/figures/figure04` (line 474). **Rewritten** in Phase 4.

## Scope and dependency policy

### Goals

- Fig 4 / claim #20 rest on the canonical `is_swr` split, a matched non-event
- Per-neuron event effects control spike history/slow drift and use temporal nulls/held-out scores rather than treating adjacent 2-ms bins as independent.
- Every consumed artifact is traceable to a code version and event-definition tag.

### Non-Goals

- **OFC.** The GLM/decoding configs support OFC, but claim #20 and Fig 4 are
  mPFC-only. Extending `load_population` and the figure to OFC is out of scope.
- **Powered equivalence / TOST.** Dropped per decision 4.
- Changing the Poisson family or lag window. Theta mechanics are corrected in Phase 0; event GLMs gain nuisance terms and calibrated inference in Phase 2b.
- **Re-decoding HPC.** This plan consumes existing HPC decoding results.
- **The two misfiled replay/behavior scripts** (`run_stage5b_retrospective.py`,
  `run_stage6_learning.py`) under `scripts/pfc/` — they import `analysis.replay`,
  not `analysis.pfc`. Not touched here.

### Dependency policy

No new dependencies. Matched-control randomness uses a seeded
`numpy.random.default_rng` (seed passed explicitly, per repo reproducibility rule).

## Metrics

- **Correctness:** Phase 1 diagnostic reports, per session, how many events change
  `is_swr` class under the canonical definition and the resulting shift in the
  non-SWR modulation coefficient. No silent denominator change.
- **Provenance:** every regenerated artifact has a sidecar recording event counts,
  thresholds, `is_swr` definition tag, git hash (see [shared-contracts.md](shared-contracts.md)).
- **Inference:** Fig 4d reports per-animal fractions (5 values) plus a combined
  animal-level statistic, not a single pooled percentage.
- **Calibration:** fixed-seed autocorrelated null simulations control false positives within the prespecified tolerance; planted post-event effects are recovered.
- **Rerun feasibility:** Phase 3 smoke test measures one-session wall-clock + peak
  memory before launching the full batch.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Canonical event table uses a different event *window* than the PFC reconstruction, silently shifting onsets and modulation. | Phase 1 diagnostic explicitly compares event starts and counts old-vs-new per session before any rerun; divergence-guard test fails if PFC event construction diverges from the canonical table. |
| Matched control leaks: time-shifted onsets land on real events or outside immobility. | Phase 2 algorithm rejects offsets overlapping any real event window or non-immobile bins; unit test asserts zero overlap and immobility containment. |
| Full rerun is expensive / GPU-contended. | Phase 3 smoke test + extrapolation gate before the batch; runs in `tmux`. |
| Per-animal aggregation over 5 heavily-imbalanced animals is underpowered. | Report per-animal values transparently (5/5-style) rather than a single p-value; document imbalance in the figure caption. |
| Stale artifacts silently consumed by the figure. | Phase 4 loader guard raises if a consumed pickle lacks the provenance sidecar or carries removed non-nested LR columns. |
| Spike autocorrelation and drift generate optimistic chi-square tests. | Phase 2b uses spike-history/drift nuisance terms, circular/block nulls, and contiguous held-out deviance. |

## Rollout Strategy

Replace-in-place, in order `0 → 1 → 2 → 2b → 3 → 4`. New artifact files (matched-control `.npy`, provenance sidecar)
are additive; the four existing `*_modulation.npy` files keep their names but are
overwritten by Phase 3's rerun. No feature flag — the old non-canonical event
construction is deleted in Phase 1, not run in parallel. Fig 4 outputs overwrite
`paper/figures/figure04/`.

## Open Questions

1. **Immobility inequality — settle on `<` or `<=`?** Current best answer: adopt
   `<` (the batch `glm.py` convention) everywhere and delete the figure's `<=`.
   Resolved in Phase 1; flagged here only because it changes a boundary bin.
2. **Matched-control offset distribution** — fixed offset vs. random within-bout?
   Best answer: random offset drawn per event within its immobility bout, seeded.
   Specified in [shared-contracts.md](shared-contracts.md); revisit only if the
   diagnostic shows the control rate structure doesn't match real events.

## Estimated Effort

- Phase 0: ~150 LOC (binning + fit preflight/status + theta cleanup) + tests.
- Phase 1: ~150 LOC (event swap + provenance + diagnostic) + tests.
- Phase 2: ~120 LOC (control generator + design matrix + wiring) + tests.
- Phase 2b: ~250 LOC (history/drift bases + temporal null/CV) + tests.
- Phase 3: mostly compute; ~40 LOC runner/provenance glue.
- Phase 4: ~200 LOC (per-animal aggregation + figure rewrite) + doc edits.
