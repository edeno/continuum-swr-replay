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
3. **Inference: neuron → session → animal** — summarize neurons within session, combine sessions
   equally within animal, report all five animal effects, and use the prespecified one-sided exact
   sign test for positive matched-control engagement; no mixed model or cluster bootstrap.
4. **Claim framing: engagement only** — drop equivalence/"regardless". No powered
   TOST. Fig 4 is retitled to an engagement claim.
5. **Source generation: regression-owned** — production PFC artifacts consume the
   frozen staged canonical-table/session-input generation created during
   `regression-integrity-fixes` Phase 13. PFC never treats pre-refresh decoder or
   fixed table presence as proof of validity.

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
  immobility control, dependence-aware inference, and all-five-animal summaries.
- Per-neuron event effects control spike history/slow drift and use temporal nulls/held-out scores rather than treating adjacent 2-ms bins as independent.
- Every consumed artifact is traceable to one frozen canonical-table generation,
  its exact session-input manifest, a code version, and an event-definition tag.
- Phase 4 emits one atomic `pfc_completion.json` inside an immutable PFC run directory only after the
  complete rerun, population inference, staged Figure-4/adjudication smoke, and fit audit pass. It creates no
  PFC-specific current pointer; regression Phase 13 is the sole authority that makes the component
  current.

### Non-Goals

- **OFC.** The GLM/decoding configs support OFC, but claim #20 and Fig 4 are
  mPFC-only. Extending `load_population` and the figure to OFC is out of scope.
- **Powered equivalence / TOST.** Dropped per decision 4.
- Changing the Poisson family or lag window. Theta mechanics are corrected in Phase 0; event GLMs gain nuisance terms and calibrated inference in Phase 2b.
- **Re-decoding HPC.** Regression-integrity Phase 13 owns source refresh and any
  required HPC re-decode. This plan consumes only its frozen staged generation and
  reruns any PFC session whose recorded source hash changes.
- **The two misfiled replay/behavior scripts** (`run_stage5b_retrospective.py`,
  `run_stage6_learning.py`) under `scripts/pfc/` — they import `analysis.replay`,
  not `analysis.pfc`. Not touched here.

### Dependency policy

No new dependencies. Matched-control randomness uses a seeded
`numpy.random.default_rng` (seed passed explicitly, per repo reproducibility rule).
All code-bearing work in Phases 0–2b, Phase-3 runner/provenance glue, and Phase-4
loader/finalizer/figure code may proceed once regression Phase 7 establishes the common manifest
fields. Those changes and synthetic tests land before the shared clean production-code freeze. The
Phase-3 and Phase-4 files each have two explicit deliverables: a pre-freeze implementation PR and a
post-freeze production execution/reporting record with no source/config changes. The
production Phase-3 batch must wait until regression Phase 13 has completed all
source/table/model runs and frozen its release-candidate manifest plus per-session input manifests.
The candidate is immutable while PFC Phases 3–4 execute their production work. Phase 4 completes
before Phase 13 attaches the validated PFC run to the global regression release.

## Metrics

- **Correctness:** Phase 1 diagnostic reports, per session, how many events change
  `is_swr` class under the canonical definition and the resulting shift in the
  non-SWR modulation coefficient. No silent denominator change.
- **Provenance:** every regenerated artifact has a sidecar recording event counts,
  thresholds, `is_swr` definition tag, git hash, canonical-table-generation hash,
  and per-session source-manifest hash (see [shared-contracts.md](shared-contracts.md)).
- **Completion:** the atomic completion manifest covers every expected session
  exactly once, all five animals, accounts for every attempted neuron/model fit with explicit
  rank/convergence status, hashes every consumed/output artifact, and is
  absent whenever any production validation remains incomplete.
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
| PFC finishes against a source generation that Phase 13 later refreshes. | Phase 3 requires the frozen staged generation hash in every sidecar; any changed session/table hash invalidates and reruns that PFC session before Phase 4 can complete. |
| An interrupted rerun looks current or damages the previous PFC output. | Phase 3 writes only to a new `<pfc_root>/runs/<pfc_run_id>.staging/`; Phase 4 writes the completion file last and atomically finalizes that directory. No PFC current pointer exists. |
| A staged Figure 4 or claims edit mutates after PFC completion. | Figure files and `claim_20_adjudication.json` are hashed in the global release-output staging tree; shared Markdown is composed once by Phase 13 and the whole tree is immutable before `CURRENT.json` advances. |

## Rollout Strategy

Land implementation PRs in order `0 → 1 → 2 → 2b → 3-code → 4-code` with synthetic fixtures, then
include them in the shared clean analysis-code freeze. Wait for regression Phase 13 to freeze the
source-valid release candidate; execute `3-production → 4-production` against that exact generation
without changing source or config. New artifact files (matched-control `.npy`,
provenance sidecar) and the existing modulation basenames all live under the new
`<pfc_root>/runs/<pfc_run_id>.staging/`; legacy files are read-only inputs and are never
overwritten. No feature flag — the old non-canonical event construction is deleted in Phase 1, not
run in parallel. Figure 4 and `claim_20_adjudication.json` are written beneath the shared
`<regression_root>/release_outputs/<release_id>.staging/` using repository-relative paths and are
hashed by both PFC completion and the global release; live paper paths are untouched. Phase 4 writes
`pfc_completion.json` last, validates it, and atomically renames the staging directory to
`<pfc_root>/runs/<pfc_run_id>/`. Regression Phase 13 then validates and records the immutable
run/path hash as the optional PFC component, composes shared claims documents from the adjudication,
and promotes the one global release. The post-promotion PFC publisher resolves
`components.pfc` through global `CURRENT.json` and copies only the already-hashed Figure-4/claim bytes
to repository destinations; it does not rerun, reimplement, or independently point at PFC analysis.

## Resolved implementation details

1. **Immobility inequality:** use `<` everywhere and delete the figure's `<=`.
2. **Matched-control offset distribution:** draw a seeded random offset per event within its
   immobility bout, exactly as specified in [shared-contracts.md](shared-contracts.md).

## Estimated Effort

- Phase 0: ~150 LOC (binning + fit preflight/status + theta cleanup) + tests.
- Phase 1: ~150 LOC (event swap + provenance + diagnostic) + tests.
- Phase 2: ~120 LOC (control generator + design matrix + wiring) + tests.
- Phase 2b: ~250 LOC (history/drift bases + temporal null/CV) + tests.
- Phase 3: mostly compute; ~40 LOC runner/provenance glue.
- Phase 4: ~200 LOC (per-animal aggregation + figure rewrite) + doc edits.
