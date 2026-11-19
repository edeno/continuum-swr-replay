# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

Everything below is on branch `feature/pfc-marked-event-kernels` at commit `55c55d9` (the plan itself is `5dac575`; the branch is now pushed, 14 commits over master). Line numbers were verified against that commit; if the branch has moved, re-grep the named function before editing.

## Current codebase integration points

- `src/continuum_swr_replay/analysis/pfc/marked_glm.py:145` — `KernelDesign`: gains a `n_baseline` field; its `__post_init__` (contiguity check, first block at column 1) moves the first block to column `1 + n_baseline`.
- `src/continuum_swr_replay/analysis/pfc/marked_glm.py:273` — `MarkedSessionContext`: gains `baseline` `(n_fit_bins, n_baseline)` and `baseline_names`.
- `src/continuum_swr_replay/analysis/pfc/marked_glm.py:307-437` — `prepare_marked_session_context`: builds the baseline columns from `cohort_stops` after the fit mask at `:381` and before spike binning at `:413`.
- `src/continuum_swr_replay/analysis/pfc/marked_glm.py:917-980` — `build_kernel_design`: the intercept column at `:954` becomes intercept plus baseline columns; takes `baseline: NDArray | None`.
- `src/continuum_swr_replay/analysis/pfc/marked_glm.py:983` — `_without_block`: slice recomputation starts at `1 + n_baseline`.
- `src/continuum_swr_replay/analysis/pfc/marked_glm.py:1006` — `build_marked_model_ladder` and the single-mark runner `scripts/pfc/run_pfc_marked_kernels.py`: pass the baseline through and use `make_ridge_penalty(width, sigma, n_unpenalized=1 + n_baseline)`; otherwise untouched by this plan.
- `src/continuum_swr_replay/analysis/pfc/marked_model_family.py:89` — `MARK_DESCRIPTIONS`, `:195-383` `RECOMMENDED_MODEL_SPECS`, `:385-390` `CORE_MODEL_NAMES` / `INTERACTION_MODEL_NAMES`: new marks and re-specified models (phase 1b), marginal specs (phase 2).
- `src/continuum_swr_replay/analysis/pfc/marked_model_family.py:609-799` — `build_recommended_event_mark_table`: new distance and size marks; interaction products at `:792-797` become centered products.
- `src/continuum_swr_replay/analysis/pfc/marked_model_family.py:980-1080` — `build_marked_family_ladder`: rare-level eligibility rule; baseline pass-through at `:1063`.
- `src/continuum_swr_replay/analysis/pfc/marked_model_family.py:1132-1160` — `DEFAULT_MARK_PENALTY_SIGMA` and `family_ridge_penalty`: `n_unpenalized` follows the baseline width.
- `src/continuum_swr_replay/analysis/pfc/marked_model_family.py:1229` — `fit_marked_family_ladder`: slope-permutation draws (phase 3) hang off the full-data fit here.
- `scripts/pfc/run_pfc_marked_model_family.py:109` — `BUNDLE_CONTRACT_VERSION = 1` becomes 2; `:166` `build_fit_contract` gains the baseline and tier keys; `:594` context preparation; `:681` ladder build; `:800` CLI.
- `scripts/pfc/summarize_pfc_marked_family.py:59-120` — `load_bundles`: reads the `tier` field; `:181` `population_curve_summary` reads the null curves (phase 3).
- `src/continuum_swr_replay/analysis/pfc/cohort.py:32` — `load_session_stop_cohort`: the stops it returns carry `stop_start`, `stop_end`, `occupancy_well_id`, `is_rewarded_outcome`, `immobility_duration_s`; untouched.
- `src/continuum_swr_replay/analysis/pfc/glm.py:1464` `build_session_fit_mask`, `:1661` `build_session_event_sets`, `:40` `get_spikecount_per_time_bin`: untouched, reused by phases 4 and 5.
- `Processed-Data/replay_content/stop_table.parquet` (107,400 stops, 55 columns) and `event_table.parquet` (`mean_abs_ahead_behind_cm`, `spatial_extent_cm`, `stop_id`, `occupancy_well_id`, `is_rewarded_outcome`): read, never written.
- `Processed-Data/pfc_marked_model_family/` — the contract-1 corpus (816 bundles, git `4d69b83`): renamed, not deleted; see rollout.

## Scope and dependency policy

### Goals

- Estimate event-locked responses separately from per-stop baseline differences, so the reward, switch, novelty and previous-arm results become interpretable.
- Report every content effect both marginally (alone) and jointly (partial), the way the content family reports total and direct.
- Give the population slope kernels a shuffled-mark reference so a 1 % consistent slope can be read against chance.
- Test the two things the family cannot: mPFC drive against replay count per stop, and reactivation of the replayed place in the mPFC population pattern.

### Non-Goals

- No p-values inside the family beyond the permutation references named here; the family stays descriptive.
- No per-arm kernels (arm-identity tuning); phase 5 answers "which place" from the population side instead.
- No change to the canonical Figure 4 model, its bundles, or claim 20.
- No OFC run in this plan; the runners accept `--brain-area OFC` and that is a separate decision.
- No time-resolved (within-event) content marks and no pre-onset "mPFC predicts content" analysis; both are listed as later questions in `.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md` and are out of scope here.

### Dependency policy

No new dependencies. `statsmodels` (already present) is used for the per-unit per-stop Poisson fits in phase 4.

## Metrics

- Phase 1a: with baseline columns included and every mark removed, the base-kernel curves equal the canonical Figure 4 curves to `atol 1e-8` on the three smoke sessions **only when the baseline is empty**; with well and reward columns present the intercept-only model's held-out deviance must not be worse than the single-intercept model on those sessions (baseline terms are unpenalized, so this is a rank check, not a tuning check).
- Phase 1b: on the adversarial mark fixture, distance and size marks match hand-computed values; centered interaction products have `|r| < 0.2` with each constituent on the corpus (median over session × class), against 0.81–0.95 today.
- Phase 1c: 71 sessions × models, one contract, zero failed sessions, blow-up units under 10 corpus-wide for every model including the interactions (37 in the exploration model today).
- Phase 2: joint and marginal estimates for every target on one table; the gap is reported, not interpreted.
- Phase 3: population mean slope kernels with a per-lag null band from at least 30 draws per session for the three models retained.
- Phase 4: per-stop coefficient of replay count on mPFC rate, aggregated unit → session → animal, with sign consistency across animals and a within-session shuffle reference.
- Phase 5: per-session reactivation statistic with a 1,000-draw within-session label shuffle and animals-consistent count.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A well never visited in a session gives an all-zero, unpenalized baseline column and a rank-deficient design. | Drop empty baseline columns per session, record the kept names in the sidecar, and let `KernelDesign.__post_init__` verify the width (phase 1a). |
| Baseline columns are unpenalized while everything else is ridge-shrunk, so a stop with few bins could carry an extreme coefficient. | Reward outcome and well indicators are session-wide (many stops each), never per-stop dummies; a per-stop random intercept is deliberately not attempted. |
| Contract 2 bundles silently mixed with contract 1 bundles in a summary. | The contract-1 tree is renamed with a `_v1_single_intercept` suffix before the refit; `load_bundles` refuses a directory whose sidecars disagree on `bundle_contract_version`. |
| Marginal-tier bundles doubling the corpus footprint. | Marginal designs are compact (45–67 columns); their bundles carry the same layout under a `tier` key and a distinct artifact name; storage is about a fifth of the joint tree. |
| Slope-permutation refits are R full-data fits per model per session. | Restricted to three models and R = 30 by default; the runner records draws so a second pass can extend R without refitting the first 30. |
| Phase 5's reactivation statistic is confounded by the current well (mPFC well tuning plus current-arm content bias). | The current well's template is excluded from both numerator and comparison set, and only events with at least half their leaf content off the current well enter. |

## Rollout Strategy

Replace in place; no deprecation window (CLAUDE.md default). Before phase 1c's refit, `Processed-Data/pfc_marked_model_family/` is renamed to `Processed-Data/pfc_marked_model_family_v1_single_intercept/` and the audits that quote it (`.claude/docs/reviews/2026-09-19-pfc-marked-family-corpus.md`, `-summary.md`) get one line saying so. Contract-1 bundles are never read by the contract-2 summary. Phases 4 and 5 write their own artifact directories and touch nothing in the family tree.

## Open Questions

1. Should the baseline include a smooth time-in-stop term (the rate family's within-stop time course) as well as well and reward? Current answer: no; the within-stop-timing mark already asks that at the event level and phase 4 asks it at the stop level.
2. Which distance mark: `mean_abs_ahead_behind_cm` (mean represented distance from the animal over the event) or `max_abs_ahead_behind_cm`? Current answer: the mean, because the maximum is one bin's excursion; both are in the event table if the executor wants a sensitivity.
3. Should phase 3's null also refit the held-out scorer (a null for the delta as well as the curve)? Current answer: no; that is the single-mark runner's job and costs K-fold refits per draw.
4. Phase 4's exposure: whole immobile stop, or only the fitted at-well bins? Current answer: the fitted bins, so the model shares the family's population exactly.

## Estimated Effort

- Phase 1a: ~250 LOC source, ~200 LOC tests.
- Phase 1b: ~200 LOC source, ~250 LOC tests.
- Phase 1c: ~50 LOC (launcher, audit script tweaks), one batch of about 15 GPU-hours on three cards, two audit documents.
- Phase 2: ~150 LOC source, ~120 LOC tests, about 4 GPU-hours.
- Phase 3: ~200 LOC source, ~150 LOC tests, about 15 GPU-hours.
- Phase 4: ~300 LOC script, ~150 LOC tests, CPU only, under an hour.
- Phase 5: ~350 LOC script, ~200 LOC tests, CPU only, about two hours.
