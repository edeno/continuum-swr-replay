# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

- `src/continuum_swr_replay/analysis/replay/tables.py:176-209` — `_prepare_dot` and `_in_event_mask`: define decoded arms, non-local weight, and strict event-retention thresholds.
- `src/continuum_swr_replay/analysis/replay/tables.py:215-669` — `build_event_table`: one retained non-local label per row, filtered-survivor aggregates, full-label SWR classification, posterior summaries, trajectory summaries, and stop assignment.
- `src/continuum_swr_replay/analysis/replay/tables.py:672-771` — `build_event_sequence_table`: one retained event time bin per row and the nine-arm posterior.
- `src/continuum_swr_replay/analysis/replay/tables.py:774-1006` — `build_stop_table`: one row per immobility bout, including zero-event and single-sample bouts; this table supplies rate-model exposures and reward lags.
- `src/continuum_swr_replay/analysis/replay/tables.py:1009-1073` — `add_nonlocal_swr_split`: assembly-time reconciliation of SWR and non-SWR event counts by stop.
- `src/continuum_swr_replay/analysis/replay/tables.py:1076-1370` — `build_event_arm_table`: nine alternatives per event, hard/soft/posterior responses, spatial/value/history covariates, and interactions.
- `src/continuum_swr_replay/analysis/replay/tables.py:1373-1538` — `build_event_patch_table`: three alternatives per event, projected from the arm table.
- `src/continuum_swr_replay/analysis/replay/tables.py:1541-1751` — private schema lists: current source of ordered output columns; Phase 1 turns these implicit lists into validated contracts.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:29-266` — trial context, immobility bouts, hashes, and posterior alignment used by multiple builders.
- `src/continuum_swr_replay/analysis/replay/covariates.py:47-374,684-803` — visit/reward history, lagged rewards, contingency values, and uncertainty predictors.
- `src/continuum_swr_replay/analysis/replay/summaries.py:59-317` — arm-posterior marginalization and event/arm response summaries.
- `src/continuum_swr_replay/analysis/replay/uncertainty.py:28-204` and `trajectory.py:541-672` — posterior uncertainty and trajectory rollups merged into event rows.
- `src/continuum_swr_replay/analysis/replay/spatial_mapping.py:60-166` — position/arm/well mappings whose failure can silently permute model alternatives.
- `src/continuum_swr_replay/analysis/hippocampus/event_table.py:206-365` — full-label event classification and strict positive-duration SWR overlap.
- `src/continuum_swr_replay/analysis/hippocampus/data_aggregation.py:215-280` — the existing partial `dot` validator; Phase 1 replaces this location with the canonical replay-table contract module and migrates callers without a compatibility alias.
- `src/continuum_swr_replay/analysis/replay/prepare_data.py:87-195,199-304,335-506` — session discovery/cache loading, builder orchestration, concatenation, the current single cross-table guard, pooled standardization, and canonical writes.
- `tests/unit/test_replay_tables.py:29-733` — current synthetic builder coverage. The focused baseline selection currently passes 77 tests, but does not independently establish most join/filter semantics.
- `tests/integration/test_from_cache_parity.py:29-57` — current cold/warm parity test; it can skip on broad loading failures and compares two executions of the same code.
- `tests/integration/test_builder_parity.py:12-20` — explicitly records that raw one-session canonical parity is absent.
- `scripts/content/fit_replay_content_models.py:279-460,658-750`, `scripts/content/fit_swr_contrast_content.py:100-200,503-584`, `scripts/rate/fit_per_stop_rate_models.py:59-185`, `scripts/rate/fit_per_bin_rate_models.py:271-327`, and `scripts/rate/fit_swr_contrast_rate.py:133-478` — current content/rate consumers; Phase 3 uses only their corrected successors after regression-integrity cohort/fit/inference gates land.
- `scripts/content/compare_event_source_models.py:450-545`, `scripts/pfc/run_stage6_learning.py:47-244`, and `scripts/rate/fit_swr_power_model.py:42-153` — legacy choice/stay/power estimators explicitly replaced by regression-integrity Phases 11 and 12; they may be historical table-sensitivity diagnostics but cannot certify a claim.
- `scripts/content/analyze_content_occupancy.py:1-166` — claim #14 mixes a canonical event-arm occupancy view with a raw-dot time-bin view; the claim matrix must separate those estimands.
- `paper/claims_table.md:25,42-57,61-64` — candidate table dependencies and adjacent claims. Initial direct set is #2, #13, the table-based component of #14, #15-#19, #21, and #22. Claim #24 is a raw timepoint ripple-density analysis owned by the SWR-continuum plan, not the event-level power regression.
- `.claude/docs/plans/regression-integrity-fixes/phase-7-run-artifacts.md` through `phase-12-swr-power.md` — prerequisite owners of source-valid coherent runs, exact cohorts/folds, fit validity, robust inference, decision-grain behavior, and identified power. Phase 3 consumes their APIs/manifests and does not add competing path/cohort/model plumbing.
- `.claude/docs/plans/regression-integrity-fixes/phase-13-refit-reporting.md` — sole owner of production rebuild/refit, promotion, and paper cutover. This audit supplies a mandatory table-specific gate and impact matrix; it never promotes candidate tables.

Current on-disk scope is large enough to require a staged design: 218 sessions from five animals, about 19 GB of `data_over_time` inputs, about 271 GB of posterior NetCDF files, and five canonical table artifacts totaling about 789 MB. The current materialized tables contain 65,913 event rows, 593,217 event-arm rows, 197,739 event-patch rows, 107,400 stop rows, and 2,971,614 sequence rows. These are baseline observations to record, not values to preserve if a validated correction changes the event population.

## Scope and dependency policy

### Goals

- Make every public table builder reject mixed-session, misaligned, duplicated, malformed, or cross-generation inputs before producing plausible output.
- Establish the event, stop, arm, patch, and sequence response arithmetic with hand-authored examples and a slow reference implementation that shares no production table logic.
- Reconcile the five canonical tables against all 218 raw `data_over_time` inputs without rebuilding or modifying the artifacts.
- Review deterministic, source-to-table visual windows across every animal and the highest-risk event/stop strata, with an auditable manual sign-off sheet.
- Validate a complete, generation-coherent noncanonical run produced through the regression-integrity run API: replay tables, behavior/Q, event-onset/time-bin, and decision components share one source/table generation.
- Compare table generations by scientific keys and quantify impact on every directly table-dependent atomic estimand using Phase 9-12 cohort/fold/fit/inference contracts; consume other claim owners' completion manifests for mixed/raw analyses.
- Issue a deterministic `PASS`, `PASS—PAPER REVISION REQUIRED`, `FAIL`, or `INCONCLUSIVE` verdict that separates table correctness from paper-impact ownership.

### Non-Goals

- Redesigning the decoder, rerunning event detection, changing SWR thresholds, or testing the broader biological reality of replay.
- Treating expected biological directions, coefficient similarity, or non-significant p-values as table-correctness oracles.
- Changing event-window, stop-exposure, trunk-fallback, probability-clipping, or junction-mass semantics inside an audit PR before the evidence and scientific decision are recorded.
- Rewriting regression estimands, promoting candidate pickles, modifying figures, or changing manuscript numbers in any of the three audit PRs.
- Reimplementing run manifests/path resolution, comparison cohorts, folds, fit diagnostics, dependence-aware inference, decision models, or SWR-power models already owned by regression-integrity Phases 7-12.
- Preserving incorrect behavior through newly introduced compatibility aliases. Callers are migrated to corrected internal APIs in the same PR; any public schema change is documented and reviewed explicitly.
- Auditing unrelated PFC feature construction or ContFrag tables. The shared patch projection used by ContFrag receives a cross-check, but Non-Local replay tables remain the review boundary.

### Dependency policy

Use existing project dependencies only: NumPy, pandas, xarray, NetworkX, matplotlib, pytest, and the current serialization stack. The test oracle may not import production replay builders or their calculation helpers. Phase 2 must not scan 271 GB of posterior NetCDF; it reads canonical tables and streams one raw `data_over_time` file at a time. Phase 3 requires the Phase-7 source audit/cache attestation and immutable run resolver; it does not add a weaker spot-check or direct Spyglass fallback.

## Metrics

- The existing focused replay-table baseline remains green: 77/77 selected tests before changes, followed by the full unit suite in each PR.
- Every public builder has a tested input contract, a schema-complete typed empty path, a tested non-mutating-input guarantee, and output validation at its return boundary.
- Hand-authored oracle cases cover every mask clause and boundary, full-label versus filtered windows, SWR overlap, zero/single-sample stops, all-junction bins, diffuse and one-hot posterior mass, arbitrary DataFrame indices, and missing/duplicated keys.
- Key/count/boolean/category identities compare exactly. Deterministic float identities use `atol=1e-12, rtol=1e-10`; raw coordinate alignment uses `atol=1e-6`; posterior row normalization uses `atol=1e-5`. NaN masks match exactly and infinities are forbidden.
- Phase 2 inventories exactly 218 current sessions and all five animals, hashes the five canonical tables, runs every hard/identity check, and obtains manual verdicts for all 40 deterministic base QC selections. Any hard or identity failure blocks sign-off.
- Canonical table hashes, sizes, and modification times are unchanged before/after every audit and rebuild stage.
- Phase 3 validates a Phase-7 noncanonical run with every expected session/component accounted for, produces no unexplained key/cohort/schema/value delta, and completes the claim-impact row for every direct table estimand (#2, #13, table-based #14, #15-#19, #21, #22) plus explicit cross-owner status for mixed/raw claims such as #24.
- A final verdict is impossible unless source/run provenance, structural/oracle gates, generation-coherent auxiliaries, Phase 9-12 cohort/fold/fit/inference gates, required owner completion manifests, and canonical before/after hashes are complete.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Production and oracle code repeat the same mistake | The oracle cannot import `tables`, `_helpers`, `summaries`, `spatial_mapping`, `uncertainty`, contracts, or producer schema helpers; expected values are literal and reviewed on tiny fixtures. |
| A transposed array or row-order bug still produces reasonable distributions | Compare on declared natural keys after uniqueness checks, test shuffled/nonconsecutive labels and duplicate indices, and assert exact per-event/per-stop conservation. |
| Full-corpus work accidentally rewrites canonical inputs | Phase 1/2 write only under ignored scratch; Phase 3 uses Phase-7 staging/finalization with no current-pointer advance, hashes legacy canonical files before/after, and cannot resolve a staging directory as production current. |
| The audit passes because it derives expectations from producer internals | Phase 2 maintains an independent, versioned schema/check registry and independently recomputes raw-dot masks/bouts instead of calling builder helpers. |
| A real data anomaly is “fixed” by clipping, filling, sorting, or relaxing tolerances | Contracts reject rather than repair; tolerances are declared here and versioned in the manifest; any change requires a reviewed contract revision before seeing candidate results. |
| The 271 GB posterior corpus makes the review impractical | Phase 2 never opens posterior NetCDF; Phase 3 measures one-session resource use, gates eight stress sessions, and only then starts the full staged rebuild. |
| Corrected tables change cohorts or conclusions | Use Phase-9 exact key manifests and strict fitters for natural/intersection cohorts, Phase-10 robust inference, and separate table correctness from claim-owner adjudication. |
| Negative claims are “confirmed” with `p > 0.05` | Require a predeclared equivalence bound and confidence interval; otherwise mark the claim `unresolved`, never supported by absence of significance. |
| Existing remediation work is duplicated or contradicted | Phase 3 has a hard dependency on Phases 7-12 and imports their run/identity/fold/fit APIs. Production compute/promotion and manuscript cutover remain exclusively in Phase 13. |
| A legacy estimator reproduces a paper number but is scientifically invalid | Legacy choice/Stage-6/power outputs are diagnostics only. A claim row remains `unresolved` until its Phase-11/12 corrected estimator and Phase-10 inference are present. |
| Candidate tables are paired with stale Q/onset artifacts | Candidate replay, behavior/Q, onset/time-bin, and decision components are regenerated/finalized under one run-generation manifest; mixed-generation components fail resolution. |

## Rollout Strategy

1. Phase 1 lands formal contracts, independent reference calculations, and adversarial tests. It intends no population/formula change for valid canonical inputs. Structural bugs such as index-based row multiplication may be fixed only with a failing hand oracle and a dedicated regression test. Any broader semantic mismatch stops the PR and is corrected in a separate, narrowly scoped table-fix PR.
2. Phase 2 lands a read-only audit engine, CLI, tests, documentation, and a sanitized report. It neither calls the table rebuild entry point nor changes `tables.py`. Automated findings and manual concerns are triaged before Phase 3.
3. Regression-integrity Phases 7-12 land first. They own coherent run storage, frozen cohorts/folds, valid estimators/inference, decision-grain analyses, and SWR power. If those APIs/manifests are missing, Phase 3's paper-impact result is `INCONCLUSIVE`; this audit does not fill the gap with legacy runners.
4. Phase 3 lands only table-specific run adapters, keyed generation comparison, claim specifications, and verdict/reporting gates. It validates an immutable noncanonical release candidate through the Phase-7 resolver; a table bug returns to a table-fix PR and new run ID.
5. Phase 13 owns the clean production rebuild/refit and may consume the Phase-3 table gate. Only authors may authorize promotion and paper/figure cutover. Sensitivity agreement alone cannot authorize promotion.

## Open Questions

1. **What is the correct stop exposure?** Current `immobility_time = stop_end - stop_start`, so an `n`-sample stop contributes `(n-1) * dt` and a single-sample stop contributes zero. Deferred: Phase 2 reports alternative exposure definitions and timestamp-gap prevalence; authors lock the estimand before any correction/refit.
2. **Should `is_mua` use the filtered survivor window or the full event-label window?** Current event rows use the filtered window while SWR classification uses the full label window. Deferred: Phase 1 pins current behavior; Phase 2 reports classification deltas under both windows; a scientific decision precedes any change.
3. **Is trunk-to-destination fallback an acceptable definition of current arm?** Current logic can turn trunk occupancy into a leaf arm. Deferred: quantify fallback frequency in Phase 2; if authors approve a sensitivity, register it before Phase 3 and run it through the corrected cohort/inference APIs.
4. **How should probabilities slightly above one be handled?** Current materialized data include `peak_nonlocal_prob` up to about 1.00039 and nine events whose weight sum exceeds their retained-bin count. Deferred: distinguish numerical state-probability drift from a calculation bug; never clip silently.
5. **What should an all-junction posterior mean?** Current arm marginalization can renormalize after excluding junction bins, while current materialized `n_junction_bins` is identically zero. Deferred: require a synthetic oracle, find or explain the lack of real examples, and record whether all-junction output is missing or uniform before changing it.
6. **Should timestamp gaps split an immobility bout?** Current bouts are contiguous by row mask, not by a maximum elapsed-time gap. Deferred: Phase 2 reports gaps and affected stops; any correction/sensitivity is registered before Phase 3 and rebuilt as a coherent generation.

## Estimated Effort

Expected implementation and test footprint is roughly 5,000-7,500 lines across three PRs: 1,500-2,200 lines for contracts/oracles/tests, 2,000-3,000 for the corpus audit/QC/reporting surface, and 1,500-2,300 for table-specific run adapters, generation comparison, claim specifications, tests, and verdict rendering. Phase-7-12 infrastructure is reused rather than duplicated. Generated candidates, refits, full failures, logs, and QC images remain outside git.
