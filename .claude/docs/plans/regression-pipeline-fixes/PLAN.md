# Regression Pipeline Fixes Implementation Plan

**Status:** Phases 1, 2a, 2b, 3, 4, 5a complete and merged to master (2026-09-06); phase 5b in progress. Follow-up parked: add `M6ra_time_arrival_histcos` so the arrival ladder has a reward-given-time+history step like the stop-clock ladder. Also parked: `models_per_bin_rate/swr_power_model.pkl` (2026-09-02) is stale against the regenerated tables (a refit moves all seven coefficient columns, up to 0.73) — refit once its runner has argparse. The `swr_power_*_audit.csv` sidecars were rewritten on 2026-09-06 by an accidental pre-argparse run and no longer match the restored pickle — regenerate both together at that refit. Also parked: unify `attach_standardized` to `ddof=0` and drop the `ddof` knob at the (already required) uncertainty-rate refit (≈5e-6 relative change to its per-SD RRs on the full cohort); investigate the separation detector marking rows with `coef_hist_rcos_0 ≈ −100` as `separated=False`; persist the per-bin runner's `none_counts` in provenance so rung coverage (M5a/M7ra fit 213 of 218 sessions) can be asserted against recorded failures. Further follow-ups from the phase ledgers: `require_fresh_inputs` does not propagate staleness through aggregate-only artifacts (the `check_fit_provenance.py` sweep does); `qvals_table` still lacks natural session keys (`stop_id` only), so `assert_qvals_aligned` scopes sessions by a padded clock span; the 21 content bundles are provenance-stale on the additive `stop_table` hash only (parity-verified, nominal); the DM result doc tabulates 14 of 32 pooled rungs (full set in the audited pickle); `covariate_registry.py` (908 lines) reaches < 400 only via a `qvals_ingestion.py` split; `prepare_data.py` still calls `logging.basicConfig` at import; `COVARIATE_SCHEMA_VERSION` was not bumped for the phase-1 column-set change; `docs/design/2026-06-23-replay-trajectory-metrics-design.md:235` names the deleted `tables.py`.

Fixes the inference-changing defects found in the 2026-09-03 regression pipeline review
(`.claude/docs/reviews/2026-09-03-regression-pipeline-review.md`), refits every ladder on the
current canonical tables, adds a second content estimator (Dirichlet-multinomial) and compares it
with the shipped one, makes Family B's clock and per-bin reward covariates honest about stop
fragmentation, ties fitted artifacts to the tables they were fit on, rewrites the docs and claims
table from the refits, and then removes the readability debt that makes the pipeline hard to read.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals/non-goals, metrics, risks, rollout, open questions.
- [shared-contracts.md](shared-contracts.md) — common-cohort freeze for rate designs, fraction response, fit-provenance sidecar, estimator result schema.
- [designs.md](designs.md) — event-clustered sandwich covariance, Dirichlet-multinomial fitter, arrival clock and per-bin reward lags in the per-bin design.
- Phases (each ships as a separable PR):
  - [phase-1-rate-runners.md](phase-1-rate-runners.md) — common cohort in both rate families, effect-scale separation in the SWR-rate runner, output-dir keying, aggregation filter, onset-bin edges, in-transit gating; refit Family A/B and SWR-rate.
  - [phase-2a-content-fractions.md](phase-2a-content-fractions.md) — per-event fraction response as the shipped content estimator, event-clustered SEs, event-length control, outcome strata, leaf-only value chain; refit all content ladders.
  - [phase-2b-dirichlet-multinomial.md](phase-2b-dirichlet-multinomial.md) — Dirichlet-multinomial content fitter with an SWR concentration term, simulation validation, three-estimator comparison report.
  - [phase-3-timing-provenance.md](phase-3-timing-provenance.md) — arrival/outcome clocks persisted on stops, arrival-anchored Family-B variant, per-bin reward lags, qvals alignment guard, fit provenance sidecars, production omission clock in fixtures.
  - [phase-4-docs-claims.md](phase-4-docs-claims.md) — every doc and the claims table rewritten from the refits; stale paths and retired names removed.
  - [phase-5a-readability-hygiene.md](phase-5a-readability-hygiene.md) — narrative out of `model_specs`, shims and dead code deleted, one standardization path, version check wired.
  - [phase-5b-readability-structure.md](phase-5b-readability-structure.md) — `_helpers` split, design-constant module, registry ledger trimmed, session keys and `is_stay` unified, CLIs and validators relocated; byte-parity gated.
