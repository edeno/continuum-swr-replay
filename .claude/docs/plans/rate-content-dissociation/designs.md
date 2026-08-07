# Designs

> **SUPERSEDED — DO NOT EXECUTE (2026-07-29).** This file is retained only as
> historical design context. Canonical corrective work is owned by
> [`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md); the only
> surviving downstream rate/content task is
> [`rate-content-followups` Phase 2](../rate-content-followups/phase-2-swr-continuum.md).

[<- back to PLAN.md](PLAN.md)

These are implementation sketches, not final APIs. Phase files decide the exact
file names and tests.

## artifact-audit-design

Inputs:

- current specs from `model_specs.py`;
- primary content artifacts under `Processed-Data/replay_content/models*/`;
- SWR-contrast content artifacts under
  `Processed-Data/replay_content/models/swr_contrast_content_*.pkl`;
- existing rate artifacts for comparison/provenance only.

Algorithm:

1. Load every target artifact into a DataFrame.
2. For each `(model, population)` row group, resolve expected predictors from
   the current spec.
3. Assert every fitted row either has `coef_<term>` for expected terms or has a
   recorded skip/unfit reason.
4. Flag deprecated coefficient names, especially old reward-rate aliases.
5. Load the companion fit-availability table and verify every missing
   key coefficient has an explained `(model, population, session)` availability
   row.
6. For any regenerated content design table sampled during audit, assert the
   retained rows per event equal 9 for arms and 3 for patches.
7. Emit `content_artifact_audit.csv`, `content_fit_availability.csv`, and a
   concise markdown report.

Hard failure conditions:

- deprecated coefficient aliases in a claimed-current artifact;
- missing current key terms with no skip reason;
- evidence of row-wise category drops in content/choice candidate sets;
- missing provenance or availability tables after Phase 1.

## summary-builder-design

Proposed module:

- `src/continuum_swr_replay/analysis/replay/model_summaries.py`

Proposed runner:

- `scripts/qa/build_rate_content_summary_tables.py`

Core functions:

```python
def summarize_coefficients(
    artifact: pd.DataFrame,
    *,
    family: str,
    artifact_path: Path,
    key_terms: set[str],
    complete_case_policy: str,
) -> pd.DataFrame:
    """Return rows matching the shared model-summary schema."""
```

Rules:

- Extract all `coef_*` columns and matching `bse_*`/`pval_*` columns when
  present.
- Preserve source sample-size columns (`n_events`, `n_bins`, `n_rows`,
  `n_stops`) without inventing missing values.
- Transform effects with `np.exp(coef)` and label the correct `effect_scale`.
- Add one deterministic `artifact_hash` per source file.
- Mark stale rows rather than dropping them; downstream headline scripts filter
  to `stale_flag == False`.
- Join fit-availability rows so unavailable key terms are visible in the same
  report surface as successful coefficients.

## rate-content-dissociation-design

Primary tests are session-paired where possible:

1. **Immediate reward rate gate:** compare `reward_lag1` or
   `is_swr:reward_lag1` in SWR-overlap vs non-SWR streams.
2. **Delayed reward history:** summarize `reward_lag2`-`reward_lag10` as a
   signed kernel per session and population.
3. **Content locality:** compare `is_current`/`is_current_patch` coefficients
   across SWR and non-SWR splits, plus pooled SWR interaction terms.
4. **Task-state content:** summarize key HMM/task terms (`rpe_x_current`,
   `switchrate_x_current`, `q_arm`) and their SWR contrasts.
5. **Joint pattern:** classify each session/covariate as rate-only,
   content-only, both, or neither based on signed, audited key terms.

Default inference:

- within-session paired deltas when two terms share a session;
- Wilcoxon signed-rank across sessions for headline paired effects;
- per-animal sign consistency as a robustness summary;
- bootstrap confidence intervals by animal for final figures.

Support classification:

- Use [shared-contracts.md](shared-contracts.md#support-classification-rule) for
  `supported`, `supported_secondary`, `not_supported`, `underpowered`,
  `blocked_stale`, and the derived `rate_only`/`content_only`/`both` labels.
- Do not collapse `underpowered` into `not_supported`; these have different
  scientific meanings.
- Correct headline p-values within each pre-registered family of terms before
  assigning support labels.

## continuous-swr-design

Question: does a binary `is_swr` label capture the effect, or does content/rate
scale continuously with ripple overlap or ripple power?

Implementation steps:

1. Audit existing event tables for a usable continuous mark, such as
   `ripple_power_z`, overlap duration, or overlap fraction.
2. If absent, derive event-level overlap fraction from event intervals and SWR
   intervals without changing event detection.
3. Add continuous-mark versions of selected terms:
   - rate: mark main effect and `mark:reward_lag1`;
   - content: `mark_x_current`, `mark_x_current_patch`, and selected value/task
     interactions only if identified.
4. Compare against binary `is_swr` using held-out likelihood/AIC and stability
   across animals.
5. If a continuous-mark fit is promoted beyond sensitivity analysis, write it as
   a normal audited artifact with provenance, fit-availability rows, summary
   rows, stale flags, and tests under the same Phase 1/2 contracts.

Gate:

- If continuous marks do not improve fit or change interpretation, use binary
  `is_swr` in final figures and document the sensitivity.
- If continuous marks improve fit and alter the conclusion, promote them to a
  primary figure panel and keep binary `is_swr` as a simplified descriptor.
  Promotion requires the same audit/provenance/summary gates as binary-SWR
  artifacts.

## choice-feature-design

Build stop-by-arm features from audited content artifacts and canonical event
tables:

- `replay_frac`: mean arm fraction over events in the stop;
- `replay_frac_swr` and `replay_frac_non_swr`: split by event SWR overlap;
- `replay_frac_residual`: residual content after a content model has predicted
  the expected arm fraction from spatial/value/task covariates;
- optional rate summaries per stop: event count, SWR count, non-SWR count, and
  rate-model expected count if needed for mediation diagnostics.

Avoid leakage:

- never use `next_arm`, `is_next`, or post-choice reward as predictors;
- keep the candidate set to well arms minus current arm;
- drop whole stops when any required candidate-row predictor is missing;
- report the behavioral baseline before any replay coefficient.
- For every nested choice comparison, refit the baseline and richer model on the
  same retained-stop set. Report the retained-stop cohort and cohort shrinkage
  before interpreting incremental likelihood.

## claim-ledger-design

Final docs/figures should be accompanied by a small table:

| claim_id | claim | source_table | source_terms | artifact_hashes | status |
| --- | --- | --- | --- | --- | --- |

Status values:

- `supported`: audited artifact, pre-registered term, validation passed;
- `supported_secondary`: audited but exploratory/sparse;
- `not_supported`: tested and did not pass;
- `blocked_stale`: artifact failed audit;
- `blocked_missing`: required source term unavailable.
