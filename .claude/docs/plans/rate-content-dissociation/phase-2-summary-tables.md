# Phase 2 - Canonical rate/content summary tables

[<- back to PLAN.md](PLAN.md) . [overview](overview.md) . [shared contracts](shared-contracts.md)

Build the shared coefficient table that every later phase uses. This removes the
current pattern of manually opening pickles and reading whichever `coef_*`
columns happen to be present.

**Questions:** What are the audited rate, content, and choice-effect estimates
on their own scales? Which key terms are available per session/population/model?

**Inputs to read first:**

- [shared-contracts.md](shared-contracts.md) - schema, estimand labels, key terms.
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:79` - rate
  output fields and `rr_*` convention.
- `scripts/rate/fit_swr_contrast_rate.py:124` - SWR-contrast rate output shape.
- `src/continuum_swr_replay/analysis/replay/fitting.py:193` - conditional
  content output shape.
- Phase 1 audit outputs.

## Tasks

- Add `src/continuum_swr_replay/analysis/replay/model_summaries.py`.
  - Implement coefficient-column extraction for `coef_*`, `bse_*`, `pval_*`,
    and sample-size fields.
  - Add effect transformations and `effect_scale` labels.
  - Add stale/provenance columns from Phase 1 audit results.
  - Join fit-availability rows so missing key coefficients have explicit
    `fit_status` and `skip_reason`.
- Add `scripts/qa/build_rate_content_summary_tables.py`.
  - Read audited rate artifacts, SWR-contrast rate artifacts, primary content
    artifacts, and SWR-contrast content artifacts.
  - Emit:
    - `Processed-Data/replay_content/model_summaries/rate_content_terms.pkl`;
    - `Processed-Data/replay_content/model_summaries/rate_content_terms.csv`;
    - `Processed-Data/replay_content/model_summaries/rate_content_availability.csv`;
    - `Processed-Data/replay_content/model_summaries/key_term_report.md`.
- Define a small key-term registry for the first pass:
  - rate: immediate reward, delayed reward-history kernel, SWR interactions;
  - content: current/current-patch, reward-rate/value, task-state current-bias
    interactions, SWR content interactions;
  - choice placeholders for Phase 4.
- Add summary-level validation helpers:
  - no duplicated `(family, artifact_path, model, population, session, term)`;
  - every non-stale key term has `effect_scale`;
  - no content row enters summary from a stale Phase 1 artifact unless
    `stale_flag=True`.
  - every unavailable key term has a matching availability row with
    `fit_status != "fit"` and a documented `skip_reason`.

## Gate

Proceed to Phase 3 only when:

- the summary table is deterministic across two local runs;
- all required key terms are present or explicitly marked unavailable;
- stale artifacts remain visible but excluded from headline report rows;
- missing key coefficients are explained by availability rows, not silent
  absence;
- downstream code can reproduce the coefficient readout from this table alone.

## Deliberately not in this phase

- No new inference beyond summary aggregation.
- No new model fits.
- No figures except a diagnostic key-term availability report.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_summarize_coefficients_extracts_terms` | Converts `coef_x`, `bse_x`, `pval_x` into long rows and `effect=exp(coef)`. |
| `test_effect_scale_labels` | Rate/content/choice families receive the correct effect-scale label. |
| `test_summary_deduplicates_keys` | Duplicate term/session rows are rejected. |
| `test_missing_key_terms_require_availability` | A missing headline coefficient without a skip row fails summary validation. |
| real smoke | Builder runs on existing audited artifacts and emits the expected terms, availability, and report files. |

## Fixtures

- Minimal rate, content, and choice-style coefficient frames.
- Phase 1 audit fixture with one stale and one current artifact.

## Review

Confirm that later phases no longer hand-open model pickles for coefficients.
The summary table is allowed to expose stale rows, but headline filters must
exclude them by default.
