# Phase 3 - Joint rate/content dissociation tests

[<- back to PLAN.md](PLAN.md) . [overview](overview.md) . [designs](designs.md)

Use the Phase 2 summary table to test the central scientific interpretation:
reward and task state may change replay **rate**, replay **content**, or both,
and SWR overlap may mark a distinct sub-process.

**Questions:** Which effects are rate-only, content-only, or shared? Is the SWR
effect binary, or does it scale with ripple overlap/power? Do the audited
content coefficients support the same qualitative story as the rate
coefficients?

**Inputs to read first:**

- Phase 2 summary outputs.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:791` - rate
  SWR-contrast terms.
- `src/continuum_swr_replay/analysis/replay/model_specs.py:867` - content
  SWR-contrast terms.
- `scripts/rate/fit_swr_contrast_rate.py:156` - stacked SWR/non-SWR rate
  contrast construction.
- `scripts/content/fit_swr_contrast_content.py:10` - why event-level scalars
  enter content only through interactions with arm/patch features.

## Tasks

- Add `scripts/analysis/test_rate_content_dissociation.py` or equivalent.
  - Load `rate_content_terms.pkl`.
  - Filter to `stale_flag == False`.
  - Build paired session tables for rate and content key terms.
- Implement headline tests:
  - immediate reward rate gate: `reward_lag1` in `non_local_swr` vs
    `non_local_no_swr`, plus stacked `is_swr:reward_lag1`;
  - delayed reward-history suppression: signed kernel over `reward_lag2` to
    `reward_lag10` by population;
  - content locality: `is_current`, `is_current_patch`, `swr_x_current`,
    `swr_x_current_patch`;
  - content task-state modulation: `rpe_x_current`, `switchrate_x_current`,
    `q_arm`, and their SWR contrasts where available;
  - rate/content classification per covariate: rate-only, content-only, both,
    neither.
- Add continuous-SWR sensitivity:
  - audit event/ripple tables for continuous overlap/power marks;
  - fit a minimal selected set of continuous-mark rate/content models only if
    the mark is available and identified;
  - compare against binary `is_swr` with AIC or held-out likelihood when
    possible.
- Emit:
  - `Processed-Data/replay_content/model_summaries/rate_content_dissociation.csv`;
  - `Processed-Data/replay_content/model_summaries/rate_content_dissociation.md`;
  - optional continuous-SWR sensitivity table.

## Gate

Proceed to Phase 4 when the dissociation report identifies the supported
rate/content claims and the SWR representation to carry forward:

- **Binary sufficient:** use existing `is_swr` split/interaction terms in Phase
  4 and final figures.
- **Continuous mark needed:** add the continuous mark to the Phase 4 feature set
  and document binary `is_swr` as a simplification.
- **Content unsupported/stale:** Phase 4 may still test rate or raw replay
  features, but final content claims remain blocked.

## Deliberately not in this phase

- No behavior prediction.
- No causal mediation language.
- No paper-layout work.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_pair_terms_by_session` | SWR/non-SWR paired deltas use only shared sessions. |
| `test_classify_rate_content_effects` | Synthetic signed terms are classified as rate-only/content-only/both/neither. |
| `test_stale_rows_excluded` | Headline tests reject rows with `stale_flag=True`. |
| real smoke | Runner produces the dissociation report from Phase 2 summary outputs. |

## Fixtures

- Synthetic summary table with planted rate-only and content-only effects.
- Optional tiny continuous-SWR mark table if a reusable mark builder is added.

## Review

Confirm that the report states estimand scales clearly. A content allocation
ratio is not interpreted as an event-rate change, and a rate ratio is not
interpreted as represented-arm content.
