# Phase 1 — Duration-controlled replay time course (cheap de-confound)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Cheapest step, reuses Family B: test whether the within-stop replay rate ramp is
a genuine time-since-stop effect or **stop-duration composition** (long ≈
rewarded stops dominate large `tc`). No new estimator.

**Questions:** Is the post-stop replay ramp real within constant-duration stops?
Do long-*destined* stops already show higher *early* replay (a first hint of
replay ↔ duration coupling)?

**Inputs to read first:**

- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:58` `fit_per_bin_rate_model` — the replay fitter to reuse.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:880` `aggregate_to_timebins` (`stop_covariates` param) — carry `immobility_time` (duration) per stop.
- `docs/per_bin_rate_results.md` — the current (composition-confounded) time course + the survival overlay already added to `per_bin_rate_ladder_summary.png`.

## Tasks

- Scratch runner `scripts/_duration_controlled_timecourse.py`: per session, label
  each stop's **duration tertile** (short/med/long, from `immobility_time`), and
  for each population (`swr`, `non_local_swr`, `non_local_no_swr`):
  - fit the M5_time / M7 replay model **within each tertile** and reconstruct the
    fitted `λ(tc)` (fixed-knot spline → reconstructable), restricted to the
    `tc` range the *shorter* strata still reach (high survival), so strata are
    compared over a shared, fully-supported window.
  - additionally fit M7 with a `duration_tertile × bs(tc)` interaction to test
    whether the time-course *shape* differs by destined duration.
- Figure `paper/figures/duration_controlled_timecourse.png`: fitted `λ(tc)` by
  duration tertile, per population, with the shared high-survival window marked.
- Report: median time-since-stop slope (early window) within each tertile, per
  population, per-animal sign; whether long-destined stops have higher early rate.

## Gate (decide whether to continue)

- If the ramp **dissolves** within tertiles (slope ≈ 0 once duration is held) →
  the time course is mostly composition; document in `per_bin_rate_results.md`
  and **stop** (no departure/joint model needed for the time-course question).
- If the ramp **survives** and/or long-destined stops show higher early replay →
  a duration process matters → proceed to Phase 2.

## Deliberately not in this phase
- No departure-hazard model (Phase 2); no joint model.
- No conditioning on the *full* duration to predict late-`tc` replay (collider
  risk) — compare only the *early*, fully-supported window across strata.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_duration_timecourse::test_tertile_labels` | Duration tertiles partition stops per session into ~equal thirds by `immobility_time`; boundaries monotone. |
| `test_duration_timecourse::test_shared_window` | The compared `tc` window is bounded by the shortest tertile's coverage (no extrapolation past a stratum's support). |
| runner smoke (slow) | Runs on `--max-per-animal 1`; emits the figure + per-tertile slope table. |

## Fixtures
- Synthetic: stops with known durations spanning the tertiles + planted onset
  rate with/without a true `tc` ramp (to confirm the test detects ramp survival
  vs dissolution). Real-data smoke via one session.

## Review
Dispatch `code-reviewer`. Confirm: comparison restricted to the shared high-
survival window (no collider conditioning on full duration); fitted curves use
the fixed-knot spline; gate result reported, not asserted; no departure/joint
code added; names free of plan references.
