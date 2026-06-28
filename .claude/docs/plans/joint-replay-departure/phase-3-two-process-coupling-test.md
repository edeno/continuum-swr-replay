# Phase 3 — Two-process independent fit + coupling test (the key gate)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Fit the replay intensity (Family B) and the departure hazard (Phase 2) on a
**shared covariate set**, independently, and test whether they are **coupled
beyond shared covariates** — i.e. whether the full joint/frailty model (Phase 4)
is needed at all. This is the gate that prevents fitting the complex model for
nothing.

**Questions:** Are replay and dwell independent given reward, or co-driven? Does
a **replay burst precede departure** (replay → leaving)? Does the stop's
*destined* duration predict its *early* replay (replay ↔ duration)? Do SWR-
coupled vs non-SWR events relate to departure differently?

**Inputs to read first:**

- [shared-contracts.md#departure-hazard-design](shared-contracts.md#departure-hazard-design) and `src/.../departure_hazard.py` (Phase 2).
- `src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py` `fit_per_bin_rate_model` (replay half).

## Tasks

- Scratch runner `scripts/_replay_departure_coupling.py`: per session, build the
  shared `(stop × bin)` design once; fit (a) the replay intensity (M7-style) and
  (b) the departure hazard (D2) with the **same** covariates. Then the
  **residual-coupling tests** (the gate evidence):
  - **replay → departure:** add lagged replay-onset counts `onset_lag{j}` (and a
    raised-cosine summary) to the departure hazard; is the recent-replay
    coefficient ≠ 0 (does replay in the last ~150 ms raise/lower the hazard of
    leaving)? Per population (does SWR-coupled replay precede leaving more than
    non-SWR?).
  - **duration → replay:** per stop, correlate the replay-rate residual (obs −
    expected from covariates, *early* window only) with the stop's total
    duration residual (obs − expected from the hazard covariates). A positive
    partial correlation = replay-rich stops are longer beyond shared covariates.
  - report both with per-animal n/5 and a signed-rank / partial-correlation p.
- Figure `paper/figures/replay_departure_coupling.png`: the replay→hazard kernel
  by population + the duration↔replay residual scatter/summary.

## Gate (the decisive one)
- If **neither** coupling test is significant (replay history doesn't predict
  the hazard; duration and replay residuals are uncorrelated) → replay and
  departure are **conditionally independent given reward**; the independent
  Phase-2/Family-B models are the reportable result → **do not build Phase 4**.
- If **either** is significant → there is structure a shared latent would
  capture → proceed to Phase 4 (and Phase 3 already says *which* direction the
  coupling runs).

## Deliberately not in this phase
- No frailty / joint estimation (Phase 4) — here the two models are fit
  separately; this phase only *measures* whether coupling exists.
- No replay direction; no change to shipped models / `all_models.pkl`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_coupling::test_replay_in_hazard_columns` | The hazard design gains `onset_lag{j}` columns = within-stop lagged replay counts (shift within `stop_id`, 0 at first bin). |
| `test_coupling::test_independence_under_null` | On simulated *independent* replay + departure (no shared driver), the coupling tests are null (replay-history hazard coef ≈ 0; residual corr ≈ 0). |
| `test_coupling::test_detects_coupling` | On simulated data with a shared per-stop multiplier (replay↑ & duration↑), both tests are positive. |
| runner smoke (slow) | Runs on `--max-per-animal 1`; emits the coupling figure + test table. |

## Fixtures
- Synthetic: paired replay/departure processes generated under (i) independence
  and (ii) a shared frailty, so the gate's null and detection are both tested.
  Real-data smoke via one session.

## Review
Dispatch `code-reviewer`. Confirm: the coupling tests genuinely test residual
(beyond-shared-covariate) dependence, not the shared covariates themselves; the
duration↔replay test uses the *early* window (no collider on full duration); the
gate verdict is reported; no frailty code; names free of plan references.
