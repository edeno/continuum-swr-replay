# Phase 3 — SWR / non-SWR continuum test (cross-type ensemble history)

> **SUPERSEDED — DO NOT EXECUTE (2026-07-29).** This phase is retained only as
> historical design context. Covariate correction and the coherent refit are
> owned by [`regression-integrity-fixes`](../regression-integrity-fixes/PLAN.md);
> the only surviving downstream continuum task is
> [`rate-content-followups` Phase 2](../rate-content-followups/phase-2-swr-continuum.md).

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Follow-up motivated by the self-history result in
[per_bin_rate_results.md](../../../../docs/per_bin_rate_results.md) §4: analyzed
*separately*, non-SWR non-local events look less bursty and more refractory
(burst 1.54× @135 ms, refractory ≤150 ms) than SWR-coupled events (2.1–2.4×,
≤60–75 ms). But the per-bin self-history is computed **within** each population,
so it cannot see a non-SWR event that follows an *SWR-coupled* event. **Question:
are non-SWR non-local events a genuinely separate, less-bursty mode, or part of
one bursty continuum that we artificially split by SWR-status?** The pooled
`non_local` kernel (2.04× @105 ms, ≤60 ms) is suggestive of a continuum but is
dominated by the SWR-coupled events — it does not show that the *non-SWR* events
themselves ride the burst. This phase tests that directly. Rate-only.

**Inputs to read first:**

- [docs/per_bin_rate_results.md](../../../../docs/per_bin_rate_results.md) §4 — the within-type self-history result this phase interrogates.
- `src/continuum_swr_replay/analysis/replay/_helpers.py:880` `aggregate_to_timebins` — builds per-bin `count` + within-population `onset_lag{j}` for ONE population; this phase needs lagged counts of a *second* population on the same bins.
- `src/.../_helpers.py` `attach_is_swr_to_onsets`, `POPULATIONS` — to split onsets into SWR-coupled vs non-SWR non-local.

## Tasks

- **Cross-correlogram (descriptive, assumption-light)** in a scratch script
  `scripts/_swr_nonswr_continuum.py`: within each stop, histogram the time of
  each non-SWR non-local onset relative to each SWR-coupled non-local onset
  (only same-`stop_id` pairs), ±300 ms at 15 ms bins, pooled across sessions
  (normalize per session then average, so long sessions don't dominate). A clear
  peak at small |lag| means non-SWR events cluster around SWR events →
  continuum. Also plot the two within-type auto-correlograms (NL-no-SWR↔NL-no-SWR,
  NL-w/SWR↔NL-w/SWR) on the same axes for comparison. Figure
  `paper/figures/swr_nonswr_crosscorr.png`.

- **Cross-type ensemble-history GLM (controlled)**: extend the (stop × bin)
  design so the *non-SWR* rate is modelled with **two** history kernels —
  lagged counts of (a) prior **non-SWR** onsets and (b) prior **SWR-coupled**
  onsets — plus the usual `bs(tc)` + `reward_lag1`. Build the cross kernel by
  computing per-bin SWR-coupled onset counts on the same bins as the non-SWR
  design and shifting within `stop_id` (a small helper, or extend
  `aggregate_to_timebins` to accept an optional second onset series for extra
  `xhist_lag{j}` columns — default off, so the shipped path is unchanged). Fit
  per session via `fit_per_bin_rate_model` with an ad-hoc spec carrying the
  cross-history columns as `extra_covariates` (raised-cosine basis preferred to
  avoid the refractory separation, per §6). Report both kernels' RR-by-lag,
  per-animal sign, and a paired comparison.
  - **Continuum verdict:** if the **SWR→non-SWR** cross-kernel is positive at
    ~100 ms (non-SWR events elevated after a recent SWR-coupled event), the
    events are one bursty process and "non-SWR is less bursty" is a splitting
    artifact. If only the within-non-SWR kernel matters (and stays weak), they
    are a genuinely separate, less-bursty mode.
  - Symmetric check: non-SWR→SWR cross-kernel for the SWR-coupled rate.

- **Document** the verdict in `per_bin_rate_results.md` (a "continuum" note in
  §4) — whichever way it resolves, with the CCG + cross-history numbers.

## Deliberately not in this phase

- No replay direction; no change to the shipped `MODEL_PER_BIN_RATE` ladder or
  `all_models.pkl` (ad-hoc spec in the scratch runner, like Phase 2).
- No spike/cell-ensemble cross-history — this is event-onset cross-history only.

## Validation slice

| Test | Asserts |
| --- | --- |
| `test_continuum::test_crosscorr_synthetic` | Two synthetic within-stop point processes coupled at +90 ms → CCG shows a peak at ~+90 ms; independent processes → flat CCG within noise. |
| `test_continuum::test_cross_history_columns` | The cross-history design has `xhist_lag{j}` = prior counts of the *second* population, shifted within stop, 0 at each stop's first bin; within-pop `onset_lag` unchanged. |
| `scripts/_swr_nonswr_continuum.py` (slow) | Runs end-to-end; emits the CCG figure + the cross-history kernel table. |

## Fixtures

- **Synthetic**: two coupled within-stop onset series with a known lag offset
  (drives the CCG + cross-history unit tests).
- **Real-data**: existing `event_onset_table` / `event_table` (the slow run).

## Review

Before opening the PR, dispatch `code-reviewer`. Confirm: same-stop-only pairing
in the CCG (no cross-stop contamination); cross-history shift is within `stop_id`
with no leakage; the shipped ladder / `all_models.pkl` untouched; the continuum
verdict is *reported*, not asserted to a magnitude; docstrings/names free of plan
references.
