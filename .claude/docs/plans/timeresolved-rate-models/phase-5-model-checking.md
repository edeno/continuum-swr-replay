# Phase 5 — Point-process model checking across the ladder

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Treat the events as a point process and check them as one. The per-fit GOF scalars
(`ks_statistic`, `ks_within_band`, `max_isi_autocorr`, `dispersion`) are already
emitted into the tidy file by Phase 3 (via the `timeresolved_gof` contract). This
phase *reads* them to tell the model-checking story across the ladder, and renders
the diagnostic plots for a few example sessions.

**Inputs to read first:**

- [shared-contracts.md#timeresolved_gof](shared-contracts.md#timeresolved_gof) — the GOF contract this phase consumes; reuses `non_local_detector.model_checking.sorted_spikes` (`TimeRescaling`, `point_process_residuals`, `plot_ks`, `plot_qq`, `plot_rescaled_ISI_autocorrelation`). **Verified present** at `non_local_detector/model_checking/sorted_spikes.py`.
- [shared-contracts.md#tidy-output-schema](shared-contracts.md#tidy-output-schema) — the GOF + `cv_bits_per_event` columns read here.
- [phase-3-orchestration.md](phase-3-orchestration.md) — produced `models_timeresolved/all_models.pkl`; confirm it carries the GOF columns and that the fitter returns `predicted` (per-bin conditional intensity).

## Tasks

- Write `scripts/plot_model_checking.py` reading `models_timeresolved/all_models.pkl`:
  - **GOF-up-the-ladder table/figure** per population: `ks_statistic`, fraction of sessions `ks_within_band`, median `dispersion`, and `cv_bits_per_event`, for M0_const → M9. The narrative to surface: `M0_const` fails the KS band (non-Poisson: refractory+bursty), the self-history block (`M3_hist`/`M6_time_hist`/`M7`) walks KS into the band and pulls `dispersion` → ~1, and `cv_bits_per_event` increases where a block adds real signal.
  - **Per-event-type comparison**: is the SWR-coupled train better fit (higher bits/event, KS in band at a simpler model) than NL-no-SWR? This is the model-checking face of the dissociation.
- For 2–3 example sessions (one rich e.g. `chimi 20200216 06_r3`, one sparse e.g. `wilbur`), render the diagnostic quartet per population at the full model: `TimeRescaling.plot_ks`, `.plot_qq`, `.plot_rescaled_ISI_autocorrelation`, and `point_process_residuals` cumulative-residual vs time-since-stop and vs trial number (the residual-vs-covariate check that reveals a missing block — e.g. residual drift vs trial number ⇒ the deferred session-drift / state-space term). Re-fit those sessions inline to get `predicted` (cheap) rather than persisting per-bin intensities for all sessions.
- Extend `docs/rate_models.md` (from Phase 4) with a **"Model checking"** section: the time-rescaling logic, what the constant-model KS failure means, the dispersion/bits-per-event read, and the explicit statement that the residual-vs-trial-number drift (if present) is the empirical motivation for the state-space model (links to the SS-GLM spec).

## Deliberately not in this phase

- No new model specs (the ladder is fixed in Phase 2).
- No persisting per-bin conditional intensities for all 218 sessions × 4 populations × 11 models (storage blow-up) — recompute `predicted` inline for the few plotted example sessions; the *scalar* GOF for every fit already lives in the tidy file.
- No state-space model — the residual drift only *motivates* it (separate spec).

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run python scripts/plot_model_checking.py` | Regenerates the GOF-up-the-ladder figure + the example-session diagnostic quartets; runs off the tidy file (+ inline refit for examples), no manual steps. |
| GOF sanity (assertion in the script or a test) | `M0_const` median `ks_within_band` fraction < the full model's; `dispersion` decreases monotonically-ish as the history block is added; `cv_bits_per_event` ≥ 0 for every model vs `M0_const`. |
| Reuse check (grep) | No re-implementation of time-rescaling / KS / ISI-autocorrelation in this repo — all GOF math imported from `non_local_detector.model_checking.sorted_spikes`. |

## Fixtures

- Real-data: `models_timeresolved/all_models.pkl` (Phase 3) + the canonical Part B tables for the inline example-session refits. No synthetic fixture — the GOF *is* the check.

## Review

Before opening the PR, dispatch `code-reviewer` against the diff. Confirm:
- GOF math is imported, not re-implemented; `adjust_for_short_trials=True` is used (the short-stop censoring correction).
- The KS-fails-for-constant / KS-in-band-for-full narrative is actually borne out by the numbers, not asserted — if the full model does *not* reach the band, say so (it's a real finding about remaining misfit, e.g. the across-session structure the SS-GLM targets).
- Docstrings/names free of plan references (no "Phase 5").
