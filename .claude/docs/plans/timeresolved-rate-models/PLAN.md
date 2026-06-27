# Time-Resolved Rate Models (Family B) Implementation Plan

**Status:** Not started.

Consolidate the scattered time-resolved per-bin event-rate analyses (currently five
scratch scripts that each re-implement the same loading, SWR-split, and binning logic)
into a clean, persisted **Family B** that mirrors the existing per-stop **Family A**
(`MODEL_RATE` → `rate_model.py` → `fit_rate_models.py`). The result is two parallel,
documented model families: **A** = simple, dwell-offset, literature-comparable
(Ambrose/Sosa); **B** = a **point-process GLM conditional-intensity ladder**
(Truccolo–Eden–Brown 2005), `MODEL_TIMERESOLVED` = M0_const … M9, toggling the four
blocks (time-since-stop spline, self-history, reward [lag-1/kernel/continuous rate],
switch) so each block's contribution is readable, plus the reward×time interaction —
all at one bin width so the ladder is nested and AIC/CV/GOF-comparable. Events are
**treated and checked as a point process** (time-rescaling KS, rescaled-ISI
independence, residuals, overdispersion — reusing `non_local_detector.model_checking`).
Fit across four populations (SWR, non-local, non-local-w-SWR, non-local-no-SWR; the
pooled non-local is the one-population null), the whole thing becomes reproducible from
a single tidy `models_timeresolved/all_models.pkl`, and the scratch scripts shrink to plotters.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — it is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md) — helper signatures, the spec shape, the fit-result dict, and the tidy-output schema that phases 2–4 share.
3. **Need broader scope / risks / decisions?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, metrics, decisions, risks.
- [shared-contracts.md](shared-contracts.md) — the 7 cross-phase contracts (helpers, spec/ladder, fitter, GOF, tidy schema, populations).
- Phases (each ships as a separable PR):
  - [phase-1-helpers.md](phase-1-helpers.md) — extract `attach_is_swr_to_onsets` + `aggregate_to_timebins` into `_helpers.py`; refactor the 5 scratch scripts to use them (behavior-preserving).
  - [phase-2-spec-fitter.md](phase-2-spec-fitter.md) — add `MODEL_TIMERESOLVED` (M0_const…M9 ladder) + `fit_timeresolved_model` / `cross_validate_timeresolved_model` (10-fold, bits/event) + `timeresolved_gof` primitives.
  - [phase-3-orchestration.md](phase-3-orchestration.md) — `fit_timeresolved_models.py` → tidy `models_timeresolved/all_models.pkl` (per-session × population × model + CV + GOF).
  - [phase-4-plotters-docs.md](phase-4-plotters-docs.md) — reduce scratch scripts to plotters reading the tidy output; `docs/rate_models.md` describing the two families.
  - [phase-5-model-checking.md](phase-5-model-checking.md) — point-process GOF sweep + diagnostic plots (KS/QQ/ISI-autocorr/residuals) across the ladder, reusing `non_local_detector.model_checking`.
