# Time-Resolved Rate Models (Family B) Implementation Plan

**Status:** Not started.

Consolidate the scattered time-resolved per-bin event-rate analyses (currently five
scratch scripts that each re-implement the same loading, SWR-split, and binning logic)
into a clean, persisted **Family B** that mirrors the existing per-stop **Family A**
(`MODEL_RATE` → `rate_model.py` → `fit_rate_models.py`). The result is two parallel,
documented model families: **A** = simple, dwell-offset, literature-comparable
(Ambrose/Sosa); **B** = per-bin, spline + reward-kernel + self-history, confound-free.
The headline reward dissociation (non-SWR non-local events are not consummatory/
reward-gated; SWR-coupled events are) becomes reproducible from a single tidy
`models_timeresolved/all_models.pkl`, and the scratch scripts shrink to plotters.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file — it is self-contained.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md) — helper signatures, the spec shape, the fit-result dict, and the tidy-output schema that phases 2–4 share.
3. **Need broader scope / risks / decisions?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — scope, integration points, metrics, decisions, risks.
- [shared-contracts.md](shared-contracts.md) — the 6 cross-phase contracts (helpers, spec, fitter, tidy schema, populations).
- Phases (each ships as a separable PR):
  - [phase-1-helpers.md](phase-1-helpers.md) — extract `attach_is_swr_to_onsets` + `aggregate_to_timebins` into `_helpers.py`; refactor the 5 scratch scripts to use them (behavior-preserving).
  - [phase-2-spec-fitter.md](phase-2-spec-fitter.md) — add `MODEL_TIMERESOLVED` (T0/T1/Tlag/Thist) + `fit_timeresolved_model` / `cross_validate_timeresolved_model` primitives.
  - [phase-3-orchestration.md](phase-3-orchestration.md) — `fit_timeresolved_models.py` → tidy `models_timeresolved/all_models.pkl` (per-session × population × model + CV).
  - [phase-4-plotters-docs.md](phase-4-plotters-docs.md) — reduce scratch scripts to plotters reading the tidy output; `docs/rate_models.md` describing the two families.
