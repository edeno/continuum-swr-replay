# Phase 2 — Uniform fit diagnostics (convergence, separation, scoped warnings)

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

Make convergence and separation first-class, checked outputs across all fitters, and stop blanket
warning suppression from hiding them. Rank guards already exist (`within_group_full_rank`,
`per_stop_rate_model.py:113`) — do **not** re-add those; this phase closes the *convergence* and
*separation* gaps only. No estimand change; fits that were silently non-converged now surface a flag.

**Inputs to read first:**

- [src/continuum_swr_replay/analysis/replay/learning.py:475-482](../../../../src/continuum_swr_replay/analysis/replay/learning.py) — the one existing convergence check to copy: `sm.Logit(...).fit(disp=False, maxiter=200)` then `model.mle_retvals.get("converged", False)`.
- [src/continuum_swr_replay/analysis/replay/fitting.py:150-170](../../../../src/continuum_swr_replay/analysis/replay/fitting.py) and [:370-385](../../../../src/continuum_swr_replay/analysis/replay/fitting.py) — conditional-Poisson / conditional-logit `.fit(disp=0)`; discrete-MLE results → convergence in `result.mle_retvals`.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:129-140](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py) — `sm.GLM(...).fit(disp=0)`; **GLM** results → convergence in `result.converged`.
- [src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:118-160](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py) and [:245-256](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py) — GLM `.fit()`; `:118` docstring: separation yields a large finite coef (not an error).
- [src/continuum_swr_replay/analysis/replay/choice_prediction.py:482-495](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py) — choice conditional-logit `.fit(disp=0)`.
- [scripts/content/fit_swr_contrast_content.py:78](../../../../scripts/content/fit_swr_contrast_content.py) — blanket `warnings.filterwarnings("ignore")`; also grep the other `scripts/{content,rate}/fit_*.py` for the same line.

## Tasks

- **Add a small shared diagnostics helper** `analysis/replay/_fit_diagnostics.py` with two functions:
  ```python
  import contextlib
  import warnings
  import numpy as np
  from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, ConvergenceWarning

  SEPARATION_COEF_ABS = 10.0  # matches contrast_aggregation.py's quasi-separation filter

  def fit_status(result, kind: str) -> dict:
      """kind: 'glm' (GLM results, use .converged) or 'mle' (discrete results, use mle_retvals)."""
      if kind == "glm":
          converged = bool(getattr(result, "converged", False))
      else:  # discrete MLE (ConditionalPoisson/ConditionalLogit/Logit)
          converged = bool((getattr(result, "mle_retvals", {}) or {}).get("converged", False))
      params = np.asarray(result.params, dtype=float)
      bse = np.asarray(result.bse, dtype=float)
      separated = bool(
          np.any(np.abs(params) >= SEPARATION_COEF_ABS)
          or np.any(~np.isfinite(bse))
          or np.any(bse > 1e3)
      )
      return {"converged": converged, "separated": separated}

  @contextlib.contextmanager
  def capture_fit_warnings():
      """Record instead of suppress; force 'always' so warnings already emitted once in the
      process (Python's default de-dups repeats) are still captured per-fit."""
      with warnings.catch_warnings(record=True) as caught:
          warnings.simplefilter("always")
          yield caught  # caller inspects `caught` for ConvergenceWarning / PerfectSeparationWarning
  ```
  (`catch_warnings(record=True)` alone is insufficient — without `simplefilter("always")` a warning that
  already fired once earlier in the run is suppressed by the default filter and never recorded.)
- **Wire `fit_status` into every fitter**, adding **flat** `converged`, `separated`, and
  `n_convergence_warnings` keys to the returned result dict. The fitters return flat dicts that become
  DataFrame rows (`fitting.py:193`), so a nested `status={...}` object would break that flattening — use
  flat scalar columns, not a sub-dict. Do **not** drop the fit here — record the flags and let the
  existing aggregation policy decide (see the exclusion-policy note below):
  - `fitting.py:160` (conditional Poisson) and `:380` (conditional logit): `kind="mle"`.
  - `per_stop_rate_model.py:130`: `kind="glm"`.
  - `per_bin_rate_model.py:153` (the primary fit): `kind="glm"` → flat `converged`/`separated`.
  - `choice_prediction.py:492`: `kind="mle"`.
  `fit_status` returns `{"converged": ..., "separated": ...}`; unpack those as flat keys and add
  `n_convergence_warnings` from the captured-warnings count.
- **CV fits are multi-fit — use FOLD-LEVEL fields, not a single flat status (Medium, round 6).** The CV
  loops fit a model *and* a baseline per fold (`per_bin_rate_model.py:249,253`; `per_stop_rate_model.py:201`),
  so one `converged`/`separated` scalar is ambiguous. Aggregate across folds into
  `cv_n_nonconverged_folds` / `cv_n_separated_folds` (count folds where either the model or baseline fit
  failed the `fit_status` check). **Deterministic rule (Medium, round 7): if ANY fold's model or baseline
  is nonconverged/separated, mark the whole CV metric invalid (`nan`/`None`) — do NOT drop the bad fold.**
  Dropping a fold changes the held-out set and makes CV metrics non-comparable across rungs (a rung that
  drops fold 3 is scored on a different test set than one that doesn't). Report the fold counts alongside
  the invalidated metric. Do not collapse per-fold status into a single top-level `converged`.
- **Exclusion policy is unchanged in this phase (no estimand change).** Non-converged and `separated`
  fits are **flagged only**; the sole thing that *excludes* a fit remains the pre-existing
  `contrast_aggregation.py:83` `|coef|>=10` filter. This phase adds the flags and *counts* how many fits
  each reason would catch, but does not add any new exclusion. If a later decision wants to exclude on
  `converged==False`, that is a separate, explicit change with its own estimand-impact review — call it
  out, don't fold it in here.
- **Propagate to aggregation as counts, not exclusions.** `contrast_aggregation.py` already drops
  `|coef|>=10` (`:83`) per (session, term) — that stays, and its counts go to the **term-diagnostics**
  table ([contract 3](shared-contracts.md#term-diagnostics)), **not** the cohort audit (it is term-level,
  not a cohort exclusion — F2). Additionally count, per (model, population), how many *modeled
  session-fits* carry `converged==False` and how many carry `separated==True`, and surface those as the
  phase-3 **session-level** informational `n_sessions_flagged_nonconverged` /
  `n_sessions_flagged_separated` columns. These are diagnostics on the retained cohort, **not** removals —
  reconciliation is unaffected (see [contract 1](shared-contracts.md#cohort-audit-schema)).
- **Scope down warning suppression.** Replace blanket `warnings.filterwarnings("ignore")` (at
  `fit_swr_contrast_content.py:78` and the equivalents in the other runners) with targeted handling:
  wrap the per-session fit call in `capture_fit_warnings()` and log a count of `ConvergenceWarning` /
  `PerfectSeparationWarning` per session; keep suppressing known-benign patsy/RuntimeWarning noise by
  narrowing to those specific categories, not `"ignore"` globally.
- **No new user-facing docs** — these are internal diagnostics. Add a one-line note to
  `docs/rate_models.md` and `docs/models_table.md`'s model-checking section that fits now carry
  `converged`/`separated` status flags, since those docs describe the fit outputs.

## Deliberately not in this phase

- Re-adding rank checks — they already exist (`fitting.py:31`, `per_stop_rate_model.py:113`). Touching
  them risks regressions.
- Changing the `|coef|>=10` threshold or the aggregation's drop policy — only *counting/logging* is added.
- Retrying non-converged fits with different starts / maxiter tuning — out of scope; flagging is enough.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/unit/test_fit_diagnostics.py::test_glm_converged` | `fit_status` on a well-posed GLM returns `converged=True, separated=False` |
| `::test_separation_flagged` | a synthetic perfectly-separated logistic design → `separated=True` (coef ≥ 10 or non-finite SE) — not silently returned as clean |
| `::test_nonconverged_flagged` | a GLM fit with `maxiter=1` on a hard design → `converged=False` |
| `::test_warnings_captured` | `capture_fit_warnings()` records a `ConvergenceWarning` rather than swallowing it |
| existing rate/content fitter unit tests | still pass (status dict is additive) |

## Fixtures

Synthesize in `conftest.py`: (a) a well-separated 2-class logistic dataset (guaranteed separation);
(b) a benign Poisson count design; (c) a design forcing `maxiter=1` non-convergence. All tiny, no real data.

## Review

Dispatch `code-reviewer` against the diff. Confirm:
- `kind="glm"` vs `kind="mle"` is correct at each call site (GLM→`.converged`, discrete→`mle_retvals`) — reviewer re-reads each site.
- No fit is *dropped* by the new code — status is additive; only aggregation counting changed.
- Warning scoping does not re-suppress the diagnostics it is meant to surface.
- Separation test actually triggers separation (not a tautology); shared setup is in fixtures.
