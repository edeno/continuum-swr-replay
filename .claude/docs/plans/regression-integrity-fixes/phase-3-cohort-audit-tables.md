# Phase 3 — Per-model cohort / drop-profile audit tables

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Contracts referenced:** [cohort-audit-table schema](shared-contracts.md#cohort-audit-schema) — this
phase is its producer; do not weaken the reconciliation invariant.

Make complete-case filtering *visible*. Every fit run emits a cohort-audit record so the estimand
(who is in the model, who was dropped and why, and whether the drop is biased) is stated, not implied.
This is reporting infrastructure — no fitted numbers change — and phases 4/5 attach it to their refits.

**Inputs to read first:**

- [scripts/content/fit_replay_content_models.py:495-530](../../../../scripts/content/fit_replay_content_models.py) — the union complete-case selection and the `Σcount==0` drop; the natural place to tap `n_in`/drops for content.
- [src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py:83-90](../../../../src/continuum_swr_replay/analysis/replay/per_stop_rate_model.py) — `dropna(required)` + `immobility>0` for Family A.
- [src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py:130-145](../../../../src/continuum_swr_replay/analysis/replay/per_bin_rate_model.py) — per-bin `dropna` + `exposure>0` + `min_bins/min_events`.
- [scripts/content/fit_swr_contrast_content.py:225-245](../../../../scripts/content/fit_swr_contrast_content.py) and [scripts/rate/fit_swr_contrast_rate.py:225-245](../../../../scripts/rate/fit_swr_contrast_rate.py) — the qvals merge (mechanism-B drop) points.
- [src/continuum_swr_replay/analysis/replay/choice_prediction.py:300-315](../../../../src/continuum_swr_replay/analysis/replay/choice_prediction.py) — valid-replay-stop selection for choice.

## Tasks

- **Add a cohort-audit emitter** `analysis/replay/cohort_audit.py` implementing the
  [cohort-audit schema](shared-contracts.md#cohort-audit-schema) exactly — mutually-exclusive drop
  buckets in the schema's priority order:
  ```python
  import pandas as pd

  DROP_FIELDS = ("n_drop_qvals_gap", "n_drop_own_predictor_missing", "n_drop_zero_response",
                 "n_drop_common_cohort_other", "n_excluded_session")  # priority order; reconcile
  FLAG_FIELDS = ("n_sessions_flagged_nonconverged", "n_sessions_flagged_separated")  # session-level; do NOT reconcile

  def cohort_record(model, population, grain, *, cohort_mode="per_rung", n_in, n_modeled,
                    sessions_in, sessions_modeled, animals_modeled, drop_notes="", **counts) -> dict:
      rec = dict(model=model, population=population, grain=grain, cohort_mode=cohort_mode,
                 n_in=int(n_in), n_modeled=int(n_modeled),
                 n_sessions_in=int(sessions_in), n_sessions_modeled=int(sessions_modeled),
                 n_animals_modeled=int(animals_modeled), drop_notes=str(drop_notes))
      for f in DROP_FIELDS:
          rec[f] = int(counts.get(f, 0))
      for f in FLAG_FIELDS:                       # carried through with defaults, but NOT subtracted
          rec[f] = int(counts.get(f, 0))
      recon = rec["n_in"] - sum(rec[f] for f in DROP_FIELDS)   # flags excluded from reconciliation
      if recon != rec["n_modeled"]:
          raise AssertionError(
              f"cohort audit does not reconcile for {model}/{population}: "
              f"{recon} != {rec['n_modeled']} (buckets must be mutually exclusive)")
      return rec

  def classify_drops(dropped_mask_by_reason: dict) -> dict:
      """Assign each dropped unit to exactly ONE bucket by DROP_FIELDS priority order,
      so a unit failing several checks is counted once (schema priority rule)."""
      counted = None  # boolean mask of already-attributed units
      out = {}
      for field in DROP_FIELDS:
          m = dropped_mask_by_reason.get(field)
          if m is None:
              out[field] = 0
              continue
          fresh = m if counted is None else (m & ~counted)
          out[field] = int(fresh.sum())
          counted = fresh if counted is None else (counted | fresh)
      return out

  def drop_profile(dropped: pd.DataFrame) -> str:
      """Summarize WHO was dropped, to detect non-random drops (mechanism A vs B)."""
      if len(dropped) == 0:
          return "no drops"
      parts = []
      if "time_since_stop" in dropped:
          parts.append(f"median_tss={dropped['time_since_stop'].median():.2f}")
      if "reward_outcome" in dropped:
          parts.append(f"rewarded_frac={dropped['reward_outcome'].mean():.3f}")
      for vcol in ("q_arm_hmm", "hmm_relative_value"):
          if vcol in dropped:
              parts.append(f"{vcol}_median={dropped[vcol].median():.3f}")
      return "; ".join(parts)
  ```
- **Instrument each runner** to build one `cohort_record` per (model, population) at its existing
  drop sites, accumulate into a list, and write `<run>_cohort_audit.csv` next to the fit outputs:
  - Content base ladder (`fit_replay_content_models.py:488-512`): tap `n_in` before the union filter.
    Because the union drops an event when **any** rung's predictor is NaN, build a per-reason boolean
    mask dict and pass it through `classify_drops` so M0–M2 events dropped for a *later* rung's predictor
    land in `n_drop_common_cohort_other` (cohort_mode=`common`), not `n_drop_own_predictor_missing` (F4).
    Value-covariate NaN (the qvals tail) goes to `n_drop_qvals_gap`; `Σcount==0` to `n_drop_zero_response`;
    the **data-level** gates `min_*`/rank/`n<2·n_pred` to `n_excluded_session`. The `|coef|>=10`
    quasi-separation filter is **term-level** and goes in the separate term-diagnostics table
    ([contract 3](shared-contracts.md#term-diagnostics)), **not** `n_excluded_session` (F2 — a model-level
    cohort row can't represent a per-(session,term) coefficient drop). The phase-2 convergence/separation
    flags are recorded as the **session-level informational** `n_sessions_flagged_nonconverged`/
    `n_sessions_flagged_separated` columns (excluded from reconciliation; see
    [contract 1](shared-contracts.md#cohort-audit-schema)).
  - Content/rate swr-contrast runners: the qvals-merge drop → `n_drop_qvals_gap` (mechanism B), kept
    distinct from own-predictor missingness.
  - Family A / Family B rate runners: `immobility<=0` / `exposure<=0` / `dropna` → `n_drop_own_predictor_missing`;
    session gates → `n_excluded_session`.
  - Choice runner: valid-replay-stop selection and ≠1-chosen drop → `n_drop_own_predictor_missing`;
    `n<2·n_pred` → `n_excluded_session`.
- **Emit the term-level diagnostics table ([contract 3](shared-contracts.md#term-diagnostics)).**
  `contrast_aggregation.py:83` already applies the per-(session, term) `|coef|>=10` filter and requires
  `>=5` surviving sessions; surface those counts as `<run>_term_diagnostics.csv` (one row per model ×
  population × term: `n_sessions_fit`, `n_sessions_quasiseparated`, `n_sessions_used`, `n_animals_used`).
  This keeps term-specific coefficient exclusions out of the cohort reconciliation.
- **Repair the content runner's missing-column path (F4).** `fit_swr_contrast_content.py:185`
  (`_fit_session`) currently calls `fit_conditional_poisson` even when required predictor columns are
  absent — a latent crash / hidden path that differs from the rate runner's clean
  `return None, f"missing:{cols}"` skip (`fit_swr_contrast_rate.py:134`). Since phase 3 is already
  instrumenting this runner, make the content path mirror the rate path: when `missing`, return a clean
  skip with reason `missing:<cols>` that the cohort audit records under `n_excluded_session` (reason in
  `drop_notes`), never a fitter call with absent columns. This makes missing predictors *auditable skips*,
  not crashes or silent path divergence.
- **Add a summary CLI** `scripts/qa/summarize_cohort_audits.py` that concatenates all
  `*_cohort_audit.csv` under the outputs dir into one table and flags rows where
  `n_modeled / n_in < 0.9` or where `drop_profile` indicates a biased drop (early-session or
  low-value). This is what a reviewer reads to answer "did complete-case change the estimand?".
- **Document** the cohort-audit outputs in `docs/README.md` ("Where results live") and add a short
  subsection to `docs/models_table.md` explaining the schema and the reconciliation invariant.

## Deliberately not in this phase

- *Changing* any drop policy (encoding mechanism A, time-gating, common-cohort refits) — that is phases
  4/5. This phase only *measures and reports* the current drops.
- Emitting the profile for the 32.5M-row timebin table — audit at the stop/event grain; the timebin
  design is aggregated per-session inside the fitter.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/unit/test_cohort_audit.py::test_reconciles` | a hand-built record with matching counts returns; a mismatched one raises `AssertionError` |
| `::test_classify_drops_mutually_exclusive` | a unit failing both qvals-gap and own-predictor-missing is counted once, in `n_drop_qvals_gap` (higher priority); Σ buckets == total dropped |
| `::test_common_cohort_bucket` | in `common` mode, an event whose own predictors are present but a later rung's predictor is NaN lands in `n_drop_common_cohort_other`, not `n_drop_own_predictor_missing` |
| `::test_flags_do_not_reconcile` | passing `n_sessions_flagged_nonconverged`/`n_sessions_flagged_separated` populates those columns but does **not** change reconciliation (nonzero flags + closing drop buckets still returns); quasi-separation counts appear only in `<run>_term_diagnostics.csv`, never in the cohort row (F1/F2) |
| `tests/unit/test_content_missing_cols.py::test_clean_skip` | `_fit_session` with an absent predictor returns a `missing:<cols>` skip (no `fit_conditional_poisson` call), matching the rate runner (F4) |
| `::test_drop_profile_early_session` | a dropped-rows frame biased to low `time_since_stop` yields a profile string exposing it |
| `tests/integration/test_content_emits_audit.py` (mark slow) | running the content base ladder on one cached session writes a `*_cohort_audit.csv` whose `n_in − Σdrops == n_modeled` for every rung |
| manual: `summarize_cohort_audits.py` on a smoke run | prints the 11.6% content drop and the 1.9% swr-contrast qvals drop as separate, flagged rows |

## Fixtures

Use one cached session (see `docs/cached_trials_for_testing.md`) for the integration test; unit tests
use tiny hand-built frames. No re-decode.

## Review

Dispatch `code-reviewer`. Confirm:
- The reconciliation invariant is asserted (fail-loud), not just logged.
- `n_drop_qvals_gap` is recorded separately from generic missing-predictor drops (mechanism B is distinguishable from mechanism A).
- No runner path can emit fit outputs without the cohort record (per [contract 1](shared-contracts.md#cohort-audit-schema)).
- Integration test uses a real cached slice, not only synthetic.
