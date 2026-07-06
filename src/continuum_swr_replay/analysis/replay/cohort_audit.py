"""Per-model cohort / drop-profile audit records.

Makes complete-case filtering *visible*: every fit run emits one record per
(model, population) stating who entered the model (``n_in``), who was dropped and
why (the mutually-exclusive ``n_drop_*`` buckets), who was fit (``n_modeled``), and
whether the drop looks biased (``drop_notes`` profile). This is reporting
infrastructure — it changes no fitted number.

Schema and the reconciliation invariant are defined in
``.claude/docs/plans/regression-integrity-fixes/shared-contracts.md`` (Contract 1).
The single hard rule: the drop buckets are **mutually exclusive** and
``n_in - Σ n_drop_* == n_modeled`` must hold, or :func:`cohort_record` raises.

Convergence/separation flags (Contract 1) and quasi-separation (Contract 3,
:func:`term_diagnostics`) are **session/term-level** and deliberately excluded from
the row reconciliation — they annotate the retained cohort, they do not remove rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Row-level drop buckets, in the schema's PRIORITY order. Each dropped unit is
# attributed to exactly the first bucket it hits, so the reconciliation closes even
# when a unit fails several checks.
DROP_FIELDS = (
    "n_drop_qvals_gap",  # [1] value/HMM covariate missing (mechanism B — the qvals tail)
    "n_drop_own_predictor_missing",  # [2] this model's own required predictor is NaN
    "n_drop_zero_response",  # [3] all responses zero (content: Σcount==0)
    "n_drop_common_cohort_other",  # [4] common mode: dropped only for ANOTHER rung's NaN
    "n_excluded_session",  # [5] data-level session gates (min_*/rank/n<2·n_pred)
)

# Session-level informational flags. Carried on the record but NOT subtracted in the
# reconciliation (they count fits, not rows).
FLAG_FIELDS = ("n_sessions_flagged_nonconverged", "n_sessions_flagged_separated")

# Full column order for the emitted CSV.
COHORT_COLUMNS = (
    "model", "population", "grain", "cohort_mode", "n_in",
    *DROP_FIELDS, "n_modeled", *FLAG_FIELDS,
    "n_sessions_in", "n_sessions_modeled", "n_animals_modeled", "drop_notes",
)

TERM_DIAGNOSTICS_COLUMNS = (
    "model", "population", "term",
    "n_sessions_fit", "n_sessions_quasiseparated", "n_sessions_used", "n_animals_used",
)

# Matches contrast_aggregation.py's quasi-separation filter and _fit_diagnostics.
_QUASI_SEP_ABS = 10.0

# Covariates that are NaN on the trailing "qvals gap" (mechanism B): the ~896-event /
# 2056-stop tail with no fitted HMM/beta value state. When one of these is a model's
# required predictor, an event/stop it drops is attributed to ``n_drop_qvals_gap`` (with
# priority over generic own-predictor missingness), keeping mechanism B distinguishable
# from mechanism A (e.g. ``reward_rate_arm_k10``'s early-session NaN). The beta ``q_arm``,
# HMM ``q_arm_hmm`` and nominal ``p_reward_arm`` share one identical missingness mask
# (verified: 8064 event-arm rows / 896 events); the stop-level ``hmm_*`` scalars and their
# ``_x_current`` interactions come off the same qvals table and share the tail.
QVALS_GAP_COLS = frozenset({
    # per-arm value / contingency (identical missingness)
    "q_arm", "q_arm_hmm", "p_reward_arm",
    "swr_x_q_arm", "swr_x_q_arm_hmm",
    # stop-level HMM/beta scalars from qvals_table (same trailing tail)
    "hmm_rpe", "rpe", "hmm_relative_value", "hmm_value_update",
    "reward_entropy", "state_entropy",
    # content _x_current interactions built from those scalars
    "rpe_x_current", "relval_x_current", "entropy_x_current", "update_x_current",
})


def cohort_record(
    model: str,
    population: str,
    grain: str,
    *,
    cohort_mode: str = "per_rung",
    n_in: int,
    n_modeled: int,
    sessions_in: int,
    sessions_modeled: int,
    animals_modeled: int,
    drop_notes: str = "",
    **counts: int,
) -> dict:
    """One cohort-audit record; raises if the drop buckets do not reconcile.

    ``counts`` supplies any of :data:`DROP_FIELDS` / :data:`FLAG_FIELDS` (missing ones
    default to 0). ``grain`` names the counting unit for every ``n_*`` (``event`` /
    ``stop`` / ``stop_bin``).
    """
    rec = {
        "model": model, "population": population, "grain": grain,
        "cohort_mode": cohort_mode, "n_in": int(n_in), "n_modeled": int(n_modeled),
        "n_sessions_in": int(sessions_in), "n_sessions_modeled": int(sessions_modeled),
        "n_animals_modeled": int(animals_modeled), "drop_notes": str(drop_notes),
    }
    for f in DROP_FIELDS:
        rec[f] = int(counts.get(f, 0))
    for f in FLAG_FIELDS:  # carried through, but NOT subtracted in the reconciliation
        rec[f] = int(counts.get(f, 0))
    recon = rec["n_in"] - sum(rec[f] for f in DROP_FIELDS)
    if recon != rec["n_modeled"]:
        raise AssertionError(
            f"cohort audit does not reconcile for {model}/{population}: "
            f"n_in - Σdrops = {recon} != n_modeled = {rec['n_modeled']} "
            f"(drop buckets must be mutually exclusive and cover every dropped unit)"
        )
    return rec


def classify_drops(dropped_mask_by_reason: dict) -> dict:
    """Attribute each dropped unit to exactly ONE bucket by :data:`DROP_FIELDS` priority.

    ``dropped_mask_by_reason`` maps a drop-field name to a boolean mask (or array) over
    the same candidate units. A unit failing several checks is counted once, in its
    highest-priority bucket.
    """
    counted = None  # boolean mask of units already attributed
    out = {}
    for field in DROP_FIELDS:
        m = dropped_mask_by_reason.get(field)
        if m is None:
            out[field] = 0
            continue
        m = np.asarray(m, dtype=bool)
        fresh = m if counted is None else (m & ~counted)
        out[field] = int(fresh.sum())
        counted = fresh if counted is None else (counted | fresh)
    return out


def event_drop_buckets(
    long_df: pd.DataFrame,
    *,
    event_col: str,
    response_col: str,
    own_predictors,
    union_predictors,
) -> dict:
    """Per-event drop classification for a **union complete-case** (common-cohort) rung.

    ``long_df`` is the *pre-filter* long frame. Under the union complete-case rule an
    event is dropped if ANY union predictor is NaN in ANY of its rows; a surviving event
    is dropped if its response sums to zero. Each dropped event is attributed to exactly
    one bucket by the schema priority (qvals_gap → own_predictor_missing →
    common_cohort_other → zero_response):

    - ``n_drop_qvals_gap`` — a qvals-tail covariate (:data:`QVALS_GAP_COLS`) is NaN;
    - ``n_drop_own_predictor_missing`` — one of *this rung's* own (non-qvals) predictors
      is NaN;
    - ``n_drop_common_cohort_other`` — dropped only because *another* rung's predictor is
      NaN (this rung's own predictors are all present) — the F4 common-cohort bucket;
    - ``n_drop_zero_response`` — survives complete-case but the response sums to zero.

    Returns the four counts plus ``dropped_event_ids`` (the union of all dropped events).
    """
    present_union = [c for c in union_predictors if c in long_df.columns]
    nan_by_col = {
        c: set(long_df.loc[long_df[c].isna(), event_col]) for c in present_union
    }
    qvals_cols = [c for c in present_union if c in QVALS_GAP_COLS]
    qvals_events = (
        set().union(*(nan_by_col[c] for c in qvals_cols)) if qvals_cols else set()
    )
    own_cols = [
        c for c in own_predictors if c in nan_by_col and c not in QVALS_GAP_COLS
    ]
    own_events = (
        set().union(*(nan_by_col[c] for c in own_cols)) if own_cols else set()
    ) - qvals_events
    all_nan_events = (
        set().union(*nan_by_col.values()) if nan_by_col else set()
    )
    common_events = all_nan_events - qvals_events - own_events
    survivors = long_df[~long_df[event_col].isin(all_nan_events)]
    totals = survivors.groupby(event_col)[response_col].sum()
    zero_events = set(totals[totals == 0].index)
    return {
        "n_drop_qvals_gap": len(qvals_events),
        "n_drop_own_predictor_missing": len(own_events),
        "n_drop_common_cohort_other": len(common_events),
        "n_drop_zero_response": len(zero_events),
        "dropped_event_ids": qvals_events | own_events | common_events | zero_events,
    }


def drop_profile(dropped: pd.DataFrame) -> str:
    """Summarize WHO was dropped, to expose non-random drops (mechanism A vs B).

    Records the median within-stop time, reward fraction, and value quantiles of the
    dropped rows where those columns are present, so a reviewer can see whether the
    drop skews early-session (mechanism A) or late-session/low-value (mechanism B).
    """
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
    return "; ".join(parts) if parts else f"n_dropped={len(dropped)}"


def write_cohort_audit(records: list[dict], path) -> pd.DataFrame:
    """Write cohort records to ``path`` as CSV in the canonical column order."""
    df = pd.DataFrame(records)
    for col in COHORT_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col not in ("drop_notes",) else ""
    df = df[list(COHORT_COLUMNS)]
    df.to_csv(path, index=False)
    return df


def term_diagnostics(
    res: pd.DataFrame, specs: list[tuple[str, str, str]]
) -> pd.DataFrame:
    """Per (model, population, term) quasi-separation diagnostics (Contract 3).

    Mirrors ``contrast_aggregation``'s per-(session, term) ``|coef| >= 10`` filter and
    ``>= 5`` surviving-session requirement, surfaced as a standalone table so
    term-specific coefficient exclusions stay OUT of the cohort reconciliation.

    Parameters
    ----------
    res : pd.DataFrame
        Per-session fits with ``model``, ``population``, ``animal`` and ``coef_<term>``.
    specs : list of (model, population, term)
    """
    rows = []
    for model, pop, term in specs:
        ccol = f"coef_{term}"
        sub = res[(res.model == model) & (res.population == pop)]
        if ccol not in sub.columns:
            rows.append({
                "model": model, "population": pop, "term": term,
                "n_sessions_fit": 0, "n_sessions_quasiseparated": 0,
                "n_sessions_used": 0, "n_animals_used": 0,
            })
            continue
        present = sub[sub[ccol].notna()]
        quasi = present[ccol].abs() >= _QUASI_SEP_ABS
        used = present[~quasi]
        n_animals_used = (
            int(used["animal"].nunique()) if "animal" in used.columns else 0
        )
        rows.append({
            "model": model, "population": pop, "term": term,
            "n_sessions_fit": int(len(present)),
            "n_sessions_quasiseparated": int(quasi.sum()),
            "n_sessions_used": int(len(used)),
            "n_animals_used": n_animals_used,
        })
    return pd.DataFrame(rows, columns=list(TERM_DIAGNOSTICS_COLUMNS))
