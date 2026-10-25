"""Per-model cohort / drop-profile audit records.

Makes complete-case filtering *visible*: every fit run emits one record per
(model, population) stating who entered the model (``n_in``), who was dropped and
why (the mutually-exclusive ``n_drop_*`` buckets), who was fit (``n_modeled``), and
whether the drop looks biased (``drop_notes`` profile). This is reporting
infrastructure — it changes no fitted number.

The single hard rule: the drop buckets are **mutually exclusive** and
``n_in - Σ n_drop_* == n_modeled`` must hold, or :func:`cohort_record` raises.

Convergence/separation flags (Contract 1) and quasi-separation (Contract 3,
:func:`term_diagnostics`) are **session/term-level** and deliberately excluded from
the row reconciliation — they annotate the retained cohort, they do not remove rows.
"""

from __future__ import annotations

from collections.abc import Container, Iterable

import numpy as np
import pandas as pd

from .covariates.definitions import QVALS_ROW_GATED_COVARIATES

# Row-level drop buckets, in the schema's PRIORITY order. Each dropped unit is
# attributed to exactly the first bucket it hits, so the reconciliation closes even
# when a unit fails several checks.
DROP_FIELDS = (
    "n_drop_qvals_gap",  # [1] value covariate missing (mechanism B — the qvals tail)
    "n_drop_own_predictor_missing",  # [2] this model's own required predictor is NaN
    "n_drop_nonfinite",  # [3] NaN-free but ±Inf/invalid numeric required value
    "n_drop_zero_response",  # [4] all responses zero (content: Σcount==0)
    "n_drop_common_cohort_other",  # [5] dropped only for another rung's NaN
    "n_excluded_session",  # [6] data-level session gates (min_*/rank/n<2·n_pred)
)

# Session-level informational flags. Carried on the record but NOT subtracted in the
# reconciliation (they count fits, not rows).
FLAG_FIELDS = ("n_sessions_flagged_nonconverged", "n_sessions_flagged_separated")

# Full column order for the emitted CSV.
COHORT_COLUMNS = (
    "model",
    "population",
    "grain",
    "cohort_mode",
    "n_in",
    *DROP_FIELDS,
    "n_modeled",
    *FLAG_FIELDS,
    "n_sessions_in",
    "n_sessions_modeled",
    "n_animals_modeled",
    "drop_notes",
)

TERM_DIAGNOSTICS_COLUMNS = (
    "model",
    "population",
    "term",
    "n_sessions_fit",
    "n_sessions_quasiseparated",
    "n_sessions_used",
    "n_animals_used",
)

# Matches contrast_aggregation.py's quasi-separation filter and _fit_diagnostics.
_QUASI_SEP_ABS = 10.0

#: Name modifiers a design column may carry over its base covariate: the fit-time
#: ``_z`` standardization and the leaf-mean ``_patch`` aggregate.
_DERIVED_SUFFIXES = ("_z", "_patch")


def is_qvals_gap_column(name: object) -> bool:
    """Whether a NaN in design column ``name`` means the stop has no qvals row.

    The qvals gap (mechanism B) is the set of stops with no ``qvals_table``
    row. A column is in it when one of its factors is a covariate catalogued
    as ``qvals_row_gated`` (:data:`QVALS_ROW_GATED_COVARIATES`). Factors are
    the ``_x_``-separated parts of the name, so ``swr_x_<base>`` and
    ``<base>_z_x_is_current_arm`` qualify, and each factor may carry a ``_z``
    or ``_patch`` suffix. A product is NaN wherever any factor is NaN.

    Parameters
    ----------
    name : object
        A design-column name.

    Returns
    -------
    bool
        True when ``name`` is built from a qvals-row-gated covariate.
    """
    if not isinstance(name, str):
        return False
    for factor in name.split("_x_"):
        for suffix in _DERIVED_SUFFIXES:
            factor = factor.removesuffix(suffix)
        if factor in QVALS_ROW_GATED_COVARIATES:
            return True
    return False


class _QvalsGapColumns(Container[str]):
    """Membership view of :func:`is_qvals_gap_column` for ``name in ...`` tests."""

    def __contains__(self, name: object) -> bool:
        return is_qvals_gap_column(name)

    def __repr__(self) -> str:
        return f"QVALS_GAP_COLS(bases={sorted(QVALS_ROW_GATED_COVARIATES)})"


# When a qvals-gap column is a model's required predictor, an event/stop it drops
# is attributed to ``n_drop_qvals_gap`` (with priority over generic own-predictor
# missingness), keeping mechanism B distinguishable from mechanism A (e.g.
# candidate-arm reward history's early-session NaN). Derived from the covariate
# catalog rather than listed, so a new qvals covariate and its variants join the
# gap once its definition is flagged.
QVALS_GAP_COLS: Container[str] = _QvalsGapColumns()


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
        "model": model,
        "population": population,
        "grain": grain,
        "cohort_mode": cohort_mode,
        "n_in": int(n_in),
        "n_modeled": int(n_modeled),
        "n_sessions_in": int(sessions_in),
        "n_sessions_modeled": int(sessions_modeled),
        "n_animals_modeled": int(animals_modeled),
        "drop_notes": str(drop_notes),
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
    event is dropped if ANY union predictor is missing or nonfinite in ANY of
    its rows; a surviving event is dropped if its response sums to zero. Each
    dropped event is attributed to exactly one bucket by the schema priority
    (qvals_gap → own_predictor_missing → nonfinite → common_cohort_other →
    zero_response):

    - ``n_drop_qvals_gap`` — a qvals-tail covariate (:data:`QVALS_GAP_COLS`) that **this
      rung itself uses** is NaN (0 for a rung with no value covariate, per Contract 1);
    - ``n_drop_own_predictor_missing`` — one of *this rung's* own (non-qvals) predictors
      is NaN;
    - ``n_drop_nonfinite`` — a required numeric value is present but ±Inf or
      otherwise nonnumeric;
    - ``n_drop_common_cohort_other`` — dropped only because *another* rung's predictor is
      NaN (this rung's own predictors are all present) — the F4 common-cohort bucket. A
      qvals-tail event where the value covariate belongs to a *different* rung lands here,
      NOT in ``n_drop_qvals_gap``.
    - ``n_drop_zero_response`` — survives complete-case but the response sums to zero.

    Returns the five counts plus ``dropped_event_ids`` (the union of all dropped events).
    """
    present_union = [c for c in union_predictors if c in long_df.columns]
    nan_by_col = {
        c: set(long_df.loc[long_df[c].isna(), event_col]) for c in present_union
    }
    nonfinite_by_col = {}
    for column in present_union:
        numeric = pd.to_numeric(long_df[column], errors="coerce")
        invalid = long_df[column].notna() & ~np.isfinite(numeric.to_numpy(dtype=float))
        nonfinite_by_col[column] = set(long_df.loc[invalid, event_col])
    # qvals gap is attributed only when THIS rung's own value covariate is missing.
    own_qvals_cols = [
        c for c in own_predictors if c in QVALS_GAP_COLS and c in nan_by_col
    ]
    qvals_events = (
        set().union(*(nan_by_col[c] for c in own_qvals_cols))
        if own_qvals_cols
        else set()
    )
    own_cols = [
        c for c in own_predictors if c in nan_by_col and c not in QVALS_GAP_COLS
    ]
    own_events = (
        set().union(*(nan_by_col[c] for c in own_cols)) if own_cols else set()
    ) - qvals_events
    all_nan_events = set().union(*nan_by_col.values()) if nan_by_col else set()
    common_events = all_nan_events - qvals_events - own_events
    all_nonfinite_events = (
        set().union(*nonfinite_by_col.values()) if nonfinite_by_col else set()
    )
    nonfinite_events = all_nonfinite_events - all_nan_events
    survivors = long_df[~long_df[event_col].isin(all_nan_events | nonfinite_events)]
    totals = survivors.groupby(event_col)[response_col].sum()
    zero_events = set(totals[totals == 0].index)
    return {
        "n_drop_qvals_gap": len(qvals_events),
        "n_drop_own_predictor_missing": len(own_events),
        "n_drop_nonfinite": len(nonfinite_events),
        "n_drop_common_cohort_other": len(common_events),
        "n_drop_zero_response": len(zero_events),
        "dropped_event_ids": (
            qvals_events | own_events | nonfinite_events | common_events | zero_events
        ),
    }


def stop_own_missing(stop_df, required_cols, *, immobility_col="immobility_duration_s"):
    """Boolean mask over ``stop_df`` rows dropped for *own-predictor* reasons.

    Mirrors what the rate fitters remove at the stop level BEFORE fitting: a stop with
    ``immobility_time <= 0`` (never enters the timebin design) or with NaN in any of this
    model's stop-level required predictors (all its bins are dropped by the fitter's
    ``dropna(needed)``, since these columns are constant within a stop). Bin-level drops
    (short self-history lags) are below this grain and fall into ``n_excluded_session``.
    """
    present = [c for c in required_cols if c in stop_df.columns]
    mask = (
        stop_df[present].isna().any(axis=1)
        if present
        else pd.Series(False, index=stop_df.index)
    )
    if immobility_col in stop_df.columns:
        mask = mask | (stop_df[immobility_col] <= 0)
    return mask


def drop_profile(dropped: pd.DataFrame) -> str:
    """Summarize WHO was dropped, to expose non-random drops (mechanism A vs B).

    Records the median within-stop time, reward fraction, and value quantiles of the
    dropped rows where those columns are present, so a reviewer can see whether the
    drop skews early-session (mechanism A) or late-session/low-value (mechanism B).
    """
    if len(dropped) == 0:
        return "no drops"
    parts = []
    if "time_since_stop_start_s" in dropped:
        parts.append(f"median_tss={dropped['time_since_stop_start_s'].median():.2f}")
    if "is_rewarded_outcome" in dropped:
        parts.append(f"rewarded_frac={dropped['is_rewarded_outcome'].mean():.3f}")
    for vcol in (
        "reward_belief_mean_candidate_arm_preoutcome",
        "relative_value_switch_minus_stay_preoutcome",
    ):
        if vcol in dropped:
            parts.append(f"{vcol}_median={dropped[vcol].median():.3f}")
    return "; ".join(parts) if parts else f"n_dropped={len(dropped)}"


#: Characters ``drop_notes`` uses as its own separators, and which therefore may
#: not appear inside one note's text.
_NOTE_SEPARATORS = ";,"


def escape_note_field(text: object) -> str:
    """Make arbitrary text safe to embed in a ``drop_notes`` string.

    ``drop_notes`` is a single free-text cell built by joining parts with ``"; "``
    and ``", "``, and it is written to CSV. An estimator's refusal message is the
    one part that is not authored here -- it quotes column names and shapes and
    routinely contains commas -- so embedding it raw makes the note's own
    structure unparseable and can split the cell in a naive CSV reader. The
    separators are replaced by spaces (never dropped: removing them would run
    two words together).

    Parameters
    ----------
    text : object
        The message to embed; coerced with ``str``.

    Returns
    -------
    str
        ``text`` with ``";"`` and ``","`` replaced by spaces and runs of
        whitespace collapsed.
    """
    cleaned = str(text)
    for separator in _NOTE_SEPARATORS:
        cleaned = cleaned.replace(separator, " ")
    return " ".join(cleaned.split())


def estimator_skip_note(
    named_skips: Iterable[tuple[str, str]], skip_classes: Container[str]
) -> str:
    """The ``drop_notes`` fragment describing sessions that produced no fit.

    One policy for every runner: the skip **classes** are summarized (a session
    that was simply too short is routine and needs no name), and only the classes
    in ``skip_classes`` -- the estimator refusing a design it cannot fit -- name
    the sessions they happened to, because those are rare and actionable.

    Parameters
    ----------
    named_skips : iterable of (str, str)
        ``(session_label, reason)`` for every session with no fit.
    skip_classes : container of str
        Reason classes whose sessions are named individually.

    Returns
    -------
    str
        ``""`` when nothing was skipped.
    """
    named_skips = list(named_skips)
    if not named_skips:
        return ""
    classes = sorted({str(reason).split(":")[0] for _label, reason in named_skips})
    named = [
        (label, reason)
        for label, reason in named_skips
        if str(reason).split(":")[0] in skip_classes
    ]
    parts = ["no fit: " + ", ".join(classes)]
    if named:
        parts.append(
            "estimator skipped "
            + ", ".join(
                f"{escape_note_field(label)} ({escape_note_field(reason)})"
                for label, reason in named
            )
        )
    return "; ".join(parts)


def records_from_common_cohort_provenance(
    prov, model: str, population: str, *, grain="stop"
):
    """Standardized cohort record from a ``_freeze_common_cohort`` provenance frame.

    ``compare_event_source_models.py`` fits its patch/rate/**choice** blocks on a frozen
    per-session complete-case cohort and records rich per-session provenance
    (``n_units_before``/``n_units_after``/``n_missing_units``/
    ``n_nonfinite_units``/``n_invalid_choice_units``/
    ``n_nonpositive_offset_units``/``n_zero_response_units``/``status``). This maps that
    native provenance onto the standardized schema so the choice cohort is on the same
    footing as the rate/content ladders. The block's up-front selection (e.g. the choice
    model's valid-replay-stop filter) is *definitional* — ``n_in`` is the post-selection
    candidate count. Returns ``None`` for an empty frame.
    """
    if prov is None or len(prov) == 0:
        return None

    def _col(c):
        return prov[c] if c in prov.columns else pd.Series(0, index=prov.index)

    n_in = int(_col("n_units_before").sum())
    n_own = int(
        (
            _col("n_missing_units")
            + _col("n_invalid_choice_units")
            + _col("n_nonpositive_offset_units")
        ).sum()
    )
    n_nonfinite = int(_col("n_nonfinite_units").sum())
    n_zero = int(_col("n_zero_response_units").sum())
    fit_mask = (
        prov["status"] == "fit"
        if "status" in prov.columns
        else pd.Series(False, index=prov.index)
    )
    n_modeled = int(prov.loc[fit_mask, "n_units_after"].sum()) if fit_mask.any() else 0
    animals = (
        int(prov.loc[fit_mask, "animal"].nunique())
        if fit_mask.any() and "animal" in prov.columns
        else 0
    )
    return cohort_record(
        model,
        population,
        grain,
        cohort_mode="common",
        n_in=n_in,
        n_modeled=n_modeled,
        sessions_in=len(prov),
        sessions_modeled=int(fit_mask.sum()),
        animals_modeled=animals,
        drop_notes=(
            "block common-cohort; up-front selection is definitional (n_in = "
            "post-selection); dropna + invalid-choice + nonpositive-offset -> own; "
            "size-gate / fit-failed sessions -> excluded"
        ),
        n_drop_own_predictor_missing=n_own,
        n_drop_nonfinite=n_nonfinite,
        n_drop_zero_response=n_zero,
        n_excluded_session=n_in - n_own - n_nonfinite - n_zero - n_modeled,
    )


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

    Reports ONLY the ``|coef| >= 10`` count (with the ``>= 5`` surviving-session
    requirement), surfaced as a standalone table so term-specific coefficient
    exclusions stay OUT of the cohort reconciliation. This is a backstop count, not
    the exclusion the aggregate is built on: ``contrast_aggregation`` drops a session
    for a term when the fitters named that term in ``separated_terms``, which is the
    effect-scale rule and catches unidentified terms this threshold misses. Read
    ``n_dropped`` in the audited aggregate as the authoritative per-term count;
    ``n_sessions_quasiseparated`` here is strictly the runaway-coefficient subset.

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
            rows.append(
                {
                    "model": model,
                    "population": pop,
                    "term": term,
                    "n_sessions_fit": 0,
                    "n_sessions_quasiseparated": 0,
                    "n_sessions_used": 0,
                    "n_animals_used": 0,
                }
            )
            continue
        present = sub[sub[ccol].notna()]
        quasi = present[ccol].abs() >= _QUASI_SEP_ABS
        used = present[~quasi]
        n_animals_used = (
            int(used["animal"].nunique()) if "animal" in used.columns else 0
        )
        rows.append(
            {
                "model": model,
                "population": pop,
                "term": term,
                "n_sessions_fit": int(len(present)),
                "n_sessions_quasiseparated": int(quasi.sum()),
                "n_sessions_used": int(len(used)),
                "n_animals_used": n_animals_used,
            }
        )
    return pd.DataFrame(rows, columns=list(TERM_DIAGNOSTICS_COLUMNS))
