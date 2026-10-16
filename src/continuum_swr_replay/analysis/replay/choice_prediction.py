"""Does replay content predict the next choice?

Builds a stop × arm choice dataset from the canonical tables, then
fits conditional logistic regression (or conditional Poisson with
binary response) to test whether arm-level replay content improves
prediction of the animal's next destination above a behavior-only
baseline.

The key predictor is ``replay_fraction`` — the mean ``arm_fraction``
across all events during a stop. This is the fraction of non-local
content on each arm, averaged over replay events within the stop.
It's known BEFORE the animal makes its next choice (replay happens
during immobility, before the next trip begins).

``is_next`` / ``next_arm`` do NOT enter as predictors — they're the
response variable. The model is strictly predictive, not
retrospective (:mod:`retrospective` handles the retrospective alignment).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import WELL_ARMS
from continuum_swr_replay.session_keys import SESSION_COLUMNS

from ._fit_diagnostics import (
    NUISANCE_TERMS,
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
)
from ._helpers import strict_mean
from .fitting import (
    CONDITIONAL_FIT_MAXITER,
    conditional_linear_predictor,
    within_group_full_rank,
)

logger = logging.getLogger(__name__)


# Leaf arms — the six arms that touch a well (one per well, since
# every well in the bandit graph is degree-1). The animal can only
# *choose* to visit a well, so these are the eligible alternatives in
# the conditional-logit choice set. Trunk arms 6/7/8 connect patch
# junctions and are transit-only — they can never be ``next_arm``,
# they have no ``reward_fraction_candidate_arm_last10_visits``, and ``arm_path_well_to_well``
# already documents that trunk arms cannot be stop endpoints. The
# learning-stage helpers use the same set under
# ``StopSummaryConfig.well_arms`` for "best arm" lookups.


def _require_session_unique_ids(
    df: pd.DataFrame, id_col: str, *, helper_name: str
) -> None:
    """Raise if ``id_col`` values are not unique across sessions.

    The canonical hashed event_id / stop_id columns mix session keys
    into the hash, so each ID maps to exactly one
    ``(animal, day, epoch)`` tuple. Helpers that do per-ID groupbys
    (softmax / mean / sum) silently produce wrong results — fractions
    summing above 1, dropped stops, cross-session averages — when
    that invariant is violated. This guard catches session-local IDs
    early with a clear message instead.

    No-op when the session columns aren't present (e.g. tests that
    pass minimal frames without animal/day/epoch).
    """
    missing = [c for c in SESSION_COLUMNS if c not in df.columns]
    if missing or id_col not in df.columns:
        return
    per_id_session_count = (
        df.groupby(id_col)[list(SESSION_COLUMNS)].nunique().max(axis=1)
    )
    bad = per_id_session_count[per_id_session_count > 1]
    if len(bad):
        sample = bad.index[: min(3, len(bad))].tolist()
        raise ValueError(
            f"{helper_name}: {id_col!r} must be globally unique across "
            f"sessions, but {len(bad)} value(s) appear in multiple "
            f"(animal, day, epoch) tuples (e.g. {sample}). Use the "
            "canonical hashed IDs from "
            "``analysis.replay.event_ids.hash_event_id`` / "
            "``hash_stop_id`` (or the ContFrag equivalents) before "
            "calling this helper."
        )


# ── Residualized replay ──


def predict_arm_fractions(
    event_arm_table: pd.DataFrame,
    fit_result: dict,
    predictors: list[str],
    event_col: str = "event_id",
    fraction_col: str = "arm_fraction",
) -> pd.DataFrame:
    """Compute model-predicted arm fractions and residuals per event-arm.

    Given a fitted content model (e.g. M5), computes
    ``softmax(X @ beta)`` within each event to get predicted fractions,
    then ``residual_frac = fraction_col - predicted_frac``.

    Parameters
    ----------
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).
    fit_result : dict
        Output of :func:`fitting.fit_conditional_poisson`. Must contain
        ``coef_{predictor}`` keys.
    predictors : list[str]
        Predictor column names used in the content model.
    event_col : str
        Event grouping column.
    fraction_col : str
        Column with observed arm fractions. Use ``"arm_fraction"``
        (default, MAP-based) or ``"arm_fraction_posterior"``
        (full-posterior-based).

    Returns
    -------
    pd.DataFrame
        Copy of *event_arm_table* with ``predicted_frac`` and
        ``residual_frac`` columns added.
    """
    eat = event_arm_table.copy()
    _require_session_unique_ids(eat, event_col, helper_name="predict_arm_fractions")

    # Event-wise NaN filtering matches ``fit_conditional_poisson``:
    # if any predictor (or the observed fraction) is NaN for any
    # arm in an event, drop the entire event from the prediction so
    # the softmax doesn't normalize over a partial choice set. The
    # event's rows still appear in the output, but with
    # ``predicted_frac`` and ``residual_frac`` set to NaN — that
    # keeps row alignment intact for downstream merges while
    # preventing residuals from being constructed against a
    # contaminated softmax.
    required_cols = list(predictors) + [fraction_col]
    required_values = eat[required_cols].to_numpy(dtype=float)
    invalid_row = ~np.isfinite(required_values).all(axis=1)
    events_with_nan = eat.loc[invalid_row, event_col].unique()
    is_dropped = eat[event_col].isin(events_with_nan)

    # A term the fit reports as unidentified is taken to its limit: a separated
    # term gives its ruled-out rows probability 0, a term constant within every
    # event contributes nothing. Any other non-finite coefficient raises.
    X = eat[predictors].values.astype(float)
    eta = conditional_linear_predictor(
        X, eat[event_col].to_numpy(), fit_result, list(predictors)
    )

    # Numerically stable softmax within each event
    eat["_eta"] = eta
    # Mask dropped events out of max/sum so the softmax of any
    # surviving event is computed only over its valid rows. (This
    # is equivalent to dropping events entirely; we keep them in
    # the frame so callers don't have to re-key on event_id.)
    eat.loc[is_dropped, "_eta"] = np.nan
    max_eta = eat.groupby(event_col)["_eta"].transform("max")
    eat["_shifted_exp"] = np.exp(eat["_eta"] - max_eta)
    sum_shifted = eat.groupby(event_col)["_shifted_exp"].transform("sum")
    eat["predicted_frac"] = eat["_shifted_exp"] / sum_shifted
    eat.loc[is_dropped, "predicted_frac"] = np.nan

    eat["residual_frac"] = eat[fraction_col] - eat["predicted_frac"]

    return eat.drop(columns=["_eta", "_shifted_exp"])


def build_residualized_choice_dataset(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    content_fit: dict,
    content_predictors: list[str],
    *,
    fraction_col: str = "arm_fraction",
) -> pd.DataFrame:
    """Build choice dataset with both raw and residualized replay.

    Fits ``predict_arm_fractions`` on the content model, aggregates
    residuals to the stop × arm level, then merges into the standard
    choice dataset from :func:`build_choice_dataset`.

    Parameters
    ----------
    event_table, event_arm_table, stop_table
        Canonical tables.
    content_fit : dict
        Fitted content model result (from ``fit_conditional_poisson``).
    content_predictors : list[str]
        Predictors used in the content model (e.g. M5 predictors).
    fraction_col : str
        Column with observed arm fractions. ``"arm_fraction"``
        (default) or ``"arm_fraction_posterior"``.

    Returns
    -------
    pd.DataFrame
        Choice dataset with ``replay_fraction`` (raw) and
        ``replay_fraction_residual`` (content-model residualized) columns.
    """
    eat_resid = predict_arm_fractions(
        event_arm_table, content_fit, content_predictors, fraction_col=fraction_col
    )

    # Aggregate residual_frac to stop × arm level. Use a NaN-aware
    # mean (``skipna=False``) so a stop containing ANY event whose
    # ``predict_arm_fractions`` dropped (NaN residual) propagates
    # NaN through to the stop-level value. A bare ``mean()`` would
    # silently average over the surviving events and reintroduce a
    # partial-event aggregate — the same contamination
    # ``predict_arm_fractions`` was just hardened against.
    stop_arm_resid = (
        eat_resid.groupby(["stop_id", "candidate_arm"])["residual_frac"]
        .apply(strict_mean)
        .rename("replay_fraction_residual")
        .reset_index()
    )

    choice_df = build_choice_dataset(
        event_table, event_arm_table, stop_table, fraction_col=fraction_col
    )
    if len(choice_df) == 0:
        return choice_df

    choice_df = choice_df.merge(
        stop_arm_resid[["stop_id", "candidate_arm", "replay_fraction_residual"]],
        on=["stop_id", "candidate_arm"],
        how="left",
    )
    # Do NOT ``fillna(0.0)`` — that fabricates a real zero from
    # missing data and defeats the event-wise NaN filter in
    # ``predict_arm_fractions``. Downstream ``fit_choice_model``
    # already drops any stop with a NaN required predictor, so
    # propagating NaN here is what makes the filter end-to-end.
    # Stops whose events were ALL filtered will have NaN
    # ``replay_fraction_residual`` for every arm and be dropped at
    # fit time.

    return choice_df


def build_choice_dataset(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    *,
    fraction_col: str = "arm_fraction",
) -> pd.DataFrame:
    """Build a stop × arm choice dataset for the choice models.

    For each stop that is explicitly labeled ``at_well``, has (a) at least
    one replay event and (b) a valid ``next_arm`` (known next destination),
    produces one row per candidate arm. The choice set is the well arms (leaf arms 0-5,
    one per well) minus ``occupancy_well_arm`` — the animal can only choose
    to visit a well, and consecutive pokes at the same port are never
    rewarded. Trunk arms (6/7/8) are transit-only — they connect
    patch junctions and can never be ``next_arm`` — so including them
    in the conditional-logit denominator would treat impossible
    destinations as unchosen alternatives, distorting the likelihood
    and coefficient interpretation.

    Parameters
    ----------
    event_table, event_arm_table, stop_table
        Canonical tables from :mod:`tables`.
    fraction_col : str
        Column with observed arm fractions. ``"arm_fraction"``
        (default, MAP-based) or ``"arm_fraction_posterior"``.

    Returns
    -------
    pd.DataFrame
        One row per (stop, candidate arm). Columns:

        - ``stop_id`` — grouping key for conditional logistic
        - ``candidate_arm`` — candidate arm ID
        - ``chosen`` — 1 if the animal chose this arm, 0 otherwise
        - ``replay_fraction`` — mean fraction across events in this stop
        - trial recency/novelty, recent reward fraction, reward-belief mean,
          scheduled reward probability, graph distance, and same-patch status
          — behavioral baseline predictors (from the last event in
          the stop)
        - ``n_selected_events`` — number of non-local replay events in this stop
        - ``occupancy_state``, ``at_well`` — explicit cohort provenance
        - Session identifiers (``animal``, ``session_date``, ``epoch``)
    """
    # Choice analysis is the first downstream consumer that intentionally
    # selects the at-well cohort. ``occupancy_well_arm`` is additionally
    # required so the candidate exclusion below removes exactly one arm from
    # the WELL_ARMS candidate set
    # (5 alternatives after the well-arm filter); when occupancy is
    # NaN the comparison evaluates True for every arm (IEEE 754),
    # silently inflating the choice set to all 6 well arms for those
    # stops.
    # Both inputs are aggregated by stop_id later, which produces
    # session-mixing means / drops if stop_id repeats across sessions.
    _require_session_unique_ids(
        stop_table, "stop_id", helper_name="build_choice_dataset"
    )
    _require_session_unique_ids(
        event_arm_table, "stop_id", helper_name="build_choice_dataset"
    )

    required_location = ["occupancy_state", "at_well"]
    missing_location = [c for c in required_location if c not in stop_table.columns]
    if missing_location:
        raise KeyError(
            "build_choice_dataset requires explicit stop location columns; "
            f"missing {missing_location}. Regenerate the canonical tables."
        )

    replay_stops = stop_table["has_replay"]
    # ``at_well`` is NA where the stop's location is unobserved, so the
    # fillna keeps those out rather than letting NA read as True.
    at_well_stops = stop_table["at_well"].eq(True).fillna(False)
    has_replay = (
        replay_stops
        & at_well_stops
        & stop_table["next_arm"].notna()
        & stop_table["occupancy_well_arm"].notna()
    )
    valid_stops = stop_table.loc[has_replay].copy()
    # The cohort shrinks with the immobility threshold and the trial
    # boundaries; log it so a changed N is traceable to its cause.
    logger.info(
        "build_choice_dataset: %d/%d stops with replay; %d retained after "
        "at-well (%d dropped) + finite next/current arm (%d dropped)",
        int(replay_stops.sum()),
        len(stop_table),
        len(valid_stops),
        int((replay_stops & ~at_well_stops).sum()),
        int((replay_stops & at_well_stops & ~has_replay).sum()),
    )

    if len(valid_stops) == 0:
        return pd.DataFrame()

    # Aggregate replay content to stop × arm: mean fraction
    # across events within each stop.
    eat = event_arm_table[
        [
            "event_id",
            "stop_id",
            "candidate_arm",
            fraction_col,
            "visit_recency_trials",
            "is_candidate_arm_never_visited",
            "reward_count_candidate_arm_last10_known_trials",
            "reward_fraction_candidate_arm_last10_visits",
            "reward_belief_mean_candidate_arm_preoutcome",
            "task_reward_probability_candidate_arm",
            "distance_to_candidate_arm_cm",
            "candidate_arm_distance_rank",
            "is_same_patch_as_current",
        ]
    ].copy()

    # ``strict_mean`` keeps a stop with any NaN event-level
    # ``arm_fraction`` surfaced as NaN — the default ``skipna=True``
    # mean would silently emit the surviving event's value, and the
    # chosen-arm sanity check below cannot recover the missingness
    # once the NaN has been collapsed away.
    stop_arm = (
        eat.groupby(["stop_id", "candidate_arm"])
        .agg(
            replay_fraction=(fraction_col, strict_mean),
            visit_recency_trials=("visit_recency_trials", "last"),
            is_candidate_arm_never_visited=("is_candidate_arm_never_visited", "last"),
            reward_count_candidate_arm_last10_known_trials=("reward_count_candidate_arm_last10_known_trials", "last"),
            reward_fraction_candidate_arm_last10_visits=("reward_fraction_candidate_arm_last10_visits", "last"),
            reward_belief_mean_candidate_arm_preoutcome=("reward_belief_mean_candidate_arm_preoutcome", "last"),
            task_reward_probability_candidate_arm=("task_reward_probability_candidate_arm", "last"),
            distance_to_candidate_arm_cm=("distance_to_candidate_arm_cm", "last"),
            candidate_arm_distance_rank=("candidate_arm_distance_rank", "last"),
            is_same_patch_as_current=("is_same_patch_as_current", "last"),
        )
        .reset_index()
    )

    # SWR-overlap-conditional replay fractions per (stop, arm). Each is
    # the mean ``arm_fraction`` restricted to events where ``is_swr`` is
    # True (or False) within the stop. A (stop, arm) with no SWR-overlap
    # events at all gets NaN for ``replay_fraction_swr`` — different from
    # "0 SWR replay on this arm despite SWR events," which would be 0;
    # honest NaN lets the fitter drop those stops from any model using
    # the split predictors.
    if "is_swr" in event_arm_table.columns:
        eat_swr = event_arm_table[["stop_id", "candidate_arm", fraction_col, "is_swr"]].copy()
        swr_frac = (
            eat_swr[eat_swr["is_swr"]]
            .groupby(["stop_id", "candidate_arm"])[fraction_col]
            .apply(strict_mean)
            .rename("replay_fraction_swr")
            .reset_index()
        )
        non_swr_frac = (
            eat_swr[~eat_swr["is_swr"]]
            .groupby(["stop_id", "candidate_arm"])[fraction_col]
            .apply(strict_mean)
            .rename("replay_fraction_non_swr")
            .reset_index()
        )
        stop_arm = stop_arm.merge(swr_frac, on=["stop_id", "candidate_arm"], how="left")
        stop_arm = stop_arm.merge(non_swr_frac, on=["stop_id", "candidate_arm"], how="left")

    # Join with stop-level info.
    stop_info = valid_stops[
        [
            "stop_id",
            "animal",
            "session_date",
            "epoch",
            "occupancy_state",
            "at_well",
            "occupancy_well_arm",
            "next_arm",
            "n_selected_events",
            "is_rewarded_outcome",
        ]
    ].copy()

    choice_df = stop_arm.merge(stop_info, on="stop_id", how="inner")

    # Restrict the choice set to well arms (leaf arms 0-5) minus
    # current_arm. Trunk arms 6/7/8 are transit-only and can never
    # be ``next_arm``; including them as unchosen alternatives in
    # the conditional logit would distort the likelihood. Filter on
    # the canonical ``WELL_ARMS`` constant rather than slicing the full
    # arm axis, so a future track-graph change can extend the
    # eligible-destination set in one place.
    choice_df = choice_df[
        choice_df["candidate_arm"].isin(WELL_ARMS)
        & (choice_df["candidate_arm"] != choice_df["occupancy_well_arm"])
    ].reset_index(drop=True)

    # Response: did the animal choose this arm?
    choice_df["chosen"] = (choice_df["candidate_arm"] == choice_df["next_arm"]).astype(int)

    # Sanity: exactly one chosen arm per stop.
    chosen_per_stop = choice_df.groupby("stop_id")["chosen"].sum()
    valid_stop_ids = chosen_per_stop[chosen_per_stop == 1].index
    choice_df = choice_df[choice_df["stop_id"].isin(valid_stop_ids)].reset_index(
        drop=True
    )

    return choice_df


def fit_choice_model(
    choice_df: pd.DataFrame,
    predictors: list[str],
    stop_col: str = "stop_id",
) -> dict | None:
    """Fit conditional logistic on the choice dataset.

    Uses ``statsmodels.ConditionalLogit`` — same conditional
    likelihood framework as the content model's conditional Poisson,
    but with binary response (chose / didn't choose).

    Parameters
    ----------
    choice_df : pd.DataFrame
        Output of :func:`build_choice_dataset`.
    predictors : list[str]
        Arm-varying predictors (e.g., ``["replay_fraction"]``).
    stop_col : str
        Grouping variable.

    Returns
    -------
    dict | None
        Coefficients, SE, p-values, fit statistics. None on failure.
    """
    from statsmodels.discrete.conditional_models import (
        ConditionalLogit,
    )

    required = ["chosen", stop_col] + predictors
    missing = [c for c in required if c not in choice_df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return None

    # Drop the whole stop if any required column is NaN on any of its
    # rows. A bare ``.dropna()`` would silently shrink the choice set
    # for that stop (e.g. 9 candidate arms → 7), so the conditional
    # logit's normalization runs over a partial set and "chosen" can
    # be ambiguous (1 of 7 vs 1 of 9). Mirrors the event-wise NaN
    # filtering in ``fit_conditional_poisson``.
    df = choice_df[required].copy()
    numeric = df[["chosen", *predictors]].to_numpy(dtype=float)
    invalid = ~np.isfinite(numeric).all(axis=1) | df[stop_col].isna().to_numpy()
    bad_stops = df.loc[invalid, stop_col].unique()
    df = df[~df[stop_col].isin(bad_stops)].reset_index(drop=True)

    # Drop stops where no arm is chosen (shouldn't happen after
    # build_choice_dataset, but defensive).
    chosen_per_stop = df.groupby(stop_col)["chosen"].sum()
    valid = chosen_per_stop[chosen_per_stop == 1].index
    df = df[df[stop_col].isin(valid)].reset_index(drop=True)

    n_stops = df[stop_col].nunique()
    if n_stops < 2 * len(predictors):
        return None

    y = df["chosen"].values.astype(float)
    X = df[predictors].values.astype(float)
    groups = df[stop_col].values

    # Reject designs the conditional logit cannot identify: globally
    # collinear predictors AND predictors constant within every stop (no
    # within-group variation), which fit with finite-but-meaningless
    # coefficients/SEs/AIC.
    if not within_group_full_rank(X, groups):
        logger.warning(
            "choice ConditionalLogit: rank-deficient or within-group-constant "
            "design; not fitting %s",
            predictors,
        )
        return None

    try:
        with capture_fit_warnings() as caught:
            model = ConditionalLogit(y, X, groups=groups)
            result = model.fit(disp=0, maxiter=CONDITIONAL_FIT_MAXITER)
    except Exception:
        logger.warning("ConditionalLogit failed for %s", predictors)
        return None
    predictor_sd = X.std(axis=0)
    status = fit_status(
        result,
        "conditional",
        caught=caught,
        term_names=predictors,
        nuisance_terms=NUISANCE_TERMS,
    )

    coefs = dict(zip(predictors, result.params, strict=True))
    bse = dict(zip(predictors, result.bse, strict=True))
    pvals = dict(zip(predictors, result.pvalues, strict=True))

    return {
        **{f"coef_{k}": float(v) for k, v in coefs.items()},
        **{f"bse_{k}": float(v) for k, v in bse.items()},
        **{f"pval_{k}": float(v) for k, v in pvals.items()},
        "n_stops": n_stops,
        "n_rows": len(df),
        "n_params": len(predictors),
        "llf": float(result.llf),
        **{
            f"sd_{name}": float(sd)
            for name, sd in zip(predictors, predictor_sd, strict=True)
        },
        "converged": status["converged"],
        "separated": status["separated"],
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        "separated_terms": "|".join(status["separated_terms"]),
        "n_convergence_warnings": count_fit_quality_warnings(caught),
    }
