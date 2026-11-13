"""Model-light estimands for reward-belief uncertainty and replay content.

The content model conditions on a replay event and asks which arm receives its
decoded content.  This module supplies a complementary, ground-up description at
the behavioral trial grain.  It deliberately distinguishes two quantities:

``*_leaf_share``
    Relative allocation among the six reward-well arms, conditional on a trial
    containing decoded replay content.

``*_event_share``
    The reward well's fraction of all decoded replay content in the trial; path
    and trunk content remains in the denominator.

``*_rate``
    Absolute arm-specific decoded content per immobility second, retaining trials
    with no replay as zeros.

The uncertainty state is the pre-outcome Beta-posterior variance.  Because that
state is broadcast to every stop in a trial, stops and events are aggregated to
one trial x physical-well row before uncertainty quantiles are assigned.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior._fit_diagnostics import (
    flagged_nonconverged,
    flagged_separated,
)
from continuum_swr_replay.analysis.shared_statistics import (
    session_median_statistics,
    signed_rank_p,
)
from continuum_swr_replay.data_loaders.constants import LEAF_ARMS, port_columns
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..covariate_timing import assert_predictor_timing

LOGGER = logging.getLogger(__name__)

UNCERTAINTY_COLUMNS: tuple[str, ...] = tuple(port_columns("reward_belief_variance"))
RESPONSE_COLUMNS: Mapping[str, str] = {
    "arm_count_hard": "hard_count",
    "arm_count_soft": "soft_count",
    "arm_posterior_mass": "posterior_mass",
}
SHARE_COLUMNS: Mapping[str, str] = {
    "hard_count": "hard_leaf_share",
    "soft_count": "soft_leaf_share",
    "posterior_mass": "posterior_leaf_share",
}
EVENT_SHARE_COLUMNS: Mapping[str, str] = {
    "hard_count": "hard_event_share",
    "soft_count": "soft_event_share",
    "posterior_mass": "posterior_event_share",
}
LEAF_FRACTION_COLUMNS: Mapping[str, str] = {
    "hard_count": "hard_leaf_fraction",
    "soft_count": "soft_leaf_fraction",
    "posterior_mass": "posterior_leaf_fraction",
}


def _require_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    duplicated = frame.duplicated(columns, keep=False)
    if duplicated.any():
        raise ValueError(
            f"{label} must be unique on {columns}; found "
            f"{int(duplicated.sum())} duplicated rows"
        )


def _session_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["animal"].astype(str)
        + "/"
        + frame["session_date"].astype(str)
        + "/"
        + frame["epoch"].astype(str)
    )


def _assert_constant(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
    description: str,
) -> None:
    counts = frame.groupby(group_columns, observed=True)[value_columns].nunique(
        dropna=False
    )
    bad = counts.gt(1).any(axis=1)
    if bad.any():
        raise ValueError(
            f"{description} must be constant within "
            f"{' x '.join(group_columns)}; found {int(bad.sum())} inconsistent groups"
        )


def assign_within_arm_quantiles(
    frame: pd.DataFrame,
    *,
    n_quantiles: int = 4,
    uncertainty_col: str = "reward_belief_variance_candidate_arm_preoutcome",
    output_col: str = "uncertainty_quartile",
) -> pd.DataFrame:
    """Assign tie-preserving uncertainty quantiles within session x well.

    A midpoint empirical CDF gives identical uncertainty values identical bins.
    Groups with no uncertainty variation are explicitly marked missing rather
    than being assigned an arbitrary middle quantile.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    require_columns(
        frame,
        {"session_id", "candidate_arm", uncertainty_col},
        "uncertainty frame",
    )

    out = frame.copy()

    def _midpoint_quantile(values: pd.Series) -> pd.Series:
        finite = values.notna()
        result = pd.Series(np.nan, index=values.index, dtype=float)
        observed = values[finite]
        if observed.nunique() < 2:
            return result
        ranks = observed.rank(method="average")
        midpoint_ecdf = (ranks - 0.5) / len(observed)
        bins = np.floor(midpoint_ecdf * n_quantiles).astype(int) + 1
        result.loc[observed.index] = bins.clip(1, n_quantiles).astype(float)
        return result

    out[output_col] = (
        out.groupby(["session_id", "candidate_arm"], observed=True, group_keys=False)[
            uncertainty_col
        ]
        .apply(_midpoint_quantile)
        .reindex(out.index)
    )
    return out


def build_trial_leaf_table(
    event_arm_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    qvals_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Build one row per mapped behavioral trial x reward-well arm.

    Events are first summed within stop x arm and then across all stops assigned
    to the same trial.  Zero-event trials are introduced from ``stop_table`` and
    ``qvals_table`` before response normalization, so absolute content rates
    have genuine zeros rather than a replay-selected cohort.

    Returns
    -------
    trial_leaf : pandas.DataFrame
        Six rows per mapped trial.  Contains hard, soft, and posterior response
        masses, full-event and leaf-conditional shares, absolute rates, and
        uncertainty quartiles assigned from all mapped trial opportunities.
    audit : dict
        Cohort and coverage counts used to verify the joins.
    """
    event_required = {
        *SESSION_COLUMNS,
        "event_id",
        "stop_id",
        "candidate_arm",
        *RESPONSE_COLUMNS.keys(),
    }
    stop_required = {
        *SESSION_COLUMNS,
        "stop_id",
        "stop_idx",
        "immobility_duration_s",
    }
    qval_required = {
        "stop_id",
        "epoch",
        "trial_number_by_epoch",
        "chosen_arm",
        "trial_reward",
        "in_poke",
        "pump_on_at_bout_mid",
        *UNCERTAINTY_COLUMNS,
    }
    require_columns(event_arm_table, event_required, "event_arm_table")
    require_columns(stop_table, stop_required, "stop_table")
    require_columns(qvals_table, qval_required, "qvals_table")
    _require_unique(event_arm_table, ["event_id", "candidate_arm"], "event_arm_table")
    _require_unique(stop_table, ["stop_id"], "stop_table")
    _require_unique(qvals_table, ["stop_id"], "qvals_table")

    stop_columns = [*SESSION_COLUMNS, "stop_id", "stop_idx", "immobility_duration_s"]
    qval_columns = [
        "stop_id",
        "epoch",
        "trial_number_by_epoch",
        "chosen_arm",
        "trial_reward",
        "in_poke",
        "pump_on_at_bout_mid",
        *UNCERTAINTY_COLUMNS,
    ]
    qvals = qvals_table[qval_columns].rename(columns={"epoch": "qvals_epoch"})
    mapped_stops_all = stop_table[stop_columns].merge(
        qvals,
        on="stop_id",
        how="inner",
        validate="one_to_one",
    )
    epoch_mismatch = mapped_stops_all["epoch"].astype(str) != mapped_stops_all[
        "qvals_epoch"
    ].astype(str)
    if epoch_mismatch.any():
        raise ValueError(
            "stop_table and qvals_table epoch labels disagree for "
            f"{int(epoch_mismatch.sum())} mapped stops"
        )
    nonpositive_exposure = mapped_stops_all["immobility_duration_s"] <= 0
    zero_exposure_stop_ids = set(mapped_stops_all.loc[nonpositive_exposure, "stop_id"])
    event_stop_ids = set(event_arm_table["stop_id"])
    invalid_event_stops = zero_exposure_stop_ids & event_stop_ids
    if invalid_event_stops:
        raise ValueError(
            "mapped stops with replay content must have positive immobility_time; "
            f"found {len(invalid_event_stops)} invalid stops"
        )
    # A zero-duration, zero-event stop carries no exposure and therefore no
    # opportunity for the absolute-rate estimand.  Exclude it explicitly and
    # record the exclusion rather than dividing by zero or treating it as a
    # negative replay observation.
    mapped_stops = mapped_stops_all.loc[~nonpositive_exposure].copy()
    if mapped_stops.empty:
        raise ValueError("no positive-exposure stops remain after q-value mapping")
    mapped_stops["session_id"] = _session_id(mapped_stops)

    long_stops = mapped_stops.melt(
        id_vars=[
            *SESSION_COLUMNS,
            "session_id",
            "stop_id",
            "stop_idx",
            "immobility_duration_s",
            "trial_number_by_epoch",
            "chosen_arm",
            "trial_reward",
            "in_poke",
            "pump_on_at_bout_mid",
        ],
        value_vars=list(UNCERTAINTY_COLUMNS),
        var_name="uncertainty_source",
        value_name="reward_belief_variance_candidate_arm_preoutcome",
    )
    long_stops["candidate_arm"] = (
        long_stops["uncertainty_source"].str.extract(r"(\d+)$")[0].astype(int)
    )
    long_stops = long_stops.drop(columns="uncertainty_source")

    leaf_events = event_arm_table[
        event_arm_table["candidate_arm"].isin(LEAF_ARMS)
    ].copy()
    content_by_stop_arm = (
        leaf_events.groupby(["stop_id", "candidate_arm"], observed=True, as_index=False)
        .agg(
            **{output: (source, "sum") for source, output in RESPONSE_COLUMNS.items()},
            n_events=("event_id", "nunique"),
        )
        .astype(dict.fromkeys(RESPONSE_COLUMNS.values(), float))
    )
    _require_unique(
        content_by_stop_arm, ["stop_id", "candidate_arm"], "content summary"
    )
    content_by_stop_total = (
        event_arm_table.groupby("stop_id", observed=True, as_index=False)
        .agg(
            **{
                f"{output}_total": (source, "sum")
                for source, output in RESPONSE_COLUMNS.items()
            }
        )
        .astype(
            dict.fromkeys(
                (f"{output}_total" for output in RESPONSE_COLUMNS.values()), float
            )
        )
    )
    _require_unique(content_by_stop_total, ["stop_id"], "total content summary")

    stop_leaf = long_stops.merge(
        content_by_stop_arm,
        on=["stop_id", "candidate_arm"],
        how="left",
        validate="one_to_one",
    )
    stop_leaf = stop_leaf.merge(
        content_by_stop_total,
        on="stop_id",
        how="left",
        validate="many_to_one",
    )
    response_names = list(RESPONSE_COLUMNS.values())
    total_response_names = [f"{name}_total" for name in response_names]
    stop_leaf[response_names + total_response_names] = (
        stop_leaf[response_names + total_response_names].fillna(0.0).astype(float)
    )
    stop_leaf["n_events"] = stop_leaf["n_events"].fillna(0).astype(int)
    _assert_constant(
        stop_leaf,
        ["stop_id"],
        ["n_events"],
        "event count",
    )
    _assert_constant(
        stop_leaf,
        ["session_id", "trial_number_by_epoch", "candidate_arm"],
        ["reward_belief_variance_candidate_arm_preoutcome"],
        "belief uncertainty",
    )
    _assert_constant(
        stop_leaf,
        ["session_id", "trial_number_by_epoch"],
        ["chosen_arm", "trial_reward"],
        "trial context",
    )

    trial_leaf = (
        stop_leaf.groupby(
            [
                *SESSION_COLUMNS,
                "session_id",
                "trial_number_by_epoch",
                "candidate_arm",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            reward_belief_variance_candidate_arm_preoutcome=(
                "reward_belief_variance_candidate_arm_preoutcome",
                "first",
            ),
            chosen_arm=("chosen_arm", "first"),
            trial_reward=("trial_reward", "first"),
            any_in_poke=("in_poke", "max"),
            any_rewarding=("pump_on_at_bout_mid", "max"),
            immobility_duration_s=("immobility_duration_s", "sum"),
            n_stops=("stop_id", "nunique"),
            n_events=("n_events", "sum"),
            hard_count=("hard_count", "sum"),
            soft_count=("soft_count", "sum"),
            posterior_mass=("posterior_mass", "sum"),
            hard_count_total=("hard_count_total", "sum"),
            soft_count_total=("soft_count_total", "sum"),
            posterior_mass_total=("posterior_mass_total", "sum"),
        )
        .sort_values([*SESSION_COLUMNS, "trial_number_by_epoch", "candidate_arm"])
        .reset_index(drop=True)
    )
    if (
        not trial_leaf.groupby(["session_id", "trial_number_by_epoch"], observed=True)[
            "candidate_arm"
        ]
        .nunique()
        .eq(len(LEAF_ARMS))
        .all()
    ):
        raise ValueError("every mapped trial must contain exactly six reward-well rows")
    _assert_constant(
        trial_leaf,
        ["session_id", "trial_number_by_epoch"],
        ["immobility_duration_s", "n_stops", "n_events", "chosen_arm", "trial_reward"],
        "trial aggregate",
    )
    _assert_constant(
        trial_leaf,
        ["session_id", "trial_number_by_epoch"],
        total_response_names,
        "trial total content",
    )

    trial_leaf["has_replay"] = trial_leaf["n_events"] > 0
    trial_leaf["is_chosen"] = trial_leaf["candidate_arm"] == trial_leaf["chosen_arm"]
    trial_keys = ["session_id", "trial_number_by_epoch"]
    for base in response_names:
        denominator = trial_leaf.groupby(trial_keys, observed=True)[base].transform(
            "sum"
        )
        share = np.full(len(trial_leaf), np.nan, dtype=float)
        np.divide(
            trial_leaf[base].to_numpy(dtype=float),
            denominator.to_numpy(dtype=float),
            out=share,
            where=denominator.to_numpy(dtype=float) > 0,
        )
        trial_leaf[SHARE_COLUMNS[base]] = share
        total = trial_leaf[f"{base}_total"].to_numpy(dtype=float)
        event_share = np.full(len(trial_leaf), np.nan, dtype=float)
        np.divide(
            trial_leaf[base].to_numpy(dtype=float),
            total,
            out=event_share,
            where=total > 0,
        )
        trial_leaf[EVENT_SHARE_COLUMNS[base]] = event_share
        leaf_fraction = np.full(len(trial_leaf), np.nan, dtype=float)
        np.divide(
            denominator.to_numpy(dtype=float),
            total,
            out=leaf_fraction,
            where=total > 0,
        )
        trial_leaf[LEAF_FRACTION_COLUMNS[base]] = leaf_fraction
        trial_leaf[f"{base}_rate"] = trial_leaf[base].to_numpy(
            dtype=float
        ) / trial_leaf["immobility_duration_s"].to_numpy(dtype=float)

    trial_leaf = assign_within_arm_quantiles(trial_leaf)

    mapped_stop_ids = set(mapped_stops_all["stop_id"])
    event_rows = event_arm_table.drop_duplicates("event_id")
    mapped_events = event_rows["stop_id"].isin(mapped_stop_ids)
    total_posterior_mass = float(event_arm_table["arm_posterior_mass"].sum())
    mapped_posterior_mass = float(
        event_arm_table.loc[
            event_arm_table["stop_id"].isin(mapped_stop_ids), "arm_posterior_mass"
        ].sum()
    )
    audit: dict[str, float | int] = {
        "n_input_stops": int(len(stop_table)),
        "n_qvals_stops": int(len(qvals_table)),
        "n_mapped_stops": int(len(mapped_stops_all)),
        "n_zero_exposure_mapped_stops": int(nonpositive_exposure.sum()),
        "n_positive_exposure_mapped_stops": int(len(mapped_stops)),
        "n_mapped_trials": int(
            trial_leaf[["session_id", "trial_number_by_epoch"]]
            .drop_duplicates()
            .shape[0]
        ),
        "n_input_events": int(event_rows["event_id"].nunique()),
        "n_mapped_events": int(mapped_events.sum()),
        "n_unmapped_events": int((~mapped_events).sum()),
        "stop_qvals_coverage": float(len(mapped_stops_all) / len(stop_table)),
        "event_qvals_coverage": float(mapped_events.mean()),
        "posterior_mass_qvals_coverage": float(
            mapped_posterior_mass / total_posterior_mass
            if total_posterior_mass > 0
            else np.nan
        ),
    }
    return trial_leaf, audit


def hierarchical_quartile_profile(
    trial_leaf: pd.DataFrame,
    *,
    numerator_col: str,
    denominator_col: str | None = None,
    scope: Literal["all", "unchosen"] = "all",
    quartile_col: str = "uncertainty_quartile",
) -> dict[str, pd.DataFrame]:
    """Summarize a quartile profile with arms, sessions, and animals balanced.

    If ``denominator_col`` is supplied, each session x arm x quartile estimate is
    a ratio of sums.  This is the appropriate exposure-weighted aggregation for
    absolute content rates.  Otherwise the estimate is the arithmetic mean of
    trial-level values, as used for conditional content shares.
    """
    required = {
        "animal",
        "session_id",
        "candidate_arm",
        "reward_belief_variance_candidate_arm_preoutcome",
        "is_chosen",
        quartile_col,
        numerator_col,
    }
    if denominator_col is not None:
        required.add(denominator_col)
    require_columns(trial_leaf, required, "trial_leaf")
    if scope not in {"all", "unchosen"}:
        raise ValueError("scope must be 'all' or 'unchosen'")

    data = trial_leaf.copy()
    if scope == "unchosen":
        data = data[~data["is_chosen"].astype(bool)]
    finite = data[quartile_col].notna() & np.isfinite(data[numerator_col])
    if denominator_col is not None:
        finite &= np.isfinite(data[denominator_col]) & data[denominator_col].gt(0)
    data = data[finite]
    if data.empty:
        empty = pd.DataFrame()
        return {
            "candidate_arm": empty,
            "session": empty,
            "animal": empty,
            "overall": empty,
        }

    arm_keys = ["animal", "session_id", "candidate_arm", quartile_col]
    grouped = data.groupby(arm_keys, observed=True)
    if denominator_col is None:
        arm_profile = grouped.agg(
            value=(numerator_col, "mean"),
            reward_belief_variance_candidate_arm_preoutcome=(
                "reward_belief_variance_candidate_arm_preoutcome",
                "mean",
            ),
            n_trials=(numerator_col, "size"),
        ).reset_index()
    else:
        sums = grouped.agg(
            numerator=(numerator_col, "sum"),
            denominator=(denominator_col, "sum"),
            reward_belief_variance_candidate_arm_preoutcome=(
                "reward_belief_variance_candidate_arm_preoutcome",
                "mean",
            ),
            n_trials=(numerator_col, "size"),
        ).reset_index()
        sums["value"] = sums["numerator"] / sums["denominator"]
        arm_profile = sums.drop(columns=["numerator", "denominator"])

    session_profile = arm_profile.groupby(
        ["animal", "session_id", quartile_col], observed=True, as_index=False
    ).agg(
        value=("value", "mean"),
        reward_belief_variance_candidate_arm_preoutcome=(
            "reward_belief_variance_candidate_arm_preoutcome",
            "mean",
        ),
        n_arms=("candidate_arm", "nunique"),
        n_trials=("n_trials", "sum"),
    )
    animal_profile = session_profile.groupby(
        ["animal", quartile_col], observed=True, as_index=False
    ).agg(
        value=("value", "mean"),
        reward_belief_variance_candidate_arm_preoutcome=(
            "reward_belief_variance_candidate_arm_preoutcome",
            "mean",
        ),
        n_sessions=("session_id", "nunique"),
    )
    overall_profile = animal_profile.groupby(
        quartile_col, observed=True, as_index=False
    ).agg(
        value=("value", "mean"),
        sem=("value", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        reward_belief_variance_candidate_arm_preoutcome=(
            "reward_belief_variance_candidate_arm_preoutcome",
            "mean",
        ),
        n_animals=("animal", "nunique"),
    )
    return {
        "candidate_arm": arm_profile,
        "session": session_profile,
        "animal": animal_profile,
        "overall": overall_profile,
    }


def paired_quartile_contrasts(
    arm_profile: pd.DataFrame,
    *,
    low: int = 1,
    high: int = 4,
    quartile_col: str = "uncertainty_quartile",
) -> dict[str, pd.DataFrame]:
    """Compute high-minus-low contrasts, paired within physical well.

    The contrast is averaged over a session's wells, then over an animal's
    sessions; ``overall`` is the mean over animals (``contrast``), its standard
    error (``sem``), the animals (``n_animals``) and the animals whose own
    contrast is above zero (``n_animals_positive``).
    """
    require_columns(
        arm_profile,
        {"animal", "session_id", "candidate_arm", quartile_col, "value"},
        "arm profile",
    )
    index = ["animal", "session_id", "candidate_arm"]
    wide = arm_profile.pivot_table(
        index=index,
        columns=quartile_col,
        values="value",
        aggfunc="first",
    )
    if low not in wide.columns or high not in wide.columns:
        empty = pd.DataFrame()
        return {
            "candidate_arm": empty,
            "session": empty,
            "animal": empty,
            "overall": empty,
        }
    arm = (wide[high] - wide[low]).dropna().rename("contrast").reset_index()
    session = arm.groupby(["animal", "session_id"], observed=True, as_index=False).agg(
        contrast=("contrast", "mean"), n_arms=("candidate_arm", "nunique")
    )
    animal = session.groupby("animal", observed=True, as_index=False).agg(
        contrast=("contrast", "mean"), n_sessions=("session_id", "nunique")
    )
    overall = pd.DataFrame(
        {
            "contrast": [animal["contrast"].mean()],
            "sem": [animal["contrast"].std(ddof=1) / np.sqrt(len(animal))],
            "n_animals": [animal["animal"].nunique()],
            "n_animals_positive": [int((animal["contrast"] > 0).sum())],
        }
    )
    return {
        "candidate_arm": arm,
        "session": session,
        "animal": animal,
        "overall": overall,
    }


def summarize_mu_coefficients(mu_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare transparent all-session and fit-quality-clean Mu summaries."""
    predictor = "reward_belief_variance_candidate_arm_preoutcome_z"
    coefficient = f"coef_{predictor}"
    standard_error = f"bse_{predictor}"
    required = {
        *SESSION_COLUMNS,
        coefficient,
        standard_error,
        "converged",
        "separated",
    }
    require_columns(mu_results, required, "Mu results")
    _require_unique(mu_results, list(SESSION_COLUMNS), "Mu results")
    sessions = mu_results.copy()
    sessions["session_id"] = _session_id(sessions)
    sessions["fit_quality"] = np.where(
        # A fit with no convergence flag has not converged; one with no
        # separation flag is not separated.
        ~flagged_nonconverged(sessions) & ~flagged_separated(sessions),
        "clean",
        "flagged",
    )

    summaries = []
    overall_rows = []
    for cohort, subset in [
        ("all", sessions),
        ("clean", sessions[sessions["fit_quality"] == "clean"]),
    ]:
        animal = (
            subset.groupby("animal", observed=True, as_index=False)
            .agg(
                coefficient=(coefficient, "median"),
                n_sessions=("session_id", "nunique"),
            )
            .assign(cohort=cohort)
        )
        summaries.append(animal)
        summary = session_median_statistics(subset[coefficient], subset["animal"])
        overall_rows.append(
            {
                "cohort": cohort,
                "coefficient": summary["animal_median"],
                "n_animals": summary["n_animals"],
            }
        )
    animal_summary = pd.concat(summaries, ignore_index=True)
    overall = pd.DataFrame(overall_rows)
    overall["relative_weight"] = np.exp(overall["coefficient"])
    return {"session": sessions, "animal": animal_summary, "overall": overall}


HORSERACE_PREDICTORS: tuple[str, ...] = (
    "reward_belief_variance_candidate_arm_preoutcome_z",  # per-leaf Beta belief variance (pre-standardized in the table)
    "reward_belief_mean_candidate_arm_preoutcome",  # beta belief value (z-scored here)
    "distance_to_candidate_arm_cm",  # geometry parallel (z-scored here)
    "visit_recency_trials",  # recency parallel (z-scored here)
)


def _scaled_condition_number(design: np.ndarray) -> float:
    """Condition number of the column-scaled design.

    Scaling each column to unit norm isolates how close the columns are to
    linearly dependent, rather than reporting their arbitrary relative units,
    and is the form the conventional 10/30 bands refer to. Shared definition
    with :func:`analysis.behavior.fitting._compute_conditional_diagnostics`.
    """
    norms = np.linalg.norm(design, axis=0)
    scale = np.where(norms > 0, norms, 1.0)
    return float(np.linalg.cond(design / scale))


def within_arm_horserace(
    event_arm: pd.DataFrame,
    *,
    predictors: tuple[str, ...] = HORSERACE_PREDICTORS,
    response: str = "arm_fraction",
    min_rows: int = 60,
) -> pd.DataFrame:
    """Within-arm (arm fixed-effects) horse-race for off-current replay content.

    Asks whether the content an event places on a **non-current** arm tracks that
    arm's belief *uncertainty* rather than its value, distance, or visit recency.
    This is the exploit-vs-explore extension to the quartile-profile estimands
    above; it operates on the **event x arm** table (one row per non-local event x
    arm), NOT the trial-leaf grain of the rest of this module.

    The routine restricts to non-current leaf arms and standardizes each
    continuous predictor **once, globally** over that pooled subset (a name already
    ending in ``_z`` is taken as pre-standardized) — so coefficients are per 1
    *global* SD, comparable across predictors. Then, **per session x SWR
    population**, it within-(session, arm) demeans the response and predictors (the
    arm fixed-effects transform) and fits OLS. The demeaning removes arm identity
    and geometry, so the uncertainty coefficient is identified from *within-arm*
    variation over the session — the confound-free frame in which beta value and
    beta uncertainty separate cleanly (they are collinear only against the arm
    dummies). ``reward_belief_variance_candidate_arm_preoutcome_z`` is
    pre-standardized over the six reward-site
    leaf arms in the pooled event-arm table; the other predictors use this
    analysis's non-current-leaf subset, so their SD populations still differ.

    Parameters
    ----------
    event_arm : pd.DataFrame
        Event x arm rows with ``animal``/``session_date``/``epoch``,
        ``event_id``, ``candidate_arm``
        (0-5 leaf, 6-8 trunk), ``is_current``, ``is_swr``, the ``response`` column,
        and each ``predictors`` column.
    predictors : tuple of str
        Continuous predictors entered together (z-scored). Default: beta
        uncertainty vs beta value vs distance vs recency.
    response : str
        Per-event per-arm content weight (default ``arm_fraction``).
    min_rows : int
        Minimum non-current-leaf rows required to fit a session.

    Returns
    -------
    pd.DataFrame
        One row per (``animal``, ``session_id``, ``population``) with
        ``coef_<z-name>`` for each predictor, ``cond_number`` (design condition
        number = identifiability evidence), and ``n_rows``. Correlational.
    """
    # Causal-hygiene gate. Every other shipped fitter asserts predictor timing
    # before fitting; this one did not, and its coefficients are cited, so a
    # leaky covariate added to HORSERACE_PREDICTORS would have been fit and
    # reported without complaint. ``predictors`` is gated rather than the module
    # constant so a caller-supplied override is checked too.
    assert_predictor_timing(predictors)
    needed = {
        "animal",
        "session_date",
        "epoch",
        "event_id",
        "candidate_arm",
        "is_current_arm",
        "is_swr",
        response,
        *predictors,
    }
    require_columns(event_arm, needed, "event-arm table")
    frame = event_arm[
        (event_arm["candidate_arm"] <= 5) & (event_arm["is_current_arm"] == 0)
    ].copy()
    frame["session_id"] = _session_id(frame)

    design_cols: list[str] = []
    for predictor in predictors:
        if predictor.endswith("_z"):
            design_cols.append(predictor)
            continue
        z_name = f"{predictor}_z"
        values = frame[predictor].astype(float)
        sd = values.std()
        frame[z_name] = (values - values.mean()) / sd if sd > 0 else 0.0
        design_cols.append(z_name)

    populations = {
        "all": np.ones(len(frame), dtype=bool),
        "SWR": frame["is_swr"].to_numpy(dtype=bool),
        "non_SWR": ~frame["is_swr"].to_numpy(dtype=bool),
    }
    rows = []
    for population, mask in populations.items():
        subset = frame[mask]
        # Count why sessions drop out so a biased exclusion is visible, not silent:
        # rank-deficiency correlates with low within-arm exploration — exactly the
        # sessions whose loss could skew the exploit-vs-explore read.
        n_seen = n_too_few = n_rank_deficient = n_fit = 0
        for (animal, session_id), group in subset.groupby(
            ["animal", "session_id"], observed=True, sort=False
        ):
            n_seen += 1
            fit = group[["candidate_arm", response, *design_cols]].dropna()
            if fit["candidate_arm"].nunique() < 2 or len(fit) < min_rows:
                n_too_few += 1
                continue
            demeaned = fit[[response, *design_cols]] - fit.groupby(
                fit["candidate_arm"]
            )[[response, *design_cols]].transform("mean")
            design = demeaned[design_cols].to_numpy()
            if np.linalg.matrix_rank(design) < design.shape[1]:
                n_rank_deficient += 1
                continue
            n_fit += 1
            beta, *_ = np.linalg.lstsq(
                design, demeaned[response].to_numpy(), rcond=None
            )
            rows.append(
                {
                    "animal": animal,
                    "session_id": session_id,
                    "population": population,
                    "n_rows": int(len(fit)),
                    # Column-scaled, matching fitting._compute_conditional_diagnostics
                    # so the two producers of this column mean the same thing and
                    # the conventional 10/30 bands apply to both. Unscaled, this
                    # also reports the columns' arbitrary relative units.
                    "cond_number": float(_scaled_condition_number(design)),
                    **{
                        f"coef_{col}": float(b)
                        for col, b in zip(design_cols, beta, strict=True)
                    },
                }
            )
        LOGGER.info(
            "within_arm_horserace[%s]: fit %d/%d sessions (%d < min_rows/arms, "
            "%d rank-deficient)",
            population,
            n_fit,
            n_seen,
            n_too_few,
            n_rank_deficient,
        )
    return pd.DataFrame(rows)


#: The file :func:`summarize_horserace`'s table is written to, in
#: ``paths.UNCERTAINTY_CONTENT_DIR``; the behavior summary and the uncertainty
#: figure read it.
HORSERACE_SUMMARY = "horserace_summary.csv"


def summarize_horserace(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Animal-balanced across-session summary of :func:`within_arm_horserace`.

    For each (population, predictor): the across-session median coefficient, the
    number of animals whose own median matches the overall sign (sign consistency,
    of those present; both from
    :func:`~continuum_swr_replay.analysis.shared_statistics.session_median_statistics`),
    the median per-session design condition number, and the
    Wilcoxon signed-rank p
    (:func:`~continuum_swr_replay.analysis.shared_statistics.signed_rank_p`, NaN below
    six non-zero coefficients). The Wilcoxon is **descriptive** —
    sessions within an animal are not independent — so read the animal
    sign-consistency as the population evidence.
    """
    coef_cols = [c for c in coefficients.columns if c.startswith("coef_")]
    rows = []
    for population, pop_df in coefficients.groupby("population", observed=True):
        for coef_col in coef_cols:
            values = pop_df[coef_col].dropna()
            if len(values) < 2:
                continue
            summary = session_median_statistics(pop_df[coef_col], pop_df["animal"])
            rows.append(
                {
                    "population": population,
                    "predictor": coef_col[len("coef_") :],
                    "median": summary["median"],
                    "n_animals_consistent": summary["n_animals_same_sign"],
                    "n_animals": summary["n_animals"],
                    "median_cond_number": float(pop_df["cond_number"].median()),
                    "wilcoxon_p": signed_rank_p(values),
                    "n_sessions": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


#: The populations the analysis may be read on: the sessions the Mu fit
#: summarizes (``mu``), or every mapped session (``all``).
ANALYSIS_COHORTS: tuple[str, ...] = ("mu", "all")


def restrict_to_analysis_cohort(
    trial_leaf: pd.DataFrame,
    horserace: pd.DataFrame,
    mu_summary: Mapping[str, pd.DataFrame],
    cohort: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the sessions of one analysis cohort.

    ``mu`` keeps the sessions present in the Mu fit's per-session summary
    (``summarize_mu_coefficients(...)["session"]``), so the model-light
    profiles and the within-arm horse race are read on the sessions the
    adjusted model was fit on; ``all`` keeps every session.

    Parameters
    ----------
    trial_leaf : pd.DataFrame
        :func:`build_trial_leaf_table` output, with ``session_id``.
    horserace : pd.DataFrame
        :func:`within_arm_horserace` output, with ``session_id``.
    mu_summary : mapping of str to pd.DataFrame
        :func:`summarize_mu_coefficients` output.
    cohort : {"mu", "all"}
        The cohort.

    Returns
    -------
    (trial_leaf, horserace)
        The two tables restricted to the cohort's sessions.

    Raises
    ------
    ValueError
        If ``cohort`` is not one of :data:`ANALYSIS_COHORTS`.
    """
    if cohort not in ANALYSIS_COHORTS:
        raise ValueError(f"cohort must be one of {ANALYSIS_COHORTS}, not {cohort!r}")
    if cohort == "mu":
        valid_sessions = set(mu_summary["session"]["session_id"])
        trial_leaf = trial_leaf[trial_leaf["session_id"].isin(valid_sessions)].copy()
        horserace = horserace[horserace["session_id"].isin(valid_sessions)].copy()
    return trial_leaf, horserace


def analysis_cohort_counts(trial_leaf: pd.DataFrame) -> dict[str, int]:
    """Sessions, trials, replay-bearing trials and rows of the analysed table.

    Parameters
    ----------
    trial_leaf : pd.DataFrame
        The trial x reward-well table after :func:`restrict_to_analysis_cohort`.

    Returns
    -------
    dict
        ``n_analysis_sessions``, ``n_analysis_trials`` (distinct session x
        trial), ``n_analysis_replay_trials`` (those with ``has_replay``) and
        ``n_analysis_trial_leaf_rows``.
    """
    return {
        "n_analysis_sessions": int(trial_leaf["session_id"].nunique()),
        "n_analysis_trials": int(
            trial_leaf[["session_id", "trial_number_by_epoch"]]
            .drop_duplicates()
            .shape[0]
        ),
        "n_analysis_replay_trials": int(
            trial_leaf.drop_duplicates(["session_id", "trial_number_by_epoch"])[
                "has_replay"
            ].sum()
        ),
        "n_analysis_trial_leaf_rows": int(len(trial_leaf)),
    }


#: The session the representative-session panel shows when the cohort holds it.
DEFAULT_EXAMPLE_SESSION = ("wilbur", "20210329", "08_r4")


def select_example_session(trial_leaf: pd.DataFrame) -> dict[str, str]:
    """The representative session, chosen without looking at the association.

    :data:`DEFAULT_EXAMPLE_SESSION` when the table holds it; otherwise the
    robust medoid of the sessions on trial count, event count and the median
    within-well uncertainty range (each feature centred on its median and
    scaled by 1.4826 x MAD, or by its SD when the MAD is zero; a feature with
    no spread is left out).

    Parameters
    ----------
    trial_leaf : pd.DataFrame
        The trial x reward-well table.

    Returns
    -------
    dict of str to str
        ``animal``, ``session_date``, ``epoch`` and the ``selection`` rule used.
    """
    animal, session_date, epoch = DEFAULT_EXAMPLE_SESSION
    present = (
        trial_leaf["animal"].astype(str).eq(animal)
        & trial_leaf["session_date"].astype(str).eq(session_date)
        & trial_leaf["epoch"].astype(str).eq(epoch)
    )
    if present.any():
        return {
            "animal": animal,
            "session_date": session_date,
            "epoch": epoch,
            "selection": "association-blind cohort medoid selected before plotting",
        }

    # Portable fallback for alternate input cohorts: select the robust medoid on
    # trial count, event count, and within-well uncertainty range.  No
    # uncertainty-content association enters this rule.
    trial = trial_leaf.drop_duplicates(["session_id", "trial_number_by_epoch"])
    counts = trial.groupby(
        ["animal", "session_date", "epoch", "session_id"], as_index=False
    ).agg(
        n_trials=("trial_number_by_epoch", "nunique"),
        n_events=("n_events", "sum"),
    )
    uncertainty_range = (
        trial_leaf.groupby(["session_id", "candidate_arm"], observed=True)[
            "reward_belief_variance_candidate_arm_preoutcome"
        ]
        .agg(lambda x: x.max() - x.min())
        .groupby("session_id")
        .median()
        .rename("uncertainty_range")
    )
    counts = counts.merge(uncertainty_range, on="session_id", validate="one_to_one")
    features = ["n_trials", "n_events", "uncertainty_range"]
    distance = np.zeros(len(counts), dtype=float)
    for feature in features:
        values = counts[feature].to_numpy(dtype=float)
        median = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - median))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(values)
        if not np.isfinite(scale) or scale == 0:
            continue
        distance += ((values - median) / scale) ** 2
    counts["distance"] = np.sqrt(distance)
    selected = counts.sort_values(["distance", "session_id"]).iloc[0]
    return {
        "animal": str(selected["animal"]),
        "session_date": str(selected["session_date"]),
        "epoch": str(selected["epoch"]),
        "selection": "fallback robust medoid; association not used",
    }


def _profile_configurations() -> tuple[dict[str, str | None], ...]:
    configurations: list[dict[str, str | None]] = []
    for metric, response_columns in [
        (
            "conditional_event_share",
            [
                ("hard", "hard_event_share"),
                ("soft", "soft_event_share"),
                ("posterior", "posterior_event_share"),
            ],
        ),
        (
            "conditional_leaf_share",
            [
                ("hard", "hard_leaf_share"),
                ("soft", "soft_leaf_share"),
                ("posterior", "posterior_leaf_share"),
            ],
        ),
    ]:
        for response, share in response_columns:
            for scope in ["all", "unchosen"]:
                configurations.append(
                    {
                        "metric": metric,
                        "response": response,
                        "scope": scope,
                        "numerator": share,
                        "denominator": None,
                    }
                )
    for response, mass in [
        ("hard", "hard_count"),
        ("soft", "soft_count"),
        ("posterior", "posterior_mass"),
    ]:
        for scope in ["all", "unchosen"]:
            configurations.append(
                {
                    "metric": "absolute_rate",
                    "response": response,
                    "scope": scope,
                    "numerator": mass,
                    "denominator": "immobility_duration_s",
                }
            )
    return tuple(configurations)


#: The quartile profiles the analysis reports, one per (metric, response,
#: scope): the conditional event share and leaf share of the hard, soft and
#: posterior responses, and the absolute rate (response mass per immobility
#: second), each over all wells and over the unchosen wells.
QUARTILE_PROFILE_CONFIGURATIONS = _profile_configurations()


def _add_result_metadata(
    frame: pd.DataFrame,
    *,
    metric: str,
    response: str,
    scope: str,
    level: str,
) -> pd.DataFrame:
    return frame.assign(
        metric=metric,
        response=response,
        scope=scope,
        level=level,
    )


def quartile_profiles_and_contrasts(
    trial_leaf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every reported quartile profile and its high-minus-low contrasts.

    For each of :data:`QUARTILE_PROFILE_CONFIGURATIONS`, the
    :func:`hierarchical_quartile_profile` at every level and the
    :func:`paired_quartile_contrasts` of its per-arm level; then the prevalence
    of the chosen well (``chosen_prevalence``, over all wells) as a profile only.

    Parameters
    ----------
    trial_leaf : pd.DataFrame
        The trial x reward-well table of the analysis cohort.

    Returns
    -------
    (profiles, contrasts)
        Long tables with ``metric``, ``response``, ``scope`` and ``level``
        columns; levels with no rows are left out.
    """
    profile_frames = []
    contrast_frames = []
    for config in QUARTILE_PROFILE_CONFIGURATIONS:
        profile = hierarchical_quartile_profile(
            trial_leaf,
            numerator_col=config["numerator"],
            denominator_col=config["denominator"],
            scope=config["scope"],
        )
        for level, frame in profile.items():
            if frame.empty:
                continue
            profile_frames.append(
                _add_result_metadata(
                    frame,
                    metric=config["metric"],
                    response=config["response"],
                    scope=config["scope"],
                    level=level,
                )
            )
        contrasts = paired_quartile_contrasts(profile["candidate_arm"])
        for level, frame in contrasts.items():
            if frame.empty:
                continue
            contrast_frames.append(
                _add_result_metadata(
                    frame,
                    metric=config["metric"],
                    response=config["response"],
                    scope=config["scope"],
                    level=level,
                )
            )

    profiles = pd.concat(profile_frames, ignore_index=True, sort=False)
    contrasts = pd.concat(contrast_frames, ignore_index=True, sort=False)

    chosen_profile = hierarchical_quartile_profile(
        trial_leaf.assign(is_chosen_value=trial_leaf["is_chosen"].astype(float)),
        numerator_col="is_chosen_value",
        scope="all",
    )
    for level, frame in chosen_profile.items():
        if frame.empty:
            continue
        profiles = pd.concat(
            [
                profiles,
                _add_result_metadata(
                    frame,
                    metric="chosen_prevalence",
                    response="binary",
                    scope="all",
                    level=level,
                ),
            ],
            ignore_index=True,
            sort=False,
        )
    return profiles, contrasts


__all__ = [
    "ANALYSIS_COHORTS",
    "DEFAULT_EXAMPLE_SESSION",
    "EVENT_SHARE_COLUMNS",
    "HORSERACE_PREDICTORS",
    "HORSERACE_SUMMARY",
    "LEAF_FRACTION_COLUMNS",
    "QUARTILE_PROFILE_CONFIGURATIONS",
    "RESPONSE_COLUMNS",
    "SHARE_COLUMNS",
    "UNCERTAINTY_COLUMNS",
    "analysis_cohort_counts",
    "assign_within_arm_quantiles",
    "build_trial_leaf_table",
    "hierarchical_quartile_profile",
    "paired_quartile_contrasts",
    "quartile_profiles_and_contrasts",
    "restrict_to_analysis_cohort",
    "select_example_session",
    "summarize_horserace",
    "summarize_mu_coefficients",
    "within_arm_horserace",
]
