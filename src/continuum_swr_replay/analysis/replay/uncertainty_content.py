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

from .covariate_timing import assert_predictor_timing

LOGGER = logging.getLogger(__name__)

LEAF_ARMS: tuple[int, ...] = tuple(range(6))
SESSION_COLUMNS: tuple[str, ...] = ("animal", "day", "epoch")
UNCERTAINTY_COLUMNS: tuple[str, ...] = tuple(
    f"betadist_var{arm + 1}" for arm in LEAF_ARMS
)
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


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


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
        + frame["day"].astype(str)
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
    uncertainty_col: str = "arm_uncertainty",
    output_col: str = "uncertainty_quartile",
) -> pd.DataFrame:
    """Assign tie-preserving uncertainty quantiles within session x well.

    A midpoint empirical CDF gives identical uncertainty values identical bins.
    Groups with no uncertainty variation are explicitly marked missing rather
    than being assigned an arbitrary middle quantile.
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    _require_columns(
        frame,
        {"session_id", "arm", uncertainty_col},
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
        out.groupby(["session_id", "arm"], observed=True, group_keys=False)[
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
        "arm",
        *RESPONSE_COLUMNS.keys(),
    }
    stop_required = {
        *SESSION_COLUMNS,
        "stop_id",
        "stop_idx",
        "immobility_time",
    }
    qval_required = {
        "stop_id",
        "epoch",
        "trial_number_by_epoch",
        "arm",
        "trial_reward",
        "in_poke",
        "is_rewarding",
        *UNCERTAINTY_COLUMNS,
    }
    _require_columns(event_arm_table, event_required, "event_arm_table")
    _require_columns(stop_table, stop_required, "stop_table")
    _require_columns(qvals_table, qval_required, "qvals_table")
    _require_unique(event_arm_table, ["event_id", "arm"], "event_arm_table")
    _require_unique(stop_table, ["stop_id"], "stop_table")
    _require_unique(qvals_table, ["stop_id"], "qvals_table")

    stop_columns = [*SESSION_COLUMNS, "stop_id", "stop_idx", "immobility_time"]
    qval_columns = [
        "stop_id",
        "epoch",
        "trial_number_by_epoch",
        "arm",
        "trial_reward",
        "in_poke",
        "is_rewarding",
        *UNCERTAINTY_COLUMNS,
    ]
    qvals = qvals_table[qval_columns].rename(
        columns={"epoch": "qvals_epoch", "arm": "chosen_arm"}
    )
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
    nonpositive_exposure = mapped_stops_all["immobility_time"] <= 0
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
            "immobility_time",
            "trial_number_by_epoch",
            "chosen_arm",
            "trial_reward",
            "in_poke",
            "is_rewarding",
        ],
        value_vars=list(UNCERTAINTY_COLUMNS),
        var_name="uncertainty_source",
        value_name="arm_uncertainty",
    )
    long_stops["arm"] = (
        long_stops["uncertainty_source"].str.extract(r"(\d+)$")[0].astype(int) - 1
    )
    long_stops = long_stops.drop(columns="uncertainty_source")

    leaf_events = event_arm_table[event_arm_table["arm"].isin(LEAF_ARMS)].copy()
    content_by_stop_arm = (
        leaf_events.groupby(["stop_id", "arm"], observed=True, as_index=False)
        .agg(
            **{output: (source, "sum") for source, output in RESPONSE_COLUMNS.items()},
            n_events=("event_id", "nunique"),
        )
        .astype(dict.fromkeys(RESPONSE_COLUMNS.values(), float))
    )
    _require_unique(content_by_stop_arm, ["stop_id", "arm"], "content summary")
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
        on=["stop_id", "arm"],
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
        ["session_id", "trial_number_by_epoch", "arm"],
        ["arm_uncertainty"],
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
                "arm",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            arm_uncertainty=("arm_uncertainty", "first"),
            chosen_arm=("chosen_arm", "first"),
            trial_reward=("trial_reward", "first"),
            any_in_poke=("in_poke", "max"),
            any_rewarding=("is_rewarding", "max"),
            immobility_time=("immobility_time", "sum"),
            n_stops=("stop_id", "nunique"),
            n_events=("n_events", "sum"),
            hard_count=("hard_count", "sum"),
            soft_count=("soft_count", "sum"),
            posterior_mass=("posterior_mass", "sum"),
            hard_count_total=("hard_count_total", "sum"),
            soft_count_total=("soft_count_total", "sum"),
            posterior_mass_total=("posterior_mass_total", "sum"),
        )
        .sort_values([*SESSION_COLUMNS, "trial_number_by_epoch", "arm"])
        .reset_index(drop=True)
    )
    if (
        not trial_leaf.groupby(["session_id", "trial_number_by_epoch"], observed=True)[
            "arm"
        ]
        .nunique()
        .eq(len(LEAF_ARMS))
        .all()
    ):
        raise ValueError("every mapped trial must contain exactly six reward-well rows")
    _assert_constant(
        trial_leaf,
        ["session_id", "trial_number_by_epoch"],
        ["immobility_time", "n_stops", "n_events", "chosen_arm", "trial_reward"],
        "trial aggregate",
    )
    _assert_constant(
        trial_leaf,
        ["session_id", "trial_number_by_epoch"],
        total_response_names,
        "trial total content",
    )

    trial_leaf["has_replay"] = trial_leaf["n_events"] > 0
    trial_leaf["is_chosen"] = trial_leaf["arm"] == trial_leaf["chosen_arm"]
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
        ) / trial_leaf["immobility_time"].to_numpy(dtype=float)

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
        "arm",
        "arm_uncertainty",
        "is_chosen",
        quartile_col,
        numerator_col,
    }
    if denominator_col is not None:
        required.add(denominator_col)
    _require_columns(trial_leaf, required, "trial_leaf")
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
        return {"arm": empty, "session": empty, "animal": empty, "overall": empty}

    arm_keys = ["animal", "session_id", "arm", quartile_col]
    grouped = data.groupby(arm_keys, observed=True)
    if denominator_col is None:
        arm_profile = grouped.agg(
            value=(numerator_col, "mean"),
            arm_uncertainty=("arm_uncertainty", "mean"),
            n_trials=(numerator_col, "size"),
        ).reset_index()
    else:
        sums = grouped.agg(
            numerator=(numerator_col, "sum"),
            denominator=(denominator_col, "sum"),
            arm_uncertainty=("arm_uncertainty", "mean"),
            n_trials=(numerator_col, "size"),
        ).reset_index()
        sums["value"] = sums["numerator"] / sums["denominator"]
        arm_profile = sums.drop(columns=["numerator", "denominator"])

    session_profile = arm_profile.groupby(
        ["animal", "session_id", quartile_col], observed=True, as_index=False
    ).agg(
        value=("value", "mean"),
        arm_uncertainty=("arm_uncertainty", "mean"),
        n_arms=("arm", "nunique"),
        n_trials=("n_trials", "sum"),
    )
    animal_profile = session_profile.groupby(
        ["animal", quartile_col], observed=True, as_index=False
    ).agg(
        value=("value", "mean"),
        arm_uncertainty=("arm_uncertainty", "mean"),
        n_sessions=("session_id", "nunique"),
    )
    overall_profile = animal_profile.groupby(
        quartile_col, observed=True, as_index=False
    ).agg(
        value=("value", "mean"),
        sem=("value", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        arm_uncertainty=("arm_uncertainty", "mean"),
        n_animals=("animal", "nunique"),
    )
    return {
        "arm": arm_profile,
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
    """Compute high-minus-low contrasts, paired within physical well."""
    _require_columns(
        arm_profile,
        {"animal", "session_id", "arm", quartile_col, "value"},
        "arm profile",
    )
    index = ["animal", "session_id", "arm"]
    wide = arm_profile.pivot_table(
        index=index,
        columns=quartile_col,
        values="value",
        aggfunc="first",
    )
    if low not in wide.columns or high not in wide.columns:
        empty = pd.DataFrame()
        return {"arm": empty, "session": empty, "animal": empty, "overall": empty}
    arm = (wide[high] - wide[low]).dropna().rename("contrast").reset_index()
    session = arm.groupby(["animal", "session_id"], observed=True, as_index=False).agg(
        contrast=("contrast", "mean"), n_arms=("arm", "nunique")
    )
    animal = session.groupby("animal", observed=True, as_index=False).agg(
        contrast=("contrast", "mean"), n_sessions=("session_id", "nunique")
    )
    overall = pd.DataFrame(
        {
            "contrast": [animal["contrast"].mean()],
            "sem": [animal["contrast"].std(ddof=1) / np.sqrt(len(animal))],
            "n_animals": [animal["animal"].nunique()],
        }
    )
    return {"arm": arm, "session": session, "animal": animal, "overall": overall}


def summarize_mu_coefficients(mu_results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Prepare transparent all-session and fit-quality-clean Mu summaries."""
    required = {
        *SESSION_COLUMNS,
        "coef_arm_uncertainty_z",
        "bse_arm_uncertainty_z",
        "converged",
        "separated",
    }
    _require_columns(mu_results, required, "Mu results")
    _require_unique(mu_results, list(SESSION_COLUMNS), "Mu results")
    sessions = mu_results.copy()
    sessions["session_id"] = _session_id(sessions)
    sessions["fit_quality"] = np.where(
        sessions["converged"].astype(bool) & ~sessions["separated"].astype(bool),
        "clean",
        "flagged",
    )

    summaries = []
    for cohort, subset in [
        ("all", sessions),
        ("clean", sessions[sessions["fit_quality"] == "clean"]),
    ]:
        animal = (
            subset.groupby("animal", observed=True, as_index=False)
            .agg(
                coefficient=("coef_arm_uncertainty_z", "median"),
                n_sessions=("session_id", "nunique"),
            )
            .assign(cohort=cohort)
        )
        summaries.append(animal)
    animal_summary = pd.concat(summaries, ignore_index=True)
    overall = animal_summary.groupby("cohort", observed=True, as_index=False).agg(
        coefficient=("coefficient", "median"),
        n_animals=("animal", "nunique"),
    )
    overall["relative_weight"] = np.exp(overall["coefficient"])
    return {"session": sessions, "animal": animal_summary, "overall": overall}


HORSERACE_PREDICTORS: tuple[str, ...] = (
    "arm_uncertainty_z",  # per-leaf Beta belief variance (pre-standardized in the table)
    "q_arm",  # beta belief value (z-scored here)
    "dist_to_arm",  # geometry parallel (z-scored here)
    "visit_recency",  # recency parallel (z-scored here)
)


def _scaled_condition_number(design: np.ndarray) -> float:
    """Condition number of the column-scaled design.

    Scaling each column to unit norm isolates how close the columns are to
    linearly dependent, rather than reporting their arbitrary relative units,
    and is the form the conventional 10/30 bands refer to. Shared definition
    with :func:`analysis.replay.fitting._compute_conditional_diagnostics`.
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
    dummies). NB: ``arm_uncertainty_z`` is pre-standardized over *all* arms (table
    ``EVENT_ARM_STANDARDIZED_COLS``) while the others are standardized over this
    non-current-leaf subset, so the "_z" SD populations differ slightly.

    Parameters
    ----------
    event_arm : pd.DataFrame
        Event x arm rows with ``animal``/``day``/``epoch``, ``event_id``, ``arm``
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
    needed = {"animal", "day", "epoch", "event_id", "arm", "is_current", "is_swr",
              response, *predictors}
    _require_columns(event_arm, needed, "event-arm table")
    frame = event_arm[(event_arm["arm"] <= 5) & (event_arm["is_current"] == 0)].copy()
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
            fit = group[["arm", response, *design_cols]].dropna()
            if fit["arm"].nunique() < 2 or len(fit) < min_rows:
                n_too_few += 1
                continue
            demeaned = fit[[response, *design_cols]] - fit.groupby(fit["arm"])[
                [response, *design_cols]
            ].transform("mean")
            design = demeaned[design_cols].to_numpy()
            if np.linalg.matrix_rank(design) < design.shape[1]:
                n_rank_deficient += 1
                continue
            n_fit += 1
            beta, *_ = np.linalg.lstsq(design, demeaned[response].to_numpy(), rcond=None)
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
            population, n_fit, n_seen, n_too_few, n_rank_deficient,
        )
    return pd.DataFrame(rows)


def summarize_horserace(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Animal-balanced across-session summary of :func:`within_arm_horserace`.

    For each (population, predictor): the across-session median coefficient, the
    number of animals whose own median matches the overall sign (sign consistency,
    of those present), the median per-session design condition number, and the
    Wilcoxon signed-rank p. The Wilcoxon is **descriptive** — sessions within an
    animal are not independent — so read the animal sign-consistency as the
    population evidence.
    """
    from scipy.stats import wilcoxon

    coef_cols = [c for c in coefficients.columns if c.startswith("coef_")]
    rows = []
    for population, pop_df in coefficients.groupby("population", observed=True):
        for coef_col in coef_cols:
            values = pop_df[coef_col].dropna()
            if len(values) < 2:
                continue
            median = float(values.median())
            per_animal = pop_df.groupby("animal", observed=True)[coef_col].median()
            try:
                pvalue = float(wilcoxon(values).pvalue)
            except ValueError:
                pvalue = float("nan")
            rows.append(
                {
                    "population": population,
                    "predictor": coef_col[len("coef_"):],
                    "median": median,
                    "n_animals_consistent": int(
                        (np.sign(per_animal) == np.sign(median)).sum()
                    ),
                    "n_animals": int(per_animal.notna().sum()),
                    "median_cond_number": float(pop_df["cond_number"].median()),
                    "wilcoxon_p": pvalue,
                    "n_sessions": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "EVENT_SHARE_COLUMNS",
    "HORSERACE_PREDICTORS",
    "LEAF_ARMS",
    "LEAF_FRACTION_COLUMNS",
    "RESPONSE_COLUMNS",
    "SHARE_COLUMNS",
    "UNCERTAINTY_COLUMNS",
    "assign_within_arm_quantiles",
    "build_trial_leaf_table",
    "hierarchical_quartile_profile",
    "paired_quartile_contrasts",
    "summarize_horserace",
    "summarize_mu_coefficients",
    "within_arm_horserace",
]
