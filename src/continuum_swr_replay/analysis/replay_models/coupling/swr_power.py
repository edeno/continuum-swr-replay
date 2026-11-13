"""The per-event SWR-power model: an event's ripple power against the behavioral covariates.

Each non-local event's SWR coupling is summarized by ``ripple_power_z``, the
event mean of the z-scored consensus ripple power (the signal the Kay detector
thresholds, so ``is_swr`` is a thresholded version of it). The model is a
per-session least-squares fit of that power on the standardized covariates of
:data:`PREDICTORS`. The covariates include the current trial's RPE, so a
session is fit only on its events that begin at or after
``outcome_known_time``, and only when both a rewarded and an omitted outcome
survive that gate and the complete-case filter.

Also here: the per-event frame the model is fit on (:func:`assemble_power_frame`),
the distribution of the power over all non-local events (its percentiles, its
bimodality, its mean and SD by SWR class) and its correlation with event
duration, and the event-grain cohort record of the fit.

The fit is the conditional power-vs-covariates relationship only; it is not a
marked point process (no joint intensity, no history dependence of the power).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import add_uncertainty_covariates
from continuum_swr_replay.analysis.constants import OUTCOME_TIME_SEMANTICS
from continuum_swr_replay.analysis.hippocampus.class_or_continuum import (
    sarle_bimodality,
)
from continuum_swr_replay.analysis.replay_models.cohort_audit import (
    fitted_cohort_record,
)
from continuum_swr_replay.analysis.replay_models.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.tables.standardization import (
    Reference,
    attach_standardized,
)

SC = list(SESSION_COLUMNS)
POWER_COL = "ripple_power_z"  # event-mean z-scored consensus ripple power
PREDICTORS = [
    "known_reward_outcome_lag_1",
    "task_reward_probability_chosen_arm",
    "rpe_postoutcome",
    "patch_switch_fraction_last5_completed_trips",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "reward_belief_mean_chosen_arm_preoutcome",
    "time_since_stop_start_s",
]
POWER_FIT_COLUMNS = (*PREDICTORS, POWER_COL)
POST_OUTCOME_GATE_RULE = "event_start_s >= outcome_known_time"
FIT_SUPPORT_LABEL = "fit_cohort_all_rows"

#: The percentiles :func:`power_distribution_statistics` reports.
POWER_PERCENTILES = (5, 25, 50, 75, 95)


def post_outcome_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(finite, eligible)`` row masks for the post-outcome gate.

    The single definition of the gate rule, shared by the fit and the
    cohort audit so the two cannot drift apart. Equality is eligible.

    Parameters
    ----------
    df : pd.DataFrame
        Event-grain frame carrying ``event_start_s`` and ``outcome_known_time``.

    Returns
    -------
    finite : np.ndarray, shape (n_events,)
        Both timestamps present.
    eligible : np.ndarray, shape (n_events,)
        Finite *and* starting at/after the outcome boundary.
    """
    require_columns(
        df,
        [*SC, "event_start_s", "outcome_known_time", "trial_reward"],
        "the SWR-power event frame",
    )

    event_start = df["event_start_s"].to_numpy(dtype=float)
    boundary = df["outcome_known_time"].to_numpy(dtype=float)
    raw_trial_reward = df["trial_reward"]
    trial_reward = pd.to_numeric(raw_trial_reward, errors="coerce").to_numpy(
        dtype=float
    )
    if (raw_trial_reward.notna() & ~np.isfinite(trial_reward)).any():
        raise ValueError("trial_reward must contain binary 0/1 values or NaN")
    known_outcome = np.isfinite(trial_reward)
    if not np.isin(trial_reward[known_outcome], (0.0, 1.0)).all():
        raise ValueError("trial_reward must contain binary 0/1 values or NaN")
    # A left-join qvals gap legitimately has no outcome label. It cannot pass
    # the physical gate and is attributed to qvals_gap by the cohort audit,
    # rather than aborting the entire paper-scale fit.
    finite = np.isfinite(event_start) & np.isfinite(boundary) & known_outcome
    return finite, finite & (event_start >= boundary)


def _power_fit_complete_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows retained by the OLS model's exact listwise-finite filter."""
    require_columns(df, POWER_FIT_COLUMNS, "the SWR-power event frame")
    return np.isfinite(df[list(POWER_FIT_COLUMNS)].to_numpy(dtype=float)).all(axis=1)


def balanced_post_outcome_selection(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """The post-outcome gate and the outcome-balance guard, per session.

    A session keeps its fit-complete events that pass the post-outcome gate
    only when both a rewarded and an omitted event remain among them;
    otherwise it keeps none.

    Parameters
    ----------
    df : pd.DataFrame
        Event-grain frame with the session columns, the timing columns of
        :func:`post_outcome_masks` and the columns of :data:`POWER_FIT_COLUMNS`.

    Returns
    -------
    finite, eligible : np.ndarray, shape (n_events,)
        As :func:`post_outcome_masks`.
    selected : np.ndarray, shape (n_events,)
        The events the fit uses.
    audit : pd.DataFrame, shape (n_sessions, n_fields)
        One row per session: the event counts before and after each gate, by
        outcome, and whether the balance guard held.
    """
    finite, eligible = post_outcome_masks(df)
    fit_complete = _power_fit_complete_mask(df)
    trial_reward = df["trial_reward"].to_numpy(dtype=float)
    selected = np.zeros(len(df), dtype=bool)
    audit_rows = []
    for keys, positions in df.groupby(SC, sort=False).indices.items():
        pos = np.asarray(positions, dtype=int)
        finite_g = finite[pos]
        eligible_g = eligible[pos]
        fit_complete_g = fit_complete[pos]
        candidate_g = eligible_g & fit_complete_g
        rewarded_g = trial_reward[pos] == 1.0
        omitted_g = trial_reward[pos] == 0.0
        reward_after = int((eligible_g & rewarded_g).sum())
        omission_after = int((eligible_g & omitted_g).sum())
        reward_complete = int((candidate_g & rewarded_g).sum())
        omission_complete = int((candidate_g & omitted_g).sum())
        balance_valid = reward_complete > 0 and omission_complete > 0
        if balance_valid:
            selected[pos] = candidate_g
            balance_status = "both_outcomes_retained"
        else:
            missing = []
            if reward_complete == 0:
                missing.append("rewarded")
            if omission_complete == 0:
                missing.append("omission")
            balance_status = "missing_after_complete_case:" + ",".join(missing)
        audit_rows.append(
            {
                **dict(zip(SC, keys, strict=True)),
                "post_outcome_gate": True,
                "post_outcome_gate_rule": POST_OUTCOME_GATE_RULE,
                "outcome_time_semantics": OUTCOME_TIME_SEMANTICS,
                "n_events_before_post_outcome_gate": int(len(pos)),
                "n_events_after_post_outcome_gate": int(eligible_g.sum()),
                "n_rewarded_events_before_post_outcome_gate": int(rewarded_g.sum()),
                "n_rewarded_events_after_post_outcome_gate": reward_after,
                "n_omission_events_before_post_outcome_gate": int(omitted_g.sum()),
                "n_omission_events_after_post_outcome_gate": omission_after,
                "n_events_before_complete_case": int(eligible_g.sum()),
                "n_rewarded_events_before_complete_case": reward_after,
                "n_omission_events_before_complete_case": omission_after,
                "n_events_after_complete_case": int(candidate_g.sum()),
                "n_rewarded_events_after_complete_case": reward_complete,
                "n_omission_events_after_complete_case": omission_complete,
                "n_events_excluded_missing_timing": int((~finite_g).sum()),
                "n_events_excluded_pre_outcome": int((finite_g & ~eligible_g).sum()),
                "n_events_excluded_incomplete_fit": int(
                    (eligible_g & ~fit_complete_g).sum()
                ),
                "outcome_balance_valid": balance_valid,
                "outcome_balance_status": balance_status,
                "n_events_after_outcome_balance_guard": int(candidate_g.sum())
                if balance_valid
                else 0,
            }
        )
    return finite, eligible, selected, pd.DataFrame(audit_rows)


def gate_events_after_outcome(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep fit-complete events whose start is at/after the current outcome.

    Equality is eligible. Rows with non-finite event or boundary timestamps do
    not pass. Outcome balance is evaluated after the OLS listwise-finite mask;
    the audit distinguishes timing, physical-gate, and complete-case losses.
    """
    _, _, selected, audit = balanced_post_outcome_selection(df)
    return df.loc[selected].copy(), audit


def fit_power_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-session least squares of ``ripple_power_z`` on the standardized covariates.

    Each session's covariates are standardized on its own fitted events, so a
    coefficient reads per 1 SD of that session's spread. A session with fewer
    than 30 events after :func:`gate_events_after_outcome` is not fit.

    Parameters
    ----------
    df : pd.DataFrame
        Event-grain frame: the session columns, the timing columns of
        :func:`post_outcome_masks` and the columns of :data:`POWER_FIT_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        One row per fitted session: the session columns, ``n_events``, the
        session's gate audit fields and one coefficient per predictor.
        ``attrs["post_outcome_gate_audit"]`` holds every session's gate audit
        (as records) and ``attrs["z_reference"]`` the standardization moments,
        keyed ``"animal|session_date|epoch"``.
    """
    assert_predictor_timing(PREDICTORS, retrospective=True)
    gated, gate_audit = gate_events_after_outcome(df)
    gate_by_session = {
        tuple(row[column] for column in SC): row
        for row in gate_audit.to_dict(orient="records")
    }
    rows = []
    z_reference: dict[str, Reference] = {}
    for keys, g in gated.groupby(SC, sort=False):
        # ``balanced_post_outcome_selection`` already applied this exact
        # listwise-finite column set before checking outcome balance.
        g = g[list(POWER_FIT_COLUMNS)]
        if len(g) < 30:
            continue
        # Per-session standardization, so a coefficient reads per 1 SD of that
        # session's own covariate spread. ``zero_std="center"`` keeps a
        # within-session constant predictor as an all-zero (unidentified,
        # coefficient 0) column rather than NaN-ing the whole session's fit.
        standardized, session_reference = attach_standardized(
            g[list(PREDICTORS)].copy(),
            PREDICTORS,
            support_label=FIT_SUPPORT_LABEL,
            zero_std="center",
        )
        z_reference["|".join(str(key) for key in keys)] = session_reference
        X = standardized[[f"{c}_z" for c in PREDICTORS]].to_numpy(dtype=float)
        y = g[POWER_COL].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(g)), X], y, rcond=None)
        gate_metadata = gate_by_session[keys]
        rows.append(
            {
                **dict(zip(SC, keys, strict=True)),
                "n_events": len(g),
                **{key: value for key, value in gate_metadata.items() if key not in SC},
                **dict(zip(PREDICTORS, beta[1:], strict=True)),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["post_outcome_gate_audit"] = gate_audit.to_dict(orient="records")
    result.attrs["z_reference"] = z_reference
    return result


def power_distribution_statistics(power: pd.Series, is_swr: pd.Series) -> dict:
    """The distribution of the events' ripple power, overall and by SWR class.

    Parameters
    ----------
    power : pd.Series, shape (n_events,)
        Each event's ``ripple_power_z``.
    is_swr : pd.Series of bool, shape (n_events,)
        Whether each event overlaps an SWR; on ``power``'s index.

    Returns
    -------
    dict
        - ``n_events``: the number of events.
        - ``percentiles``: np.ndarray, shape (5,), the power at the
          :data:`POWER_PERCENTILES` (NaN if any power is NaN).
        - ``sarle_bimodality``: Sarle's bimodality coefficient of the power.
        - ``non_swr`` and ``swr``: for each class, a dict of ``n``, ``mean``
          and ``sd`` (the sample SD, ddof 1); a NaN power is left out of the
          mean and SD but counted in ``n``.
    """
    values = power.to_numpy()
    by_class = {}
    for label, members in (("non_swr", ~is_swr), ("swr", is_swr)):
        sub = power[members]
        by_class[label] = {"n": len(sub), "mean": sub.mean(), "sd": sub.std()}
    return {
        "n_events": len(power),
        "percentiles": np.percentile(values, list(POWER_PERCENTILES)),
        "sarle_bimodality": sarle_bimodality(values),
        **by_class,
    }


def duration_power_correlation(duration: pd.Series, power: pd.Series) -> dict:
    """Pearson correlation of the events' ripple power with their duration.

    Parameters
    ----------
    duration : pd.Series, shape (n_events,)
        Each event's ``event_duration_s``; positive.
    power : pd.Series, shape (n_events,)
        Each event's ``ripple_power_z``, on ``duration``'s index.

    Returns
    -------
    dict
        ``corr`` (power against duration) and ``log_duration_corr`` (power
        against log duration), over the events where both are present.
    """
    both = pd.DataFrame({"duration": duration, "power": power}).dropna()
    return {
        "corr": np.corrcoef(both["duration"], both["power"])[0, 1],
        "log_duration_corr": np.corrcoef(np.log(both["duration"]), both["power"])[0, 1],
    }


#: The power record's own fields, written after the canonical cohort columns.
#: With the session gate's remainder (outcome-balanced events in sessions under
#: 30 events) they sum to ``n_excluded_session``.
COHORT_EXTRA_COLUMNS = (
    "n_events_excluded_pre_outcome",
    "n_events_excluded_outcome_imbalance",
)

# Predictors sourced (via left join) from qvals_table -- NaN on the mechanism-B tail.
_QVALS_PREDICTORS = (
    "outcome_known_time",
    "trial_reward",
    "task_reward_probability_chosen_arm",
    "rpe_postoutcome",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "reward_belief_mean_chosen_arm_preoutcome",
)

#: Stop-table covariates joined onto each event.
STOP_COVARIATES = (
    "known_reward_outcome_lag_1",
    "patch_switch_fraction_last5_completed_trips",
)


def assemble_power_frame(
    events: pd.DataFrame, stops: pd.DataFrame, qvals: pd.DataFrame
) -> pd.DataFrame:
    """The per-event frame the power model is fit on.

    Each event gets its stop's :data:`STOP_COVARIATES` (after
    :func:`~continuum_swr_replay.analysis.behavior.covariates.add_uncertainty_covariates`)
    and its stop's qvals columns (``outcome_known_time``, the trial's reward and
    the behavioral model's values and RPE, under their canonical names). Both
    are left joins validated many-to-one on ``stop_id``, so a duplicated stop
    raises instead of multiplying events; an event with no qvals row keeps NaN
    there (the mechanism-B tail :func:`power_cohort_records` books).

    Parameters
    ----------
    events : pd.DataFrame
        The event table, one row per non-local event, with ``stop_id``.
    stops : pd.DataFrame
        The stop table, one row per stop.
    qvals : pd.DataFrame
        The qvals table, one row per stop.

    Returns
    -------
    pd.DataFrame
        ``events`` with the joined columns, in the event table's row order.
    """
    stops = add_uncertainty_covariates(stops)
    # Many events per stop, one row per stop on both right-hand tables.
    # Without validate= a duplicated stop_id would silently multiply
    # events into the per-event OLS instead of raising.
    df = events.merge(
        stops[["stop_id", *STOP_COVARIATES]],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )
    return df.merge(
        qvals[["stop_id", *_QVALS_PREDICTORS]],
        on="stop_id",
        how="left",
        validate="many_to_one",
    )


def power_cohort_records(df: pd.DataFrame, coefs: pd.DataFrame) -> list[dict]:
    """Event-grain cohort record. The per-session ``dropna`` over PREDICTORS silently
    removed the qvals-tail events (qvals-derived predictors NaN on the left join); this
    attributes them to ``n_drop_qvals_gap`` (mechanism B), the other predictor/response
    NaNs to own, and the <30-event session gate to excluded. ``n_modeled`` is the
    fitter's own count, the sum of ``coefs["n_events"]``. ``n_excluded_session`` is
    split by the two :data:`COHORT_EXTRA_COLUMNS` (pre-outcome events, and events
    of sessions left with one outcome); the rest of it is the <30-event gate."""
    # Raises if the timing columns are absent, so ``event_start_s`` below
    # needs no membership guard: reaching this point means it exists.
    _, eligible, selected, _ = balanced_post_outcome_selection(df)
    qp = [c for c in _QVALS_PREDICTORS if c in df.columns]
    op = [
        c
        for c in (*PREDICTORS, POWER_COL, "event_start_s")
        if c not in qp and c in df.columns
    ]
    qmask = (
        ~np.isfinite(df[qp].to_numpy(dtype=float)).all(axis=1)
        if qp
        else pd.Series(False, index=df.index)
    )
    qmask = pd.Series(qmask, index=df.index)
    omask = (
        ~np.isfinite(df[op].to_numpy(dtype=float)).all(axis=1)
        if op
        else pd.Series(False, index=df.index)
    )
    omask = pd.Series(omask, index=df.index) & ~qmask
    complete = ~(qmask | omask)
    records = [
        fitted_cohort_record(
            "swr_power",
            "all",
            "event",
            coefs,
            # The fitter reports how many events each fitted session kept.
            count_column="n_events",
            n_in=len(df),
            sessions_in=df.groupby(SC).ngroups,
            own_missing=omask,
            n_qvals_gap=int(qmask.sum()),
            drop_notes=(
                "qvals-derived predictor NaN (left join) -> qvals_gap (mechanism B); "
                "other predictor/response/timing NaN -> own; pre-outcome event, "
                "one-outcome session or <30-event session -> excluded"
            ),
        )
    ]
    records[0].update(
        n_events_excluded_pre_outcome=int(
            (complete & ~pd.Series(eligible, index=df.index)).sum()
        ),
        n_events_excluded_outcome_imbalance=int(
            (
                complete
                & pd.Series(eligible, index=df.index)
                & ~pd.Series(selected, index=df.index)
            ).sum()
        ),
    )
    return records
