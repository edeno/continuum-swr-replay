"""What a per-bin rate model may see once the trial's outcome is known.

A post-outcome covariate (the trial's RPE, its belief update) is defined only
from ``outcome_known_time`` on. Two rules keep a per-bin rate fit from
conditioning on it earlier:

- :func:`mask_post_outcome_covariates` sets the post-outcome covariates to NaN
  on every stop that begins before its outcome is known, so the cohort freeze,
  not the fitter, decides which stops the rungs that carry them are fit on (the
  is_swr-contrast rate ladder).
- :func:`gate_bins_after_outcome` keeps only the bins whose left edge is at or
  after the outcome, and fits nothing when one outcome class leaves no bin (the
  uncertainty rate model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.columns import require_columns

from ..model_specs import BIN_S, POST_OUTCOME_COVARIATES, PerBinRateModelSpec
from .per_bin_rate_model import _design_columns

#: The bin gate's rule, recorded on each gated fit.
POST_OUTCOME_GATE_RULE = "stop_start + binj * BIN_S >= outcome_known_time"


def mask_post_outcome_covariates(
    stop: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """NaN the post-outcome covariates on stops that begin before the outcome.

    ``rpe_postoutcome`` and ``reward_belief_mean_update_l1_all_ports_postoutcome``
    are defined only once the trial's outcome is known (``outcome_known_time``).
    A stop whose ``stop_start`` precedes that instant spans the pre-outcome
    interval, so attaching the trial's post-outcome value to it would let the
    model condition on information the animal did not yet have. Those values are
    set to NaN here, before any freeze, so the cohort freezes -- not a silent
    per-rung drop inside the fitter -- decide which rows each rung is fit on.

    Parameters
    ----------
    stop : pd.DataFrame
        Post-qvals-join stop table; must carry ``stop_start``,
        ``outcome_known_time``, ``occupancy_state``, and both post-outcome
        covariates.

    Returns
    -------
    gated : pd.DataFrame
        Copy of ``stop`` with the post-outcome covariates set to NaN on
        pre-outcome stops.
    n_masked_by_state : pd.Series
        Number of masked stops per ``occupancy_state`` category.
    """
    require_columns(
        stop,
        [
            "stop_start",
            "outcome_known_time",
            "occupancy_state",
            *POST_OUTCOME_COVARIATES,
        ],
        "the stop table joined to qvals_table",
    )
    stop_start = stop["stop_start"].to_numpy(dtype=float)
    outcome_known = stop["outcome_known_time"].to_numpy(dtype=float)
    pre_outcome = stop_start < outcome_known
    gated = stop.copy()
    gated.loc[pre_outcome, list(POST_OUTCOME_COVARIATES)] = np.nan
    n_masked_by_state = (
        stop.loc[pre_outcome, "occupancy_state"].value_counts().astype(int)
    )
    return gated, n_masked_by_state


def retained_time_course_support(
    time_in_stop: pd.Series,
    retained: np.ndarray,
    rewarded: np.ndarray,
    omitted: np.ndarray,
) -> dict[str, float]:
    """Where the surviving within-stop time course starts, by outcome class.

    The spline's boundary knots are fixed at ``(0, TMAX)``, but the gate
    removes early bins -- and can remove different amounts by outcome class
    because rewarded and omitted outcomes have different physical cues. The low-``tc``
    region is then populated mainly by rewarded stops, so the basis is not
    orthogonal to outcome and the time-course and reward coefficients are
    partially confounded. Reporting the retained support makes the size of
    that asymmetry visible per session instead of assumed away.

    Parameters
    ----------
    time_in_stop : pd.Series, shape (n_bins,)
        Each bin's within-stop time ``tc``.
    retained, rewarded, omitted : np.ndarray of bool, shape (n_bins,)
        The bins the gate keeps, and each bin's outcome class.

    Returns
    -------
    dict
        Minimum and median retained ``tc`` for each outcome class, plus the
        rewarded-minus-omission gap in minimum retained ``tc``. NaN where a
        class retains no bin.
    """
    values = time_in_stop.to_numpy(dtype=float)
    support: dict[str, float] = {}
    for label, mask in (("rewarded", rewarded), ("omission", omitted)):
        kept = values[retained & mask]
        kept = kept[np.isfinite(kept)]
        support[f"min_retained_tc_{label}"] = (
            float(kept.min()) if kept.size else float("nan")
        )
        support[f"median_retained_tc_{label}"] = (
            float(np.median(kept)) if kept.size else float("nan")
        )
    support["min_retained_tc_omission_minus_rewarded"] = (
        support["min_retained_tc_omission"] - support["min_retained_tc_rewarded"]
    )
    return support


def gate_bins_after_outcome(
    design: pd.DataFrame,
    spec: PerBinRateModelSpec,
    *,
    bin_s: float = BIN_S,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep fit-complete bins whose left edge is at/after the current outcome.

    ``binj`` is zero-based, so a bin's physical observation time is
    ``stop_start + binj * bin_s``. Equality is eligible: information known at
    the boundary can be associated with responses beginning at that boundary.
    Non-finite timing rows are counted separately and never pass the gate. The
    outcome-balance guard is evaluated only after applying the same listwise
    column selection and positive-exposure filter as
    :func:`~.per_bin_rate_model.fit_per_bin_rate_model`.

    Parameters
    ----------
    design : pd.DataFrame, shape (n_bins, n_columns)
        One (session, population) per-bin design carrying ``stop_id``,
        ``stop_start``, ``binj``, ``outcome_known_time``, ``trial_reward``,
        ``tc``, ``exposure`` and ``spec``'s design columns.
    spec : PerBinRateModelSpec
        The rung to be fit on the gated bins; its design columns define a
        complete bin.
    bin_s : float
        Bin width in seconds.

    Returns
    -------
    gated : pd.DataFrame
        The kept bins with a ``response_time`` column and a fresh index; empty
        when one outcome class keeps no bin.
    audit : dict
        Bin counts before and after the gate and the complete-case filter, by
        outcome; whether the balance guard held; and the retained time-course
        support (:func:`retained_time_course_support`).
    """
    require_columns(
        design,
        ["stop_id", "stop_start", "binj", "outcome_known_time", "trial_reward"],
        "the per-bin design",
    )

    timing = design.groupby("stop_id", sort=False)[
        ["stop_start", "outcome_known_time", "trial_reward"]
    ]
    inconsistent = timing.nunique(dropna=False).gt(1).any(axis=1)
    if inconsistent.any():
        bad_ids = tuple(inconsistent[inconsistent].index)
        raise ValueError(
            "Post-outcome bin gating requires stop_start, "
            "outcome_known_time, and trial_reward to be constant within stop_id; "
            "invalid stop_id(s): "
            f"{bad_ids}"
        )

    fit_columns = [*_design_columns(spec), "exposure"]
    require_columns(design, fit_columns, "the per-bin design")

    output = design.copy()
    stop_start = output["stop_start"].to_numpy(dtype=float)
    binj = output["binj"].to_numpy(dtype=float)
    boundary = output["outcome_known_time"].to_numpy(dtype=float)
    trial_reward = output["trial_reward"].to_numpy(dtype=float)
    if (
        not np.isfinite(trial_reward).all()
        or not np.isin(trial_reward, (0.0, 1.0)).all()
    ):
        raise ValueError("trial_reward must contain finite binary 0/1 values")
    response_time = stop_start + binj * float(bin_s)
    finite = np.isfinite(response_time) & np.isfinite(boundary)
    eligible = finite & (response_time >= boundary)
    output["response_time"] = response_time
    # Match fit_per_bin_rate_model exactly: listwise drop over formula columns
    # plus exposure, followed by exposure > 0. Keeping this selection here
    # prevents the sole row class from disappearing inside the fitter after an
    # apparently balanced physical gate.
    # Treat +/-inf as unavailable too. Statsmodels may fail deep in fitting or
    # emit non-finite estimates if an infinite design value is allowed through;
    # the scientific complete-case contract is finite, not merely non-null.
    listwise_complete = np.isfinite(output[fit_columns].to_numpy(dtype=float)).all(
        axis=1
    )
    positive_exposure = output["exposure"].to_numpy(dtype=float) > 0
    fit_complete = listwise_complete & positive_exposure
    candidate = eligible & fit_complete
    rewarded = trial_reward == 1.0
    omitted = trial_reward == 0.0
    reward_after = int((eligible & rewarded).sum())
    omission_after = int((eligible & omitted).sum())
    reward_complete = int((candidate & rewarded).sum())
    omission_complete = int((candidate & omitted).sum())
    outcome_balance_valid = reward_complete > 0 and omission_complete > 0
    if outcome_balance_valid:
        balance_status = "both_outcomes_retained"
    else:
        missing = []
        if reward_complete == 0:
            missing.append("rewarded")
        if omission_complete == 0:
            missing.append("omission")
        balance_status = "missing_after_complete_case:" + ",".join(missing)
    audit = {
        "n_bins_before_post_outcome_gate": int(len(output)),
        "n_bins_after_post_outcome_gate": int(eligible.sum()),
        "n_rewarded_bins_before_post_outcome_gate": int(rewarded.sum()),
        "n_rewarded_bins_after_post_outcome_gate": reward_after,
        "n_omission_bins_before_post_outcome_gate": int(omitted.sum()),
        "n_omission_bins_after_post_outcome_gate": omission_after,
        "n_bins_before_complete_case": int(eligible.sum()),
        "n_rewarded_bins_before_complete_case": reward_after,
        "n_omission_bins_before_complete_case": omission_after,
        "n_bins_after_complete_case": int(candidate.sum()),
        "n_rewarded_bins_after_complete_case": reward_complete,
        "n_omission_bins_after_complete_case": omission_complete,
        "n_bins_excluded_missing_timing": int((~finite).sum()),
        "n_bins_excluded_pre_outcome": int((finite & ~eligible).sum()),
        "n_bins_excluded_incomplete_fit": int((eligible & ~listwise_complete).sum()),
        "n_bins_excluded_nonpositive_exposure": int(
            (eligible & listwise_complete & ~positive_exposure).sum()
        ),
        "outcome_balance_valid": outcome_balance_valid,
        "outcome_balance_status": balance_status,
        "n_bins_after_outcome_balance_guard": int(candidate.sum())
        if outcome_balance_valid
        else 0,
        # Where the retained within-stop time course starts, per outcome
        # class. An omission's latency boundary and a reward's pump-onset
        # boundary can truncate the two classes by different amounts
        # and the time spline is no longer orthogonal to outcome. Counts
        # alone cannot show that; these do. See docs/rate_models.md.
        **retained_time_course_support(output["tc"], candidate, rewarded, omitted),
    }
    # A one-outcome survivor set changes the estimand and makes current-outcome
    # terms uninterpretable. Keep the audit, but do not fit that cell.
    selected = candidate if outcome_balance_valid else np.zeros(len(output), dtype=bool)
    return output.loc[selected].reset_index(drop=True), audit
