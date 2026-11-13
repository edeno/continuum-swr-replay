"""Value and uncertainty covariates computed from the behavioral stream: the
canonical leaky-Q value history and the model-free behavioral-uncertainty
covariates. The qvals-artifact joins live in :mod:`value_block`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.design_constants import (
    N_REWARD_LAGS,
)
from continuum_swr_replay.data_loaders.constants import (
    WELL_ARMS,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ._common import _prepare_port_experiences

# Canonical paper-analysis settings. Patch-switch history uses every movement
# trip; local value history uses every spatial port experience but updates only
# on observed outcomes. Neither can be reconstructed correctly from a stop
# table with missing visits.
CANONICAL_SWITCH_WINDOW = 5
CANONICAL_LEAKY_Q_ALPHA = 0.3


UNCERTAINTY_COVARIATES = (
    "omission_streak",
    "patch_switch_fraction_last5_completed_trips",
    "wsls_deviation",
    "leaky_q_value_diff",
)


# ── Completed-trip switch and local-learning summaries ──


def _leaky_q_value_diff(
    completed_arm: np.ndarray,
    is_rewarded_outcome: np.ndarray,
    q_alpha: float,
) -> np.ndarray:
    """Pre-update experienced-value margin over an ordered visit sequence.

    For each arm ``a`` a leaky reward-rate is updated once per input row
    ``Q_a <- (1 - q_alpha) Q_a + q_alpha * is_rewarded_outcome`` (init 0.5 for all
    arms). The returned value is the best-minus-second-best gap evaluated
    **before** that row's update. Canonical paper tables call this on the full
    complete port-experience stream, never on the reduced stop sequence.

    Parameters
    ----------
    completed_arm : np.ndarray, shape (n_visits,)
        Arm/well 0..5 visited at each completed visit. A ``NaN`` arm still
        yields the current pre-update Q-spread margin (a real value -- the
        choice margin does not depend on which arm is visited) but does not
        update any arm's ``Q`` (the unknown arm cannot be credited).
    is_rewarded_outcome : np.ndarray, shape (n_visits,)
        Reward (0/1) at each completed visit. ``NaN`` denotes spatial
        occupancy without an observed outcome and leaves the Q state unchanged.
    q_alpha : float
        Leaky-integration learning rate.

    Returns
    -------
    np.ndarray, shape (n_visits,)
        Pre-update best-minus-second-best arm value at each visit.
    """
    q = np.full(6, 0.5)
    out = np.full(completed_arm.size, np.nan)
    for i in range(completed_arm.size):
        ordered = np.sort(q)
        out[i] = ordered[-1] - ordered[-2]
        arm = completed_arm[i]
        outcome = is_rewarded_outcome[i]
        if not np.isnan(arm) and np.isfinite(outcome):
            a = int(arm)
            q[a] = (1.0 - q_alpha) * q[a] + q_alpha * outcome
    return out


def canonical_port_value_history(port_experiences: pd.DataFrame) -> np.ndarray:
    """Pre-update local-Q margin for every canonical port experience.

    Unlike the movement-trial stream, this includes experience 0: the physical
    origin poke preceding the first complete movement. Its observed outcome
    updates the local value state.
    """
    experiences = _prepare_port_experiences(port_experiences)
    if experiences.empty:
        return np.array([], dtype=float)
    return _leaky_q_value_diff(
        experiences["port"].to_numpy(dtype=float),
        experiences["is_reward"].to_numpy(dtype=float, na_value=np.nan),
        CANONICAL_LEAKY_Q_ALPHA,
    )


def canonical_trial_switch_history(did_switch_patch: np.ndarray) -> np.ndarray:
    """Rolling patch-switch fraction over completed movement trips.

    Movement-trial history and reward-port history deliberately have different
    grains. A switch is defined only by a movement, so trial ``j`` enters this
    rolling mean at its destination arrival. Reward/value history instead uses
    :func:`canonical_port_value_history`, which also includes the port occupied
    before the first movement. Keeping these functions separate prevents the
    starting-port experience from being accidentally omitted from value state
    or incorrectly invented as a patch switch.
    """
    switched = np.asarray(did_switch_patch, dtype=float)
    if switched.ndim != 1:
        raise ValueError("completed-trip switch history must be one-dimensional")
    if not np.isfinite(switched).all() or not np.isin(switched, (0.0, 1.0)).all():
        raise ValueError(
            "did_last_completed_trip_switch_patch must contain finite binary values"
        )
    return (
        pd.Series(switched)
        .rolling(CANONICAL_SWITCH_WINDOW, min_periods=1)
        .mean()
        .to_numpy()
    )


def _completed_visit_context(stops: pd.DataFrame) -> pd.DataFrame:
    """Validate and collapse stop rows to explicitly identified visits.

    ``completed_trial_id`` is the zero-based completion-order ID regenerated
    by :func:`add_trial_context`. Stop tables need not contain every completed
    trial: a visit with no immobility bout has no stop row, so ID gaps are
    valid. IDs must nevertheless be nondecreasing in stop order, nullable rows
    may occur only before the first observed ID (true carry-in), and every row
    sharing an ID must agree on its completed-trial arm, outcome, and switch
    status.

    The returned sequence contains only visits represented in ``stops``. It is
    not a reconstruction of completed visits with no stop row; computing such
    a full sequence would require the original trials table, which this
    stop-table-only helper deliberately does not accept.
    """
    required = (
        "completed_trial_id",
        "completed_trial_arm",
        "is_rewarded_outcome",
        "completed_trial_reward",
        "did_last_completed_trip_switch_patch",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
    )
    require_columns(stops, required, "stops (regenerate stop_table)")
    if len(stops) == 0:
        return pd.DataFrame(columns=required)

    raw_ids = stops["completed_trial_id"]
    try:
        visit_ids = pd.array(raw_ids, dtype="Int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "completed_trial_id must contain nullable integral completion-order IDs"
        ) from exc
    finite_ids = visit_ids[~visit_ids.isna()].to_numpy(dtype=np.int64)
    if len(finite_ids) and (finite_ids < 0).any():
        raise ValueError("completed_trial_id must be nonnegative")
    if len(finite_ids) and np.any(np.diff(finite_ids) < 0):
        raise ValueError("completed_trial_id must be nondecreasing in stop order")
    if len(finite_ids):
        first_observed = int(np.flatnonzero(~visit_ids.isna())[0])
        if visit_ids[first_observed:].isna().any():
            raise ValueError(
                "nullable completed_trial_id carry-in rows may occur only before "
                "the first observed completed trial"
            )

    carry_in = visit_ids.isna()
    arrival_history_columns = (
        "completed_trial_arm",
        "did_last_completed_trip_switch_patch",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
    )
    history_columns = (
        *arrival_history_columns,
        "is_rewarded_outcome",
        "completed_trial_reward",
    )
    if stops.loc[carry_in, list(history_columns)].notna().to_numpy().any():
        raise ValueError(
            "carry-in rows without a completed_trial_id must have unknown "
            "completed-trial context and canonical history covariates"
        )

    working = stops.loc[~carry_in, list(required)].copy()
    working["completed_trial_id"] = finite_ids
    if working.empty:
        return working

    for column in history_columns:
        values = pd.to_numeric(working[column], errors="coerce")
        raw_finite = working[column].notna()
        if (raw_finite & values.isna()).any():
            raise ValueError(f"{column} must be numeric within completed trials")
        working[column] = values.astype(float)
        if column == "is_rewarded_outcome":
            # An arrived trip can legitimately have pre-outcome bouts (NaN)
            # followed by post-outcome bouts (one binary value). Its realized
            # value may never change, and availability may never regress from
            # finite back to missing as stop time advances.
            grouped_outcome = working.groupby("completed_trial_id", sort=False)[column]
            inconsistent = grouped_outcome.apply(
                lambda group: group.dropna().nunique() > 1
            )
            regressed = grouped_outcome.apply(
                lambda group: bool((group.notna().cummax() & group.isna()).any())
            )
            inconsistent = inconsistent | regressed
        else:
            inconsistent = working.groupby("completed_trial_id", sort=False)[
                column
            ].apply(lambda group: group.nunique(dropna=False) != 1)
        if inconsistent.any():
            bad = tuple(inconsistent[inconsistent].index[:10])
            raise ValueError(
                f"{column} must be constant within each completed_trial_id; "
                f"invalid ID(s): {bad}"
            )

    visits = working.drop_duplicates("completed_trial_id", keep="first").reset_index(
        drop=True
    )
    for column in arrival_history_columns:
        if not np.isfinite(visits[column].to_numpy(dtype=float)).all():
            raise ValueError(
                f"{column} must be finite for every observed completed_trial_id"
            )
    arm = visits["completed_trial_arm"].to_numpy(dtype=float)
    if not np.equal(arm, np.floor(arm)).all() or not np.isin(arm, WELL_ARMS).all():
        raise ValueError(
            f"completed_trial_arm must be a canonical leaf arm {tuple(WELL_ARMS)}"
        )
    for column in ("did_last_completed_trip_switch_patch",):
        values = visits[column].to_numpy(dtype=float)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{column} must be binary within completed trials")
    reward_values = working["is_rewarded_outcome"].dropna().to_numpy(dtype=float)
    if not np.isin(reward_values, (0.0, 1.0)).all():
        raise ValueError("is_rewarded_outcome must be binary whenever it is known")
    completed_reward = visits["completed_trial_reward"].to_numpy(dtype=float)
    if (
        not np.isfinite(completed_reward).all()
        or not np.isin(completed_reward, (0.0, 1.0)).all()
    ):
        raise ValueError(
            "completed_trial_reward must be finite binary within completed trials"
        )
    for column in ("patch_switch_fraction_last5_completed_trips", "leaky_q_value_diff"):
        values = visits[column].to_numpy(dtype=float)
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} must lie in [0, 1]")
    return visits


def _omission_streak(reward_lag_matrix: np.ndarray) -> np.ndarray:
    """Count leading consecutive omissions in ``known_reward_outcome_lag_1, known_reward_outcome_lag_2, ...``.

    Parameters
    ----------
    reward_lag_matrix : np.ndarray, shape (n_stops, n_lags)
        Column ``j`` is ``known_reward_outcome_lag_{j+1}`` (1 = reward, 0 = omission, NaN =
        no such prior trial).

    Returns
    -------
    np.ndarray, shape (n_stops,)
        Number of leading 0s before the first 1; a ``NaN`` lag terminates the
        streak (counts the omissions seen before it). ``NaN`` when the most
        recent outcome itself is unresolved -- with no known outcome to start
        from, a count of leading omissions is undefined, and returning 0 would
        be indistinguishable from "the last trial was rewarded".
    """
    reward_lag_matrix = np.asarray(reward_lag_matrix, dtype=float)
    if reward_lag_matrix.ndim != 2:
        raise ValueError("reward_lag_matrix must be two-dimensional")
    finite = reward_lag_matrix[np.isfinite(reward_lag_matrix)]
    if not np.isin(finite, (0.0, 1.0)).all():
        raise ValueError("reward lag values must be binary 0/1 or NaN")
    n_stops = reward_lag_matrix.shape[0]
    out = np.zeros(n_stops)
    for i in range(n_stops):
        streak = 0
        for val in reward_lag_matrix[i]:
            if np.isnan(val) or val == 1.0:
                break
            streak += 1
        out[i] = streak
    # No resolved most-recent outcome -> the streak has nothing to count from.
    # Without this the pre-first-outcome stops of every session land in the
    # streak==0 bucket alongside genuinely just-rewarded stops.
    if n_stops:
        out[~np.isfinite(reward_lag_matrix[:, 0])] = np.nan
    return out


def validate_uncertainty_inputs(
    stop_table: pd.DataFrame,
    *,
    switch_window: int = CANONICAL_SWITCH_WINDOW,
    q_alpha: float = CANONICAL_LEAKY_Q_ALPHA,
) -> None:
    """Every check :func:`add_uncertainty_covariates` makes, without its columns.

    For a reader that needs the stop table's canonical history validated but
    reads neither ``omission_streak`` nor ``wsls_deviation``.

    Raises
    ------
    KeyError, ValueError
        As :func:`add_uncertainty_covariates`.
    """
    _validated_stop_table(stop_table, switch_window=switch_window, q_alpha=q_alpha)


def _validated_stop_table(
    stop_table: pd.DataFrame, *, switch_window: int, q_alpha: float
) -> pd.DataFrame:
    """A copy of ``stop_table`` with its reward lags numeric, once every check
    of :func:`add_uncertainty_covariates` passes (per session, in its order)."""
    if switch_window != CANONICAL_SWITCH_WINDOW:
        raise ValueError(
            "noncanonical switch_window requires recomputation from the full "
            "trials table; a stop table cannot recover visits without stops"
        )
    if not np.isfinite(q_alpha) or q_alpha != CANONICAL_LEAKY_Q_ALPHA:
        raise ValueError(
            "noncanonical q_alpha requires recomputation from the full trials "
            "table; a stop table cannot recover visits without stops"
        )

    required = (
        *SESSION_COLUMNS,
        "stop_idx",
        "completed_trial_id",
        "completed_trial_arm",
        "did_last_completed_trip_switch_patch",
        # wsls_deviation reads the *realized* outcome, not the gated flags --
        # validate the column actually consumed. The gated primitive is retained
        # because a stop table lacking it predates the outcome clock entirely.
        "completed_trial_reward",
        "is_rewarded_outcome",
        "is_stay",
        "patch_switch_fraction_last5_completed_trips",
        "leaky_q_value_diff",
    )
    require_columns(stop_table, required, "stop_table (regenerate it)")

    out = stop_table.copy()
    expected_lag_cols = [
        f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)
    ]
    observed_lag_cols = [
        c for c in out.columns if c.startswith("known_reward_outcome_lag_")
    ]
    duplicate_lags = sorted(
        {column for column in observed_lag_cols if observed_lag_cols.count(column) > 1}
    )
    if set(observed_lag_cols) != set(expected_lag_cols) or duplicate_lags:
        missing_lags = [c for c in expected_lag_cols if c not in observed_lag_cols]
        unexpected_lags = [c for c in observed_lag_cols if c not in expected_lag_cols]
        raise KeyError(
            "omission_streak requires the exact contiguous canonical reward-lag "
            f"schema {expected_lag_cols}; missing={missing_lags}, "
            f"unexpected={unexpected_lags}, duplicates={duplicate_lags}"
        )
    raw_lags = out[expected_lag_cols]
    numeric_lags = raw_lags.apply(pd.to_numeric, errors="coerce")
    if (raw_lags.notna() & numeric_lags.isna()).to_numpy().any():
        raise ValueError("reward lag values must be numeric binary 0/1 or NaN")
    finite_lags = numeric_lags.to_numpy(dtype=float)
    finite_values = finite_lags[np.isfinite(finite_lags)]
    if not np.isin(finite_values, (0.0, 1.0)).all():
        raise ValueError("reward lag values must be binary 0/1 or NaN")
    out.loc[:, expected_lag_cols] = numeric_lags

    for _, idx in out.groupby(list(SESSION_COLUMNS), sort=False).groups.items():
        g = out.loc[idx].sort_values("stop_idx")
        _completed_visit_context(g)
        stay = g["is_stay"].to_numpy(dtype=float)
        finite_stay = stay[np.isfinite(stay)]
        if not np.isin(finite_stay, (0.0, 1.0)).all():
            raise ValueError("is_stay must be binary whenever it is known")
    return out


def add_uncertainty_covariates(
    stop_table: pd.DataFrame,
    *,
    switch_window: int = 5,
    q_alpha: float = 0.3,
) -> pd.DataFrame:
    """Complete the four model-free behavioral-uncertainty covariates per stop.

    Only two columns are *added* -- ``omission_streak`` and ``wsls_deviation``,
    the stop-local summaries. The two visit-history covariates
    (``patch_switch_fraction_last5_completed_trips``,
    ``leaky_q_value_diff``) must already be present:
    switch history is computed from every movement trip and value history from
    every port experience. They are validated and passed through unchanged,
    never recomputed. A reduced
    stop table cannot reconstruct history across completed visits that contain
    no immobility bout, so missing canonical columns fail loudly.
    The input is not mutated; the returned frame keeps its index and row count.

    The four covariates:

    - ``omission_streak``: leading consecutive omissions = count of leading 0s
      in ``known_reward_outcome_lag_1, known_reward_outcome_lag_2, ...`` (stop at the first 1; a ``NaN`` lag
      terminates the streak).
    - ``patch_switch_fraction_last5_completed_trips``: canonical rolling mean
      of whether each of the last five completed trips changed patch, through
      the identified completed trial. It is validated and retained, not
      recomputed.
    - ``wsls_deviation``: 1.0 (deviated from win-stay/lose-shift) unless the stop
      is ``(rewarded & is_stay)`` or ``(omitted & ~is_stay)``, in which case
      0.0. Uses ``completed_trial_reward``, the trip's *realized* outcome, not
      the bout-start-gated ``is_rewarded_outcome`` -- an at-well bout that
      starts inside the arrival -> outcome window has no outcome yet, which is
      the case for about a fifth of them, so the gated form drops those stops.
      ``NaN`` where no physical trip has completed or the outbound
      choice is unresolved -- such rows are dropped at fit time rather than
      counted as a deviation. Descriptive only: it embeds the eventual choice.
    - ``leaky_q_value_diff``: canonical best-minus-second-best leaky-Q margin
      (init 0.5, alpha 0.3) immediately before the identified completed trial's
      update. It is validated and retained, not recomputed.

    Noncanonical ``q_alpha`` or ``switch_window`` requests fail: sensitivity
    values require recomputation from the complete trials table, not this
    reduced stop table.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Per-stop table. Must carry ``animal``, ``session_date``, ``epoch``,
        ``stop_idx``, ``completed_trial_id``, ``completed_trial_arm``,
        ``did_last_completed_trip_switch_patch``, ``is_stay``,
        ``completed_trial_reward``, ``is_rewarded_outcome``,
        ``patch_switch_fraction_last5_completed_trips``,
        ``leaky_q_value_diff``, and ``known_reward_outcome_lag_1..`` columns.
    switch_window : int
        Must equal the canonical value 5.
    q_alpha : float
        Must equal the canonical value 0.3.

    Returns
    -------
    pd.DataFrame
        Copy of ``stop_table`` with canonical upstream history validated and
        the stop-local ``omission_streak`` and ``wsls_deviation`` columns added.
    """
    out = _validated_stop_table(
        stop_table, switch_window=switch_window, q_alpha=q_alpha
    )
    expected_lag_cols = [
        f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)
    ]
    omission = pd.Series(np.nan, index=out.index, dtype=float)
    wsls = pd.Series(np.nan, index=out.index, dtype=float)

    for _, idx in out.groupby(list(SESSION_COLUMNS), sort=False).groups.items():
        g = out.loc[idx].sort_values("stop_idx")
        pos = g.index

        omission.loc[pos] = _omission_streak(g[expected_lag_cols].to_numpy(dtype=float))

        # Read the trip's *realized* outcome, not the bout-start-gated flags.
        # A bout at a well begins at arrival, and the outcome is already
        # observable at 80.06 % of at-well bout starts (measured 2026-09-06);
        # the remaining fifth
        # start inside the arrival -> outcome window, where
        # ``is_rewarded_outcome`` is legitimately missing. Gating here would
        # drop those. Win-stay/lose-shift is by
        # definition about the outcome the animal just experienced, paired with
        # the choice it then makes -- and that choice, ``is_stay``, is future
        # information regardless, so this construct is descriptive-only however
        # the outcome is clocked. See covariate_timing.FUTURE_REFERENCING.
        outcome = g["completed_trial_reward"].to_numpy(dtype=float)
        stay = g["is_stay"].to_numpy(dtype=float)
        win_stay = (outcome == 1.0) & (stay == 1.0)
        lose_shift = (outcome == 0.0) & (stay == 0.0)
        wsls_vals = np.where(win_stay | lose_shift, 0.0, 1.0)
        # No completed trip (e.g. before the session's first arrival) or no
        # resolved outbound choice -> NaN, so the row drops at fit time instead
        # of being counted as a deviation.
        undefined = ~np.isfinite(outcome) | ~np.isfinite(stay)
        wsls_vals[undefined] = np.nan
        wsls.loc[pos] = wsls_vals

    out["omission_streak"] = omission
    out["wsls_deviation"] = wsls
    return out
