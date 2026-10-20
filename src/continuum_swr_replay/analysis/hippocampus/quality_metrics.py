"""Quality metrics for hippocampal decoding analysis.

This module provides functions for evaluating the quality of clusterless
hippocampal decoding results. Metrics assess whether the decoder accurately
tracks the animal's position during local states, whether non-local events
occur in biologically expected contexts (SWR, theta sequences), and whether
state classifications are internally consistent.

All functions operate on the ``data_over_time`` DataFrame produced by
:func:`~continuum_swr_replay.analysis.hippocampus.get_data_over_time`,
which is already saved as pickle files for all sessions, read through the SWR
source (``swr_source.read_dot_with_swr_source``, or ``load_data_over_time``).

Excluded time
-------------
Nothing in excluded time is analysed. The read marks those bins with
``in_excluded_time``, and every ``compute_*`` function requires that column
(:func:`_analysed_rows`). A point-wise statistic (an error, a mean, a
calibration bin, a correlation) leaves the excluded rows out. A function that
labels runs of bins (collapse episodes, non-local runs during movement, running
bouts) keeps the rows and treats the excluded bins as a gap that breaks a run:
removing the rows would join the runs on either side.

Design Principles
-----------------
- **Graph-based distances**: Decoding error uses ``abs(ahead_behind_distance)``,
  which is computed via ``non_local_detector.analysis.get_ahead_behind_distance``
  using track-graph shortest paths.
- **Continuous metrics**: All functions return continuous values. Binary flags
  are added only in :func:`compute_session_quality_summary` as a convenience.
- **Non-local during movement is expected**: Theta sequences produce non-local
  activity during running. We quantify rate and duration rather than penalize.

Main Functions
--------------
compute_local_decoding_error
    Decoding error (graph distance) during local state.
compute_state_fraction_during_movement
    Fraction of movement time in each decoder state.
compute_non_local_during_movement
    Rate and duration of non-local events during movement.
compute_behavior_state_breakdown
    Per-session running vs. immobile time split and decoder state fractions
    during immobility.
compute_bout_nonlocal_dominance
    Per-session fraction of running bouts dominated by non-local state.
    Catches localized decoder failures hidden by session-level means.
compute_running_error_tail
    Per-session tail percentiles of decoding error during running (local
    state only). Catches catastrophic episodic errors hidden by the median.
compute_error_by_track_segment
    Decoding error broken down by track segment.
compute_no_spike_firing_correspondence
    Correlation between no-spike state and multiunit firing rate.
compute_non_local_event_context
    Classification of non-local events by neural context (SWR, theta, neither).
compute_local_state_collapse_details
    Detailed characterization of local state collapse episodes.
compute_calibration_curve
    Relationship between decoder confidence and error.
compute_session_quality_summary
    Aggregate all metrics into a single per-session summary.

Examples
--------
>>> from continuum_swr_replay.analysis.hippocampus.quality_metrics import (
...     compute_session_quality_summary,
... )
>>> from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
...     load_data_over_time,
... )
>>> from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
>>> # The analysed epochs, with the SWR source's labels
>>> data = load_data_over_time(PROCESSED_DATA_DIR)
>>> summary = compute_session_quality_summary(data)
>>> print(summary[["animal", "day", "epoch", "median_error_cm", "local_fraction", "n_flags"]])

"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import find_objects, label

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.analysis.events.swr_source import analysed_bins
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

logger = logging.getLogger(__name__)

_DEFAULT_GROUP_COLS = list(SESSION_COLUMNS_DAY)


def _analysed_rows(data: pd.DataFrame) -> np.ndarray:
    """The rows outside excluded time: the one analysed-time rule for a bin.

    Every ``compute_*`` function applies this at entry. A point-wise statistic
    keeps only these rows. A run-labelling function ANDs this into its run
    condition per session, so an excluded bin breaks a run instead of being
    removed.

    Parameters
    ----------
    data : pd.DataFrame
        A ``data_over_time`` frame read through the SWR source, which adds
        ``in_excluded_time`` (``swr_source.with_swr_source``).

    Returns
    -------
    np.ndarray of bool, shape (n_rows,)
        True where the row lies outside excluded time.

    Raises
    ------
    KeyError, TypeError
        As :func:`~.swr_source.analysed_bins`, which this is: a missing or
        non-boolean column is refused.
    """
    return analysed_bins(data)


def _run_slices(labeled: np.ndarray) -> list[slice]:
    """The slice of each run ``scipy.ndimage.label`` numbered, in label order.

    On a 1-D mask each label is one contiguous run, so label ``i`` is exactly
    the bins of the ``i - 1``-th slice.

    Parameters
    ----------
    labeled : np.ndarray of int, shape (n_time,)
        Labels from ``label`` with at least one run.

    Returns
    -------
    list of slice
        One slice per label ``1..n_runs``.
    """
    return [run for (run,) in find_objects(labeled)]


def _run_durations_sec(
    labeled: np.ndarray, times: np.ndarray, sampling_frequency: float
) -> np.ndarray:
    """Duration of each labelled run: last minus first bin time, plus one bin.

    Parameters
    ----------
    labeled : np.ndarray of int, shape (n_time,)
        Labels from ``label`` with at least one run.
    times : np.ndarray, shape (n_time,)
        Bin times in seconds.
    sampling_frequency : float
        Bins per second.

    Returns
    -------
    np.ndarray, shape (n_runs,)
        Duration in seconds, in label order.
    """
    runs = _run_slices(labeled)
    first = np.array([run.start for run in runs])
    last = np.array([run.stop - 1 for run in runs])
    return times[last] - times[first] + 1.0 / sampling_frequency


def compute_local_decoding_error(
    data: pd.DataFrame,
    local_prob_threshold: float = 0.5,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute decoding error during local state for each session.

    Filters timebins where ``Local_prob > local_prob_threshold`` and computes
    ``abs(ahead_behind_distance)`` (graph-based shortest-path distance between
    decoded and actual position).

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame with columns: ``Local_prob``,
        ``ahead_behind_distance``, ``animal``, ``day``, ``epoch``.
    local_prob_threshold : float, optional
        Minimum ``Local_prob`` to consider a timebin as local state.
    group_cols : list[str] | None, optional
        Session grouping columns, by default ``["animal", "day", "epoch"]``.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``median_error_cm``,
        ``p90_error_cm``, ``mean_error_cm``, ``n_local_bins``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    local_mask = (data["Local_prob"] > local_prob_threshold) & _analysed_rows(data)
    local_data = data.loc[local_mask].copy()
    local_data["error_cm"] = local_data["ahead_behind_distance"].abs()

    if local_data.empty:
        return pd.DataFrame(
            columns=group_cols
            + ["median_error_cm", "p90_error_cm", "mean_error_cm", "n_local_bins"]
        )

    return (
        local_data.groupby(group_cols, sort=False)["error_cm"]
        .agg(
            median_error_cm="median",
            p90_error_cm=lambda x: np.nanpercentile(x, 90),
            mean_error_cm="mean",
            n_local_bins="count",
        )
        .reset_index()
    )


def compute_state_fraction_during_movement(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    local_prob_threshold: float = 0.5,
    sampling_frequency: float = 500.0,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Fraction of movement time in each decoder state, plus collapse detection.

    During movement (``head_speed > speed_threshold``), computes the fraction
    of time the decoder spends in each state (Local, No-Spike, Non-Local
    Continuous, Non-Local Fragmented). Also detects episodes where the local
    state collapses (``Local_prob < local_prob_threshold``) during movement.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame.
    speed_threshold : float, optional
        Minimum ``head_speed`` (cm/s) to consider the animal as moving.
    local_prob_threshold : float, optional
        Threshold below which local state is considered collapsed.
    sampling_frequency : float, optional
        Sampling rate in Hz.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``local_fraction``,
        ``nospike_fraction``, ``nonlocal_continuous_fraction``,
        ``nonlocal_fragmented_fraction``, ``n_moving_bins``,
        ``max_collapse_duration_sec``, ``n_collapse_episodes``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    is_moving_full = (data["head_speed"] > speed_threshold) & _analysed_rows(data)

    if not is_moving_full.any():
        return pd.DataFrame(
            columns=group_cols
            + [
                "local_fraction",
                "nospike_fraction",
                "nonlocal_continuous_fraction",
                "nonlocal_fragmented_fraction",
                "n_moving_bins",
                "max_collapse_duration_sec",
                "n_collapse_episodes",
            ]
        )

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        # Moving bins outside excluded time: the means leave the excluded bins
        # out, and the collapse runs below treat them as gaps.
        analysed = _analysed_rows(group)
        is_moving = (group["head_speed"] > speed_threshold).to_numpy() & analysed
        n_moving = int(is_moving.sum())
        moving_block = group.loc[is_moving]
        result = {
            "local_fraction": moving_block["Local_prob"].mean(),
            "nospike_fraction": moving_block["No-Spike_prob"].mean(),
            "nonlocal_continuous_fraction": moving_block[
                "Non-Local_Continuous_prob"
            ].mean(),
            "nonlocal_fragmented_fraction": moving_block[
                "Non-Local_Fragmented_prob"
            ].mean(),
            "n_moving_bins": n_moving,
        }

        # Label on the FULL timeline so immobile rows and excluded rows act
        # as gaps that break collapse runs. Filtering first and labeling on
        # the filtered frame would treat 5 cm/s | 0 cm/s | 5 cm/s as one
        # contiguous moving stretch and merge two 1-bin events into
        # one 3-second event.
        collapsed_and_moving = (
            group["Local_prob"] < local_prob_threshold
        ).to_numpy() & is_moving
        times = group.index.to_numpy()
        labeled, n_episodes = label(collapsed_and_moving)
        if n_episodes > 0:
            episode_durations = _run_durations_sec(labeled, times, sampling_frequency)
            result["max_collapse_duration_sec"] = float(episode_durations.max())
            result["n_collapse_episodes"] = n_episodes
        else:
            result["max_collapse_duration_sec"] = 0.0
            result["n_collapse_episodes"] = 0

        return pd.Series(result)

    return (
        data.groupby(group_cols, sort=False)
        .apply(
            _session_stats,
            include_groups=False,
        )
        .reset_index()
    )


def compute_non_local_during_movement(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    sampling_frequency: float = 500.0,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Rate and duration of non-local events during movement.

    Non-local events during movement are expected (theta sequences). This
    function quantifies their rate and duration rather than treating them as
    errors.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame.
    speed_threshold : float, optional
        Minimum ``head_speed`` (cm/s) to consider the animal as moving.
    non_local_threshold : float, optional
        Combined non-local probability threshold for event detection.
    sampling_frequency : float, optional
        Sampling rate in Hz.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``nonlocal_movement_rate_per_min``,
        ``nonlocal_movement_mean_duration_ms``,
        ``nonlocal_movement_median_duration_ms``,
        ``nonlocal_movement_fraction``, ``n_nonlocal_movement_events``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    is_moving_full = (data["head_speed"] > speed_threshold) & _analysed_rows(data)

    if not is_moving_full.any():
        return pd.DataFrame(
            columns=group_cols
            + [
                "nonlocal_movement_rate_per_min",
                "nonlocal_movement_mean_duration_ms",
                "nonlocal_movement_median_duration_ms",
                "nonlocal_movement_fraction",
                "n_nonlocal_movement_events",
            ]
        )

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        # Moving bins outside excluded time: the rate is per minute of them,
        # and the runs below treat excluded bins as gaps.
        analysed = _analysed_rows(group)
        is_moving = (group["head_speed"] > speed_threshold).to_numpy() & analysed
        n_moving = int(is_moving.sum())
        if n_moving == 0:
            return pd.Series(
                {
                    "nonlocal_movement_rate_per_min": 0.0,
                    "nonlocal_movement_mean_duration_ms": 0.0,
                    "nonlocal_movement_median_duration_ms": 0.0,
                    "nonlocal_movement_fraction": 0.0,
                    "n_nonlocal_movement_events": 0,
                }
            )

        nonlocal_prob = (
            group["Non-Local_Continuous_prob"] + group["Non-Local_Fragmented_prob"]
        )
        # Label on the FULL timeline so immobile rows and excluded rows act
        # as gaps that break event runs. Filtering first treated 5/0/5 cm/s
        # as one contiguous moving stretch and merged separate events.
        is_nonlocal_and_moving = (
            nonlocal_prob > non_local_threshold
        ).to_numpy() & is_moving
        times = group.index.to_numpy()
        labeled, n_events = label(is_nonlocal_and_moving)

        total_duration_min = n_moving / sampling_frequency / 60.0

        if n_events > 0:
            durations_sec = _run_durations_sec(labeled, times, sampling_frequency)
            durations_ms = durations_sec * 1000.0
            return pd.Series(
                {
                    "nonlocal_movement_rate_per_min": n_events / total_duration_min,
                    "nonlocal_movement_mean_duration_ms": durations_ms.mean(),
                    "nonlocal_movement_median_duration_ms": np.median(durations_ms),
                    "nonlocal_movement_fraction": is_nonlocal_and_moving.sum()
                    / n_moving,
                    "n_nonlocal_movement_events": n_events,
                }
            )
        else:
            return pd.Series(
                {
                    "nonlocal_movement_rate_per_min": 0.0,
                    "nonlocal_movement_mean_duration_ms": 0.0,
                    "nonlocal_movement_median_duration_ms": 0.0,
                    "nonlocal_movement_fraction": 0.0,
                    "n_nonlocal_movement_events": 0,
                }
            )

    return (
        data.groupby(group_cols, sort=False)
        .apply(
            _session_stats,
            include_groups=False,
        )
        .reset_index()
    )


def compute_behavior_state_breakdown(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-session breakdown of time spent running vs. immobile and in each state.

    Complements :func:`compute_state_fraction_during_movement` (which covers
    only the running side) by adding:

    - the overall running / immobile time split, and
    - the decoder state fractions during **immobility**
      (``head_speed <= speed_threshold``).

    State fractions are expected fractions computed as the mean of the
    per-timebin state probabilities, matching the convention in
    :func:`compute_state_fraction_during_movement`. A combined
    ``immobile_nonlocal_fraction`` (continuous + fragmented) is also reported
    since for SWR-replay QC the continuous/fragmented split is often less
    important than the local vs. non-local contrast.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame with columns ``head_speed``, ``Local_prob``,
        ``No-Spike_prob``, ``Non-Local_Continuous_prob``,
        ``Non-Local_Fragmented_prob``.
    speed_threshold : float, optional
        Speed (cm/s) above which the animal is considered running.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``running_fraction``,
        ``immobile_fraction``, ``n_running_bins``, ``n_immobile_bins``,
        ``n_total_bins``, ``immobile_local_fraction``,
        ``immobile_nospike_fraction``, ``immobile_nonlocal_continuous_fraction``,
        ``immobile_nonlocal_fragmented_fraction``, ``immobile_nonlocal_fraction``.
        Every count, split and mean is over the bins outside excluded time.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    out_cols = group_cols + [
        "running_fraction",
        "immobile_fraction",
        "n_running_bins",
        "n_immobile_bins",
        "n_total_bins",
        "immobile_local_fraction",
        "immobile_nospike_fraction",
        "immobile_nonlocal_continuous_fraction",
        "immobile_nonlocal_fragmented_fraction",
        "immobile_nonlocal_fraction",
    ]

    # Only the check here: each session's rows are filtered below, so the
    # multi-session frame is not copied whole.
    _analysed_rows(data)
    if data.empty:
        return pd.DataFrame(columns=out_cols)

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        # Every count and mean is over the session's bins outside excluded time.
        group = group.loc[_analysed_rows(group)]
        n_total = len(group)
        running_mask = group["head_speed"] > speed_threshold
        n_running = int(running_mask.sum())
        n_immobile = n_total - n_running

        immobile = group.loc[~running_mask]
        if n_immobile > 0:
            imm_local = float(immobile["Local_prob"].mean())
            imm_nospike = float(immobile["No-Spike_prob"].mean())
            imm_nlc = float(immobile["Non-Local_Continuous_prob"].mean())
            imm_nlf = float(immobile["Non-Local_Fragmented_prob"].mean())
        else:
            imm_local = imm_nospike = imm_nlc = imm_nlf = np.nan

        return pd.Series(
            {
                "running_fraction": n_running / n_total if n_total > 0 else np.nan,
                "immobile_fraction": n_immobile / n_total if n_total > 0 else np.nan,
                "n_running_bins": n_running,
                "n_immobile_bins": n_immobile,
                "n_total_bins": n_total,
                "immobile_local_fraction": imm_local,
                "immobile_nospike_fraction": imm_nospike,
                "immobile_nonlocal_continuous_fraction": imm_nlc,
                "immobile_nonlocal_fragmented_fraction": imm_nlf,
                "immobile_nonlocal_fraction": (
                    imm_nlc + imm_nlf if np.isfinite(imm_nlc) else np.nan
                ),
            }
        )

    return (
        data.groupby(group_cols, sort=False)
        .apply(
            _session_stats,
            include_groups=False,
        )
        .reset_index()
    )


def compute_bout_nonlocal_dominance(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    min_bout_duration_sec: float = 0.5,
    bout_nonlocal_threshold: float = 0.5,
    sampling_frequency: float = 500.0,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Per-session fraction of running bouts dominated by non-local state.

    A running bout is a contiguous interval where ``head_speed > speed_threshold``.
    Bouts shorter than ``min_bout_duration_sec`` are excluded. For each remaining
    bout, the mean non-local probability (continuous + fragmented) is computed;
    bouts where that mean exceeds ``bout_nonlocal_threshold`` are counted as
    "non-local dominant".

    This complements :func:`compute_state_fraction_during_movement`, whose
    session-level mean can mask directional, arm-specific, or episodic failures
    where a subset of runs is fully mis-decoded while the aggregate mean still
    looks healthy. For example, a session with local_fraction=0.77 overall can
    still have 10% of its running bouts at >50% non-local.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame with columns ``head_speed``, ``Local_prob``,
        ``Non-Local_Continuous_prob``, ``Non-Local_Fragmented_prob``.
    speed_threshold : float, optional
        Speed (cm/s) above which the animal is considered running.
    min_bout_duration_sec : float, optional
        Minimum duration (seconds) for a bout to be considered.
    bout_nonlocal_threshold : float, optional
        Mean non-local probability above which a bout is counted as dominated
        by non-local state.
    sampling_frequency : float, optional
        Sampling rate in Hz.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``n_running_bouts``,
        ``n_running_bouts_nonlocal_dominant``,
        ``frac_running_bouts_nonlocal_dominant``,
        ``max_bout_nonlocal_fraction``, ``median_bout_local_fraction``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    out_cols = group_cols + [
        "n_running_bouts",
        "n_running_bouts_nonlocal_dominant",
        "frac_running_bouts_nonlocal_dominant",
        "max_bout_nonlocal_fraction",
        "median_bout_local_fraction",
    ]

    # Only the check here: the rows are kept, and the bouts below break at
    # excluded bins.
    _analysed_rows(data)
    if data.empty:
        return pd.DataFrame(columns=out_cols)

    min_bout_bins = int(round(min_bout_duration_sec * sampling_frequency))

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        # A bout is a run of running bins outside excluded time: an excluded
        # bin ends a bout, as a stop does.
        analysed = _analysed_rows(group)
        running_mask = (group["head_speed"] > speed_threshold).to_numpy() & analysed
        labeled, n_raw_bouts = label(running_mask)
        if n_raw_bouts == 0:
            return pd.Series(
                {
                    "n_running_bouts": 0,
                    "n_running_bouts_nonlocal_dominant": 0,
                    "frac_running_bouts_nonlocal_dominant": np.nan,
                    "max_bout_nonlocal_fraction": np.nan,
                    "median_bout_local_fraction": np.nan,
                }
            )

        local = group["Local_prob"].to_numpy()
        nonlocal_prob = (
            group["Non-Local_Continuous_prob"].to_numpy()
            + group["Non-Local_Fragmented_prob"].to_numpy()
        )

        bout_nonlocal_means: list[float] = []
        bout_local_means: list[float] = []
        for run in _run_slices(labeled):
            if run.stop - run.start < min_bout_bins:
                continue
            bout_nonlocal_means.append(float(nonlocal_prob[run].mean()))
            bout_local_means.append(float(local[run].mean()))

        n_bouts = len(bout_nonlocal_means)
        if n_bouts == 0:
            return pd.Series(
                {
                    "n_running_bouts": 0,
                    "n_running_bouts_nonlocal_dominant": 0,
                    "frac_running_bouts_nonlocal_dominant": np.nan,
                    "max_bout_nonlocal_fraction": np.nan,
                    "median_bout_local_fraction": np.nan,
                }
            )

        bout_nl = np.asarray(bout_nonlocal_means)
        bout_lc = np.asarray(bout_local_means)
        n_dominant = int((bout_nl > bout_nonlocal_threshold).sum())
        return pd.Series(
            {
                "n_running_bouts": n_bouts,
                "n_running_bouts_nonlocal_dominant": n_dominant,
                "frac_running_bouts_nonlocal_dominant": n_dominant / n_bouts,
                "max_bout_nonlocal_fraction": float(bout_nl.max()),
                "median_bout_local_fraction": float(np.median(bout_lc)),
            }
        )

    return (
        data.groupby(group_cols, sort=False)
        .apply(
            _session_stats,
            include_groups=False,
        )
        .reset_index()
    )


def compute_running_error_tail(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    local_prob_threshold: float = 0.5,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Upper-tail decoding error during running, restricted to local state.

    Complements ``median_error_cm`` (from :func:`compute_local_decoding_error`)
    by reporting tail percentiles of ``abs(ahead_behind_distance)`` during
    running. Catastrophic episodic failures contribute heavily to the tail
    while leaving the median near baseline — these are the cases where the
    decoder loses the animal for a single run rather than drifting globally.

    Only timebins with ``head_speed > speed_threshold`` AND
    ``Local_prob > local_prob_threshold`` are included, so the percentiles
    measure error at moments when the decoder itself claims to be confident.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame with ``head_speed``, ``Local_prob``,
        ``ahead_behind_distance``.
    speed_threshold : float, optional
        Speed (cm/s) above which the animal is considered running.
    local_prob_threshold : float, optional
        Minimum ``Local_prob`` for local state.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``p90_running_local_error_cm``,
        ``p99_running_local_error_cm``, ``frac_running_local_error_gt20cm``,
        ``frac_running_local_error_gt50cm``, ``n_running_local_bins``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    out_cols = group_cols + [
        "p90_running_local_error_cm",
        "p99_running_local_error_cm",
        "frac_running_local_error_gt20cm",
        "frac_running_local_error_gt50cm",
        "n_running_local_bins",
    ]

    mask = (
        (data["head_speed"] > speed_threshold)
        & (data["Local_prob"] > local_prob_threshold)
        & _analysed_rows(data)
    )
    subset = data.loc[mask].copy()
    if subset.empty:
        return pd.DataFrame(columns=out_cols)

    subset["abs_err"] = subset["ahead_behind_distance"].abs()

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        err = group["abs_err"].to_numpy()
        valid = err[np.isfinite(err)]
        if valid.size == 0:
            return pd.Series(
                {
                    "p90_running_local_error_cm": np.nan,
                    "p99_running_local_error_cm": np.nan,
                    "frac_running_local_error_gt20cm": np.nan,
                    "frac_running_local_error_gt50cm": np.nan,
                    "n_running_local_bins": 0,
                }
            )
        return pd.Series(
            {
                "p90_running_local_error_cm": float(np.percentile(valid, 90)),
                "p99_running_local_error_cm": float(np.percentile(valid, 99)),
                "frac_running_local_error_gt20cm": float((valid > 20).mean()),
                "frac_running_local_error_gt50cm": float((valid > 50).mean()),
                "n_running_local_bins": int(valid.size),
            }
        )

    return (
        subset.groupby(group_cols, sort=False)
        .apply(
            _session_stats,
            include_groups=False,
        )
        .reset_index()
    )


def compute_error_by_track_segment(
    data: pd.DataFrame,
    local_prob_threshold: float = 0.5,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute decoding error broken down by track segment.

    Identifies spatial blind spots where place field coverage is poor by
    computing per-segment decoding error during local state.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame with ``track_segment_id``.
    local_prob_threshold : float, optional
        Minimum ``Local_prob`` for local state.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per (session, track_segment_id) with columns:
        ``median_error_cm``, ``n_bins``, ``segment_fraction``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    local_mask = (data["Local_prob"] > local_prob_threshold) & _analysed_rows(data)
    local_data = data.loc[local_mask].copy()
    local_data["error_cm"] = local_data["ahead_behind_distance"].abs()

    if local_data.empty:
        return pd.DataFrame(
            columns=group_cols
            + ["track_segment_id", "median_error_cm", "n_bins", "segment_fraction"]
        )

    segment_group_cols = group_cols + ["track_segment_id"]

    segment_stats = (
        local_data.groupby(segment_group_cols, sort=False)["error_cm"]
        .agg(median_error_cm="median", n_bins="count")
        .reset_index()
    )

    session_totals = (
        local_data.groupby(group_cols, sort=False)["error_cm"]
        .count()
        .rename("session_total_bins")
        .reset_index()
    )

    segment_stats = segment_stats.merge(session_totals, on=group_cols)
    segment_stats["segment_fraction"] = (
        segment_stats["n_bins"] / segment_stats["session_total_bins"]
    )
    segment_stats = segment_stats.drop(columns="session_total_bins")

    return segment_stats


def compute_no_spike_firing_correspondence(
    data: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Check that no-spike probability anti-correlates with multiunit firing.

    The no-spike state should correspond to periods of low multiunit activity.
    A Spearman correlation near zero or positive indicates a problem with the
    encoding model.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame with ``No-Spike_prob`` and ``MUA_zscore``.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``nospike_mua_rho``,
        ``nospike_mua_pvalue``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)
    # Only the check here: each session's excluded bins leave ``valid`` below.
    _analysed_rows(data)

    def _session_corr(group: pd.DataFrame) -> pd.Series:
        nospike = group["No-Spike_prob"]
        mua = group["MUA_zscore"]

        # bins outside excluded time with both values
        valid = nospike.notna() & mua.notna() & _analysed_rows(group)
        if valid.sum() < 10:
            return pd.Series({"nospike_mua_rho": np.nan, "nospike_mua_pvalue": np.nan})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            rho, pvalue = stats.spearmanr(nospike[valid], mua[valid])
        return pd.Series({"nospike_mua_rho": rho, "nospike_mua_pvalue": pvalue})

    return (
        data.groupby(group_cols, sort=False)
        .apply(
            _session_corr,
            include_groups=False,
        )
        .reset_index()
    )


def compute_non_local_event_context(
    data: pd.DataFrame,
    theta_power_threshold: float = 2.0,
    group_cols: list[str] | None = None,
    event_col: str = "non_local_event",
) -> pd.DataFrame:
    """Classify events by neural context (SWR, theta, neither).

    Each event is classified as:
    - **SWR-associated**: strict positive-duration overlap with a
      whole-event-speed-eligible sharp-wave ripple
    - **Theta-associated**: high theta power (no SWR overlap)
    - **Neither**: no SWR, low theta power

    For theta-associated events, computes circular statistics on theta phase
    to check whether activity clusters at specific theta phases.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame, read through the SWR source, which holds
        its event labels to the whole-event speed rule and the Non-Local
        probability contract.
    theta_power_threshold : float, optional
        Z-score threshold for high theta power.
    group_cols : list[str] | None, optional
        Session grouping columns.
    event_col : str, optional
        Event label column, by default ``"non_local_event"``.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``n_events``,
        ``frac_swr``, ``frac_theta``, ``frac_neither``, ``theta_mrl``,
        ``theta_mean_angle``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)
    # Event-based: the read zeroes every event label in excluded time and
    # renormalizes ``theta_power_zscore`` outside the artifact mask, so the
    # events here are the analysed ones. The rows are kept: removing some
    # could only cut an event, never drop it.
    _analysed_rows(data)

    event_group_cols = group_cols + [event_col]
    non_local = data.loc[data[event_col] > 0]

    if non_local.empty:
        return pd.DataFrame(
            columns=group_cols
            + [
                "n_events",
                "frac_swr",
                "frac_theta",
                "frac_neither",
                "theta_mrl",
                "theta_mean_angle",
            ]
        )

    def _circular_mean(phases: pd.Series) -> float:
        """Circular mean via complex exponentials.

        Drops non-finite phases before averaging. ``_nearest`` in
        ``data_aggregation`` returns NaN when no theta sample is
        within tolerance of an event timebin; without this filter,
        ``np.exp(1j * NaN) = NaN`` and ``np.mean`` propagates NaN to
        the per-event mean phase. That single NaN then poisons the
        session-level ``np.mean(np.exp(1j * phases))``, masking
        otherwise-valid theta locking across every other event in
        the session. Returns NaN only if no finite phase exists.
        """
        arr = phases.to_numpy()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.angle(np.mean(np.exp(1j * finite))))

    event_summary = (
        non_local.groupby(event_group_cols, sort=False)
        .agg(
            mean_theta_power=("theta_power_zscore", "mean"),
            circular_mean_theta_phase=("theta_phase", _circular_mean),
        )
        .reset_index()
    )

    # Strict positive-duration interval overlap with any SWR. The
    # previous label-based ``(SWR_event > 0).any()`` aggregator counted
    # boundary-touch contact (a single shared timestamp from the
    # inclusive ``get_labels`` writer) as overlap, which conflicts
    # with ``classify_swr_overlap`` / ``get_overlap_times`` and
    # inflates ``frac_swr``. Use the shared event-bound overlap helper
    # so this stays consistent with the rest of the SWR-overlap path.
    from continuum_swr_replay.analysis.events.detection import event_strict_overlap_keys

    swr_overlap_keys = (
        event_strict_overlap_keys(
            data,
            event_col,
            "SWR_event",
            session_cols=tuple(group_cols),
        )
        if "SWR_event" in data.columns
        else set()
    )
    if swr_overlap_keys:
        keys_iter = zip(
            *(event_summary[c].to_numpy() for c in group_cols),
            event_summary[event_col].to_numpy(),
            strict=True,
        )
        event_summary["has_swr"] = [tuple(k) in swr_overlap_keys for k in keys_iter]
    else:
        event_summary["has_swr"] = False

    event_summary["is_swr"] = event_summary["has_swr"]
    event_summary["is_theta"] = ~event_summary["has_swr"] & (
        event_summary["mean_theta_power"] > theta_power_threshold
    )
    event_summary["is_neither"] = ~event_summary["is_swr"] & ~event_summary["is_theta"]

    def _session_context(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        result = {
            "n_events": n,
            "frac_swr": group["is_swr"].sum() / n,
            "frac_theta": group["is_theta"].sum() / n,
            "frac_neither": group["is_neither"].sum() / n,
        }

        theta_events = group.loc[group["is_theta"]]
        # Restrict to events with a finite per-event mean phase. The
        # per-event ``_circular_mean`` drops NaN-phase samples, but a
        # whole event whose every theta-phase sample was NaN
        # (e.g. ``_nearest`` returned NaN for every bin) still emits
        # ``circular_mean_theta_phase = NaN``. Without this filter,
        # ``np.mean(np.exp(1j * phases))`` propagates that NaN to the
        # session-level ``theta_mrl`` and ``theta_mean_angle``,
        # masking otherwise-valid locking across all other events in
        # the session. Apply the ``>= 3`` threshold to the count of
        # finite-phase events so the gate matches what's actually
        # being averaged.
        finite_phases = theta_events["circular_mean_theta_phase"].to_numpy()
        finite_phases = finite_phases[np.isfinite(finite_phases)]
        if finite_phases.size >= 3:
            mean_vector = np.mean(np.exp(1j * finite_phases))
            result["theta_mrl"] = float(np.abs(mean_vector))
            result["theta_mean_angle"] = float(np.angle(mean_vector))
        else:
            result["theta_mrl"] = np.nan
            result["theta_mean_angle"] = np.nan

        return pd.Series(result)

    return (
        event_summary.groupby(group_cols, sort=False)
        .apply(
            _session_context,
            include_groups=False,
        )
        .reset_index()
    )


def compute_local_state_collapse_details(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    local_prob_threshold: float = 0.5,
    sampling_frequency: float = 500.0,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Detailed characterization of local state collapse episodes.

    Identifies contiguous periods during movement where ``Local_prob`` drops
    below threshold and classifies each episode as:

    - **uncertain**: diffuse posterior (high HPD coverage) — the decoder
      doesn't know where the animal is
    - **confidently_wrong**: focused posterior (low HPD) but high error —
      the decoder is confident but wrong

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame.
    speed_threshold : float, optional
        Minimum ``head_speed`` for movement.
    local_prob_threshold : float, optional
        Threshold below which local state is considered collapsed.
    sampling_frequency : float, optional
        Sampling rate in Hz.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per collapse episode with columns from ``group_cols`` plus:
        ``episode_id``, ``start_time``, ``end_time``, ``duration_sec``,
        ``mean_hpd``, ``mean_error_cm``, ``median_nonlocal_prob``,
        ``classification``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    is_moving_full = (data["head_speed"] > speed_threshold) & _analysed_rows(data)

    if not is_moving_full.any():
        return pd.DataFrame(
            columns=group_cols
            + [
                "episode_id",
                "start_time",
                "end_time",
                "duration_sec",
                "mean_hpd",
                "mean_error_cm",
                "median_nonlocal_prob",
                "classification",
            ]
        )

    all_episodes = []

    for keys, group in data.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Label on the FULL session timeline so immobile rows and excluded
        # rows act as gaps that break collapse runs. Filtering first treated
        # 5/0/5 cm/s as one contiguous moving stretch and merged
        # separate 1-bin episodes.
        analysed = _analysed_rows(group)
        is_moving = (group["head_speed"] > speed_threshold).to_numpy() & analysed
        collapsed = (group["Local_prob"] < local_prob_threshold).to_numpy() & is_moving
        labeled, n_episodes = label(collapsed)

        if n_episodes == 0:
            continue

        session_median_hpd = group.loc[analysed, "hpd"].median()
        error = group["ahead_behind_distance"].abs().to_numpy()
        nonlocal_prob = (
            group["Non-Local_Continuous_prob"] + group["Non-Local_Fragmented_prob"]
        ).to_numpy()
        hpd = group["hpd"].to_numpy()
        times = group.index.to_numpy()

        for i, run in enumerate(_run_slices(labeled), start=1):
            episode_times = times[run]
            episode_hpd = hpd[run]
            episode_error = error[run]
            episode_nonlocal = nonlocal_prob[run]

            mean_hpd = float(np.nanmean(episode_hpd))
            mean_error = float(np.nanmean(episode_error))

            classification = (
                "uncertain" if mean_hpd > session_median_hpd else "confidently_wrong"
            )

            episode = dict(zip(group_cols, keys, strict=True))
            episode.update(
                {
                    "episode_id": i,
                    "start_time": float(episode_times[0]),
                    "end_time": float(episode_times[-1]),
                    "duration_sec": float(episode_times[-1] - episode_times[0])
                    + 1.0 / sampling_frequency,
                    "mean_hpd": mean_hpd,
                    "mean_error_cm": mean_error,
                    "median_nonlocal_prob": float(np.nanmedian(episode_nonlocal)),
                    "classification": classification,
                }
            )
            all_episodes.append(episode)

    if not all_episodes:
        return pd.DataFrame(
            columns=group_cols
            + [
                "episode_id",
                "start_time",
                "end_time",
                "duration_sec",
                "mean_hpd",
                "mean_error_cm",
                "median_nonlocal_prob",
                "classification",
            ]
        )

    return pd.DataFrame(all_episodes)


def compute_calibration_curve(
    data: pd.DataFrame,
    n_bins: int = 10,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calibration: decoding error should decrease as local probability increases.

    Bins timebins by ``Local_prob`` decile and computes median error per bin.
    A well-calibrated decoder shows monotonically decreasing error with
    increasing confidence. Also computes a linear regression slope of error
    vs. probability (should be negative).

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame.
    n_bins : int, optional
        Number of bins for ``Local_prob``.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per (session, prob_bin) with columns: ``prob_bin_center``,
        ``median_error_cm``, ``n_bins_in_group``, ``calibration_slope``,
        ``is_monotonic_decreasing``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    needed_cols = group_cols + ["Local_prob", "ahead_behind_distance"]
    working = data.loc[_analysed_rows(data), needed_cols].copy()
    # Drop rows with NaN ``Local_prob`` BEFORE binning. ``np.digitize``
    # returns ``len(bin_edges)`` for NaN inputs, which after the
    # ``.clip(0, n_bins-1)`` below maps every NaN row into the
    # *highest* probability bin (e.g. ``prob_bin_center=0.95``).
    # Decoder errors at NaN-Local_prob timepoints (often crashes /
    # data drops) would then dominate the highest bin's median error.
    working = working.dropna(subset=["Local_prob"])
    working["error_cm"] = working["ahead_behind_distance"].abs()
    # Also drop rows where the decoded error is NaN (e.g. timebins
    # outside the linearized track). Otherwise a probability bin
    # that happens to contain only NaN-error rows yields
    # ``median_error_cm = NaN`` with ``n_bins_in_group = 0``, which
    # then poisons ``stats.linregress`` (NaN slope) and
    # ``np.diff(errors) <= 0`` (False monotonicity). A session with
    # otherwise usable calibration bins loses the diagnostic and
    # ``compute_session_quality_summary``'s
    # ``flag_poor_calibration`` silently never trips. Drop the
    # NaN-error rows before binning so each emitted bin reflects
    # actual decoded errors.
    working = working.dropna(subset=["error_cm"])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    working["prob_bin"] = np.digitize(working["Local_prob"], bin_edges[1:-1])
    working["prob_bin"] = working["prob_bin"].clip(0, n_bins - 1)
    working["prob_bin_center"] = bin_centers[working["prob_bin"]]

    binned_group_cols = group_cols + ["prob_bin", "prob_bin_center"]
    binned = (
        working.groupby(binned_group_cols, sort=False)["error_cm"]
        .agg(median_error_cm="median", n_bins_in_group="count")
        .reset_index()
    )

    def _add_session_stats(group: pd.DataFrame) -> pd.DataFrame:
        sorted_group = group.sort_values("prob_bin_center")
        errors = sorted_group["median_error_cm"].to_numpy()

        if len(errors) >= 2:
            slope, _, _, _, _ = stats.linregress(
                sorted_group["prob_bin_center"], errors
            )
            diffs = np.diff(errors)
            is_monotonic = bool(np.all(diffs <= 0))
        else:
            slope = np.nan
            is_monotonic = True

        group = group.copy()
        group["calibration_slope"] = slope
        group["is_monotonic_decreasing"] = is_monotonic
        return group

    return (
        binned.groupby(group_cols, sort=False)
        .apply(
            _add_session_stats,
            include_groups=False,
        )
        .reset_index()
    )


def compute_session_quality_summary(
    data: pd.DataFrame,
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    local_prob_threshold: float = 0.5,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    theta_power_threshold: float = 2.0,
    sampling_frequency: float = 500.0,
    error_flag_threshold_cm: float = 15.0,
    local_fraction_flag_threshold: float = 0.5,
    swr_overlap_flag_threshold: float = 0.2,
    bout_dominant_flag_threshold: float = 0.15,
    running_error_p90_flag_threshold_cm: float = 175.0,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate all quality metrics into a single per-session summary.

    Calls all individual metric functions and merges results into one row per
    session. Continuous metric values are the primary output; boolean flags
    are added as a convenience for identifying problematic sessions.

    Parameters
    ----------
    data : pd.DataFrame
        Concatenated ``data_over_time`` from all sessions.
    speed_threshold : float, optional
        Movement speed threshold (cm/s).
    local_prob_threshold : float, optional
        Local probability threshold.
    non_local_threshold : float, optional
        Non-local probability threshold for event detection.
    theta_power_threshold : float, optional
        Z-score threshold for high theta power.
    sampling_frequency : float, optional
        Sampling rate in Hz.
    error_flag_threshold_cm : float, optional
        Median error above this triggers ``flag_high_error``.
    local_fraction_flag_threshold : float, optional
        Local fraction below this triggers ``flag_local_collapse``.
    swr_overlap_flag_threshold : float, optional
        SWR overlap fraction below this triggers ``flag_low_swr_overlap``.
    bout_dominant_flag_threshold : float, optional
        Fraction of running bouts dominated by non-local state above which
        ``flag_bout_nonlocal_dominant`` triggers. Catches localized decoder
        failures that are hidden in session-level state means. The default
        (0.15) was calibrated against the dataset: the median across the
        sessions it was calibrated on was ~6% and 0.15 flagged the worst ~10%.
    running_error_p90_flag_threshold_cm : float, optional
        p90 of running-local decoding error above which
        ``flag_running_error_tail`` triggers. Catches catastrophic episodic
        errors that are hidden in the median. The default (175 cm) flags
        the worst ~10% of sessions; the dataset median is ~97 cm because
        the decoder has a baseline tail of ~1% of local-state bins decoded
        far from the animal on every session.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with all continuous metric columns plus:
        ``flag_high_error``, ``flag_local_collapse``,
        ``flag_nospike_wrong_sign``, ``flag_poor_calibration``,
        ``flag_low_swr_overlap``, ``n_flags``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)
    # Each metric below applies the rule itself (point-wise or as gaps); this
    # only refuses a frame not read through the source before any work.
    _analysed_rows(data)

    error_df = compute_local_decoding_error(
        data,
        local_prob_threshold=local_prob_threshold,
        group_cols=group_cols,
    )

    state_df = compute_state_fraction_during_movement(
        data,
        speed_threshold=speed_threshold,
        local_prob_threshold=local_prob_threshold,
        sampling_frequency=sampling_frequency,
        group_cols=group_cols,
    )

    nonlocal_move_df = compute_non_local_during_movement(
        data,
        speed_threshold=speed_threshold,
        non_local_threshold=non_local_threshold,
        sampling_frequency=sampling_frequency,
        group_cols=group_cols,
    )

    behavior_state_df = compute_behavior_state_breakdown(
        data,
        speed_threshold=speed_threshold,
        group_cols=group_cols,
    )

    bout_dominance_df = compute_bout_nonlocal_dominance(
        data,
        speed_threshold=speed_threshold,
        sampling_frequency=sampling_frequency,
        group_cols=group_cols,
    )

    running_error_df = compute_running_error_tail(
        data,
        speed_threshold=speed_threshold,
        local_prob_threshold=local_prob_threshold,
        group_cols=group_cols,
    )

    nospike_df = compute_no_spike_firing_correspondence(data, group_cols=group_cols)

    context_df = compute_non_local_event_context(
        data,
        theta_power_threshold=theta_power_threshold,
        group_cols=group_cols,
    )

    calib_df = compute_calibration_curve(data, group_cols=group_cols)
    calib_cols = ["calibration_slope", "is_monotonic_decreasing"]
    if calib_df.empty or any(c not in calib_df.columns for c in calib_cols):
        # ``compute_calibration_curve`` returns a strictly empty frame
        # (no per-session columns at all) when there's nothing to
        # calibrate; the downstream join would otherwise raise
        # ``KeyError: ['calibration_slope', 'is_monotonic_decreasing']``.
        calib_session = pd.DataFrame(columns=group_cols + calib_cols)
    else:
        calib_session = (
            calib_df.groupby(group_cols, sort=False).first()[calib_cols].reset_index()
        )

    summary = error_df
    for df in [
        state_df,
        nonlocal_move_df,
        behavior_state_df,
        bout_dominance_df,
        running_error_df,
        nospike_df,
        context_df,
        calib_session,
    ]:
        if not df.empty:
            summary = summary.merge(df, on=group_cols, how="outer")

    # Safe flag computation: missing columns get NaN -> False
    def _flag(col: str, condition) -> pd.Series:
        if col not in summary.columns:
            return pd.Series(False, index=summary.index)
        return condition(summary[col]).fillna(False)

    summary["flag_high_error"] = _flag(
        "median_error_cm",
        lambda x: x > error_flag_threshold_cm,
    )
    summary["flag_local_collapse"] = _flag(
        "local_fraction",
        lambda x: x < local_fraction_flag_threshold,
    )
    summary["flag_nospike_wrong_sign"] = _flag(
        "nospike_mua_rho",
        lambda x: x > 0,
    )
    summary["flag_poor_calibration"] = _flag(
        "calibration_slope",
        lambda x: x >= 0,
    )
    summary["flag_low_swr_overlap"] = _flag(
        "frac_swr",
        lambda x: x < swr_overlap_flag_threshold,
    )
    summary["flag_bout_nonlocal_dominant"] = _flag(
        "frac_running_bouts_nonlocal_dominant",
        lambda x: x > bout_dominant_flag_threshold,
    )
    summary["flag_running_error_tail"] = _flag(
        "p90_running_local_error_cm",
        lambda x: x > running_error_p90_flag_threshold_cm,
    )

    flag_cols = [c for c in summary.columns if c.startswith("flag_")]
    summary["n_flags"] = summary[flag_cols].sum(axis=1).astype(int)

    return summary
