"""Quality metrics for hippocampal decoding analysis.

This module provides functions for evaluating the quality of clusterless
hippocampal decoding results. Metrics assess whether the decoder accurately
tracks the animal's position during local states, whether non-local events
occur in biologically expected contexts (SWR, theta sequences), and whether
state classifications are internally consistent.

All functions operate on the ``data_over_time`` DataFrame produced by
:func:`~continuum_swr_replay.analysis.hippocampus.get_data_over_time`,
which is already saved as pickle files for all sessions.

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
>>> import pandas as pd
>>> from pathlib import Path
>>> # Load all sessions
>>> data = pd.concat(
...     [pd.read_pickle(p) for p in Path("Processed-Data").glob("*_HPC_80_data_over_time.pkl")]
... )
>>> summary = compute_session_quality_summary(data)
>>> print(summary[["animal", "day", "epoch", "median_error_cm", "local_fraction", "n_flags"]])

"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label

logger = logging.getLogger(__name__)

_DEFAULT_GROUP_COLS = ["animal", "day", "epoch"]


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

    local_mask = data["Local_prob"] > local_prob_threshold
    local_data = data.loc[local_mask].copy()
    local_data["error_cm"] = local_data["ahead_behind_distance"].abs()

    if local_data.empty:
        return pd.DataFrame(
            columns=group_cols + ["median_error_cm", "p90_error_cm", "mean_error_cm", "n_local_bins"]
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
    speed_threshold: float = 4.0,
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

    moving = data.loc[data["head_speed"] > speed_threshold]

    if moving.empty:
        return pd.DataFrame(
            columns=group_cols + [
                "local_fraction", "nospike_fraction",
                "nonlocal_continuous_fraction", "nonlocal_fragmented_fraction",
                "n_moving_bins", "max_collapse_duration_sec", "n_collapse_episodes",
            ]
        )

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        result = {
            "local_fraction": group["Local_prob"].mean(),
            "nospike_fraction": group["No-Spike_prob"].mean(),
            "nonlocal_continuous_fraction": group["Non-Local_Continuous_prob"].mean(),
            "nonlocal_fragmented_fraction": group["Non-Local_Fragmented_prob"].mean(),
            "n_moving_bins": n,
        }

        collapsed = (group["Local_prob"] < local_prob_threshold).to_numpy()
        times = group.index.to_numpy()
        labeled, n_episodes = label(collapsed)
        if n_episodes > 0:
            episode_durations = np.array([
                times[labeled == i][-1] - times[labeled == i][0]
                + 1.0 / sampling_frequency
                for i in range(1, n_episodes + 1)
            ])
            result["max_collapse_duration_sec"] = float(episode_durations.max())
            result["n_collapse_episodes"] = n_episodes
        else:
            result["max_collapse_duration_sec"] = 0.0
            result["n_collapse_episodes"] = 0

        return pd.Series(result)

    return moving.groupby(group_cols, sort=False).apply(
        _session_stats, include_groups=False,
    ).reset_index()


def compute_non_local_during_movement(
    data: pd.DataFrame,
    speed_threshold: float = 4.0,
    non_local_threshold: float = 0.8,
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

    moving = data.loc[data["head_speed"] > speed_threshold]

    if moving.empty:
        return pd.DataFrame(
            columns=group_cols + [
                "nonlocal_movement_rate_per_min",
                "nonlocal_movement_mean_duration_ms",
                "nonlocal_movement_median_duration_ms",
                "nonlocal_movement_fraction",
                "n_nonlocal_movement_events",
            ]
        )

    def _session_stats(group: pd.DataFrame) -> pd.Series:
        nonlocal_prob = (
            group["Non-Local_Continuous_prob"] + group["Non-Local_Fragmented_prob"]
        )
        is_nonlocal = (nonlocal_prob > non_local_threshold).to_numpy()
        times = group.index.to_numpy()
        labeled, n_events = label(is_nonlocal)

        total_duration_min = len(group) / sampling_frequency / 60.0

        if n_events > 0:
            durations_sec = np.array([
                times[labeled == i][-1] - times[labeled == i][0]
                + 1.0 / sampling_frequency
                for i in range(1, n_events + 1)
            ])
            durations_ms = durations_sec * 1000.0
            return pd.Series({
                "nonlocal_movement_rate_per_min": n_events / total_duration_min,
                "nonlocal_movement_mean_duration_ms": durations_ms.mean(),
                "nonlocal_movement_median_duration_ms": np.median(durations_ms),
                "nonlocal_movement_fraction": is_nonlocal.sum() / len(group),
                "n_nonlocal_movement_events": n_events,
            })
        else:
            return pd.Series({
                "nonlocal_movement_rate_per_min": 0.0,
                "nonlocal_movement_mean_duration_ms": 0.0,
                "nonlocal_movement_median_duration_ms": 0.0,
                "nonlocal_movement_fraction": 0.0,
                "n_nonlocal_movement_events": 0,
            })

    return moving.groupby(group_cols, sort=False).apply(
        _session_stats, include_groups=False,
    ).reset_index()


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

    local_data = data.loc[data["Local_prob"] > local_prob_threshold].copy()
    local_data["error_cm"] = local_data["ahead_behind_distance"].abs()

    if local_data.empty:
        return pd.DataFrame(
            columns=group_cols + ["track_segment_id", "median_error_cm", "n_bins", "segment_fraction"]
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
    segment_stats["segment_fraction"] = segment_stats["n_bins"] / segment_stats["session_total_bins"]
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

    def _session_corr(group: pd.DataFrame) -> pd.Series:
        nospike = group["No-Spike_prob"]
        mua = group["MUA_zscore"]

        valid = nospike.notna() & mua.notna()
        if valid.sum() < 10:
            return pd.Series({"nospike_mua_rho": np.nan, "nospike_mua_pvalue": np.nan})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.ConstantInputWarning)
            rho, pvalue = stats.spearmanr(nospike[valid], mua[valid])
        return pd.Series({"nospike_mua_rho": rho, "nospike_mua_pvalue": pvalue})

    return data.groupby(group_cols, sort=False).apply(
        _session_corr, include_groups=False,
    ).reset_index()


def compute_non_local_event_context(
    data: pd.DataFrame,
    theta_power_threshold: float = 2.0,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Classify non-local events by neural context.

    Each non-local event is classified as:
    - **SWR-associated**: overlaps with a sharp-wave ripple (``SWR_event > 0``)
    - **Theta-associated**: high theta power (no SWR overlap)
    - **Neither**: no SWR, low theta power

    For theta-associated events, computes circular statistics on theta phase
    to check whether non-local activity clusters at specific theta phases
    (expected for theta sequences) rather than being uniformly distributed.

    Parameters
    ----------
    data : pd.DataFrame
        ``data_over_time`` DataFrame.
    theta_power_threshold : float, optional
        Z-score threshold for high theta power.
    group_cols : list[str] | None, optional
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with columns: ``n_non_local_events``,
        ``frac_swr``, ``frac_theta``, ``frac_neither``, ``theta_mrl``,
        ``theta_mean_angle``.

    """
    if group_cols is None:
        group_cols = list(_DEFAULT_GROUP_COLS)

    event_group_cols = group_cols + ["non_local_event"]

    non_local = data.loc[data["non_local_event"] > 0]

    if non_local.empty:
        return pd.DataFrame(
            columns=group_cols + [
                "n_non_local_events", "frac_swr", "frac_theta", "frac_neither",
                "theta_mrl", "theta_mean_angle",
            ]
        )

    def _circular_mean(phases: pd.Series) -> float:
        """Circular mean via complex exponentials."""
        return float(np.angle(np.mean(np.exp(1j * phases.to_numpy()))))

    event_summary = non_local.groupby(event_group_cols, sort=False).agg(
        has_swr=("SWR_event", lambda x: (x > 0).any()),
        mean_theta_power=("theta_power_zscore", "mean"),
        circular_mean_theta_phase=("theta_phase", _circular_mean),
    ).reset_index()

    event_summary["is_swr"] = event_summary["has_swr"]
    event_summary["is_theta"] = (
        ~event_summary["has_swr"]
        & (event_summary["mean_theta_power"] > theta_power_threshold)
    )
    event_summary["is_neither"] = ~event_summary["is_swr"] & ~event_summary["is_theta"]

    def _session_context(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        result = {
            "n_non_local_events": n,
            "frac_swr": group["is_swr"].sum() / n,
            "frac_theta": group["is_theta"].sum() / n,
            "frac_neither": group["is_neither"].sum() / n,
        }

        theta_events = group.loc[group["is_theta"]]
        if len(theta_events) >= 3:
            phases = theta_events["circular_mean_theta_phase"].to_numpy()
            mean_vector = np.mean(np.exp(1j * phases))
            result["theta_mrl"] = float(np.abs(mean_vector))
            result["theta_mean_angle"] = float(np.angle(mean_vector))
        else:
            result["theta_mrl"] = np.nan
            result["theta_mean_angle"] = np.nan

        return pd.Series(result)

    return event_summary.groupby(group_cols, sort=False).apply(
        _session_context, include_groups=False,
    ).reset_index()


def compute_local_state_collapse_details(
    data: pd.DataFrame,
    speed_threshold: float = 4.0,
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

    moving = data.loc[data["head_speed"] > speed_threshold]

    if moving.empty:
        return pd.DataFrame(
            columns=group_cols + [
                "episode_id", "start_time", "end_time", "duration_sec",
                "mean_hpd", "mean_error_cm", "median_nonlocal_prob",
                "classification",
            ]
        )

    all_episodes = []

    for keys, group in moving.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        collapsed = (group["Local_prob"] < local_prob_threshold).to_numpy()
        labeled, n_episodes = label(collapsed)

        if n_episodes == 0:
            continue

        session_median_hpd = group["hpd"].median()
        error = group["ahead_behind_distance"].abs()
        nonlocal_prob = group["Non-Local_Continuous_prob"] + group["Non-Local_Fragmented_prob"]
        times = group.index.to_numpy()

        for i in range(1, n_episodes + 1):
            mask = labeled == i
            episode_times = times[mask]
            episode_hpd = group["hpd"].to_numpy()[mask]
            episode_error = error.to_numpy()[mask]
            episode_nonlocal = nonlocal_prob.to_numpy()[mask]

            mean_hpd = float(np.nanmean(episode_hpd))
            mean_error = float(np.nanmean(episode_error))

            classification = (
                "uncertain" if mean_hpd > session_median_hpd else "confidently_wrong"
            )

            episode = dict(zip(group_cols, keys, strict=True))
            episode.update({
                "episode_id": i,
                "start_time": float(episode_times[0]),
                "end_time": float(episode_times[-1]),
                "duration_sec": float(episode_times[-1] - episode_times[0]) + 1.0 / sampling_frequency,
                "mean_hpd": mean_hpd,
                "mean_error_cm": mean_error,
                "median_nonlocal_prob": float(np.nanmedian(episode_nonlocal)),
                "classification": classification,
            })
            all_episodes.append(episode)

    if not all_episodes:
        return pd.DataFrame(
            columns=group_cols + [
                "episode_id", "start_time", "end_time", "duration_sec",
                "mean_hpd", "mean_error_cm", "median_nonlocal_prob",
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
    working = data[needed_cols].copy()
    working["error_cm"] = working["ahead_behind_distance"].abs()

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

    return binned.groupby(group_cols, sort=False).apply(
        _add_session_stats, include_groups=False,
    ).reset_index()


def compute_session_quality_summary(
    data: pd.DataFrame,
    speed_threshold: float = 4.0,
    local_prob_threshold: float = 0.5,
    non_local_threshold: float = 0.8,
    theta_power_threshold: float = 2.0,
    sampling_frequency: float = 500.0,
    error_flag_threshold_cm: float = 15.0,
    local_fraction_flag_threshold: float = 0.5,
    swr_overlap_flag_threshold: float = 0.2,
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

    error_df = compute_local_decoding_error(
        data, local_prob_threshold=local_prob_threshold, group_cols=group_cols,
    )

    state_df = compute_state_fraction_during_movement(
        data, speed_threshold=speed_threshold,
        local_prob_threshold=local_prob_threshold,
        sampling_frequency=sampling_frequency, group_cols=group_cols,
    )

    nonlocal_move_df = compute_non_local_during_movement(
        data, speed_threshold=speed_threshold,
        non_local_threshold=non_local_threshold,
        sampling_frequency=sampling_frequency, group_cols=group_cols,
    )

    nospike_df = compute_no_spike_firing_correspondence(data, group_cols=group_cols)

    context_df = compute_non_local_event_context(
        data, theta_power_threshold=theta_power_threshold, group_cols=group_cols,
    )

    calib_df = compute_calibration_curve(data, group_cols=group_cols)
    calib_session = (
        calib_df.groupby(group_cols, sort=False)
        .first()[["calibration_slope", "is_monotonic_decreasing"]]
        .reset_index()
    )

    summary = error_df
    for df in [state_df, nonlocal_move_df, nospike_df, context_df, calib_session]:
        if not df.empty:
            summary = summary.merge(df, on=group_cols, how="outer")

    # Safe flag computation: missing columns get NaN -> False
    def _flag(col: str, condition) -> pd.Series:
        if col not in summary.columns:
            return pd.Series(False, index=summary.index)
        return condition(summary[col]).fillna(False)

    summary["flag_high_error"] = _flag(
        "median_error_cm", lambda x: x > error_flag_threshold_cm,
    )
    summary["flag_local_collapse"] = _flag(
        "local_fraction", lambda x: x < local_fraction_flag_threshold,
    )
    summary["flag_nospike_wrong_sign"] = _flag(
        "nospike_mua_rho", lambda x: x > 0,
    )
    summary["flag_poor_calibration"] = _flag(
        "calibration_slope", lambda x: x >= 0,
    )
    summary["flag_low_swr_overlap"] = _flag(
        "frac_swr", lambda x: x < swr_overlap_flag_threshold,
    )

    flag_cols = [c for c in summary.columns if c.startswith("flag_")]
    summary["n_flags"] = summary[flag_cols].sum(axis=1).astype(int)

    return summary
