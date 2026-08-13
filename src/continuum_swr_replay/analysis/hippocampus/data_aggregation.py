"""Aggregation of analysis results over time.

This module aggregates decoding results, posterior statistics, event detections, and
behavioral data into comprehensive time-indexed DataFrames ready for visualization
and statistical analysis. It serves as the bridge between raw decoding output and
downstream analyses.

The main function, get_data_over_time, combines:
- Decoding state probabilities (local, non-local)
- Posterior statistics (mean position, variance, HPD coverage)
- Spatial variables (ahead/behind distance, MAP speed)
- Event labels (ripples, confident events, movement)
- Behavioral data (position, speed, task variables)

Main Function
-------------
get_data_over_time
    Primary aggregation function that combines all analysis components into a single
    time-aligned DataFrame. Returns both the aggregated DataFrame and the conditional
    non-local posterior distribution.

Aggregated Data
---------------
The output DataFrame contains time-indexed columns for:

Event labels (integer event numbers):
    - non_local_event: Non-local event number (0 if not in event)
    - SWR_event: Sharp-wave ripple event number
    - MUA_event: Multiunit activity high-synchrony event number

State probabilities (0-1):
    - Local_prob: Probability of local state
    - No-Spike_prob: Probability of no-spike state
    - Non-Local_Continuous_prob: Probability of continuous non-local state
    - Non-Local_Fragmented_prob: Probability of fragmented non-local state

Spatial statistics:
    - map_estimate: Maximum a posteriori position estimate (cm)
    - map_speed: Speed of MAP position estimate (cm/s)
    - ahead_behind_distance: Signed distance from actual to decoded position (cm)
    - hpd: Highest posterior density threshold value
    - max_coverage: Maximum HPD spatial coverage
    - max_distance: Maximum distance in posterior
    - max_linear_position: Maximum linear position

Neural signals (z-scored or phase):
    - theta_phase: Theta oscillation phase (radians)
    - theta_power_zscore: Z-scored theta power
    - SWR_zscore: Z-scored ripple consensus trace
    - MUA_zscore: Z-scored multiunit firing rate

Event counts (session-level):
    - n_non_local, n_SWR, n_MUA: Whole-event-speed-eligible event counts
    - n_non_local_SWR, n_non_local_no_SWR: Non-local overlaps with/without SWR
    - n_non_local_MUA, n_non_local_no_MUA: Non-local overlaps with/without MUA
    - n_SWR_MUA, n_SWR_no_MUA: SWR overlaps with/without MUA

Session metadata:
    - animal: Animal name (str)
    - day: Recording date (int, YYYYMMDD)
    - epoch: Epoch name (str)
    - nwb_file_name: NWB file name (str)

Position and behavioral data:
    - All columns from position_info DataFrame (position, speed, track segment, etc.)

Filtering and Processing
-------------------------
- Non-Local, SWR, and MUA cohorts require every event speed sample to be
  finite and strictly below 4 cm/s; an ineligible event is removed atomically
- Missing values are handled via nearest-neighbor reindexing with tolerance
- Z-scoring is applied to relevant continuous variables
- Event overlap detection uses interval-based logic
- All data is aligned to the same temporal resolution (default 500 Hz)

Notes
-----
- All data is time-indexed in seconds
- Non-local threshold typically set to 0.8 for classification
- Ahead/behind distance uses graph shortest path (can be negative)
- MAP speed is smoothed before computation to reduce noise
- Missing behavioral data is handled via nearest-neighbor alignment

See Also
--------
continuum_swr_replay.analysis.hippocampus.posterior_analysis : Posterior statistics
continuum_swr_replay.analysis.hippocampus.event_detection : Event detection functions
continuum_swr_replay.data_loaders.bandit_task : Session data loading

Examples
--------
>>> from continuum_swr_replay.analysis.hippocampus import load_results, get_data_over_time
>>> results, detector, env = load_results("chimi20200212_.nwb", "02_r1")
>>> data_df, posterior = get_data_over_time(
...     data, results, env, "chimi20200212_.nwb", "02_r1",
...     non_local_threshold=0.80
... )
>>> # Access aggregated data
>>> data_df[["non_local_event", "ahead_behind_distance", "map_speed"]].head()
>>> # Access state probabilities
>>> data_df[["Local_prob", "Non-Local_Continuous_prob"]].head()

"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector.environment import Environment
from scipy.stats import zscore

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename

from ...types import SessionData
from .event_detection import (
    event_strict_overlap_keys,
    get_confident_times,
    get_overlap_times,
)
from .event_immobility import (
    EventDefinitionError,
    build_speed_eligible_event_cohort,
    filter_event_labels_by_speed,
    validate_nonlocal_event_probability_contract,
)
from .posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_step_distance,
    get_posterior_stats,
)


def theta_power_zscore(theta_power: np.ndarray) -> np.ndarray:
    """Z-score a theta-power trace, ignoring NaN entries.

    ``_nearest``-interpolated traces can leave NaN at any timebin
    where no LFP sample fell within tolerance. The default
    ``scipy.stats.zscore(...)`` would propagate that NaN and zero out
    the entire session's z-score, silently suppressing theta event
    classification downstream in
    :func:`quality_metrics.compute_non_local_event_context`. Pass
    ``nan_policy="omit"`` so finite values keep finite z-scores.
    Shared between :func:`get_data_over_time` (this module) and the
    ContFrag aggregator so a regression in either path fails the
    same regression test.
    """
    return np.asarray(zscore(np.asarray(theta_power), nan_policy="omit"))


def validate_required_session_data_keys(
    data: Mapping[str, object],
    required_keys: list[str],
) -> None:
    """Refuse to aggregate when ``data`` is missing required keys.

    Both ``get_data_over_time`` (this module) and
    :func:`continuum_swr_replay.analysis.contfrag.data_aggregation.get_contfrag_data_over_time`
    require the same set of session-data keys (position, theta,
    ripple, MUA, …). Routing both call sites through this helper
    keeps the contract — and the error message — in one place so a
    test can pin the validator behavior directly without
    instantiating the full aggregation pipeline.
    """
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise KeyError(
            f"Missing required data keys: {missing_keys}. "
            "Ensure load_data() was called with "
            "DataLoadingPreset.DECODING or equivalent."
        )


def validate_results_position_time_alignment(
    results_time: np.ndarray,
    position_index: pd.Index,
) -> None:
    """Refuse to attach decoder outputs to position rows on a
    different timebase.

    The aggregations in this module and in
    :mod:`continuum_swr_replay.analysis.contfrag.data_aggregation`
    build ``data_over_time`` on ``position_info.index`` while pulling
    column values from ``results.time``-indexed arrays. A length-only
    check passes when the two axes share row count but are time-
    shifted, silently attaching event labels / probabilities / MAP
    stats to the wrong position rows. Pinning the stricter
    ``np.abs(results.time - position.index).max() > 1e-6`` guard
    here — and calling it from both aggregators — keeps the two paths
    in lockstep so a regression on either path fails the same
    regression test (mirrors the alignment check used in
    ``analysis.replay._helpers._align_posterior_to_dot``).
    """
    if len(results_time) != len(position_index):
        raise ValueError(
            f"results.time ({len(results_time)}) and position_info "
            f"({len(position_index)}) have different lengths; "
            "they must be aligned 1-to-1."
        )
    drift = float(
        np.abs(
            np.asarray(results_time, dtype=float)
            - position_index.to_numpy(dtype=float)
        ).max()
    )
    if drift > 1e-6:
        raise ValueError(
            f"results.time drifts from position_info.index by up to "
            f"{drift:.3g}s — same-length axes do not imply same time "
            "origin; refusing to attach decoder outputs to position "
            "rows that don't share their timebase."
        )


def validate_dot_event_contract(
    dot: pd.DataFrame,
    *,
    event_cols: tuple[str, ...] = ("non_local_event", "SWR_event", "MUA_event"),
) -> dict[str, int]:
    """Check a ``data_over_time`` frame against the table-regen input contract.

    The table builders assume each ``data_over_time`` (``dot``) frame holds a
    single session, has a finite strictly increasing time axis, and labels each event as one
    contiguous run of timebins (``label > 0``; the builders take an event's span
    from the first and last labelled bin). A violation silently corrupts the
    regenerated tables — wrong bout assignment, an event split across a gap, or
    rows attributed to the wrong session — so this **fails closed**.

    Parameters
    ----------
    dot : pd.DataFrame
        Time-indexed per-bin frame (``get_data_over_time`` output) carrying
        ``animal`` / ``day`` / ``epoch`` and the event-label columns.
    event_cols : tuple of str
        Event-label columns to check for contiguity (absent columns are skipped).

    Returns
    -------
    dict[str, int]
        Violation counts (empty when clean). Always raises before returning if
        any fatal violation is present.

    Raises
    ------
    EventDefinitionError
        If the frame spans multiple sessions, has a non-finite/non-increasing
        time axis, or any event label is non-contiguous.
    """
    violations: dict[str, int] = {}
    for col in ("animal", "day", "epoch"):
        if col in dot.columns and dot[col].nunique(dropna=False) > 1:
            violations[f"multiple_{col}"] = int(dot[col].nunique(dropna=False))

    t = dot.index.to_numpy(dtype=float)
    n_nonfinite = int((~np.isfinite(t)).sum())
    if n_nonfinite:
        violations["nonfinite_time"] = n_nonfinite
    # Keep the established ``unsorted_time`` diagnostic key, but include equal
    # adjacent timestamps: duplicate samples are just as unsafe as reversals
    # because builders would double-count their content and event rows.
    n_unsorted = int((np.diff(t) <= 0).sum()) if t.size > 1 else 0
    if n_unsorted:
        violations["unsorted_time"] = n_unsorted

    for col in event_cols:
        if col not in dot.columns:
            continue
        lab = dot[col].to_numpy()
        # One pass per column: group positions of labelled bins by label. The
        # positions are ascending (flatnonzero), so per label first()/last() are
        # its min/max bin index; a contiguous run has last - first + 1 == count.
        pos = np.flatnonzero(lab > 0)
        if pos.size:
            grp = pd.Series(pos).groupby(lab[pos])
            span = grp.last() - grp.first() + 1
            noncontig = int((span.to_numpy() != grp.size().to_numpy()).sum())
        else:
            noncontig = 0
        if noncontig:
            violations[f"noncontiguous_{col}"] = noncontig

    if violations:
        raise EventDefinitionError(
            f"validate_dot_event_contract: input-contract violations {violations}"
        )
    return violations


def get_data_over_time(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "HPC",
    sampling_frequency: int = 500,
    non_local_threshold: float = 0.8,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Aggregate all analysis data into time-indexed DataFrame.

    Combines decoding results, event detection, posterior statistics,
    and behavioral data into a single DataFrame aligned by time.

    Parameters
    ----------
    data : SessionData
        Data dictionary from load_data() containing position, spikes, etc.
    results : xr.Dataset
        Decoding results
    env : Environment
        Environment configuration from detector
    nwb_file_name : str
        Name of the NWB file (e.g., "chimi20200212_.nwb")
    epoch_name : str
        Name of the epoch (e.g., "02_r1")
    brain_area : str, optional
        Brain area for analysis, by default "HPC"
    sampling_frequency : int, optional
        Samples per second, by default 500
    non_local_threshold : float, optional
        Probability threshold for non-local classification, by default 0.8

    Returns
    -------
    data_over_time : pd.DataFrame
        Time-indexed DataFrame containing:
        - Event labels (non_local_event, SWR_event, MUA_event)
        - State probabilities (Local_prob, No-Spike_prob, etc.)
        - Posterior statistics (map_estimate, map_speed, step_distance,
          ahead_behind_distance, hpd)
        - Neural signals (theta_phase, theta_power_zscore, SWR_zscore,
          MUA_zscore)
        - Per-session run-speed baseline (run_speed_baseline, cm/s)
        - Event counts (n_non_local, n_SWR, n_MUA, etc.)
        - Session metadata (animal, day, epoch, nwb_file_name)
        - Position and behavioral variables (from position_info)
    posterior : xr.DataArray
        Conditional non-local posterior distribution

    """
    validate_required_session_data_keys(
        data,
        required_keys=[
            "position_info", "ripple_times", "hse_times",
            "theta_phase", "theta_power",
            "zscored_ripple_consensus_trace", "multiunit_rate_zscore",
        ],
    )

    # Decoder values and position samples are paired row-for-row. Validate that
    # relationship, then use the position axis for detection, speed gating,
    # label writing, and the final data-over-time index. One canonical axis
    # prevents sub-tolerance drift from changing endpoint membership between
    # those operations.
    validate_results_position_time_alignment(
        np.asarray(results.time.values), data["position_info"].index
    )
    position_time = data["position_info"].index
    head_speed = data["position_info"].head_speed.to_numpy()

    non_local_prob = results.acausal_state_probabilities.sel(
        states="Non-Local Continuous"
    ) + results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
    non_local_prob = non_local_prob.assign_coords(
        time=position_time.to_numpy(dtype=float)
    )

    posterior = get_conditional_non_local_posterior(results)

    detected_non_local_times = get_confident_times(
        non_local_prob,
        head_speed,
        position_time.to_numpy(dtype=float),
        non_local_threshold,
    )
    non_local_times, non_local_labels = build_speed_eligible_event_cohort(
        detected_non_local_times,
        speed=head_speed,
        time=position_time,
        event_col="non_local_event",
        preserve_source_ids=False,
    )
    swr_times, swr_labels = build_speed_eligible_event_cohort(
        data["ripple_times"],
        speed=head_speed,
        time=position_time,
        event_col="SWR_event",
    )
    mua_times, mua_labels = build_speed_eligible_event_cohort(
        data["hse_times"][brain_area],
        speed=head_speed,
        time=position_time,
        event_col="MUA_event",
    )

    non_local_swr_times, non_local_no_swr_times = get_overlap_times(
        non_local_times, swr_times
    )
    non_local_mua_times, non_local_no_mua_times = get_overlap_times(
        non_local_times, mua_times
    )

    swr_mua_times, swr_no_mua_times = get_overlap_times(
        swr_times, mua_times
    )
    (
        map_estimate,
        map_speed,
        ahead_behind_distance,
        hpd,
        max_coverage,
        max_distance,
        max_linear_position,
        posterior,
    ) = get_posterior_stats(
        data, env, sampling_frequency=sampling_frequency, posterior=posterior
    )
    # Raw per-timebin track-graph displacement of the MAP position (cm). Kept
    # unsmoothed so downstream summaries can sum it (path length) or take
    # robust order statistics (jumpiness); ``map_speed`` is the smoothed rate.
    step_distance = get_map_step_distance(
        posterior.values,
        env.track_graph_with_bin_centers_edges_,
        np.asarray(env.place_bin_centers_nodes_df_.node_id),
    )
    # Per-session "typical run speed" baseline (cm/s): a robust, threshold-free
    # 90th-percentile of head speed, intended as a per-session speed normalizer
    # for downstream replay-speed/compression measures. Constant across the
    # session.
    run_speed_baseline = float(
        np.nanpercentile(data["position_info"].head_speed.to_numpy(), 90)
    )
    if not np.isfinite(run_speed_baseline):
        raise ValueError("run_speed_baseline is not finite (head_speed all-NaN?)")
    animal, day = parse_nwb_filename(nwb_file_name)

    def _nearest(series):
        # Return a positional ndarray. ``reindex`` returns a Series
        # indexed by ``results.time``; embedding that Series into the
        # DataFrame below would make pandas re-align it against
        # ``position_info.index``, producing NaNs whenever the two
        # axes differ by even epsilon despite identical lengths. The
        # length guard above ensures the two are 1-to-1, so a positional
        # ndarray is safe and matches the bare-numpy columns used here.
        return (
            series.reindex(results.time, method="nearest", tolerance=0.004).to_numpy()
        )

    data_over_time = pd.DataFrame(
        {
            "non_local_event": non_local_labels.to_numpy(),
            "SWR_event": swr_labels.to_numpy(),
            "MUA_event": mua_labels.to_numpy(),
            "Local_prob": results.sel(
                states="Local"
            ).acausal_state_probabilities.squeeze(),
            "No-Spike_prob": results.sel(
                states="No-Spike"
            ).acausal_state_probabilities.squeeze(),
            "Non-Local_Continuous_prob": results.sel(
                states="Non-Local Continuous"
            ).acausal_state_probabilities.squeeze(),
            "Non-Local_Fragmented_prob": results.sel(
                states="Non-Local Fragmented"
            ).acausal_state_probabilities.squeeze(),
            "map_estimate": map_estimate.squeeze(),
            "map_speed": map_speed.squeeze(),
            "step_distance": step_distance.squeeze(),
            "ahead_behind_distance": ahead_behind_distance.squeeze(),
            "hpd": hpd.squeeze(),
            "run_speed_baseline": run_speed_baseline,
            "max_coverage": max_coverage,
            "max_distance": max_distance,
            "max_linear_position": max_linear_position,
            "theta_phase": _nearest(data["theta_phase"]).squeeze(),
            "theta_power_zscore": theta_power_zscore(
                _nearest(data["theta_power"].squeeze()).squeeze()
            ),
            "SWR_zscore": _nearest(data["zscored_ripple_consensus_trace"]).squeeze(),
            "MUA_zscore": _nearest(
                data["multiunit_rate_zscore"][brain_area].squeeze()
            ),
            "n_non_local": len(non_local_times),
            "n_SWR": len(swr_times),
            "n_MUA": len(mua_times),
            "n_non_local_SWR": len(non_local_swr_times),
            "n_non_local_no_SWR": len(non_local_no_swr_times),
            "n_non_local_MUA": len(non_local_mua_times),
            "n_non_local_no_MUA": len(non_local_no_mua_times),
            "n_SWR_MUA": len(swr_mua_times),
            "n_SWR_no_MUA": len(swr_no_mua_times),
            "animal": animal,
            "day": day,
            "epoch": epoch_name,
            "nwb_file_name": nwb_file_name,
        },
        index=data["position_info"].index,
    )

    return pd.concat((data_over_time, data["position_info"]), axis=1), posterior


def load_data_over_time(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all NonLocal processed session data.

    This loads data from the 4-state NonLocal detector only. For
    ContFrag (2-state) data, use
    :func:`continuum_swr_replay.analysis.contfrag.load_contfrag_data_over_time`.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``*_HPC_80_data_over_time.pkl`` files.

    Returns
    -------
    pd.DataFrame
        Concatenated data with derived Non-Local_prob and distance columns.

    """
    # Sort and check explicitly so an empty directory raises a
    # targeted FileNotFoundError instead of pandas' generic
    # "No objects to concatenate" — matches the contfrag loader.
    files = sorted(data_dir.glob("*_HPC_80_data_over_time.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No *_HPC_80_data_over_time.pkl files found in {data_dir}"
        )
    data_over_time = pd.concat([pd.read_pickle(p) for p in files])
    data_over_time["Non-Local_prob"] = (
        data_over_time["Non-Local_Continuous_prob"]
        + data_over_time["Non-Local_Fragmented_prob"]
    )
    data_over_time["distance"] = np.abs(data_over_time["ahead_behind_distance"])
    return data_over_time


def get_event_types(
    data_over_time: pd.DataFrame,
    event_col: str = "non_local_event",
    *,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> dict:
    """Classify events by overlap with other detectors (SWR, MUA).

    Works for any primary event column: ``non_local_event`` (model-
    detected), ``SWR_event`` (ripple-detected), or ``MUA_event``
    (high-synchrony-detected). The two "other" detector columns are
    inferred from the remaining pair among {SWR_event, MUA_event,
    non_local_event}.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data with ``event_col``, ``SWR_event``,
        ``MUA_event``, and ``head_speed`` columns.
    event_col : str, optional
        Primary event column, by default ``"non_local_event"``.
    nonlocal_threshold : float, optional
        Minimum combined Non-Local posterior required at every labeled
        bin. Must match the threshold used to create the labels — it is
        validated, not re-applied.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys use ``event_col`` as prefix, e.g. for
        ``event_col="SWR_event"``:
        ``SWR_event_immobile``, ``SWR_event_immobile_overlap_MUA_event``,
        etc. Each value is a row subset of a *copy* in which ineligible
        event labels have been set to 0 — the label columns are mutated
        relative to the input, not merely filtered. ``_immobile`` means
        the canonical whole-event rule (every sampled speed finite and
        strictly below the threshold), not the retired endpoint rule.

    Raises
    ------
    EventDefinitionError
        When the saved labels violate the probability contract.
        Regenerate the data-over-time file.

    """
    all_event_cols = {"non_local_event", "SWR_event", "MUA_event"}
    if event_col not in all_event_cols:
        raise ValueError(
            f"event_col must be one of {sorted(all_event_cols)}; got {event_col!r}"
        )
    other_cols = sorted(all_event_cols - {event_col})

    if "non_local_event" in data_over_time.columns:
        validate_nonlocal_event_probability_contract(
            data_over_time,
            probability_threshold=nonlocal_threshold,
        )
    present_event_cols = sorted(all_event_cols.intersection(data_over_time.columns))
    eligible_data = filter_event_labels_by_speed(
        data_over_time,
        present_event_cols,
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )
    immobile = eligible_data.loc[eligible_data[event_col] > 0]

    result: dict[str, pd.DataFrame] = {
        f"{event_col}_immobile": immobile,
    }

    # Canonically eligible events for each "other" detector.
    for oc in other_cols:
        if oc in data_over_time.columns:
            result[f"{oc}_events"] = eligible_data.loc[
                eligible_data[oc] > 0
            ]

    # Overlap / no-overlap with each "other" detector. Use strict
    # positive-duration interval overlap rather than per-timebin label
    # presence; otherwise events whose intervals only touch at a
    # boundary timestamp (where ``get_labels`` writes both labels
    # inclusively) get classified as overlapping. Matches the strict
    # semantics used by ``classify_swr_overlap`` /
    # ``get_overlap_times``.
    def _strict_overlap_mask(df: pd.DataFrame, primary: str, other: str) -> np.ndarray:
        keys = event_strict_overlap_keys(eligible_data, primary, other)
        if not keys:
            return np.zeros(len(df), dtype=bool)
        return np.array(
            [
                (a, d, e, p) in keys
                for a, d, e, p in zip(
                    df["animal"].to_numpy(),
                    df["day"].to_numpy(),
                    df["epoch"].to_numpy(),
                    df[primary].to_numpy(),
                    strict=True,
                )
            ]
        )

    for oc in other_cols:
        if oc not in data_over_time.columns:
            continue
        overlap_mask = _strict_overlap_mask(immobile, event_col, oc)
        result[f"{event_col}_immobile_overlap_{oc}"] = immobile.loc[overlap_mask]
        result[f"{event_col}_immobile_no_overlap_{oc}"] = immobile.loc[~overlap_mask]

    # No overlap with ANY other detector
    if all(oc in data_over_time.columns for oc in other_cols):
        any_overlap = np.zeros(len(immobile), dtype=bool)
        for oc in other_cols:
            any_overlap |= _strict_overlap_mask(immobile, event_col, oc)
        result[f"{event_col}_immobile_no_overlap"] = immobile.loc[~any_overlap]

    # Other-detector events that do NOT overlap the primary
    for oc in other_cols:
        if oc not in data_over_time.columns:
            continue
        oc_events = eligible_data.loc[eligible_data[oc] > 0]
        if len(oc_events) == 0:
            result[f"{oc}_events_no_{event_col}"] = oc_events
            continue
        overlap_keys = event_strict_overlap_keys(eligible_data, oc, event_col)
        if not overlap_keys:
            result[f"{oc}_events_no_{event_col}"] = oc_events
            continue
        oc_keys = list(
            zip(
                oc_events["animal"].to_numpy(),
                oc_events["day"].to_numpy(),
                oc_events["epoch"].to_numpy(),
                oc_events[oc].to_numpy(),
                strict=True,
            )
        )
        no_overlap_mask = np.array([k not in overlap_keys for k in oc_keys])
        result[f"{oc}_events_no_{event_col}"] = oc_events.loc[no_overlap_mask]

    return result


def compute_swr_overlap_class_counts(
    classified_events: pd.DataFrame,
) -> dict[str, int]:
    """Count immobile non-local events by SWR-overlap class.

    Operates on the output of
    :func:`continuum_swr_replay.analysis.hippocampus.event_table.build_classified_nonlocal_event_table`,
    restricted to immobile events (``whole_event_speed_eligible``), and
    tallies the ``overlap_class`` column. The single overlap convention
    (strict positive-duration overlap, "contained" = some single SWR spans
    the event) lives in that builder, so this is a thin tally with no
    classification logic of its own.

    Parameters
    ----------
    classified_events : pd.DataFrame
        Per-non-local-event table with ``overlap_class`` and
        ``whole_event_speed_eligible`` columns. May be the builder's
        zero-row empty frame.

    Returns
    -------
    dict[str, int]
        Mapping ``{"swr_contained": n, "swr_partial": n,
        "non_swr": n, "total": n}``.

    """
    zero = {"swr_contained": 0, "swr_partial": 0, "non_swr": 0, "total": 0}
    if classified_events.empty:
        return zero
    # ``.loc`` + ``astype(bool)`` so the builder's zero-row empty frame
    # (where ``whole_event_speed_eligible`` is object-dtype) still indexes
    # cleanly rather than raising on a non-boolean mask.
    immobile = classified_events.loc[
        classified_events["whole_event_speed_eligible"].astype(bool)
    ]
    if immobile.empty:
        return zero
    counts = immobile["overlap_class"].value_counts().to_dict()
    return {
        "swr_contained": int(counts.get("swr_contained", 0)),
        "swr_partial": int(counts.get("swr_partial", 0)),
        "non_swr": int(counts.get("non_swr", 0)),
        "total": int(len(immobile)),
    }


def compute_swr_nl_event_counts(
    data_over_time: pd.DataFrame,
    event_col_a: str = "non_local_event",
    event_col_b: str = "SWR_event",
    *,
    immobile_col_a: bool = True,
    nonlocal_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
) -> dict[str, int]:
    """Event-level totals and overlap for two event detectors.

    Designed as the data source for a 2-circle Venn diagram. Reports
    four numbers: total count for each detector and how many of each
    overlap with the other.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated data with both event columns, ``head_speed``, and
        session group columns.
    event_col_a : str, optional
        First event column, by default ``"non_local_event"``.
    event_col_b : str, optional
        Second event column, by default ``"SWR_event"``.
    immobile_col_a : bool, optional
        Deprecated compatibility argument. It must remain ``True`` because all
        event sources now share the canonical whole-event speed criterion.
    nonlocal_threshold : float, optional
        Minimum combined Non-Local posterior required at every labeled
        bin. Must match the threshold used to create the labels — it is
        validated, not re-applied.

    Raises
    ------
    ValueError
        When ``immobile_col_a`` is ``False``.
    EventDefinitionError
        When the saved Non-Local labels violate the probability contract.

    Returns
    -------
    dict[str, int]
        Keys: ``n_a``, ``n_a_with_b``, ``n_b``, ``n_b_with_a``. Both totals
        are speed-eligible counts, not raw detector counts.

    """
    if not immobile_col_a:
        raise ValueError(
            "immobile_col_a=False is incompatible with the unified event "
            "definition; Non-Local, SWR, and MUA events all require whole-event "
            "speed < 4 cm/s"
        )

    if "non_local_event" in (event_col_a, event_col_b):
        validate_nonlocal_event_probability_contract(
            data_over_time,
            probability_threshold=nonlocal_threshold,
        )
    eligible_data = filter_event_labels_by_speed(
        data_over_time,
        [event_col_a, event_col_b],
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )

    sess_cols = ["animal", "day", "epoch"]

    # Strict positive-duration interval overlap on per-event bounds
    # derived from the timebin labels. The previous form used
    # ``(g[other] > 0).any()`` over timebins, which counted
    # boundary-touch contact (a single shared timestamp from the
    # inclusive ``get_labels`` writer) as overlap — diverging from
    # the strict semantics enforced by ``classify_swr_overlap`` and
    # ``get_overlap_times``.
    #
    # Both totals and both overlap directions use the same globally eligible
    # populations; an ineligible event cannot reappear as the secondary
    # detector in the reverse overlap calculation.
    a_active = eligible_data.loc[eligible_data[event_col_a] > 0]
    n_a = (
        int(a_active.groupby([*sess_cols, event_col_a], sort=False).ngroups)
        if len(a_active)
        else 0
    )
    b_active = eligible_data.loc[eligible_data[event_col_b] > 0]
    n_b = (
        int(b_active.groupby([*sess_cols, event_col_b], sort=False).ngroups)
        if len(b_active)
        else 0
    )

    if n_a == 0 or n_b == 0:
        return {"n_a": n_a, "n_a_with_b": 0, "n_b": n_b, "n_b_with_a": 0}

    a_with_b_keys = event_strict_overlap_keys(
        eligible_data, event_col_a, event_col_b
    )
    n_a_with_b = len(a_with_b_keys)

    b_with_a_keys = event_strict_overlap_keys(
        eligible_data, event_col_b, event_col_a
    )
    n_b_with_a = len(b_with_a_keys)

    return {
        "n_a": n_a,
        "n_a_with_b": n_a_with_b,
        "n_b": n_b,
        "n_b_with_a": n_b_with_a,
    }
